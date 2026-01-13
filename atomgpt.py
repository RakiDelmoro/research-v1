import math
import torch
from contextlib import nullcontext
import torch.nn.functional as func
from data_utils import fetch_data, get_batch

# Use float16/mix32 for (memory and speed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = ("bfloat16" if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else "float16")
PTDTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
ctx = (torch.autocast(device_type=DEVICE.type, dtype=PTDTYPE) if DEVICE.type == "cuda" else nullcontext())
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and DTYPE == "float16"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {DEVICE} with dtype {DTYPE}")

# Print features
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

# Training configs
CONTEXT_SIZE = 512
BATCH_SIZE = 32
MAX_EPOCHS = 3000
LEARNING_RATE = 1e-3
LOG_RESULTS_FREQ = 10

# Model configs
NUM_ITER_REFINE = 6
NUM_EMBED_DIM = 256
DROPOUT = 0.1
NUM_ATTN_HEADS = 4 
BIND_MULTIPLIER = 128
VOCAB_SIZE = 256
NUM_NEURONS = BIND_MULTIPLIER * NUM_EMBED_DIM // NUM_ATTN_HEADS

def rotary_pos_emb(device):
    def get_freq(dim, theta=10000.0, dtype=torch.float32):
        def quantize(t, q=2):
            return (t / q).floor() * q
        
        return (1.0 / (theta ** (quantize(torch.arange(0, dim, 1, dtype=dtype, device=device)) / dim)) / (2 * math.pi))

    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        cos = torch.cos(phases)
        sin = torch.sin(phases)

        return cos, sin

    def forward_pass(v, phases=None):
        if phases is None:
            inv_freq = get_freq(v.shape[-1], dtype=v.dtype)
            positions = torch.arange(0, v.shape[2], device=v.device, dtype=v.dtype).view(1, 1, -1, 1)
            phases = positions * inv_freq

        v_rot = torch.stack([-v[:, 1::2], v[:, ::2]], dim=-1).view(*v.size())
        cos, sin = phases_cos_sin(phases)
        
        return v * cos + v_rot * sin

    return forward_pass

def attention(device):
    rotary_position_emb = rotary_pos_emb(device)

    def forward_pass(query, key, value):
        assert key is query
        query_with_rope = rotary_position_emb(query)
        key_with_rope = query_with_rope

        scores = ((query_with_rope @ key_with_rope.mT) / math.sqrt(query_with_rope.shape[-1])).tril(diagonal=-1)

        return scores @ value

    return forward_pass

def model_and_parameters(device):
    # Parameters
    token_to_embedding = torch.nn.Parameter(torch.zeros((VOCAB_SIZE, NUM_EMBED_DIM), device=device).normal_(std=0.02))
    layer_norm_weight = torch.nn.Parameter(torch.empty(NUM_EMBED_DIM, device=device).normal_(std=0.02))
    layer_norm_bias = torch.nn.Parameter(torch.empty(NUM_EMBED_DIM, device=device).normal_(std=0.02))
    pre_synaptic = torch.nn.Parameter(torch.zeros((NUM_ATTN_HEADS, NUM_EMBED_DIM, NUM_NEURONS), device=device).normal_(std=0.02))
    post_synaptic = torch.nn.Parameter(torch.zeros((NUM_ATTN_HEADS, NUM_EMBED_DIM, NUM_NEURONS), device=device).normal_(std=0.02))
    bind_synaptic = torch.nn.Parameter(torch.zeros((NUM_ATTN_HEADS * NUM_NEURONS, NUM_EMBED_DIM), device=device).normal_(std=0.02))
    read_out = torch.nn.Parameter(torch.zeros((NUM_EMBED_DIM, VOCAB_SIZE), device=device).normal_(std=0.02))

    parameters = torch.nn.ParameterList([token_to_embedding, layer_norm_weight, layer_norm_bias, pre_synaptic, post_synaptic, bind_synaptic, read_out])

    def forward(token_ids, training=True):
        batch, num_tokens = token_ids.size()

        token_embeddings = token_to_embedding[token_ids].unsqueeze(1)
        token_embeddings = func.layer_norm(token_embeddings, [NUM_EMBED_DIM], layer_norm_weight, layer_norm_bias, 1e-5)

        for _ in range(NUM_ITER_REFINE):
            pre_neuron_activation = func.relu((token_embeddings @ pre_synaptic))

            neighbor_aware_act = attention(device)(pre_neuron_activation, pre_neuron_activation, token_embeddings)
            neighbor_aware_act = func.layer_norm(neighbor_aware_act, [NUM_EMBED_DIM], layer_norm_weight, layer_norm_bias, 1e-5)

            post_neuron_activation = func.relu((neighbor_aware_act @ post_synaptic))

            bind_neuron_activation = func.dropout((pre_neuron_activation * post_neuron_activation), p=0.1, training=training)
            bind_neuron_activation = bind_neuron_activation.transpose(1, 2).reshape(batch, 1, num_tokens, NUM_NEURONS*NUM_ATTN_HEADS) @ bind_synaptic
            bind_neuron_activation = func.layer_norm(bind_neuron_activation, [NUM_EMBED_DIM], layer_norm_weight, layer_norm_bias, 1e-5)

            token_embeddings = func.layer_norm((token_embeddings + bind_neuron_activation), [NUM_EMBED_DIM], layer_norm_weight, layer_norm_bias, 1e-5)

        logits = token_embeddings.view(batch, num_tokens, NUM_EMBED_DIM) @ read_out

        return logits

    return forward, parameters

def generate_tokens(model_runner, tokens, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        input_tokens = tokens
        model_output = model_runner(input_tokens)[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(model_output, min(top_k, model_output.size(-1)))
            model_output[model_output < values[:, [-1]]] = float('-inf')
        probs = func.softmax(model_output, dim=-1)
        token_next = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, token_next], dim=-1)

    return tokens

def train_per_batch(model_forward_pass, optimizer, input_batch, expected_batch):

    def calculate_loss(model_output, expected_batch):
        return func.cross_entropy(model_output.view(-1, model_output.size(-1)), expected_batch.view(-1))

    model_output = model_forward_pass(input_batch, training=True)
    loss = calculate_loss(model_output, expected_batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return loss.item()

def main():
    fetch_data()

    model_runner, model_parameters = model_and_parameters(DEVICE)
    optimizer = torch.optim.AdamW(model_parameters, lr=LEARNING_RATE)

    steps = 0
    loss_per_batch = 0
    for epoch in range(MAX_EPOCHS):
        input_batch, expected_batch = get_batch('train', BATCH_SIZE, CONTEXT_SIZE)

        with ctx:
            loss = train_per_batch(model_runner, optimizer, input_batch, expected_batch)

        steps += 1
        loss_per_batch += loss

        if epoch % LOG_RESULTS_FREQ == 0:
            print(f'Epoch: {epoch}/{MAX_EPOCHS} loss {loss_per_batch / steps:.3}')

            prompt = 'To be or '
            input_tokens = torch.tensor(bytearray(prompt, "utf-8"), dtype=torch.long, device=DEVICE).unsqueeze(0)
            ret = generate_tokens(model_runner, input_tokens, max_new_tokens=100, top_k=3)
            ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
            print(f'{GREEN}Generated{RESET}')
            print(ret_decoded)

            loss_per_batch = 0
            steps = 0

    print('Training done!')

main()
