import torch
import torch.nn.functional as func
from node import universal_node
from data_utils import fetch_data, text_get_batch
from utils import save_checkpoint, load_checkpoint

# Print features
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

# Training configs
CONTEXT_SIZE = 8
BATCH_SIZE = 32
MAX_EPOCHS = 3000
LEARNING_RATE = 1e-3
LOG_RESULTS_FREQ = 10

# Model configs
NUM_ITER_REFINE = 6
INPUT_FEATURE_SIZE = 256
NUM_MINI_POOL = 16
NEURONS_PER_MINI_POOL = 1024
DROPOUT = 0.1
NUM_CLASSES = 256

def model_and_parameters(device):
    node_runner, parameters = universal_node(INPUT_FEATURE_SIZE, NUM_MINI_POOL, NEURONS_PER_MINI_POOL, DROPOUT, NUM_ITER_REFINE, device)

    # For text parameters
    parameters['token_embedding'] = torch.nn.Parameter(torch.zeros((NUM_CLASSES, INPUT_FEATURE_SIZE), device=device).normal_(std=0.02))
    parameters['read_out'] = torch.nn.Parameter(torch.zeros((INPUT_FEATURE_SIZE, NUM_CLASSES), device=device).normal_(std=0.02))

    total_parameters = sum([params.view(-1).size(-1) for _, params in parameters.items()])

    def forward(token_ids, training=True):
        token_embeddings = parameters['token_embedding'][token_ids]
        node_output = node_runner(token_embeddings, training)
        logits = node_output @ parameters['read_out']

        return logits

    return forward, parameters

def generate_tokens(model_runner, tokens, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        input_tokens = tokens
        model_output = model_runner(input_tokens, False)[:, -1, :] / temperature
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
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def main(load_from_checkpoint=False):
    DEVICE = 'cuda'
    
    fetch_data()

    model_runner, model_parameters = model_and_parameters(DEVICE)
    optimizer = torch.optim.AdamW(torch.nn.ParameterList(model_parameters.values()), lr=LEARNING_RATE)

    if load_from_checkpoint:
        _, _ = load_checkpoint(model_parameters, 'model-checkpoints', DEVICE)

    steps = 0
    loss_per_batch = 0
    for epoch in range(MAX_EPOCHS):
        input_batch, expected_batch = text_get_batch('train', BATCH_SIZE, CONTEXT_SIZE)

        loss = train_per_batch(model_runner, optimizer, input_batch, expected_batch)

        steps += 1
        loss_per_batch += loss

        save_checkpoint('model-checkpoints', epoch, model_parameters, optimizer=optimizer)

        if epoch % LOG_RESULTS_FREQ == 0:
            print(f'Epoch: {epoch}/{MAX_EPOCHS} loss {loss_per_batch / steps:.3}')

            prompt = 'To be or '
            input_tokens = torch.tensor(bytearray(prompt, "utf-8"), dtype=torch.long, device=DEVICE).unsqueeze(0)
            ret = generate_tokens(model_runner, input_tokens, max_new_tokens=100, top_k=3)
            ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
            print(f'{GREEN}Generated{RESET}:')
            print(ret_decoded)
            loss_per_batch = 0
            steps = 0

    print('Training done!')

main()
