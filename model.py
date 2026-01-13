import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

def get_frequency(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype, device='cuda')) / n)) / (2 * math.pi))

# Configurations
NUM_ITER = 6
NUM_EMBED = 256
DROPOUT = 0.1
NUM_HEADS = 4
MLP_INTERNAL_DIM_MULTIPLIER = 128
VOCAB_SIZE = 256

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        num_heads = NUM_HEADS
        num_dim = NUM_EMBED
        self.num_neurons = MLP_INTERNAL_DIM_MULTIPLIER * num_dim // num_heads

        self.frequency = get_frequency(self.num_neurons, theta=2**16, dtype=torch.float32).view(1, 1, 1, self.num_neurons)

    def phases_cos_sin(self, phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)

        return phases_cos, phases_sin

    def rotary_pos_emb(self, phases, v):
        v_rot = torch.stack([-v[:, 1::2], v[:, ::2]], dim=-1).view(*v.size())
        phases_cos, phases_sin = self.phases_cos_sin(phases)

        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, query, key, value):
        assert self.frequency.dtype == torch.float32
        assert key is query
        _, _, T, _ = query.size()

        r_phases = (torch.arange(0, T, device=self.frequency.device, dtype=self.frequency.dtype).view(1, 1, -1, 1)) * self.frequency
        query_with_rope = self.rotary_pos_emb(r_phases, query)
        key_with_rope = query_with_rope

        scores = ((query_with_rope @ key_with_rope.mT) / math.sqrt(self.num_neurons)).tril(diagonal=-1)

        return scores @ value

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_dim = NUM_EMBED
        self.num_neurons = MLP_INTERNAL_DIM_MULTIPLIER * self.num_dim // self.num_heads

        self.encoder = nn.Parameter(torch.zeros((self.num_heads, self.num_dim, self.num_neurons)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((self.num_heads * self.num_neurons, self.num_dim)).normal_(std=0.02))

        self.attention_layer = LinearAttention()
        self.layer_norm = nn.LayerNorm(self.num_dim, elementwise_affine=False, bias=False)
        self.embeddings = nn.Embedding(VOCAB_SIZE, self.num_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.encoder_v = nn.Parameter(torch.zeros((self.num_heads, self.num_dim, self.num_neurons)).normal_(std=0.02))

        self.read_out = nn.Parameter(torch.zeros((self.num_dim, VOCAB_SIZE)).normal_(std=0.02))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        batch, num_tokens = token_ids.size()

        token_embeddings = self.embeddings(token_ids).unsqueeze(1)
        token_embeddings = self.layer_norm(token_embeddings)

        for _ in range(NUM_ITER):
            x_latent_space = token_embeddings @ self.encoder
            x_latent_space_sparse = functional.relu(x_latent_space)

            attention_features = self.attention_layer(x_latent_space_sparse, x_latent_space_sparse, token_embeddings)
            attention_features = self.layer_norm(attention_features)

            y_latent_space = attention_features @ self.encoder_v
            y_latent_space_sparse = functional.relu(y_latent_space)

            bind_latent_space = self.dropout(x_latent_space_sparse * y_latent_space_sparse)

            mlp_features = bind_latent_space.transpose(1, 2).reshape(batch, 1, num_tokens, self.num_neurons*self.num_heads) @ self.decoder

            embeddings = self.layer_norm(mlp_features)
            token_embeddings = self.layer_norm(token_embeddings + embeddings)

        logits = token_embeddings.view(batch, num_tokens, self.num_dim) @ self.read_out
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ids_condition = token_ids
            logits, _ = self(ids_condition)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')
            probs = functional.softmax(logits, dim=-1)
            ids_next = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, ids_next], dim=-1)

        return token_ids
