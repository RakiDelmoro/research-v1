import math
import torch
from torch.nn.functional import relu, layer_norm, dropout

def apply_rotary_pos_emb(input_embeddings):
    def get_angular_frequency(dim, theta=10000.0, dtype=torch.float32):

        def quantize(tensor, quantum=2):
            return (tensor / quantum).floor() * quantum

        indices = torch.arange(0, dim, 1, dtype=dtype, device=input_embeddings.device)
        quantized_indices = quantize(indices)
        normalized_positions = quantized_indices / dim
        frequencies = 1.0 / (theta ** normalized_positions)
        angular_frequencies = frequencies / (2 * math.pi)

        return  angular_frequencies

    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        cos = torch.cos(phases)
        sin = torch.sin(phases)

        return cos, sin

    _, _, num_tokens, embed_dim = input_embeddings.shape
    inv_freq = get_angular_frequency(embed_dim, dtype=input_embeddings.dtype)
    positions = torch.arange(0, num_tokens, device=input_embeddings.device, dtype=input_embeddings.dtype).view(1, 1, -1, 1)
    phases = positions * inv_freq

    rotated_embeddings = torch.stack([-input_embeddings[..., 1::2], input_embeddings[..., ::2]], dim=-1).view(*input_embeddings.shape)
    cos, sin = phases_cos_sin(phases)

    return input_embeddings * cos + rotated_embeddings * sin

def attention(query, key, value):
    assert key is query
    query_with_rope = apply_rotary_pos_emb(query)
    key_with_rope = query_with_rope

    scores = ((query_with_rope @ key_with_rope.mT) / math.sqrt(query_with_rope.shape[-1]))

    return scores @ value

def universal_node(input_feature_size, num_mini_pool, neurons_per_pool, possible_predictions=10, dropout_prob=0.1, num_refine=1, device='cuda'):
    total_neurons = num_mini_pool * neurons_per_pool
    parameters = {
        'layer_norm_weight': torch.nn.Parameter(torch.empty(input_feature_size, device=device).normal_(std=0.02)),
        'layer_norm_bias': torch.nn.Parameter(torch.empty(input_feature_size, device=device).normal_(std=0.02)),
        'pre_synaptic': torch.nn.Parameter(torch.zeros((num_mini_pool, input_feature_size, neurons_per_pool), device=device).normal_(std=0.02)),
        'post_synaptic': torch.nn.Parameter(torch.zeros((num_mini_pool, input_feature_size, neurons_per_pool), device=device).normal_(std=0.02)),
        'bind_synapse': torch.nn.Parameter(torch.zeros((total_neurons, input_feature_size), device=device).normal_(std=0.02)),
    }

    feedback_params = {}
    for name, params in parameters.items():
        if params.ndim == 1: shape = (possible_predictions,) + tuple(params.shape)
        elif params.ndim == 2: shape = (possible_predictions,) + tuple([params.shape[-1]])
        else: shape = (possible_predictions,) + tuple([params.shape[0]*params.shape[-1]])
        feedback_params[name] = torch.zeros(shape, device=device)

    def forward(input_features, train_mode=True):
        batch, num_tokens, _ = input_features.shape
        normalized_features = layer_norm(input_features.unsqueeze(1), [input_feature_size], parameters['layer_norm_weight'], parameters['layer_norm_bias'], 1e-5)
        for _ in range(num_refine):
            pre_neurons_activity = relu(normalized_features @ parameters['pre_synaptic'])
            attention_output = attention(pre_neurons_activity, pre_neurons_activity, normalized_features)
            normalized_attn_output = layer_norm(attention_output, [input_feature_size], parameters['layer_norm_weight'], parameters['layer_norm_bias'], 1e-5)
            post_neurons_activity = relu(normalized_attn_output @ parameters['post_synaptic'])

            bind_neurons_activity = dropout((pre_neurons_activity * post_neurons_activity), dropout_prob, train_mode)
            bind_neurons_activity = bind_neurons_activity.reshape(batch, 1, num_tokens, num_mini_pool*neurons_per_pool) @ parameters['bind_synapse']

            normalized_features = layer_norm(bind_neurons_activity, [input_feature_size], parameters['layer_norm_weight'], parameters['layer_norm_bias'], 1e-5)

        return normalized_features.view(batch, num_tokens, input_feature_size)

    return forward, parameters
