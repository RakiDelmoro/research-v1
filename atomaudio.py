import torch
import pickle
from data_utils import audio_get_batch
from node import universal_node

# Training configs
AUDIO_PATCH_SIZE = 16
BATCH_SIZE = 32
MAX_EPOCHS = 3000
LEARNING_RATE = 1e-3

# Model configs
REFINEMENT_STEPS = 1
NUM_EMBED_DIM = 256
NUM_MINI_POOL = 1
NEURONS_PER_MINI_POOL = 1024
DROPOUT = 0.1
NUM_CLASSES = 10

def audio_to_patch_embeddings(weight, bias=None, patch_size=4, stride=None):
    if stride is None:
        stride = patch_size

    def forward_pass(input_batch):
        conv2d_out = torch.nn.functional.conv2d(input_batch, weight, bias, stride=stride)
        flattened = conv2d_out.flatten(2)

        return flattened.transpose(1, 2)

    return forward_pass

def model_and_parameters(device):
    node_runner, parameters, = universal_node(NUM_EMBED_DIM, NUM_MINI_POOL, NEURONS_PER_MINI_POOL, NUM_CLASSES, DROPOUT, REFINEMENT_STEPS, device)

    # For vision parameters
    parameters['conv2d_weights'] = torch.nn.Parameter(torch.zeros(NUM_EMBED_DIM, 1, 16, 16, device=device).normal_(std=0.02), requires_grad=False)
    parameters['conv2d_bias'] = torch.nn.Parameter(torch.zeros(NUM_EMBED_DIM, device=device).normal_(std=0.02), requires_grad=False)
    parameters['read_out'] = torch.nn.Parameter(torch.zeros((NUM_EMBED_DIM, NUM_CLASSES), device=device).normal_(std=0.02))

    total_parameters = sum([params.view(-1).size(-1) for _, params in parameters.items()])

    def forward_pass(input_batch, training=True):
        patch_embeddings = audio_to_patch_embeddings(parameters['conv2d_weights'], parameters['conv2d_bias'])(input_batch.permute(0, 3, 2, 1))
        node_output = node_runner(patch_embeddings, training)
        logits = node_output.mean(1) @ parameters['read_out']

        return logits

    return forward_pass, parameters

def train_per_batch(model_runner, optimizer, input_batch, expected_batch):
    model_output = model_runner(input_batch)
    loss = torch.nn.functional.cross_entropy(model_output, expected_batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

def iterate_over_loader(dataloader, model_runner, optimizer=None):
    training_mode = optimizer is not None
    result_per_batch  = []
    for batched_input, batched_expected in dataloader:
        batched_input = torch.tensor(batched_input, requires_grad=True, device='cuda')
        batched_expected = torch.tensor(batched_expected, device='cuda')

        if training_mode:
            avg_loss = train_per_batch(model_runner, optimizer, batched_input, batched_expected)
            result_per_batch.append(avg_loss.item())
        else:
            model_output = model_runner(batched_input)
            avg_accuracy = (model_output.argmax(axis=-1) == batched_expected).float().mean()
            result_per_batch.append(avg_accuracy.item())

    return round(sum(result_per_batch) / len(result_per_batch), 3)

def main():
    with open('./datasets/audio.pkl', 'rb') as f: ((train_x, train_y), (test_x, test_y)) = pickle.load(f, encoding='latin1')

    model_runner, model_parameters = model_and_parameters(device='cuda')
    optimizer = torch.optim.AdamW(torch.nn.ParameterList(model_parameters.values()), lr=0.001)

    for epoch in range(1, 100+1):
        train_loader = audio_get_batch(train_x, train_y, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = audio_get_batch(test_x, test_y, batch_size=BATCH_SIZE, shuffle=False)

        avg_loss = iterate_over_loader(train_loader, model_runner, optimizer)
        avg_accuracy = iterate_over_loader(validation_loader, model_runner)

        print(f'EPOCH: {epoch} loss {avg_loss} accuracy {avg_accuracy}')

main()
