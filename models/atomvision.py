import torch
import pickle
from models.node import universal_node
from utils.model_utils import save_checkpoint
from utils.data_utils import vision_dataloader
from models.configs import VisionConfigs

configs = VisionConfigs

def image_to_patch_embeddings(weight, bias=None, patch_size=4, stride=None):
    if stride is None:
        stride = patch_size

    def forward_pass(images):
        conv2d_out = torch.nn.functional.conv2d(images, weight, bias, stride=stride)
        flattened = conv2d_out.flatten(2)

        return flattened.transpose(1, 2)

    return forward_pass

def model_and_parameters(device):
    # Model configs
    REFINEMENT_STEPS = 3
    NEURONS_PER_MINI_POOL = 512
    NUM_MINI_POOL = 2
    DROPOUT = 0.1
    NUM_CLASSES = 10
    PATCH_SIZE = 4

    node_runner, parameters, = universal_node(configs.num_mini_pool, configs.neurons_per_mini_pool, configs.dropout, configs.refinement_steps)
    parameters['conv2d_weights'] = torch.nn.Parameter(torch.zeros(configs.neurons_per_mini_pool, 1, configs.patch_size, configs.patch_size, device=device).normal_(std=0.02), requires_grad=False)
    parameters['conv2d_bias'] = torch.nn.Parameter(torch.zeros(configs.neurons_per_mini_pool, device=device).normal_(std=0.02), requires_grad=False)
    parameters['read_out'] = torch.nn.Parameter(torch.zeros((configs.neurons_per_mini_pool, configs.possible_predictions), device=device).normal_(std=0.02), requires_grad=False)

    total_parameters = sum([params.view(-1).size(-1) for _, params in parameters.items()])

    def forward_pass(input_batch, training=True):
        patch_embeddings = image_to_patch_embeddings(parameters['conv2d_weights'], parameters['conv2d_bias'])(input_batch)
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

def iterate_over_loader(dataloader, model_runner, device, optimizer=None):
    training_mode = optimizer is not None
    result_per_batch  = []
    for batched_input, batched_expected in dataloader:
        batched_input = torch.tensor(batched_input, requires_grad=True, device=device)
        batched_expected = torch.tensor(batched_expected, device=device)

        if training_mode:
            avg_loss = train_per_batch(model_runner, optimizer, batched_input, batched_expected)
            result_per_batch.append(avg_loss.item())
        else:
            model_output = model_runner(batched_input)
            avg_accuracy = (model_output.argmax(axis=-1) == batched_expected).float().mean()
            result_per_batch.append(avg_accuracy.item())

    return round(sum(result_per_batch) / len(result_per_batch), 3)

def vision_model_runner(max_epochs, batch_size, device='cuda', learning_rate=1e-3):

    with open('./datasets/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')

    model_runner, model_parameters = model_and_parameters(device)
    optimizer = torch.optim.AdamW(torch.nn.ParameterList(model_parameters.values()), lr=learning_rate)

    for epoch in range(1, max_epochs+1):
        train_loader = vision_dataloader(train_images, train_labels, batch_size=batch_size, shuffle=True)
        validation_loader = vision_dataloader(test_images, test_labels, batch_size=batch_size, shuffle=False)

        avg_loss = iterate_over_loader(train_loader, model_runner, device, optimizer)
        avg_accuracy = iterate_over_loader(validation_loader, model_runner)

        save_checkpoint('vision-checkpoints', epoch, model_parameters, optimizer)
        print(f'EPOCH: {epoch} loss {avg_loss} accuracy {avg_accuracy}')
