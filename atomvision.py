import torch
import pickle
from node import universal_node
from data_utils import image_get_batch

# Training configs
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
PATCH_SIZE = 4
BATCH_SIZE = 512
MAX_EPOCHS = 3000
LEARNING_RATE = 1e-3
LOG_RESULTS_FREQ = 10
TOTAL_IMAGE_PATCHES = (IMAGE_HEIGHT // PATCH_SIZE) * (IMAGE_WIDTH // PATCH_SIZE)

# Model configs
NUM_ITER_REFINE = 1
NUM_EMBED_DIM = 256
NUM_MINI_POOL = 1
NEURONS_PER_MINI_POOL = 512
DROPOUT = 0.1
NUM_CLASSES = 10

def image_to_patch_embeddings(weight, bias=None, patch_size=4, stride=None):
    if stride is None:
        stride = patch_size

    def forward_pass(images):
        conv2d_out = torch.nn.functional.conv2d(images, weight, bias, stride=stride)
        flattened = conv2d_out.flatten(2)

        return flattened.transpose(1, 2)

    return forward_pass

def model_and_parameters(device):
    node_runner, parameters, = universal_node(NUM_EMBED_DIM, NUM_MINI_POOL, NEURONS_PER_MINI_POOL, NUM_CLASSES, DROPOUT, NUM_ITER_REFINE, device)

    # For vision parameters
    parameters['conv2d_weights'] = torch.nn.Parameter(torch.zeros(NUM_EMBED_DIM, 1, PATCH_SIZE, PATCH_SIZE, device=device).normal_(std=0.02), requires_grad=False)
    parameters['conv2d_bias'] = torch.nn.Parameter(torch.zeros(NUM_EMBED_DIM, device=device).normal_(std=0.02), requires_grad=False)
    parameters['read_out'] = torch.nn.Parameter(torch.zeros((NUM_EMBED_DIM, NUM_CLASSES), device=device).normal_(std=0.02))

    total_parameters = sum([params.view(-1).size(-1) for _, params in parameters.items()])

    def forward_pass(images, training=True):
        patch_embeddings = image_to_patch_embeddings(parameters['conv2d_weights'], parameters['conv2d_bias'])(images)
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
    for batched_image, batched_label in dataloader:
        batched_image = torch.tensor(batched_image, requires_grad=True, device='cuda')
        batched_label = torch.tensor(batched_label, device='cuda')

        if training_mode:
            avg_loss = train_per_batch(model_runner, optimizer, batched_image, batched_label)
            result_per_batch.append(avg_loss.item())
        else:
            model_output = model_runner(batched_image)
            avg_accuracy = (model_output.argmax(axis=-1) == batched_label).float().mean()
            result_per_batch.append(avg_accuracy.item())

    return round(sum(result_per_batch) / len(result_per_batch), 3)

def main():
    with open('./datasets/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    model_runner, model_parameters = model_and_parameters(device='cuda')
    optimizer = torch.optim.AdamW(torch.nn.ParameterList(model_parameters.values()), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        train_loader = image_get_batch(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = image_get_batch(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False)

        avg_loss = iterate_over_loader(train_loader, model_runner, optimizer)
        avg_accuracy = iterate_over_loader(validation_loader, model_runner)

        print(f'EPOCH: {epoch} loss {avg_loss} accuracy {avg_accuracy}')

main()
