import torch
import pickle
from models.configs import SudokuConfigs
from utils.data_utils import sudoku_dataloader
from models.node import universal_node

configs = SudokuConfigs

def model_and_parameters(device):
    node_runner, parameters = universal_node(configs.num_mini_pool, configs.neurons_per_mini_pool, configs.dropout, configs.refinement_steps)

    # For text parameters
    parameters['token_embedding'] = torch.nn.Parameter(torch.zeros((configs.num_chars, configs.neurons_per_mini_pool), device=device).normal_(std=0.02))
    parameters['row_embedding'] = torch.nn.Parameter(torch.zeros((configs.row_positions, configs.neurons_per_mini_pool), device=device).normal_(std=0.02))
    parameters['col_embedding'] = torch.nn.Parameter(torch.zeros((configs.column_positions, configs.neurons_per_mini_pool), device=device).normal_(std=0.02))
    parameters['box_embedding'] = torch.nn.Parameter(torch.zeros((configs.box_positions, configs.neurons_per_mini_pool), device=device).normal_(std=0.02))
    parameters['read_out'] = torch.nn.Parameter(torch.zeros((configs.neurons_per_mini_pool, configs.possible_predictions), device=device).normal_(std=0.02))

    total_parameters = sum([params.view(-1).size(-1) for _, params in parameters.items()])

    def forward(token_ids, training=True):
        position = torch.arange(token_ids.shape[-1], device=device).unsqueeze(0).expand(token_ids.shape[0], -1)
        rows = position // configs.row_positions
        cols = position % configs.column_positions
        boxes = (rows // 3) * 3 + (cols // 3) 
        token_embeddings = parameters['token_embedding'][token_ids]
        token_embeddings = token_embeddings + parameters['row_embedding'][rows] + parameters['col_embedding'][cols] + parameters['box_embedding'][boxes]
        node_output = node_runner(token_embeddings, training)
        logits = node_output @ parameters['read_out']

        return logits

    return forward, parameters

def train_per_batch(model_forward_pass, optimizer, input_batch, expected_batch):

    def calculate_loss(model_output, expected_batch):
        loss_mask = input_batch == 0

        model_output_flat = model_output.view(-1, model_output.size(-1))
        expected_flat = expected_batch.view(-1)
        mask_flat = loss_mask.view(-1)

        loss = torch.nn.functional.cross_entropy(model_output_flat[mask_flat], expected_flat[mask_flat])
 
        return loss

    model_output = model_forward_pass(input_batch, training=True)
    loss = calculate_loss(model_output, expected_batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

def iterate_over_loader(dataloader, model_runner, device, optimizer=None):
    training_mode = optimizer is not None
    result_per_batch  = []
    for i, (batched_input, batched_output) in enumerate(dataloader):
        batched_input = torch.tensor(batched_input, device=device, dtype=torch.long)
        batched_output = torch.tensor(batched_output, device=device, dtype=torch.long)

        if training_mode:
            result = train_per_batch(model_runner, optimizer, batched_input, batched_output)
            result_per_batch.append(result.item())
        else:
            mask = batched_input == 0
            model_output = model_runner(batched_input)
            result = ((model_output.argmax(axis=-1) == batched_output) & mask).float().mean()
            result_per_batch.append(result.item())
        # print(i)
        if i % 100 == 0:
            mask = batched_input == 0
            model_output = model_runner(batched_input)
            log_accuracy = ((model_output.argmax(axis=-1) == batched_output) & mask).float().mean()
            print(f'batched {i} training_mode: {training_mode} result: {result} accuracy {log_accuracy}')
        
    return round(sum(result_per_batch) / len(result_per_batch), 3)

def sudoku_model_runner(max_epochs, batch_size, device='cuda', learning_rate=1e-3):
    with open('./datasets/sudoku_task.pkl', 'rb') as f: ((train_x, train_y), (test_x, test_y), _) = pickle.load(f, encoding='latin1')

    model_runner, model_parameters = model_and_parameters(device)
    optimizer = torch.optim.AdamW(torch.nn.ParameterList(model_parameters.values()), lr=learning_rate)

    for epoch in range(1, max_epochs+1):
        train_loader = sudoku_dataloader(train_x, train_y, batch_size=batch_size, shuffle=True)
        validation_loader = sudoku_dataloader(test_x, test_y, batch_size=batch_size, shuffle=False)

        avg_loss = iterate_over_loader(train_loader, model_runner, device, optimizer)
        avg_accuracy = iterate_over_loader(validation_loader, model_runner)

        print(f'EPOCH: {epoch} loss {avg_loss} accuracy {avg_accuracy}')

