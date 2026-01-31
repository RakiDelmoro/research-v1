import os
import torch
import requests
import numpy as np

def fetch_txt_data():
    if not os.path.exists('./datasets/input.txt'):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open('input.txt', "w") as f:
            f.write(requests.get(data_url).text)

def text_dataloader(split, batch_size, context_size):
    data = np.memmap('./datasets/input.txt', dtype=np.uint8, mode='r')
    if split == 'train':
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]

    idx = torch.randint(len(data) - context_size, (batch_size,))
    batched_input_token = torch.stack([torch.from_numpy((data[i : i + context_size]).astype(np.int64)) for i in idx])
    batched_next_token = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + context_size]).astype(np.int64)) for i in idx])

    if torch.cuda.is_available():
        # pin arrays x, y which allows us to move them to GPU asynchronously
        batched_input_token, batched_next_token = batched_input_token.pin_memory().to('cuda', non_blocking=True), batched_next_token.pin_memory().to('cuda', non_blocking=True)
    else:
        batched_input_token, batched_next_token = batched_input_token.to('cpu'), batched_next_token.to('cpu')

    return batched_input_token, batched_next_token

def vision_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_samples = img_arr.shape[0]    
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        batched_img = img_arr[indices[start_idx:end_idx]].reshape(current_batch_size, 1, 28, 28)
        batched_label = label_arr[indices[start_idx:end_idx]]

        yield batched_img, batched_label

def audio_dataloader(audio_arr, label_arr, batch_size, shuffle):
    num_samples = audio_arr.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        batched_audio_arr = audio_arr[indices[start_idx:end_idx]]
        batched_label = label_arr[indices[start_idx:end_idx]]

        yield batched_audio_arr, batched_label

def sudoku_dataloader(puzzles_arr, solutions_arr, batch_size, shuffle):
    num_puzzles = puzzles_arr.shape[0]
    indices = np.arange(num_puzzles)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_puzzles, batch_size):
        end_idx = start_idx + batch_size

        batched_puzzles = puzzles_arr[indices[start_idx:end_idx]]
        batched_solutions = solutions_arr[indices[start_idx:end_idx]]-1

        yield batched_puzzles, batched_solutions
