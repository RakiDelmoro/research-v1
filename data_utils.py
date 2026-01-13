import os
import torch
import requests
import numpy as np

def fetch_data():
    if not os.path.exists('input.txt'):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open('input.txt', "w") as f:
            f.write(requests.get(data_url).text)
        
def get_batch(split, batch_size, context_size):
    data = np.memmap('input.txt', dtype=np.uint8, mode='r')
    if split == 'train':
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    
    idx = torch.randint(len(data) - context_size, (batch_size,))
    batched_x = torch.stack([torch.from_numpy((data[i : i + context_size]).astype(np.int64)) for i in idx])
    batched_y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + context_size]).astype(np.int64)) for i in idx])

    if torch.cuda.is_available():
        # pin arrays x, y which allows us to move them to GPU asynchronously
        x, y = batched_x.pin_memory().to('cuda', non_blocking=True), batched_y.pin_memory().to('cuda', non_blocking=True)
    else:
        x, y = batched_x.to('cpu'), batched_y.to('cpu')

    return x, y
