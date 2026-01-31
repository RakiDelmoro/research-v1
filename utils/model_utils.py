import os
import glob
import torch

def save_checkpoint(checkpoint_dir, epoch, param_dict, optimizer=None, keep_last=5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model-epoch-{epoch}.pt')

    model_state = {'epoch': epoch,
                  'parameters': {k: v.detach().cpu() for k, v in param_dict.items()}
                  }

    if optimizer is not None: model_state['optimizer'] = optimizer.state_dict()

    torch.save(model_state, checkpoint_path)

    print(f'Saved checkpoint-file: {checkpoint_path}')

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model-epoch-*.pt')), key=os.path.getatime)
    if len(checkpoint_files) > keep_last:
        for old_file in checkpoint_files[:-keep_last]:
            os.remove(old_file)

def load_checkpoint(model_parameters, checkpoint_dir, device):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model-epoch-*.pt'))
    if not checkpoint_files:
        raise FileNotFoundError('No checkpoints found')

    latest_checkpoint = max(checkpoint_files, key=os.path.getatime)
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    print(f'Loaded checkpoint from: {latest_checkpoint}')

    for name, param in model_parameters.items():
        param.data.copy_(checkpoint['parameters'][name].to(device))

    return model_parameters, checkpoint.get('epoch', 0)
