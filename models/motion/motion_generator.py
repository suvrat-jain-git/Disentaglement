import torch

def generate_motion(x):
    diff = torch.abs(x[:, 1:] - x[:, :-1])
    motion = diff.permute(0, 2, 1, 3, 4).contiguous()
    return motion
