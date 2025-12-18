import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 和你train.py里一样的数据集定义
class TrajectoryDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        history = torch.tensor(data['history'], dtype=torch.float32)
        if history.dim() == 2:
            history = history.unsqueeze(1)
        adj = torch.tensor(data['history_graph'], dtype=torch.float32)
        last_pos = history[-1, 0, :2]
        future = torch.tensor(data['future'], dtype=torch.float32)
        return history, adj, last_pos, future

def get_in_channels(dataloader):
    for history, adj, last_pos, future in dataloader:
        print("history.shape:", history.shape)  # (B, T, N, C)
        print("Detected in_channels:", history.shape[-1])
        return

if __name__ == "__main__":
    dataset = TrajectoryDataset(os.path.expanduser("~/Desktop/processed_trajectory/train"))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    get_in_channels(loader)
