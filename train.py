# train.py
import os, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import TrajectoryGNNTransformer

# ==== 评估指标 ====
def ade(pred, gt): return torch.mean(torch.norm(pred-gt,dim=-1)).item()
def fde(pred, gt): return torch.mean(torch.norm(pred[:,-1]-gt[:,-1],dim=-1)).item()
def rmse(pred, gt): return torch.sqrt(torch.mean((pred-gt)**2)).item()

# ==== 数据集 ====
class TrajectoryDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".npy")]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        history = torch.tensor(data['history'],dtype=torch.float32)  # (T,N,2)
        adj = torch.tensor(data['history_graph'],dtype=torch.float32)  # (T,N,N)
        future = torch.tensor(data['future'],dtype=torch.float32)  # (future_len,2)
        last_pos = history[-1,0,:2]
        return history, adj, last_pos, future

# ==== 训练 ====
BATCH_SIZE, EPOCHS, LR = 16, 50, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TrajectoryDataset(os.path.expanduser("~/Desktop/processed_trajectory/train"))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = TrajectoryGNNTransformer().to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_ade = float('inf')
for epoch in range(1,EPOCHS+1):
    model.train(); total_loss=0; epoch_ade,epoch_fde,epoch_rmse=[],[],[]
    for history,adj,last_pos,future in train_loader:
        history,adj,last_pos,future = history.to(DEVICE),adj.to(DEVICE),last_pos.to(DEVICE),future.to(DEVICE)
        history = history.unsqueeze(0) if history.dim()==3 else history
        pred = model(history, adj, last_pos)
        loss = criterion(pred, future)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        epoch_ade.append(ade(pred,future)); epoch_fde.append(fde(pred,future)); epoch_rmse.append(rmse(pred,future))
    mean_ade,mean_fde,mean_rmse = np.mean(epoch_ade),np.mean(epoch_fde),np.mean(epoch_rmse)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} | ADE: {mean_ade:.4f} | FDE: {mean_fde:.4f} | RMSE: {mean_rmse:.4f}")
    if mean_ade<best_ade:
        best_ade=mean_ade
        torch.save(model.state_dict(), os.path.expanduser("~/Desktop/processed_trajectory/best_model.pth"))
