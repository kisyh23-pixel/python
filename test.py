# test.py
import os, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import TrajectoryGNNTransformer

# ==== 数据加载 ====
class TrajectoryDataset:
    def __init__(self, folder):
        self.files = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".npy")]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        history = torch.tensor(data['history'],dtype=torch.float32)
        adj = torch.tensor(data['history_graph'],dtype=torch.float32)
        future = torch.tensor(data['future'],dtype=torch.float32)
        last_pos = history[-1,0,:2]
        return history,adj,last_pos,future

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = TrajectoryDataset(os.path.expanduser("~/Desktop/processed_trajectory/test"))
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
model = TrajectoryGNNTransformer().to(DEVICE)
model.load_state_dict(torch.load(os.path.expanduser("~/Desktop/processed_trajectory/best_model.pth")))
model.eval()

# ==== 评估 ====
def ade(pred, gt): return torch.mean(torch.norm(pred-gt,dim=-1)).item()
def fde(pred, gt): return torch.mean(torch.norm(pred[:,-1]-gt[:,-1],dim=-1)).item()
def rmse(pred, gt): return torch.sqrt(torch.mean((pred-gt)**2)).item()

all_ade,all_fde,all_rmse = [],[],[]
os.makedirs(os.path.expanduser("~/Desktop/trajectory_vis"),exist_ok=True)
for idx,(history,adj,last_pos,future) in enumerate(test_loader):
    history,adj,last_pos,future = history.to(DEVICE),adj.to(DEVICE),last_pos.to(DEVICE),future.to(DEVICE)
    with torch.no_grad(): pred = model(history,adj,last_pos)
    all_ade.append(ade(pred,future)); all_fde.append(fde(pred,future)); all_rmse.append(rmse(pred,future))
    # ==== 可视化 ====
    plt.figure()
    hist = history[0,:,0,:2].cpu().numpy()
    fut = future[0].cpu().numpy()
    prd = pred[0].cpu().numpy()
    plt.plot(hist[:,0],hist[:,1],'k.-',label='History')
    plt.plot(fut[:,0],fut[:,1],'b.-',label='GT Future')
    plt.plot(prd[:,0],prd[:,1],'r.-',label='Predicted')
    plt.legend(); plt.title(f"Sample {idx}")
    plt.savefig(os.path.expanduser(f"~/Desktop/trajectory_vis/sample_{idx}.png"))
    plt.close()

print(f"Test ADE: {np.mean(all_ade):.4f} | FDE: {np.mean(all_fde):.4f} | RMSE: {np.mean(all_rmse):.4f}")
