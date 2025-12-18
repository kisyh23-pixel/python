import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ====================== 参数配置 =======================
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
SAVE_DIR = os.path.join(DESKTOP_PATH, "processed_trajectory")
TRAIN_DIR = os.path.join(SAVE_DIR, "train")
TEST_DIR = os.path.join(SAVE_DIR, "test")

HIST_LEN = 75
FUTURE_LEN = 50
STEP = 10
MAX_NEIGHBORS = 7  # 除主车外最大邻居数
MAX_NODES = MAX_NEIGHBORS + 1  # 包含主车共8个节点
SPEED_THRESHOLD = 30.0  # m/s
MIN_SPEED = 0.5  # m/s

# 创建保存目录
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ====================== 读取数据 =======================
print("Loading trajectory CSV...")
df = pd.read_csv(r"C:/Users/shiyuhang/Desktop/Trajectory.csv")
df.sort_values(by=["trackId", "frameId"], inplace=True)

# 按车辆ID分组
groups = df.groupby("trackId")
sample_id = 0

print("Generating samples...")
for track_id, group in tqdm(groups):
    group = group.reset_index(drop=True)
    frame_ids = group["frameId"].values

    # 帧连续性判断 + 最少长度判断
    if len(frame_ids) < HIST_LEN + FUTURE_LEN:
        continue
    if not np.all(np.diff(frame_ids[:HIST_LEN + FUTURE_LEN]) == 1):
        continue

    # 去除突变/静止轨迹
    dx = np.diff(group["localX"].values)
    dy = np.diff(group["localY"].values)
    speed = np.sqrt(dx**2 + dy**2)
    if np.any(speed > SPEED_THRESHOLD) or np.mean(speed) < MIN_SPEED:
        continue

    # 滑动窗口切片
    for start in range(0, len(group) - HIST_LEN - FUTURE_LEN + 1, STEP):
        sub = group.iloc[start: start + HIST_LEN + FUTURE_LEN]
        history = sub.iloc[:HIST_LEN]
        future = sub.iloc[HIST_LEN:]

        # 获取历史帧区间
        frame_range = range(history["frameId"].min(), history["frameId"].max() + 1)
        nodes_by_frame = defaultdict(list)

        for f in frame_range:
            frame_df = df[df["frameId"] == f]
            main_row = df[(df["trackId"] == track_id) & (df["frameId"] == f)]
            if main_row.empty:
                continue
            mx, my = main_row.iloc[0]["localX"], main_row.iloc[0]["localY"]
            frame_df = frame_df[frame_df["trackId"] != track_id].copy()
            frame_df["dist"] = np.sqrt((frame_df["localX"] - mx) ** 2 + (frame_df["localY"] - my) ** 2)
            neighbors = frame_df.nsmallest(MAX_NEIGHBORS, "dist")
            neighbor_nodes = [[x, y, 0.0] for x, y in neighbors[["localX", "localY"]].values.tolist()]
            # 补齐邻居节点数量到MAX_NEIGHBORS
            if len(neighbor_nodes) < MAX_NEIGHBORS:
                neighbor_nodes += [[0.0, 0.0, 0.0]] * (MAX_NEIGHBORS - len(neighbor_nodes))
            # 主车节点 + 邻居节点
            nodes_by_frame[f] = [[mx, my, 1.0]] + neighbor_nodes  # 总数固定为 MAX_NODES

        # 构建输入张量（节点特征 + 邻接图）
        hist_tensor = np.zeros((HIST_LEN, MAX_NODES, 3), dtype=np.float32)
        adj_tensor = np.zeros((HIST_LEN, MAX_NODES, MAX_NODES), dtype=np.float32)

        for t, f in enumerate(frame_range):
            nodes = nodes_by_frame[f]
            for i, node in enumerate(nodes):
                hist_tensor[t, i, :] = node
            for i in range(MAX_NODES):
                for j in range(MAX_NODES):
                    if i != j:
                        adj_tensor[t, i, j] = 1.0

        future_xy = future[["localX", "localY"]].values.astype(np.float32)

        sample = {
            "trackId": int(track_id),
            "history": hist_tensor,         # [75, 8, 3]
            "history_graph": adj_tensor,    # [75, 8, 8]
            "future": future_xy             # [50, 2]
        }

        folder = TRAIN_DIR if sample_id % 10 < 8 else TEST_DIR
        np.save(os.path.join(folder, f"sample_{sample_id:05d}.npy"), sample)
        sample_id += 1

print(f"✅ 预处理完成，样本总数: {sample_id}")
