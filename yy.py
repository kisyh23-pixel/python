import os
import numpy as np

DATA_DIR = r"C:/Users/shiyuhang/Desktop/processed_trajectory/test"  # 你的训练样本文件夹

max_nodes_overall = 0
sample_count = 0

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".npy"):
        data = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True).item()
        hist = data["history"]  # [75, N, 3]
        max_nodes_sample = hist.shape[1]
        if max_nodes_sample > max_nodes_overall:
            max_nodes_overall = max_nodes_sample
        sample_count += 1

print(f"总样本数：{sample_count}")
print(f"所有样本中最大节点数（N）：{max_nodes_overall}")
