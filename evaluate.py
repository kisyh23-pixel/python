import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

# 引入你的模型和数据集定义
from model import TrajectoryGNNTransformer
from train import TrajectoryDataset

# ==========================================
# 1. 配置与日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. 核心可视化函数 (拟真风格 + 横向布局)
# ==========================================

def transform_coordinates(data):
    """
    坐标转换：将纵向行驶(Y轴)转换为横向(X轴)
    输入: (N, 2) -> (x, y)
    输出: (N, 2) -> (y, -x)  (顺时针旋转90度)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    new_x = data[:, 1]  # 原来的纵向距离变为横坐标
    new_y = -data[:, 0] # 原来的横向位置变为纵坐标
    return np.stack([new_x, new_y], axis=1)

def get_yaw_from_trajectory(traj):
    """计算轨迹每个点的航向角"""
    dx = traj[1:, 0] - traj[:-1, 0]
    dy = traj[1:, 1] - traj[:-1, 1]
    # 补齐最后一个点
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])
    return np.arctan2(dy, dx)

def draw_detailed_car(ax, x, y, yaw, color='red', scale=1.0, alpha=1.0, zorder=10):
    """
    绘制拟真车辆 (带车窗、轮胎、圆角)
    """
    # 车辆尺寸 (标准轿车)
    LENGTH = 4.8 * scale
    WIDTH = 2.0 * scale
    
    # 创建变换对象 (旋转 + 平移)
    tr = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData

    # 1. 绘制轮胎 (4个黑色小矩形)
    wheel_len = LENGTH * 0.18
    wheel_wid = WIDTH * 0.22
    wheel_color = '#333333'
    # 左前, 右前, 左后, 右后
    wheel_positions = [
        (LENGTH/2 - wheel_len, WIDTH/2 - wheel_wid/2),
        (LENGTH/2 - wheel_len, -WIDTH/2 - wheel_wid/2),
        (-LENGTH/2, WIDTH/2 - wheel_wid/2),
        (-LENGTH/2, -WIDTH/2 - wheel_wid/2)
    ]
    for wx, wy in wheel_positions:
        rect = Rectangle((wx, wy), wheel_len, wheel_wid, color=wheel_color, transform=tr, zorder=zorder-1)
        ax.add_patch(rect)

    # 2. 绘制车身 (圆角矩形)
    body = FancyBboxPatch((-LENGTH/2, -WIDTH/2), LENGTH, WIDTH,
                          boxstyle="round,pad=0,rounding_size=0.3",
                          ec='black', fc=color, alpha=alpha, transform=tr, zorder=zorder)
    ax.add_patch(body)

    # 3. 绘制车顶/挡风玻璃 (深色圆角矩形)
    roof_len = LENGTH * 0.55
    roof_wid = WIDTH * 0.8
    roof = FancyBboxPatch((-roof_len/2 - LENGTH*0.05, -roof_wid/2), roof_len, roof_wid,
                          boxstyle="round,pad=0,rounding_size=0.2",
                          ec='none', fc='#222222', alpha=alpha*0.9, transform=tr, zorder=zorder+1)
    ax.add_patch(roof)

def draw_paper_plot(history, future, pred, other_vehicles=None, save_path=None, title=None):
    """
    生成对齐参考图风格的车辆轨迹预测图 (聚焦视图，无任何文字标签)
    """
    # --- A. 数据坐标转换 (竖 -> 横) ---
    hist_rot = transform_coordinates(history)
    fut_rot = transform_coordinates(future)
    pred_rot = transform_coordinates(pred)
    
    others_rot = []
    if other_vehicles:
        for ov in other_vehicles:
            others_rot.append(transform_coordinates(ov))

    # --- B. 画布设置 ---
    # 调整比例为 10:6，视觉更协调
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300) 
    ax.set_facecolor('white')

    # --- 计算聚焦视野 (Zoom In) ---
    # 以"当前时刻"为中心
    curr_pos = hist_rot[-1]
    center_x = curr_pos[0]
    center_y = curr_pos[1]

    # 设定显示范围：前后各 35米，上下各 8米
    x_lim_min = center_x - 35 
    x_lim_max = center_x + 35
    y_lim_min = center_y - 8
    y_lim_max = center_y + 8

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_aspect('equal') # 保持物理等比例

    # --- C. 绘制车道线 ---
    base_lane_y = np.round(center_y / 3.75) * 3.75
    lane_offsets = [-7.5, -3.75, 0, 3.75, 7.5] 
    
    for offset in lane_offsets:
        y_lane = base_lane_y + offset
        ax.axhline(y_lane, color='black', linestyle='--', linewidth=1.2, dashes=(5, 5), zorder=1)

    # --- D. 绘制其他车辆 ---
    for ov in others_rot:
        if np.sum(np.abs(ov)) < 0.1: continue
        # 视野外过滤
        if np.mean(ov[:,0]) < x_lim_min - 10 or np.mean(ov[:,0]) > x_lim_max + 10:
            continue
            
        ax.plot(ov[:,0], ov[:,1], color='#808080', linewidth=1.5, alpha=0.6, zorder=2)
        yaw = get_yaw_from_trajectory(ov)
        draw_detailed_car(ax, ov[-1,0], ov[-1,1], yaw[-1], color='#A0A0A0', scale=1.1, zorder=3)

    # --- E. 绘制主车 ---
    # 1. 历史轨迹 (深灰实线)
    ax.plot(hist_rot[:,0], hist_rot[:,1], color='#555555', linewidth=2.5, label='历史轨迹', zorder=3)
    
    # 2. 真实未来 (黑色实线)
    ax.plot(fut_rot[:,0], fut_rot[:,1], color='black', linewidth=2.5, label='真实轨迹', zorder=3)
    
    # 3. 预测轨迹 (彩虹渐变)
    points = np.array([pred_rot[:,0], pred_rot[:,1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = plt.get_cmap('jet') 
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.0, alpha=0.9, zorder=4)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)

    # 4. 主车实体 (红色)
    ego_yaw = get_yaw_from_trajectory(hist_rot)
    draw_detailed_car(ax, hist_rot[-1,0], hist_rot[-1,1], ego_yaw[-1], color='#D93025', scale=1.1, zorder=10)

    # --- F. 装饰 ---
    # 坐标轴标签 (Times New Roman 字体)
    ax.set_xlabel('y/m', fontsize=16, fontweight='normal', fontfamily='Times New Roman')
    ax.set_ylabel('x/m', fontsize=16, fontweight='normal', fontfamily='Times New Roman')
    
    # 刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontname('Times New Roman')
    
    # 图例 (放在顶部内部)
    legend_elements = [
        Line2D([0], [0], color='#555555', lw=2.5, label='历史轨迹'),
        Line2D([0], [0], color='black', lw=2.5, label='真实轨迹')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=2, frameon=False, fontsize=14, prop={'family': 'SimHei'}) 

    # 标题 (关键修改：完全移除标题绘制逻辑，即使传入 title 也不画)
    # 这里的 title 参数虽然保留在函数签名里以免调用报错，但我们不使用它
    # if title:
    #     plt.figtext(0.5, 0.01, title, ...) <--- 已注释掉

    plt.tight_layout()
    # 移除底部的额外留白
    # plt.subplots_adjust(bottom=0.1) 
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

# ==========================================
# 3. 评估指标计算
# ==========================================

def ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1)).item()

def fde(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1)).item()

def rmse(pred, gt):
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()

# ==========================================
# 4. 评估流程逻辑
# ==========================================

def evaluate_model(model, test_loader, device):
    """计算整体指标"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (history, adj, last_pos, future) in enumerate(test_loader):
            history, adj, last_pos, future = history.to(device), adj.to(device), last_pos.to(device), future.to(device)
            
            pred = model(history, adj, last_pos)
            
            batch_size = pred.shape[0]
            start_idx = batch_idx * test_loader.batch_size
            
            for i in range(batch_size):
                sample_pred = pred[i:i+1]
                sample_future = future[i:i+1]
                
                s_ade = ade(sample_pred, sample_future)
                s_fde = fde(sample_pred, sample_future)
                s_rmse = rmse(sample_pred, sample_future)
                
                all_metrics.append((s_ade, s_fde, s_rmse, start_idx + i))
            
            if (batch_idx + 1) % 50 == 0:
                logging.info(f"评估进度: {batch_idx+1}/{len(test_loader)}")
    
    # 汇总
    all_metrics = np.array(all_metrics, dtype=[('ade', float), ('fde', float), ('rmse', float), ('index', int)])
    return np.mean(all_metrics['ade']), np.mean(all_metrics['fde']), np.mean(all_metrics['rmse']), all_metrics

def visualize_selected_samples(model, dataset, indices, device, save_dir, prefix="Sample"):
    """可视化指定索引的样本"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取数据
            history, adj, last_pos, future = dataset[idx]
            
            # 转Tensor加Batch维度
            history_t = history.unsqueeze(0).to(device)
            adj_t = adj.unsqueeze(0).to(device)
            last_pos_t = last_pos.unsqueeze(0).to(device)
            
            # 预测
            pred = model(history_t, adj_t, last_pos_t)
            
            # 转Numpy准备画图
            hist_np = history_t[0, :, 0, :2].cpu().numpy()  # 主车历史 (T, 2)
            fut_np = future.cpu().numpy()                   # 真实未来 (T, 2)
            pred_np = pred[0].cpu().numpy()                 # 预测未来 (T, 2)
            
            # 提取周围车辆
            others_np = []
            num_nodes = history_t.shape[2]
            for n in range(1, num_nodes):
                track = history_t[0, :, n, :2].cpu().numpy()
                if np.sum(np.abs(track)) > 0.1: 
                    others_np.append(track)
            
            # ==== 关键修改：强制 title 为 None ====
            title = None
            
            # 调用画图
            save_path = os.path.join(save_dir, f"{prefix}_{idx}.png")
            try:
                draw_paper_plot(hist_np, fut_np, pred_np, others_np, save_path, title)
            except Exception as e:
                logging.error(f"绘图失败 Sample {idx}: {e}")
            
            if (i+1) % 10 == 0:
                logging.info(f"已可视化 {i+1}/{len(indices)} 张图 -> {save_dir}")

# ==========================================
# 5. 主程序
# ==========================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 路径设置
    save_root = os.path.expanduser(args.output_dir)
    os.makedirs(save_root, exist_ok=True)
    
    # 1. 加载模型
    model_path = os.path.expanduser(args.model_path)
    model = TrajectoryGNNTransformer().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("模型加载成功！")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    # 2. 加载数据
    data_root = os.path.expanduser(args.data_dir)
    test_dataset = TrajectoryDataset(os.path.join(data_root, "test"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"测试集加载完成，共 {len(test_dataset)} 个样本")
    
    # 3. 定量评估
    logging.info(">>> 开始定量评估...")
    mean_ade, mean_fde, mean_rmse, all_metrics = evaluate_model(model, test_loader, device)
    
    logging.info(f"\n======== 评估结果 ========")
    logging.info(f"ADE : {mean_ade:.4f} m")
    logging.info(f"FDE : {mean_fde:.4f} m")
    logging.info(f"RMSE: {mean_rmse:.4f} m")
    logging.info(f"==========================\n")
    
    # 保存结果
    np.save(os.path.join(save_root, "metrics.npy"), all_metrics)
    with open(os.path.join(save_root, "report.txt"), "w") as f:
        f.write(f"ADE: {mean_ade:.4f}\nFDE: {mean_fde:.4f}\nRMSE: {mean_rmse:.4f}\n")
    
    # 4. 定性可视化
    if not args.skip_viz:
        logging.info(">>> 开始生成可视化图表...")
        
        sorted_indices = np.argsort(all_metrics['ade'])
        best_indices = all_metrics['index'][sorted_indices[:args.num_viz]]
        worst_indices = all_metrics['index'][sorted_indices[-args.num_viz:]]
        random_indices = np.random.choice(all_metrics['index'], args.num_viz, replace=False)
        
        visualize_selected_samples(model, test_dataset, best_indices, device, 
                                   os.path.join(save_root, "viz_best"), "Best")
        visualize_selected_samples(model, test_dataset, worst_indices, device, 
                                   os.path.join(save_root, "viz_worst"), "Worst")
        visualize_selected_samples(model, test_dataset, random_indices, device, 
                                   os.path.join(save_root, "viz_random"), "Random")
        
        logging.info("可视化完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="~/Desktop/processed_trajectory/best_model.pth")
    parser.add_argument('--data_dir', type=str, default="~/Desktop/processed_trajectory")
    parser.add_argument('--output_dir', type=str, default="~/Desktop/eval_results")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num_viz', type=int, default=20, help="可视化多少张Best/Worst/Random样本")
    parser.add_argument('--skip_viz', action='store_true', help="跳过可视化步骤")
    
    args = parser.parse_args()
    main(args)