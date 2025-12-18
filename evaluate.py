import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline  # 用于轨迹平滑
from model import TrajectoryGNNTransformer
from train import TrajectoryDataset  # 导入数据集类

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# 设置matplotlib风格
plt.style.use('seaborn-v0_8-white')
# 设置中文字体支持和全局字体大小
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ==== 评估指标函数 ====
def ade(pred, gt):
    """计算平均位移误差"""
    return torch.mean(torch.norm(pred - gt, dim=-1)).item()

def fde(pred, gt):
    """计算最终位移误差"""
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1)).item()

def rmse(pred, gt):
    """计算均方根误差"""
    return torch.sqrt(torch.mean((pred - gt) **2)).item()

# ==== 场景分类辅助函数 ====
def classify_scene(history):
    """根据历史轨迹简单分类场景类型"""
    hist_target = history[:, 0, :]  # 目标车辆历史轨迹
    dy_total = abs(hist_target[-1, 0] - hist_target[0, 0])
    dx_total = abs(hist_target[-1, 1] - hist_target[0, 1])
    
    if dx_total > dy_total * 1.5:
        return "horizontal"
    elif dy_total > dx_total * 1.5:
        return "vertical"
    else:
        return "mixed"

# ==== 轨迹平滑函数 ====
def smooth_trajectory(x, y, num_points=100):
    if len(x) <= 3:
        return x, y
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# ==== 三条车道可视化函数 ====
def plot_trajectories_clean(history, future, prediction, save_path=None, title=None, 
                           show_other_vehicles=True, figsize=(16, 8), smooth=True):
    """
    简洁版轨迹可视化 - 四条虚线车道线（三条车道），无网格无指标
    """
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    
    # 简洁的颜色配置
    colors = {
        'history': '#2E86AB',      # 深蓝色历史轨迹
        'future': '#A23B72',       # 紫色真实轨迹  
        'prediction': '#FF6B6B',   # 红色预测轨迹
        'other': '#7D7D7D',        # 灰色其他车辆
        'lane': '#8D99AE'          # 浅灰色车道线
    }
    
    linewidths = {
        'history': 3.5, 
        'future': 3.5, 
        'prediction': 4.0, 
        'other': 1.8
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置纯白背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 提取目标车辆轨迹
    hist_target = history[:, 0, :]
    
    # 计算合适的坐标范围
    all_x = np.concatenate([hist_target[:, 1], future[:, 1], prediction[:, 1]])
    all_y = np.concatenate([hist_target[:, 0], future[:, 0], prediction[:, 0]])
    
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    # 减少边距，使轨迹图更紧凑
    x_margin = max(x_range * 0.1, 5)  # 减少x轴边距
    y_margin = max(y_range * 0.1, 8)  # 减少y轴边距
    
    x_min, x_max = all_x.min() - x_margin, all_x.max() + x_margin
    y_min, y_max = all_y.min() - y_margin, all_y.max() + y_margin
    
    # 绘制四条虚线车道线（三条车道）
    lane_width = 3.5  # 每条车道宽度3.5米
    total_lane_width = 3 * lane_width  # 三条车道总宽度
    
    # 计算车道中心位置，确保车道在视图中央
    lane_center_y = (y_min + y_max) / 2
    
    # 四条车道线的y坐标
    lane_positions = [
        lane_center_y - 1.5 * lane_width,  # 最左侧车道线
        lane_center_y - 0.5 * lane_width,  # 左侧车道分隔线
        lane_center_y + 0.5 * lane_width,  # 右侧车道分隔线
        lane_center_y + 1.5 * lane_width   # 最右侧车道线
    ]
    
    # 绘制贯穿整个x轴范围的车道线
    for lane_y in lane_positions:
        ax.axhline(lane_y, color=colors['lane'], linewidth=1.5, 
                  linestyle='--', alpha=0.8, zorder=1)
    
    # 历史轨迹
    x_hist, y_hist = hist_target[:, 1], hist_target[:, 0]
    if smooth and len(x_hist) > 3:
        x_hist, y_hist = smooth_trajectory(x_hist, y_hist, 100)
    
    # 历史轨迹渐变效果
    points_hist = np.array([x_hist, y_hist]).T.reshape(-1, 1, 2)
    segments_hist = np.concatenate([points_hist[:-1], points_hist[1:]], axis=1)
    lc_hist = LineCollection(segments_hist, cmap='Blues', linewidth=linewidths['history'])
    lc_hist.set_array(np.linspace(0, 1, len(segments_hist)))
    ax.add_collection(lc_hist)
    
    # 历史轨迹起点和终点标记
    ax.scatter(x_hist[0], y_hist[0], color=colors['history'], 
               s=60, zorder=5, marker='o', edgecolors='white', linewidth=1.5)
    ax.scatter(x_hist[-1], y_hist[-1], color=colors['history'], 
               s=60, zorder=5, marker='s', edgecolors='white', linewidth=1.5)
    
    # 其他车辆轨迹
    if show_other_vehicles:
        other_count = 0
        for i in range(1, min(history.shape[1], 4)):
            hist_other = history[:, i, :]
            if not np.all(hist_other == 0) and other_count < 3:
                x_o, y_o = hist_other[:, 1], hist_other[:, 0]
                if smooth and len(x_o) > 3:
                    x_o, y_o = smooth_trajectory(x_o, y_o, 80)
                ax.plot(x_o, y_o, '--', color=colors['other'], 
                       linewidth=linewidths['other'], alpha=0.6, zorder=2)
                ax.scatter(x_o[-1], y_o[-1], color=colors['other'], 
                          s=30, zorder=4, marker='^', alpha=0.7)
                other_count += 1
    
    # 真实未来轨迹
    x_f, y_f = future[:, 1], future[:, 0]
    if smooth and len(x_f) > 3:
        x_f, y_f = smooth_trajectory(x_f, y_f, 100)
    
    # 真实轨迹渐变效果
    points_future = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
    segments_future = np.concatenate([points_future[:-1], points_future[1:]], axis=1)
    lc_future = LineCollection(segments_future, cmap='Purples', linewidth=linewidths['future'])
    lc_future.set_array(np.linspace(0, 1, len(segments_future)))
    ax.add_collection(lc_future)
    
    # 真实轨迹终点标记
    ax.scatter(x_f[-1], y_f[-1], color=colors['future'], 
               s=80, zorder=5, marker='*', edgecolors='white', linewidth=1.5)
    
    # 预测轨迹
    x_p, y_p = prediction[:, 1], prediction[:, 0]
    if smooth and len(x_p) > 3:
        x_p, y_p = smooth_trajectory(x_p, y_p, 120)
    
    # 预测轨迹渐变效果
    points_pred = np.array([x_p, y_p]).T.reshape(-1, 1, 2)
    segments_pred = np.concatenate([points_pred[:-1], points_pred[1:]], axis=1)
    
    # 红色到橙色渐变
    colors_pred = ['#FF6B6B', '#FF8E6B', '#FFB16B', '#FFD46B']
    cmap_pred = mcolors.LinearSegmentedColormap.from_list('pred_cmap', colors_pred)
    
    lc_pred = LineCollection(segments_pred, cmap=cmap_pred, linewidth=linewidths['prediction'])
    lc_pred.set_array(np.linspace(0, 1, len(segments_pred)))
    ax.add_collection(lc_pred)
    
    # 预测轨迹终点标记
    ax.scatter(x_p[-1], y_p[-1], color=colors['prediction'], 
               s=100, zorder=6, marker='D', edgecolors='white', linewidth=1.5)
    
    # 设置坐标轴范围，确保车道线可见但减少空白
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 坐标轴标签 - 加大字体并加粗
    ax.set_xlabel('Y (m)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('X (m)', fontsize=18, fontweight='bold', labelpad=10)
    
    # 坐标轴刻度 - 加大字体并加粗
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=14, width=1, length=4)
    # 显式加粗刻度文字
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 加粗坐标轴线
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 标题
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    else:
        ax.set_title('Trajectory Prediction', fontsize=20, fontweight='bold', pad=20)
    
    # 简洁图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['history'], lw=3, marker='o', 
               markersize=6, label='History'),
        Line2D([0], [0], color=colors['future'], lw=3, marker='*', 
               markersize=8, label='Ground Truth'),
        Line2D([0], [0], color=colors['prediction'], lw=3, marker='D', 
               markersize=8, label='Prediction'),
    ]
    
    if show_other_vehicles:
        legend_elements.append(
            Line2D([0], [0], color=colors['other'], lw=2, linestyle='--', 
                   label='Other Vehicles')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, 
              framealpha=0.9, fancybox=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

# ==== 评估函数 ====
def evaluate_model(model, test_loader, device):
    """评估模型在整个测试集上的性能"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (history, adj, last_pos, future) in enumerate(test_loader):
            # 将数据移动到设备
            history = history.to(device)
            adj = adj.to(device)
            last_pos = last_pos.to(device)
            future = future.to(device)
            
            # 预测
            pred = model(history, adj, last_pos)
            
            # 计算每个样本的指标
            batch_size = pred.shape[0]
            start_idx = batch_idx * test_loader.batch_size
            
            for i in range(batch_size):  # 遍历批次中的每个样本
                sample_idx = start_idx + i
                sample_pred = pred[i:i+1]  # 保持维度
                sample_future = future[i:i+1]
                
                sample_ade = ade(sample_pred, sample_future)
                sample_fde = fde(sample_pred, sample_future)
                sample_rmse = rmse(sample_pred, sample_future)
                
                all_metrics.append((sample_ade, sample_fde, sample_rmse, sample_idx))
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"已处理 {batch_idx+1}/{len(test_loader)} 批次")
    
    # 计算平均指标
    all_metrics = np.array(all_metrics, dtype=[
        ('ade', float), ('fde', float), ('rmse', float), ('index', int)
    ])
    mean_ade = np.mean(all_metrics['ade'])
    mean_fde = np.mean(all_metrics['fde'])
    mean_rmse = np.mean(all_metrics['rmse'])
    
    logging.info(f"评估完成. 平均ADE: {mean_ade:.4f}, 平均FDE: {mean_fde:.4f}, 平均RMSE: {mean_rmse:.4f}")
    
    return mean_ade, mean_fde, mean_rmse, all_metrics

# ==== 样本选择函数 ====
def select_representative_samples(dataset, num_per_category=10, random_seed=None):
    """根据场景类型选择代表性样本"""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    scene_categories = {
        "horizontal": [],  # 横向移动
        "vertical": [],    # 纵向移动
        "mixed": []        # 混合方向
    }
    
    # 遍历数据集进行分类
    logging.info("按移动方向分类样本中...")
    for i in range(len(dataset)):
        try:
            history, _, _, _ = dataset[i]
            scene_type = classify_scene(history.numpy())
            scene_categories[scene_type].append(i)
            
            # 进度显示
            if (i + 1) % 100 == 0:
                logging.info(f"已分类 {i + 1}/{len(dataset)} 样本")
        except Exception as e:
            logging.warning(f"处理样本 {i} 时出错: {str(e)}")
            continue
    
    # 为每个类别选择样本
    selected_indices = []
    for scene_type, indices in scene_categories.items():
        if len(indices) > 0:
            num_select = min(num_per_category, len(indices))
            selected = np.random.choice(indices, size=num_select, replace=False)
            selected_indices.extend(selected)
            logging.info(f"为 {scene_type} 移动类型选择了 {num_select} 个样本")
        else:
            logging.warning(f"未找到 {scene_type} 移动类型的样本")
    
    return np.array(selected_indices)

# ==== 可视化函数 ====
def visualize_predictions(model, test_dataset, device, num_plots=10, indices=None, 
                          save_dir="visualizations", category_name="", show_other_vehicles=True):
    """可视化模型的预测结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 确定要可视化的样本索引
    if indices is None:
        indices = np.random.choice(len(test_dataset), size=min(num_plots, len(test_dataset)), replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # 获取数据
                history, adj, last_pos, future = test_dataset[idx]
                
                # 添加批次维度
                history_tensor = history.unsqueeze(0).to(device)
                adj_tensor = adj.unsqueeze(0).to(device)
                last_pos_tensor = last_pos.unsqueeze(0).to(device)
                future_tensor = future.unsqueeze(0).to(device)
                
                # 预测
                pred = model(history_tensor, adj_tensor, last_pos_tensor)
                
                # 转换回numpy
                history_np = history_tensor.squeeze(0).cpu().numpy()
                future_np = future_tensor.squeeze(0).cpu().numpy()
                pred_np = pred.squeeze(0).cpu().numpy()
                
                # 计算场景类型和误差
                scene_type = classify_scene(history_np)
                sample_ade = ade(torch.tensor(pred_np).unsqueeze(0), torch.tensor(future_np).unsqueeze(0))
                
                # 绘制并保存，使用简洁版可视化
                title = f"{category_name} Sample {idx} ({scene_type}) - ADE: {sample_ade:.2f}"
                save_path = os.path.join(save_dir, f"sample_{idx}.png")
                plot_trajectories_clean(history_np, future_np, pred_np, save_path=save_path, 
                                       title=title, show_other_vehicles=show_other_vehicles,
                                       smooth=True)
                
                # 打印进度
                if (i + 1) % 10 == 0:
                    logging.info(f"已保存 {i + 1}/{len(indices)} 个可视化结果至 {save_dir}")
            except Exception as e:
                logging.error(f"可视化样本 {idx} 时出错: {str(e)}")
                continue

# ==== 轨迹预测保存函数 ====
def save_trajectory_predictions(model, test_loader, device, save_path="trajectory_predictions"):
    """保存模型对所有测试样本的轨迹预测结果"""
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (history, adj, last_pos, future) in enumerate(test_loader):
            try:
                history = history.to(device)
                adj = adj.to(device)
                last_pos = last_pos.to(device)
                future = future.to(device)
                
                pred = model(history, adj, last_pos)
                
                batch_predictions = {
                    'history': history.cpu().numpy(),
                    'adjacency': adj.cpu().numpy(),
                    'last_pos': last_pos.cpu().numpy(),
                    'future_gt': future.cpu().numpy(),
                    'future_pred': pred.cpu().numpy()
                }
                
                all_predictions.append(batch_predictions)
                
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"已保存 {batch_idx+1}/{len(test_loader)} 批次的预测结果")
            except Exception as e:
                logging.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                continue
    
    try:
        np.save(os.path.join(save_path, "all_predictions.npy"), all_predictions)
        logging.info(f"所有轨迹预测已保存至 {os.path.join(save_path, 'all_predictions.npy')}")
    except Exception as e:
        logging.error(f"保存预测结果失败: {str(e)}")
    
    return all_predictions

# ==== 主函数 ====
def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 设置保存路径
    base_save_path = os.path.expanduser(args.output_dir)
    visualization_path = os.path.join(base_save_path, "visualizations")
    prediction_path = os.path.join(base_save_path, "predictions")
    
    # 创建保存目录
    for path in [base_save_path, visualization_path, prediction_path]:
        os.makedirs(path, exist_ok=True)
    
    # 加载模型
    model = TrajectoryGNNTransformer().to(device)
    model_path = os.path.expanduser(args.model_path)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"成功从 {model_path} 加载模型")
    except FileNotFoundError:
        logging.error(f"在 {model_path} 未找到模型。请先训练模型。")
        return
    except Exception as e:
        logging.error(f"加载模型时出错: {str(e)}")
        return
    
    # 加载测试数据
    data_root = os.path.expanduser(args.data_dir)
    try:
        test_dataset = TrajectoryDataset(os.path.join(data_root, "test"))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        logging.info(f"已加载测试数据集，包含 {len(test_dataset)} 个样本")
    except Exception as e:
        logging.error(f"加载测试数据集时出错: {str(e)}")
        return
    
    # 1. 定量评估
    logging.info("\n=== 定量评估 ===")
    try:
        mean_ade, mean_fde, mean_rmse, all_metrics = evaluate_model(model, test_loader, device)
        
        logging.info(f"\n测试集结果:")
        logging.info(f"ADE:  {mean_ade:.4f}")
        logging.info(f"FDE:  {mean_fde:.4f}")
        logging.info(f"RMSE: {mean_rmse:.4f}")
        
        # 保存评估结果
        evaluation_results = {
            'mean_ade': mean_ade,
            'mean_fde': mean_fde,
            'mean_rmse': mean_rmse,
            'all_metrics': all_metrics
        }
        np.save(os.path.join(base_save_path, "evaluation_results.npy"), evaluation_results)
        logging.info(f"评估结果已保存至 {os.path.join(base_save_path, 'evaluation_results.npy')}")
    except Exception as e:
        logging.error(f"定量评估过程中出错: {str(e)}")
        return
    
    # 2. 保存所有轨迹预测
    if not args.skip_predictions:
        logging.info("\n=== 保存所有轨迹预测 ===")
        save_trajectory_predictions(model, test_loader, device, save_path=prediction_path)
    
    # 3. 定性可视化 - 按场景类型选择
    if not args.skip_visualization:
        logging.info("\n=== 定性分析 - 代表性样本 ===")
        scene_indices = select_representative_samples(
            test_dataset, 
            num_per_category=args.samples_per_category,
            random_seed=args.random_seed
        )
        visualize_predictions(
            model, test_dataset, device, 
            indices=scene_indices,
            save_dir=os.path.join(visualization_path, "by_direction"),
            category_name="Direction",
            show_other_vehicles=args.show_other_vehicles
        )
        
        # 4. 定性可视化 - 随机样本
        logging.info("\n=== 定性分析 - 随机样本 ===")
        visualize_predictions(
            model, test_dataset, device, 
            num_plots=args.random_samples,
            save_dir=os.path.join(visualization_path, "random"),
            category_name="Random",
            show_other_vehicles=args.show_other_vehicles
        )
        
        # 5. 定性可视化 - 最佳和最差样本
        logging.info("\n=== 定性分析 - 最佳和最差样本 ===")
        sorted_indices = np.argsort(all_metrics['ade'])
        n_samples = min(args.best_worst_samples, len(all_metrics) // 2)
        best_indices = all_metrics['index'][sorted_indices[:n_samples]]
        worst_indices = all_metrics['index'][sorted_indices[-n_samples:]]
        
        logging.info(f"可视化 {n_samples} 个最佳样本 (最低ADE)...")
        visualize_predictions(
            model, test_dataset, device, 
            indices=best_indices, 
            save_dir=os.path.join(visualization_path, "best"),
            category_name="Best",
            show_other_vehicles=args.show_other_vehicles
        )
        
        logging.info(f"可视化 {n_samples} 个最差样本 (最高ADE)...")
        visualize_predictions(
            model, test_dataset, device, 
            indices=worst_indices, 
            save_dir=os.path.join(visualization_path, "worst"),
            category_name="Worst",
            show_other_vehicles=args.show_other_vehicles
        )
    
    # 6. 生成评估报告
    logging.info("\n=== 生成评估报告 ===")
    try:
        report = f"""
        轨迹预测模型评估报告
        =============================================
        
        数据集: {os.path.join(data_root, "test")}
        样本数量: {len(test_dataset)}
        模型: {model_path}
        
        性能指标:
        - ADE (平均位移误差): {mean_ade:.4f}
        - FDE (最终位移误差): {mean_fde:.4f}
        - RMSE (均方根误差): {mean_rmse:.4f}
        
        生成的可视化结果:
        - 按移动方向: {len(scene_indices) if 'scene_indices' in locals() else 0} 个样本
        - 随机样本: {args.random_samples} 个样本
        - 最佳性能: {n_samples if 'n_samples' in locals() else 0} 个样本
        - 最差性能: {n_samples if 'n_samples' in locals() else 0} 个样本
        
        结果保存至:
        - 评估指标: {os.path.join(base_save_path, "evaluation_results.npy")}
        {'- 轨迹预测: ' + os.path.join(prediction_path, "all_predictions.npy") if not args.skip_predictions else ''}
        {'- 可视化结果: ' + visualization_path if not args.skip_visualization else ''}
        
        评估成功完成。
        """
        
        with open(os.path.join(base_save_path, "evaluation_report.txt"), "w") as f:
            f.write(report)
        
        logging.info(report)
        logging.info(f"\n=== 评估完成 ===")
        logging.info(f"所有结果已保存至: {base_save_path}")
    except Exception as e:
        logging.error(f"生成评估报告时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估轨迹预测模型')
    
    # 路径配置
    parser.add_argument('--model_path', type=str, 
                      default="~/Desktop/processed_trajectory/best_model.pth",
                      help='训练好的模型路径')
    parser.add_argument('--data_dir', type=str, 
                      default="~/Desktop/processed_trajectory",
                      help='包含测试数据的目录')
    parser.add_argument('--output_dir', type=str, 
                      default="~/Desktop/trajectory_prediction_results",
                      help='保存评估结果的目录')
    
    # 评估配置
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='评估时的批次大小')
    parser.add_argument('--device', type=str, default="cuda", 
                      help='评估使用的设备 (cuda 或 cpu)')
    
    # 可视化配置
    parser.add_argument('--samples_per_category', type=int, default=50,
                      help='每个移动类别要可视化的样本数量')
    parser.add_argument('--random_samples', type=int, default=100,
                      help='要可视化的随机样本数量')
    parser.add_argument('--best_worst_samples', type=int, default=50,
                      help='要可视化的最佳和最差样本数量')
    parser.add_argument('--show_other_vehicles', action='store_true', default=False,
                      help='是否在可视化中显示其他车辆')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='用于重现性的随机种子')
    
    # 可选功能
    parser.add_argument('--skip_predictions', action='store_true',
                      help='跳过保存所有轨迹预测')
    parser.add_argument('--skip_visualization', action='store_true',
                      help='跳过生成可视化结果')
    
    args = parser.parse_args()
    main(args)