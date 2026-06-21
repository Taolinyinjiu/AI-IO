"""
AI-IO filter 输出可视化与指标汇总脚本。

输入:
- `FilterManager.save_logs()` 生成的 `stamped_traj_estimate.txt`
- `stamped_vel_estimate.txt`
- `stamped_bias_estimate.txt`
- 数据集中的 `stamped_groundtruth_imu.txt`

输出:
- 每个序列的 plots/ 目录，包含 position/trajectory/bias/velocity/attitude 图。
- 当前工作目录下的 `ekf_metrics.csv`，汇总 ATE/AVE/RTE/RVE 等指标。
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from learning.utils import pose
from learning.utils.plot_utils import xyztPlot
import src.utils.plotting as plotting
from learning.utils.error_analyze import *

from pyhocon import ConfigFactory
import pandas as pd

def process_sequence(seq_name, seq_path, dataset_name, result_dir):
    """
    处理单个序列的 filter 输出。

    主要步骤:
    1. 读取估计轨迹、速度和 bias。
    2. 读取 ground truth，并按估计轨迹时间戳插值对齐。
    3. 生成位置、轨迹、bias、速度、姿态图。
    4. 计算并打印误差指标。
    """
    print(f"Processing sequence: {seq_name}")
    
    # filter 结果目录约定与主运行脚本一致: results/<dataset>/<seq>/pyfilter。
    out_dir = os.path.join(result_dir, dataset_name, seq_name, 'pyfilter')
    traj_fn = os.path.join(out_dir, "stamped_traj_estimate.txt")
    bias_fn = os.path.join(out_dir, "stamped_bias_estimate.txt")
    vel_fn = os.path.join(out_dir, "stamped_vel_estimate.txt")

    if not (os.path.exists(traj_fn) and os.path.exists(bias_fn) and os.path.exists(vel_fn)):
        # 缺少任一结果文件时跳过该序列，避免整批绘图中断。
        print(f"Missing files for sequence {seq_name}. Skipping.")
        return

    # 读取 filter 输出: traj=[ts,p,q]，bias=[ts,bg,ba]，vel=[ts,v]。
    traj = np.loadtxt(traj_fn)
    bias = np.loadtxt(bias_fn)
    ts = bias[:, 0]
    bg = bias[:, 1:4]
    ba = bias[:, 4:]
    vel = np.loadtxt(vel_fn)

    gt_fn = os.path.join(seq_path, 'stamped_groundtruth_imu.txt')
    if not os.path.exists(gt_fn):
        # 没有 ground truth 无法插值对齐，也无法计算误差指标。
        print(f"Groundtruth file not found: {gt_fn}")
        return
    gt_traj = np.loadtxt(gt_fn)
    gt_ts = gt_traj[:, 0]

    # 裁剪估计轨迹，使其时间范围落在 ground truth 覆盖区间内。
    if traj[0, 0] < gt_ts[0]:
        idxs = np.argwhere(traj[:, 0] > gt_ts[0])[0][0]
    else:
        idxs = 0
    if traj[-1, 0] > gt_ts[-1]:
        idxe = np.argwhere(traj[:, 0] > gt_ts[-1])[0][0]
    else:
        idxe = traj.shape[0]
    traj = traj[idxs:idxe]
    vel = vel[idxs:idxe]

    # 按估计轨迹时间戳插值 ground truth 位置、姿态和速度。
    gt_pos_data = interp1d(gt_traj[:, 0], gt_traj[:, 1:4], axis=0)(traj[:, 0])
    gt_rot_data = Slerp(gt_traj[:, 0], Rotation.from_quat(gt_traj[:, 4:8]))(traj[:, 0])
    gt_vel_data = interp1d(gt_traj[:, 0], gt_traj[:, 8:11], axis=0)(traj[:, 0])
    gt_ts = traj[:, 0]
    gt_traj_interp = np.concatenate((gt_ts.reshape((-1, 1)), gt_pos_data, gt_rot_data.as_quat()), axis=1)
    vel_gt = np.concatenate((gt_ts.reshape((-1, 1)), gt_vel_data), axis=1)

    # 将四元数转成 yaw/pitch/roll，便于姿态曲线对比。
    ypr_gt = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in gt_traj_interp])
    ypr_est = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in traj])
    atti_gt = np.concatenate((gt_ts.reshape((-1, 1)), ypr_gt), axis=1)
    atti_est = np.concatenate((gt_ts.reshape((-1, 1)), ypr_est), axis=1)

    # 每个序列单独保存 plots，避免覆盖其他序列结果。
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # xyz time plots
    plt.figure('XYZt view')
    xyztPlot('Position', traj[:,:4], 'estim. traj', gt_traj_interp[:,:4], 'gt')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "position.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "position.png"))
    plt.close()
    plotting.make_position_plots(traj, gt_traj_interp)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajectory.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "trajectory.png"))
    plt.close()
    plotting.plotBiases(ts, bg, ba)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "bias.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "bias.png"))
    plt.close()
    plotting.make_velocity_plots(vel, vel_gt)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "velocity.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "velocity.png"))
    plt.close()
    plotting.make_ori_euler_plots(atti_est, atti_gt)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "attitude.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "attitude.png"))
    plt.close()

    # 计算位置、速度、姿态误差，具体指标定义在 learning.utils.error_analyze 中。
    errors = compute_position_velocity_orientation_errors(
        est_pos=traj[:,:4], gt_pos=gt_traj_interp[:,:4],
        est_vel=vel, gt_vel=vel_gt,
        est_euler=atti_est, gt_euler=atti_gt
    )
    print_rmse_summary(errors)
    return errors


if __name__ == "__main__":
    # 命令行入口: 根据 data_config 中的 test.data_list 批量处理序列。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, help="Path to data config")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.data_config)
    config = conf["test"]

    data_list = []
    all_results = []
    # 将 hocon 配置里的 root + drive 展开成实际 processed_data 路径。
    for entry in config["data_list"]:
        root = entry["data_root"]
        drives = entry["data_drive"]
        for drive in drives:
            path = os.path.join(root, drive, "processed_data", config["mode"])
            data_list.append((drive, path))

    for seq_name, seq_path in data_list:
        try:
            errors = process_sequence(seq_name, seq_path, args.dataset, args.result_dir)
            # 取各误差数组最后一个元素作为整段序列汇总指标。
            row = {
                "ATE_our": errors['position_rmse'][-1],
                "AVE_our": errors['velocity_rmse'][-1],
                "RTE_our": errors['position_rrmse'][-1],
                "RVE_our": errors['velocity_rrmse'][-1],
            }
            all_results.append(row)
        except Exception as e:
            # 单条序列失败不影响其他序列继续绘图。
            print(f"Error processing {seq_path}: {e}")

    df = pd.DataFrame(all_results)
    # 汇总 CSV 写在当前运行目录，适合批量实验后做表格对比。
    df.to_csv("ekf_metrics.csv", index=False)
