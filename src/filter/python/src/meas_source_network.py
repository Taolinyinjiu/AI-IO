"""
神经网络测量源。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/meas_source_network.py

AI-IO 的网络不直接输出完整 odometry，而是输出:
- 当前机体系速度测量 `v_body`
- 该速度测量的协方差

`FilterRunner` 会把这里得到的速度测量传给 `ImuMSCKF.learnt_model_update()`，
由 EKF/SCEKF 负责融合 IMU propagation 与 learned velocity update。
"""

import numpy as np
import torch

from learning.network.model_factory import get_model
from filter.python.src.utils.logging import logging
from learning.network.covariance_parametrization import DiagonalParam

class MeasSourceNetwork:
    """
    加载训练好的速度测量网络，并提供推理接口。

    输入窗口:
    - `net_accl_b`: `[N, 3]`
    - `net_gyr_b`: `[N, 3]`
    - `net_rotor`: `[N, 4]`

    当前拼接后的网络输入是 `[N, 10]`，再转成 PyTorch Conv1d 习惯的
    `[B, C, T] = [1, 10, N]`。
    """

    def __init__(self, model_path, force_cpu=False):
        # get_model(100) 创建与训练时一致的模型结构；100 对应 1s * 100Hz 输入窗口。
        self.net = get_model(100)

        # 根据运行环境选择 CPU/GPU。Sunray 当前 CPU 路线应传入 force_cpu=True 或使用 --cpu。
        if not torch.cuda.is_available() or force_cpu:
            self.device = torch.device("cpu")
            checkpoint = torch.load(
                model_path, map_location=lambda storage, location: storage
            )
        else:
            self.device = torch.device("cuda:0")
            checkpoint = torch.load(model_path)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval().to(self.device)
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))

    def get_measurement(self, net_t_s, net_accl_b, net_gyr_b, net_rotor):
        """统一测量接口，当前只实现 learned body velocity measurement。"""
        meas, meas_cov = self.get_vb_measurement_model_net(
            net_t_s, net_accl_b, net_gyr_b, net_rotor)
        return meas, meas_cov

    def get_vb_measurement_model_net(self, net_t_s, net_accl_b, net_gyr_b, net_rotor):
        """
        运行神经网络并返回机体系速度测量和协方差。

        `net_t_s` 当前没有被网络直接使用，但保留在接口中，便于后续加入时间特征、
        非均匀采样检查或调试。
        """
        # 当前特征通道约定: acc[3] + gyro[3] + rotor_spd[4] = 10。
        # 如果后续加入 normalized_propulsion，必须从这里同步扩展。
        features = np.concatenate([net_accl_b, net_gyr_b, net_rotor], axis=1)  # N x 10
        # PyTorch 模型输入形状为 [batch, channels, time]。
        features_t = torch.unsqueeze(
            torch.from_numpy(features.T).float().to(self.device), 0
        )

        # 网络输出: 机体系速度预测，以及协方差参数化向量。
        vb_learnt, vb_cov_learned = self.net(features_t)

        # EKF update 期望速度测量是 3x1 列向量。
        meas = vb_learnt.cpu().detach().numpy()
        meas = meas.reshape((3, 1))

        # 对协方差 log 参数做下限裁剪，避免协方差过小导致滤波器过度相信网络。
        vb_cov_learned[vb_cov_learned < -4] = -4  # exp(2 * -4) =~ 0.00034
        meas_cov = DiagonalParam.vec2Cov(vb_cov_learned).cpu().detach().numpy()[0, :, :]

        return meas, meas_cov
