"""
离线 filter 的数据读取器。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/dataloader/data_io.py

本文件不负责从 rosbag/mcap 预处理数据；它只读取已经生成好的
`processed_data/<split>/data.hdf5`。在 filter 运行时，`FilterManager`
会用 `DataIO` 顺序取出每一帧 IMU 和电机转速，然后喂给 `FilterRunner`。
"""

import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from learning.utils import pose


class DataIO:
    """
    单个离线序列的数据访问类。

    主要职责:
    - 读取 `data.hdf5`。
    - 暴露逐帧 IMU/rotor 输入给 filter。
    - 提供 ground truth 位置/姿态/速度用于初始化和评估。
    """

    def __init__(self, quad_name=None):
        # quad_name 当前未被实际使用，保留给多机型/多数据集扩展。
        self.quad_name = quad_name
        self.ts = None
        self.accel_raw = None
        self.gyro_raw = None
        self.accel_calib = None
        self.gyro_calib = None
        self.dataset_size = None
        self.gyro_bias = None
        self.accel_bias = None
        self.gt_traj = None

    def load(self, sequence):
        """
        从指定序列目录读取 `data.hdf5`。

        参数:
        - `sequence`: 形如 `.../processed_data/test` 的目录。

        读取字段:
        - `gyro_raw`, `accel_raw`: filter propagation 使用的原始 IMU。
        - `gyro_calib`, `accel_calib`: 网络训练/调试时可用的标定后 IMU。
        - `rotor_spd`: 四个电机转速，是 AI-IO 网络输入的一部分。
        - `traj_target`: 真值轨迹，格式 `[p, q, v]`。
        - `gyro_bias`, `accel_bias`: 离线标定 bias，可用于 filter 初始化。
        """
        indir = sequence
        with h5py.File(os.path.join(indir, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            accel_raw = np.copy(f["accel_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            accel_calib = np.copy(f["accel_calib"])
            gt_traj = np.copy(f["traj_target"])
            gyro_bias = np.copy(f["gyro_bias"])
            accel_bias = np.copy(f["accel_bias"])
            rotor_spd = np.copy(f["rotor_spd"])

        self.ts = np.round(ts, 5)
        self.accel_raw = accel_raw
        self.gyro_raw = gyro_raw
        self.accel_calib = accel_calib
        self.gyro_calib = gyro_calib
        self.dataset_size = self.ts.shape[0]
        self.gyro_bias = gyro_bias
        self.accel_bias = accel_bias
        self.rotor_spd = rotor_spd
        self.gt_ts = np.round(ts, 5)
        self.gt_p = gt_traj[:, 0:3]
        self.gt_q = gt_traj[:, 3:7]
        self.gt_vb = gt_traj[:, 7:10]

        # gt_traj 中保存的速度字段命名为 gt_vb，但这里重新用位置差分得到世界系速度。
        # FilterManager 在 ground truth 初始化时使用 self.gt_v。
        gt_v = (self.gt_p[2:] - self.gt_p[:-2]) / (self.gt_ts[2:] - self.gt_ts[:-2])[:, None]
        vs = (self.gt_p[1] - self.gt_p[0]) / (self.gt_ts[1] - self.gt_ts[0])
        gt_v = np.concatenate((vs.reshape((1,3)), gt_v), axis=0)
        vf = (self.gt_p[-1] - self.gt_p[-2]) / (self.gt_ts[-1] - self.gt_ts[-2])
        gt_v = np.concatenate((gt_v, vf.reshape((1,3))), axis=0)
        self.gt_v = gt_v

    def get_datai(self, idx):
        """
        返回第 idx 帧 filter 输入。

        输出:
        - `ts`: 秒级时间戳。
        - `acc`: `[3, 1]` 原始加速度。
        - `gyr`: `[3, 1]` 原始角速度。
        - `rotor`: `[4, 1]` 电机转速。
        """
        ts = self.ts[idx]
        acc = self.accel_raw[idx].reshape((3, 1))
        gyr = self.gyro_raw[idx].reshape((3, 1))
        rotor = self.rotor_spd[idx].reshape((4,1))
        return ts, acc, gyr, rotor

    def get_imu_calibration(self):
        """返回离线 IMU bias，用于 `ImuCalib.from_dic()` 初始化。"""
        imu_calib = {}
        imu_calib["gyro_bias"] = self.gyro_bias
        imu_calib["accel_bias"] = self.accel_bias
        return imu_calib

    def get_groundtruth_pose(self, ts):
        """
        在指定时间戳插值得到 ground truth pose。

        当前返回:
        - `is_available`: 请求时间是否在 gt 时间范围内。
        - `meas`: `[R | p]` 形式的 3x4 位姿矩阵。
        - `meas_cov`: 6x6 的模拟测量协方差。

        这个接口可用于构造外部位姿观测或评估，不是主 filter 路径的必要输入。
        """
        # 检查请求时间是否落在真值轨迹覆盖范围内。
        is_available = self.gt_ts[0] <= ts <= self.gt_ts[-1]
        if not is_available:
            return False, None, None

        # 姿态用球面线性插值，位置用线性插值。
        idx_left = np.where(self.gt_ts <= ts)[0][-1]
        idx_right = np.where(self.gt_ts > ts)[0][0]
        interp_gt_ts = self.gt_ts[idx_left : idx_right + 1]
        slerp = Slerp(interp_gt_ts, Rotation.from_quat(self.gt_q[idx_left : idx_right + 1]))
        
        gt_p_interp = interp1d(self.gt_ts, self.gt_p, axis=0)(ts)
        gt_rot_interp = slerp(ts)

        # 构造一个模拟位姿测量，meas[:, :3] 是旋转，meas[:, 3] 是位置。
        meas = np.zeros((3,4))
        meas[0:3,0:3] = gt_rot_interp.as_matrix()
        meas[:,3] = gt_p_interp
        meas_cov = np.eye(6)
        meas_cov[0:3,0:3] = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # rot
        meas_cov[3:6,3:6] = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # pos
        
        return True, meas, meas_cov
