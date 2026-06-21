"""
网络输入缓冲与 IMU 标定工具。

Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/net_input_utils.py

AI-IO 网络需要固定频率、固定窗口长度的输入；实际 IMU/ESC 数据时间戳可能并不刚好落在
网络采样时间点上。因此 `NetInputBuffer` 负责把相邻两帧测量线性插值到网络时间轴，
并保存最近一段 `[acc, gyro, rotor]` 窗口供网络推理。
"""

import numpy as np
from scipy.interpolate import interp1d


class ImuCalib:
    """
    简化 IMU 标定模型。

    当前主要使用离线 bias:
    - `accelBias`: 加速度 bias。
    - `gyroBias`: 陀螺仪 bias。

    scale 和 g-sensitivity 矩阵保留为单位/零矩阵，便于未来接入更完整的 IMU 标定。
    """

    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.zeros((3,3))
        self.accelBias = np.zeros((3,1))
        self.gyroBias = np.zeros((3,1))

    def from_dic(self, imu_calib_dic):
        """从 DataIO 读取到的字典中加载 gyro/accel bias。"""
        self.gyroBias = imu_calib_dic["gyro_bias"].reshape((3,1))
        self.accelBias = imu_calib_dic["accel_bias"].reshape((3,1))

    def calibrate_raw(self, acc, gyr):
        """
        对原始 IMU 做完整标定补偿，用于网络输入。

        输出:
        - `acc_cal`: 已减 accel bias 的加速度。
        - `gyr_cal`: 已减 gyro bias，并预留 g-sensitivity 补偿的角速度。
        """
        acc_cal = np.dot(self.accelScaleInv, acc) - self.accelBias
        gyr_cal = (
            np.dot(self.gyroScaleInv, gyr)
            - np.dot(self.gyroGSense, acc)
            - self.gyroBias
        )
        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        """
        只做 scale/g-sensitivity，不减 bias。

        Filter propagation 使用的状态中已经包含 ba/bg bias，因此这里不再从测量中减去 bias，
        让滤波器用状态里的 bias 项进行传播。
        """
        acc_cal = np.dot(self.accelScaleInv, acc)
        gyr_cal = np.dot(self.gyroScaleInv, gyr) - np.dot(self.gyroGSense, acc)
        return acc_cal, gyr_cal


class NetInputBuffer:
    """
    网络输入环节的时间序列缓冲区。

    保存内容:
    - `net_t_us`: 插值后的网络采样时间戳，单位 us。
    - `net_accl`: 插值后的加速度，形状 `[M, 3]`。
    - `net_gyr`: 插值后的角速度，形状 `[M, 3]`。
    - `net_rotor`: 插值后的电机转速，形状 `[M, 4]`。
    """

    def __init__(self):
        self.net_t_us = np.array([])
        self.net_accl = np.array([])
        self.net_gyr = np.array([])
        self.net_rotor = np.array([])

    def add_data_interpolated(
        self, last_t_us, t_us, last_gyr, gyr, last_accl, accl, last_rotor, rotor, requested_interpolated_t_us
    ):
        """
        将当前原始测量区间插值到网络需要的时间戳。

        `FilterRunner` 会按照网络输入频率维护 `requested_interpolated_t_us`。
        当新 IMU 帧到来时，如果该时间戳落在 `[last_t_us, t_us]` 区间内，就用线性插值
        得到网络输入点。
        """
        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)

        if last_t_us < 0:
            # 第一帧没有前一帧可插值，直接使用当前测量。
            accl_interp = accl.T
            gyr_interp = gyr.T
            rotor_interp = rotor.T
        else:
            try:
                accl_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_accl.T, accl.T]), axis=0)(requested_interpolated_t_us)
                gyr_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_gyr.T, gyr.T]), axis=0)(requested_interpolated_t_us)
                rotor_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_rotor.T, rotor.T]), axis=0)(requested_interpolated_t_us)
            except ValueError as e:
                print(
                    f"Trying to do interpolation at {requested_interpolated_t_us} between {last_t_us} and {t_us}"
                )
                raise e
        self._add_data(requested_interpolated_t_us, accl_interp, gyr_interp, rotor_interp)

    def _add_data(self, t_us, accl, gyr, rotor):
        """把一个插值后的网络输入点追加到缓冲区。"""
        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert (
                t_us > self.net_t_us[-1]
            ), f"trying to insert a data at time {t_us} which is before {self.net_t_us[-1]}"

        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_accl = np.append(self.net_accl, accl).reshape(-1, 3)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)
        self.net_rotor = np.append(self.net_rotor, rotor).reshape(-1, 4)

    def get_last_k_data(self, size):
        """返回缓冲区最后 size 个网络输入点。"""
        net_accl = self.net_accl[-size:, :]
        net_gyr = self.net_gyr[-size:, :]
        net_rotor = self.net_rotor[-size:, :]
        net_t_us = self.net_t_us[-size:]
        return net_accl, net_gyr, net_rotor, net_t_us

    def get_data_from_to(self, t_begin_us: int, t_us_end: int):
        """
        按起止时间戳取出网络窗口。

        这里用最近邻查找 begin/end index，并要求误差小于 1ms，避免时间轴错位导致网络输入
        窗口长度或内容不可靠。
        """
        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)

        begin_idx = np.argmin(np.abs(self.net_t_us - t_begin_us))
        end_idx   = np.argmin(np.abs(self.net_t_us - t_us_end))

        if abs(self.net_t_us[begin_idx] - t_begin_us) > 1000:
            raise ValueError(f"No suitable begin_idx found within 1ms for t_begin_us={t_begin_us}")
        if abs(self.net_t_us[end_idx] - t_us_end) > 1000:
            raise ValueError(f"No suitable end_idx found within 1ms for t_us_end={t_us_end}")
        net_accl = self.net_accl[begin_idx : end_idx + 1, :]
        net_gyr = self.net_gyr[begin_idx : end_idx + 1, :]
        net_rotor = self.net_rotor[begin_idx : end_idx + 1, :]
        net_t_us = self.net_t_us[begin_idx : end_idx + 1]
        return net_accl, net_gyr, net_rotor, net_t_us

    def throw_data_before(self, t_begin_us: int):
        """
        丢弃指定时间戳之前的缓存数据。

        网络只需要最近一个窗口附近的数据，及时丢弃旧数据可以避免缓冲区无限增长。
        """
        assert isinstance(t_begin_us, int)
        begin_idx = np.argmin(np.abs(self.net_t_us - t_begin_us))
        self.net_accl = self.net_accl[begin_idx:, :]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_rotor = self.net_rotor[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        """返回当前缓存中的网络输入点数量。"""
        return self.net_t_us.shape[0]

    def debugstring(self, query_us):
        """打印缓冲区时间范围，辅助排查窗口取样失败。"""
        print(f"min:{self.net_t_us[0]}")
        print(f"max:{self.net_t_us[-1]}")
        print(f"que:{query_us}")
        print(f"all:{self.net_t_us}")
