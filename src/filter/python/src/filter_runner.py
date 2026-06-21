"""
AI-IO filter 的在线运行器。

Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/filter_runner.py

`FilterRunner` 是单帧数据进入滤波器的主入口。它负责:
- 接收逐帧 IMU 和 rotor speed。
- 按网络要求插值/缓存固定频率输入窗口。
- 调用神经网络得到 learned body velocity measurement。
- 调用 `ImuMSCKF` 做 propagation 和 update。
"""

import json

from numba import jit
import numpy as np

from filter.python.src.meas_source_network import MeasSourceNetwork
from filter.python.src.net_input_utils import NetInputBuffer, ImuCalib
from filter.python.src.scekf import ImuMSCKF
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.math_utils import mat_exp
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterRunner:
    """
    管理单条数据流上的 AI-IO 滤波过程。

    典型调用顺序:
    1. `on_imu_measurement()` 收到一帧 IMU + rotor。
    2. 如果 filter 未初始化，则用当前 IMU 初始化姿态/状态。
    3. 如果已初始化，则先用 IMU propagation。
    4. 到达网络 update 时间点时，取出最近窗口并调用网络。
    5. 把网络输出作为速度观测更新 EKF。
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning,
        imu_calib_dic=None,
        force_cpu=False,
    ):
        # 从训练保存的 model_net_parameters.json 中读取网络输入频率和窗口长度。
        # 这两个参数必须和 checkpoint 训练时一致。
        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["sampling_freq"]
            config_from_network["window_time"] = data_json["window_time"]

        # 网络输入频率和窗口点数换算。
        self.imu_freq_net = config_from_network.imu_freq_net
        window_size = int(
            (config_from_network.window_time * config_from_network.imu_freq_net) )
        self.net_input_size = window_size

        # update_freq 是 learned measurement 的更新频率。
        # 代码要求网络输入频率能整除 update 频率，保证每次 update 窗口端点落在插值网格上。
        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError("window_time cannot be represented by integer number of updates.")
        self.update_freq = update_freq

        # 时间间隔统一用微秒整数，避免长序列浮点时间累积误差。
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(1.0 / self.update_freq * 1e6)
        self.dt_window_us = int(config_from_network.window_time * 1e6)

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} (samples)"
        )
        logging.info("IMU and rotor speed input to the network frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            f"Interpolating IMU and rotor speed measurements every {self.dt_interp_us} [us] for the network input"
        )

        # IMU 离线标定参数，用于给网络输入减 bias，同时让 filter 状态持有 bias。
        self.icalib = ImuCalib()
        self.icalib.from_dic(imu_calib_dic)

        # SCEKF/MSCKF 滤波器核心。
        self.filter = ImuMSCKF(filter_tuning)

        # 神经网络测量源，输出 body velocity measurement + covariance。
        self.meas_source = MeasSourceNetwork(model_path, force_cpu)

        # 缓存插值后的网络输入窗口。
        self.inputs_buffer = NetInputBuffer()

        # 可选回调: 第一次 learned update 前执行，用于离线评估时用 ground truth 重置状态。
        self.callback_first_update = None

        # 保存上一帧测量，用于对当前帧与上一帧之间做线性插值。
        self.last_t_us, self.last_acc, self.last_gyr = -1, None, None
        self.last_rotor = None
        self.next_interp_t_us = None
        self.next_update_t_us = None
        self.has_done_first_update = False

    @jit(forceobj=True, parallel=False, cache=False)
    def _get_inputs_samples_for_network(self, t_begin_us, t_end_us):
        """
        从缓冲区取出一个完整网络输入窗口。

        注意:
        - 网络输入使用经过离线 bias 补偿的 IMU。
        - 输出时间 `net_t_s` 目前不直接进入网络，但保留给扩展/调试。
        """
        net_ts_begin = t_begin_us
        net_ts_end = t_end_us - self.dt_interp_us

        net_accl, net_gyr, net_rotor, net_t_us = self.inputs_buffer.get_data_from_to(
            net_ts_begin, net_ts_end
        )

        assert net_gyr.shape[0] == self.net_input_size
        assert net_accl.shape[0] == self.net_input_size
        assert net_rotor.shape[0] == self.net_input_size
        
        net_t_s = from_usec_to_sec(net_t_us)

        return net_accl, net_gyr, net_rotor, net_t_s

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw, rotor_spd):
        """
        单帧 IMU + rotor 输入入口。

        返回值:
        - `False`: 本帧只做初始化或 propagation，没有成功完成 learned update。
        - `True`: 本帧完成 learned measurement update。
        """
        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw, rotor_spd)
        else:
            logging.info(f"Initializing filter at time {t_us} [us]")
            if self.icalib:
                logging.info(f"Using bias from initial calibration")
                init_ba = self.icalib.accelBias
                init_bg = self.icalib.gyroBias
                # 网络输入使用减 bias 的 IMU；filter 初始 bias 用离线标定值。
                acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                    acc_raw, gyr_raw) 
            else:
                logging.info(f"Using zero bias")
                init_ba = np.zeros((3,1))
                init_bg = np.zeros((3,1))
                acc_biascpst, gyr_biascpst = acc_raw, gyr_raw

            # 用第一帧加速度方向估计重力方向并初始化姿态，位置/速度初值为零。
            self.filter.initialize(acc_biascpst, t_us, init_ba, init_bg)
            self.next_interp_t_us = t_us
            self.next_update_t_us = t_us
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, t_us)
            self.next_update_t_us = t_us + self.dt_update_us
            self.last_t_us, self.last_acc, self.last_gyr = (
                t_us,
                acc_biascpst,
                gyr_biascpst,
            )
            self.last_rotor = rotor_spd
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw, rotor_spd):
        """
        filter 初始化后的单帧处理流程。

        该函数同时维护两条 IMU 数据流:
        - 网络输入: 使用离线 bias 补偿后的 IMU。
        - EKF propagation: 使用 scale 后但不减 bias 的 IMU，由状态里的 ba/bg 参与传播。
        """
        if self.icalib:
            # 给网络输入使用的 IMU: 减去离线 bias。
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )

            # 给 filter propagation 使用的 IMU: 不减 bias，bias 由滤波器状态估计。
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # 判断当前帧时间是否已经覆盖下一个网络插值点/更新点。
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_update = t_us >= self.next_update_t_us

        # 如果要做 learned update，必须也已经完成对应网络端点的插值。
        assert (
            do_update and do_interpolation_of_imu
        ) or not do_update, (
            "Update and interpolation does not match!"
        )

        # 将当前测量插值到网络输入时间轴并缓存。
        if do_interpolation_of_imu:
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, rotor_spd, t_us)
                
        # IMU propagation 每帧都执行，更新 R/v/p/bias covariance。
        self.filter.propagate(
            acc_raw, gyr_raw, t_us
        )

        # 到达 learned measurement 更新时刻才调用神经网络。
        did_update = False
        if do_update:
            did_update = self._process_update(t_us)
            # 规划下一次 learned update 时间。
            self.next_update_t_us += self.dt_update_us

        # 保存当前帧，供下一帧插值使用。
        self.last_t_us, self.last_acc, self.last_gyr = t_us, acc_biascpst, gyr_biascpst
        self.last_rotor = rotor_spd

        return did_update

    def _process_update(self, t_us):
        """
        执行一次 learned velocity measurement update。

        流程:
        1. 根据窗口长度确定 `[t_begin_us, t_us)`。
        2. 从缓冲区取出 acc/gyro/rotor 窗口。
        3. 神经网络输出 body velocity measurement 和 covariance。
        4. `ImuMSCKF` 计算 innovation/Jacobian。
        5. 通过 Kalman update 修正状态。
        """
        t_end_us = t_us
        t_begin_us = t_end_us - self.dt_window_us

        # 初始阶段窗口长度不足时跳过 learned update。
        if t_begin_us < self.inputs_buffer.net_t_us[0]:
            return False

        # 离线评估可在第一次 update 前用 ground truth 重置状态。
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)

        # 从缓冲区取固定长度网络输入窗口。
        net_accl_b, net_gyr_b, net_rotor, net_t_s = self._get_inputs_samples_for_network(
            t_begin_us, t_end_us)

        # 网络输出机体系速度测量与测量协方差。
        meas, meas_cov = self.meas_source.get_measurement(
            net_t_s, net_accl_b, net_gyr_b, net_rotor)

        # 把网络测量转换为 EKF innovation/Jacobian/noise。
        is_available, innovation, jac, noise_mat = \
            self.filter.learnt_model_update(meas, meas_cov)
        success = False
        if is_available:
            success = self.filter.apply_update(innovation, jac, noise_mat)

        self.has_done_first_update = True
        # 丢弃窗口起点之前的旧缓存，控制内存增长。
        self.inputs_buffer.throw_data_before(t_begin_us)
        return success

    def _add_interpolated_inputs_to_buffer(self, accl_biascpst, gyr_biascpst, rotor_spd, t_us):
        """把当前测量区间插值到网络时间轴，并推进下一个插值时间点。"""
        self.inputs_buffer.add_data_interpolated(
            self.last_t_us,
            t_us,
            self.last_gyr,
            gyr_biascpst,
            self.last_acc,
            accl_biascpst,
            self.last_rotor,
            rotor_spd,
            self.next_interp_t_us,
        )
        self.next_interp_t_us += self.dt_interp_us
