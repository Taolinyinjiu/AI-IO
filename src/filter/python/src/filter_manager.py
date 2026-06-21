"""
离线 filter 序列管理器。

Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/filter_manager.py

`FilterManager` 是离线测试入口中承上启下的一层:
- 通过 `DataIO` 读取单个序列的 hdf5 数据。
- 创建 `FilterRunner`，逐帧喂入 IMU 和电机转速。
- 在需要时用 ground truth 初始化 filter。
- 收集并保存 trajectory、velocity、bias、full_state 等结果文件。
"""

import os

import numpy as np
import progressbar
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from filter.python.src.data_io import DataIO
from filter.python.src.filter_runner import FilterRunner
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterManager:
    """
    单个序列的 filter 调度与日志管理类。

    这里不实现 EKF 数学，也不直接调用神经网络；它负责把离线数据集中的每一帧
    输入按时间顺序送入 `FilterRunner`，并把 filter 输出整理成评估脚本可读取的
    文本文件。
    """

    def __init__(self, args, sequence_name, outdir):
        self.dataset = args.dataset
        # 初始化数据读取器，并加载当前序列的 processed_data/<mode>/data.hdf5。
        self.input = DataIO()
        self.input.load(sequence_name)

        # 轨迹日志: 每行保存 ts、position、quaternion。
        outfile = os.path.join(outdir, "stamped_traj_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous trajectory log files erased")
        self.traj_outfile = outfile
        self.f_traj_logs = []

        # 速度日志: 每行保存世界系速度估计。
        outfile = os.path.join(outdir, "stamped_vel_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous velocity log files erased")
        self.vel_outfile = outfile
        self.f_vel_logs = []

        # bias 日志: 保存 gyro/accel bias 的在线估计结果。
        outfile = os.path.join(outdir, "stamped_bias_estimate.txt")
        if os.path.exists(outfile):
            os.remove(outfile)
            logging.warning("previous bias log files erased")
        self.bias_outfile = outfile
        self.f_bias_logs = []
        
        # full_state 体积较大，只在显式开启 `--log_full_state` 时写出。
        self.log_full_state = args.log_full_state
        if self.log_full_state:
            outfile = os.path.join(outdir, "full_state.txt")
            if os.path.exists(outfile):
                os.remove(outfile)
                logging.warning("previous full state log files erased")
            self.full_state_outfile = outfile
            self.full_state_logs_file = open(self.full_state_outfile, "w")
        
        # 离线标定 bias 来自数据预处理结果；若 args.initialize_with_offline_calib=True，
        # 后续会传给 FilterRunner/ImuCalib。
        imu_calibration = self.input.get_imu_calibration()

        # 汇总 filter tuning 参数。dotdict 的作用是让 ImuMSCKF 可用 config.xxx 访问。
        filter_tuning = dotdict(
            {
                "g_norm": args.g_norm, # m/s^2
                "sigma_na": args.sigma_na, # m/s^2
                "sigma_ng": args.sigma_ng, # rad/s
                "sigma_nba": args.sigma_nba, # m/s^2/sqrt(s)
                "sigma_nbg": args.sigma_nbg, # rad/s/sqrt(s)
                "init_attitude_sigma": args.init_attitude_sigma,  # rad
                "init_yaw_sigma": args.init_yaw_sigma,  # rad
                "init_vel_sigma": args.init_vel_sigma,  # m/s
                "init_pos_sigma": args.init_pos_sigma,  # m
                "init_bg_sigma": args.init_bg_sigma,  # rad/s
                "init_ba_sigma": args.init_ba_sigma,  # m/s^2
                "use_const_cov": args.use_const_cov,
                "const_cov_val_x": args.const_cov_val_x, # sigma^2
                "const_cov_val_y": args.const_cov_val_y, # sigma^2
                "const_cov_val_z": args.const_cov_val_z, # sigma^2
                "meascov_scale": args.meascov_scale,
                "mahalanobis_factor": args.mahalanobis_factor,
                "mahalanobis_fail_scale": args.mahalanobis_fail_scale
            }
        )

        # 创建单序列在线运行器。FilterRunner 负责网络输入缓存、神经网络测量和 EKF。
        if args.initialize_with_offline_calib:
            self.runner = FilterRunner(
                model_path=args.model_path,
                model_param_path=args.model_param_path,
                update_freq=args.update_freq,
                filter_tuning=filter_tuning,
                imu_calib_dic=imu_calibration)
        else:
            self.runner = FilterRunner(
                model_path=args.model_path,
                model_param_path=args.model_param_path,
                update_freq=args.update_freq,
                filter_tuning=filter_tuning)

        # full_state 的批量写缓冲，避免每一帧都触发磁盘写入。
        self.log_fullstate_buffer = None

    def __del__(self):
        """对象销毁时关闭 full_state 文件句柄，避免异常退出时文件未 flush。"""
        if self.log_full_state:
            try:
                self.full_state_logs_file.close()
            except Exception as e:
                logging.exception(e)

    def add_data_to_be_logged(self, ts, acc, gyr, with_update):
        """
        把当前 filter 状态转换成日志行并暂存在内存中。

        参数:
        - `ts`: 秒级时间戳，直接写入日志第一列。
        - `acc/gyr`: 当前 propagation 使用的 IMU 数据，仅 full_state 日志会记录。
        - `with_update`: 本帧是否完成 learned update；若没有 update，debug innovation 字段写 NaN。
        """
        # 从 EKF 中取当前演化状态: R_wi、v_wi、p_wi、ba、bg。
        R_wi, v_wi, p_wi, ba, bg = self.runner.filter.get_evolving_state()

        # 将 IMU frame 转到 body frame，便于和常见机体系轨迹定义对齐。
        # 当前 R_ib/p_ib 是单位变换，表示默认 IMU frame 与 body frame 重合。
        # 如果新无人机的 IMU 安装方向/位置不同，应在这里替换外参。
        R_ib = np.eye(3)
        p_ib = np.zeros((3,))

        # 输出姿态为 body 相对 world 的四元数，输出位置考虑 IMU-body 杆臂。
        R = R_wi @ R_ib
        v = v_wi
        p = p_wi.flatten() + R_wi @ p_ib
        q = Rotation.from_matrix(R).as_quat()
        ba = ba
        bg = bg

        traj_datapoint = np.array([
            ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
        self.f_traj_logs.append(traj_datapoint)

        # bias/velocity 日志分开保存，方便单独画图和评估。
        bias_datapoint = np.array([ts, bg[0,0], bg[1,0], bg[2,0], ba[0,0], ba[1,0], ba[2,0]])
        self.f_bias_logs.append(bias_datapoint)
        vel_datapoint = np.array([ts, v[0,0], v[1,0], v[2,0]])
        self.f_vel_logs.append(vel_datapoint)

        if self.log_full_state:
            # full_state 记录更完整的调试量: 协方差对角线、innovation、测量/预测值等。
            _, Sigma15 = self.runner.filter.get_covariance()
            sigmas = np.diag(Sigma15).reshape(15, 1)
            sigmasyawp = self.runner.filter.get_covariance_yawp().reshape(16, 1)
            inno, meas, pred, meas_sigma, inno_sigma = self.runner.filter.get_debug()

            # propagation-only 帧没有神经网络更新，相关 debug 字段没有物理意义。
            if not with_update:
                inno *= np.nan
                meas *= np.nan
                pred *= np.nan
                meas_sigma *= np.nan
                inno_sigma *= np.nan

            ts_temp = ts.reshape(1, 1)
            temp = np.concatenate(
                [v, p, ba, bg, acc, gyr, ts_temp, sigmas, inno, \
                    meas, pred, meas_sigma, inno_sigma, sigmasyawp], axis=0)
            vec_flat = np.append(R.ravel(), temp.ravel(), axis=0)

            # 批量缓存超过 100 行再写入文件，降低 I/O 频率。
            if self.log_fullstate_buffer is None:
                self.log_fullstate_buffer = vec_flat
            else:
                self.log_fullstate_buffer = np.vstack((self.log_fullstate_buffer, vec_flat))

            if self.log_fullstate_buffer.shape[0] > 100:
                np.savetxt(self.full_state_logs_file, self.log_fullstate_buffer, delimiter=",")
                self.log_fullstate_buffer = None

    def save_logs(self, save_as_npy):
        """把内存中的 trajectory、velocity、bias、full_state 写到输出目录。"""
        logging.info("Saving logs!")
        np.savetxt(self.traj_outfile, np.array(self.f_traj_logs),
                   header="ts x y z qx qy qz qw", fmt=["%.3f"] + ["%.12f"] * 7)
        np.savetxt(self.bias_outfile, np.array(self.f_bias_logs),
                   header="ts bg_x bg_y bg_z ba_x ba_y ba_z", fmt="%.3f")
        np.savetxt(self.vel_outfile, np.array(self.f_vel_logs),
                   header="ts v_x v_y v_z", fmt="%.3f")

        if self.log_full_state:
            # 写出最后不足 100 行的 full_state 缓冲。
            np.savetxt(self.full_state_logs_file, self.log_fullstate_buffer, delimiter=",")
            self.log_fullstate_buffer = None
            self.full_state_logs_file.close()

            if save_as_npy:
                # 将 .txt 转成 .npy 节省空间，并提升后续加载速度。
                states = np.loadtxt(self.full_state_outfile, delimiter=",")
                np.save(self.full_state_outfile[:-4] + ".npy", states)
                os.remove(self.full_state_outfile)

    def run(self, args):
        """
        遍历完整离线序列并驱动 filter。

        如果 `initialize_with_gt=True`，第一帧会直接用 ground truth 的 R/v/p 初始化，
        这通常用于离线评估，能够把重点放在模型测量和 propagation 本身；真实在线系统
        不能依赖 ground truth，应使用普通 IMU 初始化或外部定位初始化。
        """
        # 顺序遍历整个数据集，把每帧 IMU/rotor 送给 FilterRunner。
        n_data = self.input.dataset_size
        for i in progressbar.progressbar(range(n_data), redirect_stdout=True):
            # 从 DataIO 取下一帧原始 IMU 和四电机转速。
            ts, acc_raw, gyr_raw, rotor_spd = self.input.get_datai(i)
            t_us = from_sec_to_usec(ts)

            if self.runner.filter.initialized:
                # 常规路径: propagation 每帧执行，learned update 按 update_freq 间隔执行。
                did_update = self.runner.on_imu_measurement(t_us, gyr_raw, acc_raw, rotor_spd)
                self.add_data_to_be_logged(
                    ts,
                    self.runner.last_acc,
                    self.runner.last_gyr,
                    with_update=did_update
                )
            else:
                # 初始化路径: 可选用 IMU 自初始化，也可用 ground truth 初始化。
                if not args.initialize_with_gt:
                    self.runner.on_imu_measurement(t_us, gyr_raw, acc_raw, rotor_spd)
                else:
                    # bias 初值优先使用离线标定；否则使用零 bias。
                    if args.initialize_with_offline_calib:
                        init_ba = self.runner.icalib.accelBias
                        init_bg = self.runner.icalib.gyroBias
                    else:
                        init_ba = np.zeros((3, 1))
                        init_bg = np.zeros((3, 1))

                    # 在当前 IMU 时间戳上插值 ground truth 的位置、速度和姿态。
                    gt_p = interp1d(self.input.gt_ts, self.input.gt_p, axis=0)(ts)
                    gt_v = interp1d(self.input.gt_ts, self.input.gt_v, axis=0)(ts)
                    gt_rot = Slerp(self.input.gt_ts, Rotation.from_quat(self.input.gt_q))(ts)
                    gt_R = gt_rot.as_matrix()
                    self.runner.filter.initialize_with_state(
                        t_us,
                        gt_R,
                        np.atleast_2d(gt_v).T,
                        np.atleast_2d(gt_p).T,
                        init_ba,
                        init_bg,
                    )
                    # 下一次插值/update 从初始化时间开始对齐。
                    self.runner.next_update_t_us = t_us
                    self.runner.next_interp_t_us = t_us

        self.save_logs(args.save_as_npy)

    def reset_filter_state_from_groundtruth(self, this: FilterRunner):
        """
        用 ground truth 重置当前状态和历史 clone 状态。

        这个函数通常作为 `FilterRunner.callback_first_update` 使用，便于在第一次网络
        测量更新前把状态对齐到真值，减少初始误差对离线评估的影响。
        """
        # 对所有历史 clone 的时间戳分别插值真值 R/p/v。
        inp = self.input
        state = this.filter.state
        gt_ps = []
        gt_Rs = []
        gt_vs = []
        for _, ts_i_us in enumerate(state.si_timestamps_us):
            ts_i = from_usec_to_sec(ts_i_us)
            ps = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_p, axis=0)(ts_i)).T
            gt_ps.append(ps)
            gt_rots = Slerp(self.input.gt_ts, Rotation.from_quat(inp.gt_q))(ts_i)
            gt_Rs.append(gt_rots.as_matrix())
            vs = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_v, axis=0)(ts_i)).T
            gt_vs.append(vs)

        # 对当前 evolving state 的时间戳插值真值 R/p/v。
        ts = from_usec_to_sec(state.s_timestamp_us)
        gt_p = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_p, axis=0)(ts)).T
        gt_v = np.atleast_2d(interp1d(inp.gt_ts, inp.gt_v, axis=0)(ts)).T
        gt_rot = Slerp(self.input.gt_ts, Rotation.from_quat(inp.gt_q))(ts)
        gt_R = gt_rot.as_matrix()

        this.filter.reset_state_and_covariance(
            gt_Rs, gt_ps, gt_vs, gt_R, gt_v, gt_p, state.s_ba, state.s_bg
        )

    def reset_filter_state_pv(self):
        """
        只把 position/velocity 重置为零，保留姿态和 bias。

        该函数更像调试辅助接口，主离线测试流程默认不会调用。
        """
        state = self.runner.filter.state
        ps = []
        vs = []
        for i in state.si_timestamps:
            ps.append(np.zeros((3, 1)))
            vs.append(np.zeros((3, 1)))
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        self.runner.filter.reset_state_and_covariance(
            state.si_Rs, ps, vs, state.s_R, v, p, state.s_ba, state.s_bg
        )
