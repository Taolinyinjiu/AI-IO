"""
AI-IO 的 SCEKF/MSCKF 滤波器核心。

Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/scekf.py

本文件实现的关键思想是:
- 用 IMU 原始测量做连续 propagation，维护姿态、速度、位置和 IMU bias。
- 用神经网络输出的机体系速度 `v_body` 作为 measurement update。
- 用协方差矩阵描述状态不确定性，并用 Kalman update 修正状态。
- 保留历史 clone state 的结构，便于扩展相对位置/视觉类约束。
"""

from numba import jit
import numpy as np

from filter.python.src.utils.logging import logging
from filter.python.src.utils.math_utils import Jr_exp, hat, mat_exp, mat_exp_vec, rot_2vec


class State(object):
    """
    EKF 状态容器。

    当前 evolving state 为 15 维误差状态:
    - attitude error: 3
    - velocity error: 3
    - position error: 3
    - gyro bias error: 3
    - accel bias error: 3

    历史 clone state 每个为 9 维:
    - attitude / velocity / position，各 3 维。
    """

    def __init__(self):
        super(State, self).__init__()
        # 当前时刻的名义状态: R_wi、v_wi、p_wi、ba、bg。
        self.s_R = None
        self.s_v = None
        self.s_p = None
        self.s_ba = None
        self.s_bg = None
        self.s_timestamp_us = -1  # current state time
        self.N = 0  # 历史 clone state 数量。
        self.si_Rs = []  # 历史 clone 姿态。
        self.si_ps = []  # 历史 clone 位置。
        self.si_vs = []  # 历史 clone 速度。
        # FEJ(first-estimates Jacobian) 版本的 clone 状态，用于保持可观性一致性。
        self.si_Rs_fej = []
        self.si_ps_fej = []
        self.si_vs_fej = []
        self.si_timestamps_us = []
        # 不可观方向基，用于调试 yaw/position 等不可观自由度的信息量。
        self.unobs_shift = None

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        """用给定 R/v/p/bias 初始化当前状态，并清空所有历史 clone。"""
        assert isinstance(t_us, int)
        self.s_R = R
        self.s_v = v  # m/s
        self.s_p = p  # m
        self.s_bg = bg_init  # rad/s
        self.s_ba = ba_init  # m/s^2
        self.s_timestamp_us = t_us
        self.si_Rs = []
        self.si_ps = []
        self.si_ss = []
        self.si_Rs_fej = []
        self.si_ps_fej = []
        self.si_vs_fej = []

        self.si_timestamps_us = []
        self.unobs_shift = self.generate_unobservable_shift()

    def reset_state(self, Rs, ps, vs, R, v, p, ba_init, bg_init):
        """重置当前状态和已有历史 clone，通常用于 ground truth 对齐调试。"""
        self.s_R = R
        self.s_v = v  # m/s
        self.s_p = p  # m
        self.s_bg = bg_init  # rad/s
        self.s_ba = ba_init  # m/s^2
        self.si_Rs = Rs
        self.si_ps = ps
        self.si_vs = vs
        self.si_Rs_fej = Rs
        self.si_ps_fej = ps
        self.si_vs_fej = vs

    def __repr__(self):
        return f"R:\n{self.s_R}\nv:\n{self.s_v}\np:\n{self.s_p}\nbg:\n{self.s_bg}\nba:\n{self.s_ba}"

    def apply_correction(self, dX):
        """
        将 Kalman update 得到的误差状态 `dX` 注入名义状态。

        姿态误差使用 SO(3) 指数映射左乘修正；速度、位置和 bias 使用加法修正。
        `dX` 前半部分对应历史 clone，最后 15 维对应当前 evolving state。
        """
        dX_past = dX[:-15]
        dX_evol = dX[-15:]
        assert dX_past.flatten().shape[0] == (
            self.N * 9
        ), f"number of past error states {dX_past.flatten().shape[0]} does not match the number of states in the filter! {self.N * 9}"

        if self.N > 0:
            # 历史 clone 每个 9 维: dtheta, dv, dp。
            temp = dX_past.reshape((self.N, 9))
            dvs = np.expand_dims(temp[:, 3:6], axis=2)  # Nx6x1
            dps = np.expand_dims(temp[:, 6:9], axis=2)  # Nx6x1
            dthetas = temp[:, 0:3]
            # 批量把小角度误差映射为旋转矩阵，再作用到历史姿态上。
            dRs = mat_exp_vec(dthetas)  # Nx3x3
            Rs_past = np.stack(self.si_Rs, axis=0)  # Nx3x3
            vs_past = np.stack(self.si_vs, axis=0)  # Nx3x3
            ps_past = np.stack(self.si_ps, axis=0)  # Nx3x3
            Rs_past_new = np.matmul(dRs, Rs_past)
            ps_past_new = ps_past + dps
            vs_past_new = vs_past + dvs

            N = Rs_past.shape[0]
            
            self.si_Rs = np.split(Rs_past_new.reshape(N * 3, 3), N, 0)
            self.si_vs = np.split(vs_past_new.reshape(N * 3, 1), N, 0)
            self.si_ps = np.split(ps_past_new.reshape(N * 3, 1), N, 0)

        # 更新当前 evolving state。误差状态顺序必须与 H/Jacobian 的定义一致。
        dtheta = dX_evol[:3]
        dv = dX_evol[3:6]
        dp = dX_evol[6:9]
        dbg = dX_evol[9:12]
        dba = dX_evol[12:15]
        dR = mat_exp(dtheta)

        self.s_R = dR.dot(self.s_R)
        self.s_v = self.s_v + dv
        self.s_p = self.s_p + dp
        self.s_bg = self.s_bg + dbg
        self.s_ba = self.s_ba + dba

    def generate_unobservable_shift(self):
        """
        生成不可观方向的误差状态基。

        对纯惯性/速度观测系统而言，全局 yaw 和全局平移通常不可由内部测量唯一确定。
        这里返回 15x4 基向量，用于后续 `get_info_along_unobservable_shift()` 调试。
        """
        assert self.N == 0
        g = np.array([[0], [0], [1]])
        dX = np.zeros((15, 4))
        dX[0:3, [0]] = g
        dX[3:6, [0]] = -hat(self.s_v) @ g
        dX[6:9, [0]] = -hat(self.s_p) @ g
        dX[6:9, 1:4] = np.eye(3)
        return dX


@jit(nopython=True, parallel=False, cache=True)
def propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, g, dt):
    """
    单步 IMU propagation，并返回状态转移 Jacobian。

    输入使用当前名义状态和一帧 IMU 测量:
    - gyro 积分更新姿态。
    - accel 经姿态旋到世界系后积分速度/位置。
    - bias 在 propagation 中假设常值，协方差里用随机游走建模。
    """
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    # 扣除当前 bias 后积分角增量。
    dtheta = (gyr - b_gk) * dt
    dRd = mat_exp(dtheta)
    Rd = R_k @ dRd

    # 加速度从 IMU/body 系旋到 world 系，再叠加重力。
    dv_w = R_k @ (acc - b_ak) * dt
    dp_w = 0.5 * dv_w * dt
    gdt = g * dt
    gdt22 = 0.5 * gdt * dt
    vd = v_k + dv_w + gdt
    pd = p_k + v_k * dt + dp_w + gdt22

    # A 是 15 维误差状态的离散状态转移矩阵。
    A = np.eye(15)
    A[3:6, 0:3] = -hat(dv_w)
    A[6:9, 0:3] = -hat(dp_w)
    A[6:9, 3:6] = np.eye(3) * dt
    A[0:3, 9:12] = -Rd @ Jr_exp(dtheta) * dt
    A[3:6, 12:15] = -R_k * dt
    A[6:9, 12:15] = -0.5 * R_k * dt * dt

    return Rd, vd, pd, A


def get_rotation_from_gravity(acc):
    """用第一帧加速度方向估计初始姿态，使 IMU 测到的重力方向对齐世界 z 轴。"""
    # take the first accel data to get gravity direction
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return rot_2vec(acc, ig_w)


class ImuMSCKF:
    """
    AI-IO 使用的误差状态 Kalman filter。

    名义状态由 `State` 保存；协方差 `Sigma` 保存历史 clone + 当前 15 维状态的
    联合不确定性。网络速度测量进入 `learnt_model_update()`，最终由
    `apply_update()` 执行标准 Kalman 修正。
    """

    def __init__(self, config=None):

        # 参数完整性检查。缺失项会使用本类中的默认值。
        expected_attribute = [
            "sigma_na",
            "sigma_ng",
            "sigma_nba",
            "sigma_nbg",
            "init_attitude_sigma",
            "init_yaw_sigma",
            "init_vel_sigma",
            "init_pos_sigma",
            "init_bg_sigma",
            "init_ba_sigma",
            "zero_vel_sigma",
            "use_const_cov",
            "const_cov_val_x",
            "const_cov_val_y",
            "const_cov_val_z",
            "meascov_scale",
            "mahalanobis_fail_scale",
            "mahalanobis_factor"
        ]
        if not all(hasattr(config, attr) for attr in expected_attribute):
            logging.warning(
                "At least one filter parameter tuning will be left at its default value."
            )

        # 重力常量，世界系 z 轴向上时重力为负 z。
        g_norm = getattr(config, "g_norm", 9.8082)
        self.g = np.array([0, 0, -g_norm]).reshape((3, 1))

        # IMU 白噪声和 bias 随机游走噪声，进入 propagation covariance。
        self.sigma_na = getattr(config, "sigma_na", np.sqrt(1e-3))  # accel noise m/s^2
        self.sigma_ng = getattr(config, "sigma_ng", np.sqrt(1e-4))  # gyro noise rad/s
        self.sigma_nba = getattr(config, "sigma_nba", 1e-4)  # accel bias noise m/s^2/sqrt(s)
        self.sigma_nbg = getattr(config, "sigma_nbg", 1e-6)  # gyro bias noise rad/s/sqrt(s)

        # 初始不确定性，决定初始化后滤波器对初始姿态/速度/位置/bias 的信任程度。
        self.init_attitude_sigma = getattr(
            config, "init_attitude_sigma", 10.0 / 180.0 * np.pi
        )  # rad
        self.init_yaw_sigma = getattr(
            config, "init_yaw_sigma", 0.1 / 180.0 * np.pi
        )  # rad
        self.init_vel_sigma = getattr(config, "init_vel_sigma", 1.0)  # m/s
        self.init_pos_sigma = getattr(config, "init_pos_sigma", 0.001)  # m
        self.init_bg_sigma = getattr(config, "init_bg_sigma", 0.0001)  # rad/s
        self.init_ba_sigma = getattr(config, "init_ba_sigma", 0.2)  # m/s^2

        self.zero_vel_sigma = getattr(config, "zero_vel_sigma", 1e-2)  # m/s

        # measurement covariance 缩放和 Mahalanobis gating 参数。
        self.meascov_scale = getattr(config, "meascov_scale", 1.0)
        self.mahalanobis_factor = getattr(config, "mahalanobis_factor", 1.0)
        if self.mahalanobis_factor <= 0.0:
            logging.warning("Mahalanobis gating test deactivated!")

        self.use_const_cov = getattr(config, "use_const_cov", False)
        if self.use_const_cov:
            self.const_cov_val_x = config.const_cov_val_x
            self.const_cov_val_y = config.const_cov_val_y
            self.const_cov_val_z = config.const_cov_val_z

        # Mahalanobis 失败后的处理策略: 0 表示直接跳过该次 update，非 0 表示放大 R。
        self.mahalanobis_fail_scale = getattr(config, "mahalanobis_fail_scale", 0)
        self.last_success_mahalanobis = None
        self.force_mahalanobis_until = None

        # 噪声/协方差矩阵。Sigma 是完整状态协方差，Sigma15 是当前 evolving state 的引用块。
        self.W = None  # IMU measurement noise
        self.Q = None  # stochastic noise (random walk)
        self.R = None  # measurement noise
        self.Sigma = None  # full state covariance
        self.Sigma15 = None  # evolving state covariance
        self.state = State()
        self.last_timestamp_reset_us = None

        # 预留的在线 IMU 插值数据缓存，目前主流程未使用。
        self.imu_data_int = np.array([])

        # filter 生命周期标志。
        self.initialized = False
        self.converged = False
        self.first_update = True

        # debug log: full_state 日志会读取这些字段。
        self.innovation = np.zeros((3, 1))
        self.meas = np.zeros((3, 1))
        self.pred = np.zeros((3, 1))
        self.meas_sigma = np.zeros((3, 1))
        self.inno_sigma = np.zeros((3, 1))

    def reset_covariance(self):
        """
        根据当前状态维度重置协方差矩阵。

        维度为 `15 + 9 * N`:
        - 前面 `9 * N` 是历史 clone。
        - 最后 15 维是当前姿态/速度/位置/gyro bias/accel bias。
        """
        var_atti = np.power(self.init_attitude_sigma, 2.0)
        var_yaw = np.power(self.init_yaw_sigma, 2.0)
        var_vel = np.power(self.init_vel_sigma, 2.0)
        var_pos = np.power(self.init_pos_sigma, 2.0)
        var_bg_init = np.power(self.init_bg_sigma, 2.0)
        var_ba_init = np.power(self.init_ba_sigma, 2.0)

        Cov = np.zeros((15 + 9 * self.state.N, 15 + 9 * self.state.N))
        for i, _ in enumerate(self.state.si_timestamps_us):
            # 每个历史 clone 只包含 attitude/velocity/position 三类误差。
            Cov[9 * i :, 9 * i :][0:3, 0:3] = np.diag([var_atti, var_atti, var_yaw])
            Cov[9 * i :, 9 * i :][3:6, 3:6] = np.diag([var_vel, var_vel, var_vel])
            Cov[9 * i :, 9 * i :][6:9, 6:9] = np.diag([var_pos, var_pos, var_pos])

        # 最后 15 维是当前 evolving state，其中 bias 只有当前状态持有。
        Cov15 = Cov[-15:, -15:]  # no copy, ref
        Cov15[:3, :3] = np.diag(np.array([var_atti, var_atti, var_yaw]))
        Cov15[3:6, 3:6] = np.diag(np.array([var_vel, var_vel, var_vel]))
        Cov15[6:9, 6:9] = np.diag(np.array([var_pos, var_pos, var_pos]))
        Cov15[9:12, 9:12] = np.diag(np.array([var_bg_init, var_bg_init, var_bg_init]))
        Cov15[12:15, 12:15] = np.diag(np.array([var_ba_init, var_ba_init, var_ba_init]))
        self.Sigma = Cov
        self.Sigma15 = Cov[-15:, -15:]

    def prepare_filter(self):
        """构造 propagation 所需噪声矩阵，并初始化协方差。"""
        # define noise covariances
        var_a = np.power(self.sigma_na, 2.0)
        var_g = np.power(self.sigma_ng, 2.0)
        var_ba = np.power(self.sigma_nba, 2.0)
        var_bg = np.power(self.sigma_nbg, 2.0)
        self.W = np.diag(np.array([var_g, var_g, var_g, var_a, var_a, var_a]))
        self.Q = np.diag(np.array([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba]))
        # initialize state covariance
        self.reset_covariance()

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        """只初始化名义状态，不单独重置协方差。"""
        self.state.initialize_state(t_us, R, v, p, ba_init, bg_init)
        self.last_timestamp_reset_us = t_us

    def reset_state_and_covariance(self, Rs, ps, vs, R, v, p, ba_init, bg_init):
        """重置名义状态并重新初始化协方差，常用于离线 ground truth reset。"""
        assert len(Rs) == self.state.N
        assert len(ps) == self.state.N
        assert len(vs) == self.state.N

        self.state.reset_state(Rs, ps, vs, R, v, p, ba_init, bg_init)
        self.reset_covariance()
        self.last_timestamp_reset_us = self.state.s_timestamp_us

        assert len(Rs) == self.state.N
        assert len(ps) == self.state.N
        assert len(vs) == self.state.N
        assert self.Sigma.shape[0] == self.state.N * 9 + 15
        assert self.Sigma.shape[1] == self.state.N * 9 + 15

    def initialize_with_state(self, t_us, R, v, p, ba_init, bg_init):
        """用外部给定完整状态初始化 filter，例如离线评估用 ground truth 初始化。"""
        assert isinstance(t_us, int)
        self.prepare_filter()
        self.initialized = True
        self.initialize_state(t_us, R, v, p, ba_init, bg_init)
        logging.info("filter initialized with full state!")

    def initialize(self, acc, t_us, ba_init, bg_init):
        """用第一帧加速度估计初始姿态，速度和位置设为零。"""
        assert isinstance(t_us, int)
        self.prepare_filter()
        self.initialized = True
        self.initialize_state(
            t_us,
            get_rotation_from_gravity(acc),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            ba_init,
            bg_init,
        )
        logging.info("filter initialized with gravity!")

    def get_past_state(self, t_us):
        """按时间戳取一个历史 clone state。"""
        assert isinstance(t_us, int)
        state_idx = self.state.si_timestamps_us.index(t_us)
        R = self.state.si_Rs[state_idx]
        p = self.state.si_ps[state_idx]
        v = self.state.si_vs[state_idx]

        return R, p, v

    def get_evolving_state(self):
        """返回当前名义状态，供日志和外部调用读取。"""
        R = self.state.s_R
        v = self.state.s_v
        p = self.state.s_p
        bg = self.state.s_bg
        ba = self.state.s_ba
        return R, v, p, ba, bg

    def get_covariance(self):
        """返回完整协方差和当前 15 维状态协方差。"""
        Sigma = self.Sigma
        Sigma15 = self.Sigma15
        return Sigma, Sigma15

    def get_covariance_yawp(self):
        """返回 yaw + position 相关的 4x4 协方差块，便于观察不可观自由度。"""
        return self.Sigma15[[2, 6, 7, 8], :][:, [2, 6, 7, 8]]

    def get_info_along_unobservable_shift(self):
        """计算不可观方向上的信息量，用于调试滤波器可观性一致性。"""
        return np.diag(
            self.state.unobs_shift.T
            @ np.linalg.pinv(self.Sigma)
            @ self.state.unobs_shift
        )

    def get_debug(self):
        """返回最近一次 measurement update 的调试量。"""
        return self.innovation, self.meas, self.pred, self.meas_sigma, self.inno_sigma

    def check_filter_convergence(self):
        """简化假设: 重置后运行超过 10 秒认为滤波器已收敛。"""
        return self.state.si_timestamps_us[0] - self.last_timestamp_reset_us > int(
            10 * 1e6
        )

    def is_mahalanobis_activated(self):
        """
        判断当前是否启用 Mahalanobis gating。

        逻辑:
        - filter 未收敛时不启用，避免初始阶段误拒绝。
        - 如果连续失败太久，临时关闭一段时间，防止滤波器长期没有更新。
        """
        if not self.converged:
            return False

        # 强制开启窗口内，直接启用 gating。
        if self.force_mahalanobis_until is not None:
            if self.state.s_timestamp_us < self.force_mahalanobis_until:
                return True

        # 如果距离上次成功 gating 已经超过 0.5 秒，认为失败过久，临时关闭。
        if self.last_success_mahalanobis is not None:
            if self.state.s_timestamp_us > self.last_success_mahalanobis + int(
                0.5 * 1e6
            ):
                logging.warning(
                    "Deactivating Mahalanobis test, because failed for too long."
                )
                self.last_success_mahalanobis = None
                self.force_mahalanobis_until = self.state.s_timestamp_us + int(1e6)
                return False
        return True

    def propagate(self, acc, gyr, t_us, t_augmentation_us=None):
        """
        用一帧 IMU 测量推进 filter 状态和协方差。

        参数:
        - `acc/gyr`: 原始或仅 scale 后的 IMU 测量，bias 由状态 `s_ba/s_bg` 扣除。
        - `t_us`: 当前测量时间戳。
        - `t_augmentation_us`: 若给定，则在该中间时刻插入一个历史 clone。
        """

        R_k, v_k, p_k, b_ak, b_gk = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
            self.state.s_ba,
            self.state.s_bg,
        )

        N = self.state.N
        # 当前 15 维 evolving state propagation。
        dt_us = t_us - self.state.s_timestamp_us
        R_kp1, v_kp1, p_kp1, Akp1 = propagate_rvt_and_jac(
            R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dt_us * 1e-6
        )
        b_gkp1 = b_gk
        b_akp1 = b_ak

        # B 将 IMU 白噪声映射到 15 维误差状态。
        B = np.zeros((15, 6))
        B[0:3, 0:3] = -Akp1[0:3, 9:12]
        B[3:6, 3:6] = -Akp1[3:6, 12:15]
        B[6:9, 3:6] = -Akp1[6:9, 12:15]

        # 如果需要状态扩展，则在 t_augmentation_us 对当前状态做部分积分并插入 clone。
        if t_augmentation_us:
            # past state propagation (partial integration)
            dtd_us = t_augmentation_us - self.state.s_timestamp_us
            Rd, vd, pd, Ad = propagate_rvt_and_jac(
                R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dtd_us * 1e-6
            )

            # JA 是 clone state 对当前 15 维误差状态的 Jacobian。
            JA = np.zeros((9, 15))
            JA[0:3, :] = Ad[0:3, :]
            JA[3:6, :] = Ad[3:6, :]
            JA[6:9, :] = Ad[6:9, :]

            # A_aug 同时描述“保留旧 clone、插入新 clone、推进当前状态”的线性映射。
            A_aug = np.zeros(((15 + 9 * (N + 1)), (15 + 9 * N)))
            A_aug[0 : 9 * N, 0 : 9 * N] = np.eye(9 * N)
            A_aug[-15 - 9 : -15, -15:] = JA
            A_aug[-15:, -15:] = Akp1

            # 新 clone 和当前状态都受同一段 IMU 噪声影响。
            BJ = np.zeros((9, 6))
            BJ[0:3, 0:3] = -Ad[0:3, 9:12]
            BJ[3:6, 3:6] = -Ad[3:6, 12:15]
            BJ[6:9, 3:6] = -Ad[6:9, 12:15]

            B_aug = np.zeros(((15 + 9), 6))
            B_aug[-15 - 9 : -15, :] = BJ
            B_aug[-15:, :] = B

            # 把部分积分得到的状态加入历史 clone 列表。
            assert Rd.shape == (3, 3), "inserted past rotation state shape incorrect"
            self.state.si_Rs.append(Rd)
            self.state.si_ps.append(pd)
            self.state.si_vs.append(vd)
            self.state.si_Rs_fej.append(Rd)
            self.state.si_ps_fej.append(pd)
            self.state.si_vs_fej.append(vd)
            self.state.si_timestamps_us.append(t_augmentation_us)

            self.state.N += 1

            # print("state augmented, current number of past states: %s" % self.N)
        else:  # aug==False
            # 不插入 clone 时，只推进最后 15 维当前状态。
            A_aug = np.eye((15 + 9 * N))
            A_aug[-15:, -15:] = Akp1

            B_aug = np.zeros((15, 6))
            B_aug[-15:, :] = B

        Sigma_kp1 = propagate_covariance(
            A_aug, B_aug, dt_us * 1e-6, self.Sigma, self.W, self.Q
        )

        # 写回名义状态和协方差。
        self.state.s_R = R_kp1
        self.state.s_v = v_kp1
        self.state.s_p = p_kp1
        self.state.s_ba = b_akp1
        self.state.s_bg = b_gkp1
        self.state.s_timestamp_us = t_us
        self.Sigma = Sigma_kp1
        self.Sigma15 = self.Sigma[-15:, -15:]
        # 不可观方向也按同一个线性系统传播，用于调试。
        self.state.unobs_shift = A_aug @ self.state.unobs_shift  # propagate unobs shift

    """
    计算 learned velocity measurement update 的 innovation 和 Jacobian。
    [in]: meas = v_b，维度 3x1，神经网络输出的机体系速度测量。
    [in]: meas_cov = 3x3 covariance，神经网络输出的测量协方差。
    [out]: innovation = meas - pred
    [out]: jacobian = H
    [out]: noise matrix = R
    """
    def learnt_model_update(self, meas, meas_cov):
        """
        把网络输出的机体系速度测量转换成 EKF update 所需的 H/R/innovation。

        预测模型:
        - 当前状态保存的是世界系速度 `v_wi`。
        - 机体系速度预测为 `R_wi.T @ v_wi`。
        - 网络测量 `meas` 与该预测作差得到 innovation。
        """
        R = self.meascov_scale * meas_cov
        # 可选: 忽略网络协方差，改用常数协方差做消融/调参。
        if self.use_const_cov:
            val_x = self.const_cov_val_x * self.const_cov_val_x
            val_y = self.const_cov_val_y * self.const_cov_val_y
            val_z = self.const_cov_val_z * self.const_cov_val_z
            R = self.meascov_scale * np.diag(np.array([val_x, val_y, val_z]))

        # 协方差数值对称化，并清除极小负数/数值噪声。
        R = 0.5 * (R + R.T)
        R[R < 1e-10] = 0

        # 预测机体系速度: v_b = R_wi^T * v_w。
        pred = self.state.s_R.T @ self.state.s_v

        # H 只对当前 15 维状态生效，不直接约束历史 clone。
        H = np.zeros((3, 15))
        # 姿态误差影响速度从世界系投影到机体系的结果。
        H[:, 0 : 3] = self.state.s_R.T @ hat(self.state.s_v)
        # 速度误差项的 Jacobian。
        H[:, 3 : 6] = self.state.s_R.T

        assert (
            self.Sigma.shape[0] == H.shape[1]
        ), "state covariance and matrix H does not match shape!"

        if self.mahalanobis_factor > 0:
            # Mahalanobis gating 用 innovation 的归一化平方误差判断网络观测是否离群。
            S_temp = np.linalg.multi_dot([H, self.Sigma, H.T]) + R
            Sinv_temp = np.linalg.inv(S_temp)
            normalized_square_error = np.linalg.multi_dot(
                [(meas - pred).T, Sinv_temp, meas - pred]
            )
            # threshold from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm for nu=3, p =99
            test_failed = normalized_square_error > self.mahalanobis_factor * 11.345
            # filter 收敛后才允许拒绝观测，避免初始化阶段过早拒绝。
            if self.is_mahalanobis_activated() and test_failed:
                if self.mahalanobis_fail_scale == 0:
                    # 直接跳过本次网络 update。
                    print("Mahalanobis test failed... xi2 =", normalized_square_error)
                    return False, None, None, None
                else:
                    # 不跳过观测，但通过放大 R 降低该观测权重。
                    R = self.mahalanobis_fail_scale * R
            else:
                self.last_success_mahalanobis = self.state.s_timestamp_us

        # measurement residual。
        innovation = meas - pred
        
        self.meas = meas
        self.pred = pred

        return True, innovation, H, R

    """Zero-velocity update"""
    def zero_vel_update(self):
        """构造零速观测 update，主要用于静止检测类扩展，主 AI-IO 路径默认不用。"""
        # innovation
        meas = np.zeros((3,1))
        curr_vel = self.state.s_v
        innovation = meas - curr_vel

        # H 对当前速度误差块取单位阵。
        H = np.zeros((3, 15 + 9 * self.state.N))
        H[:, -12:-9] = np.eye(3)

        # 零速测量噪声。
        R = self.zero_vel_sigma * self.zero_vel_sigma * np.eye(3)

        return innovation, H, R

    """Relative position update"""
    def rel_pos_update(self, meas, p_w1, clone_idx_1, p_w0, clone_idx_0):
        """构造两个历史 clone 之间的相对位置观测 update。"""
        # innovation
        assert p_w1.shape == (3,1)
        assert p_w0.shape == (3,1)
        assert meas.shape == (3,1)
        dp = p_w1 - p_w0
        innovation = meas - dp

        # H 只作用于两个 clone 的位置误差块。
        H = np.zeros((3, 15 + 9 * self.state.N))
        H[:, (9 * clone_idx_1 + 6) : (9 * clone_idx_1 + 9)] = np.eye(3)  # der. wrt p_w1
        H[:, (9 * clone_idx_0 + 6) : (9 * clone_idx_0 + 9)] = -1.0 * np.eye(3)  # der. wrt p_w0

        # 相对位置观测噪声，目前写死为 1 mm 标准差。
        R = 0.001 * 0.001 * np.eye(3)

        return innovation, H, R

    """World position update"""
    def pos_update(self, meas, p_wi_est, clone_idx):
        """构造单个历史 clone 的世界系位置观测 update。"""
        # innovation
        assert p_wi_est.shape == (3,1)
        assert meas.shape == (3,1)
        innovation = meas - p_wi_est

        # H 只作用于指定 clone 的 position 误差块。
        H = np.zeros((3, 15 + 9 * self.state.N))
        H[:, (9 * clone_idx + 6) : (9 * clone_idx + 9)] = np.eye(3)  # der. wrt p_w0

        # 世界系位置观测噪声，目前写死为 1 cm 标准差。
        R = 0.01 * 0.01 * np.eye(3)

        return innovation, H, R

    """
    执行标准 EKF measurement update。
    [in]: stacked_innovations = Mx1 innovation 向量。
    [in]: stacked_H = M x (9*self.state.N+15) Jacobian。
    [in]: stacked_R = MxM measurement noise covariance。
    """
    def apply_update(self, stacked_innovations, stacked_H, stacked_R):
        """根据 innovation/H/R 计算 Kalman gain，修正状态并更新协方差。"""
        m = stacked_innovations.shape[0]
        assert m == stacked_H.shape[0] == stacked_R.shape[0]
        assert len(stacked_innovations.shape) == 2

        # innovation covariance: S = H Sigma H^T + R。
        S = np.linalg.multi_dot([stacked_H, self.Sigma, stacked_H.T]) + stacked_R
        Sinv = np.linalg.inv(S)

        # 保存调试字段，供 full_state 日志输出。
        self.innovation = stacked_innovations
        
        self.R = stacked_R
        self.meas_sigma = np.sqrt(np.diag(self.R)).reshape(m, 1)
        self.inno_sigma = np.sqrt(np.diag(S)).reshape(m, 1)

        # Kalman gain: K = Sigma H^T S^-1。
        K = np.linalg.multi_dot([self.Sigma, stacked_H.T, Sinv])
        # K[-15:-12, :] = np.zeros_like(K[-15:-12, :])
        delta_X = K.dot(self.innovation)
        # 把误差状态注入名义状态。
        self.state.apply_correction(delta_X)

        # 协方差更新，并做对称化减少数值误差。
        Sigma_up = self.Sigma - np.linalg.multi_dot([K, stacked_H, self.Sigma])
        Sigma_up = 0.5 * (Sigma_up + Sigma_up.T)
        self.Sigma = Sigma_up
        self.Sigma15 = self.Sigma[-15:, -15:]

        return True

    def marginalize(self, cut_idx):
        """
        删除 cut_idx 及其之前的历史 clone，并同步裁剪协方差。

        这用于控制 clone state 数量，避免协方差维度无限增长。
        """
        # marginalize states prior to cut_idx
        self.state.si_Rs = self.state.si_Rs[cut_idx + 1 :]
        self.state.si_ps = self.state.si_ps[cut_idx + 1 :]
        self.state.si_vs = self.state.si_vs[cut_idx + 1 :]
        self.state.si_Rs_fej = self.state.si_Rs_fej[cut_idx + 1 :]
        self.state.si_ps_fej = self.state.si_ps_fej[cut_idx + 1 :]
        self.state.si_vs_fej = self.state.si_vs_fej[cut_idx + 1 :]
        self.state.si_timestamps_us = self.state.si_timestamps_us[cut_idx + 1 :]
        self.state.unobs_shift = self.state.unobs_shift[9 * (cut_idx + 1) :, :]
        self.Sigma = self.Sigma[9 * (cut_idx + 1) :, 9 * (cut_idx + 1) :]
        self.state.N = self.state.N - (cut_idx + 1)


@jit(nopython=True, parallel=False, cache=True)
def propagate_covariance(A_aug, B_aug, dt, Sigma, W, Q):
    """
    按离散线性系统传播协方差。

    参数:
    - `A_aug`: 旧误差状态到新误差状态的状态转移矩阵。
    - `B_aug`: IMU 白噪声到新状态的噪声输入矩阵，形状为 [15 x 6] 或 [24 x 6]。
    - `dt`: 秒级传播时间。
    - `Sigma`: 上一时刻完整协方差。
    - `W`: gyro/accel 白噪声协方差。
    - `Q`: gyro/accel bias 随机游走协方差。
    """
    dim_new_state = A_aug.shape[0] - A_aug.shape[1]  # either 0 or 9
    assert B_aug.shape[0] == 15 + dim_new_state
    A = A_aug[-15 - dim_new_state :, -15:]
    AT = A_aug[-15 - dim_new_state :, -15:].T
    ret = np.zeros((A_aug.shape[0], A_aug.shape[0]))
    # clone-clone 旧协方差块直接保留。
    ret[: -15 - dim_new_state, : -15 - dim_new_state] = Sigma[
        :-15, :-15
    ]  # copy top-left block
    # clone-current 交叉协方差按当前状态转移传播。
    ret[: -15 - dim_new_state, -15 - dim_new_state :] = (
        Sigma[:-15, -15:] @ AT
    )  # top-right corner
    ret[-15 - dim_new_state :, : -15 - dim_new_state] = (
        A @ Sigma[-15:, :-15]
    )  # bottom-left corner
    ret[-15 - dim_new_state :, -15 - dim_new_state :] = (
        A @ Sigma[-15:, -15:] @ AT
    )  # bottom-right corner
    # 加入 IMU 测量白噪声。
    ret[-15 - dim_new_state :, -15 - dim_new_state :] += (
        B_aug @ W @ B_aug.T
    )  # only non zero on last block
    # 加入 bias 随机游走噪声，只影响最后 6 维 bias 块。
    ret[-6:, -6:] += dt * Q

    return ret
