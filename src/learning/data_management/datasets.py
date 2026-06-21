"""
Reference:
https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/learning/data_management/datasets.py

本文件负责把已经预处理好的 `data.hdf5` 序列封装成 PyTorch Dataset。

整体数据流:
1. `prepare_datasets/our2.py` 先把 rosbag/mcap 中的 IMU、ESC、真值轨迹等数据
   插值到统一时间轴，并写成每个序列目录下的 `processed_data/<split>/data.hdf5`。
2. `ModelSequence` 读取一个 `data.hdf5`，构造:
   - 网络输入 feature: accel_calib + gyro_calib + rotor_spd
   - 监督目标 target: 机体系速度 `v_body`
   - 辅助信息: 时间戳、原始 IMU、真值轨迹
3. `ModelOur2Dataset` / `ModelDIDODataset` 把多个序列拼成一个可被
   `torch.utils.data.DataLoader` 迭代的数据集，并按滑动窗口返回训练样本。

这里需要重点理解:
- 网络学的不是位置，也不是完整 odom，而是“当前机体系速度”。
- `traj_target` 里的速度通常是世界系速度；训练前会用姿态旋转到 body frame。
- 每个样本不是单帧输入，而是一段时间窗口 `[B, C, T]` 中的一个窗口。
"""

from abc import ABC, abstractmethod  # 定义抽象基类和抽象方法，用于约束序列读取接口。
import os  # 处理文件路径，例如拼接 data.hdf5 的实际位置。
import random  # 在训练/验证模式下打乱滑动窗口样本顺序。

import h5py  # 读取预处理后保存的 HDF5 数据文件。
import numpy as np  # 进行数组拼接、类型转换和速度标签计算。
from torch.utils.data import Dataset  # PyTorch Dataset 基类，供 DataLoader 迭代采样。

import learning.utils.pose as pose  # 姿态工具，用四元数把世界系速度转换到机体系。


class CompiledSequence(ABC):
    """
    单个已编译序列的抽象接口。

    “已编译序列”指的是已经从原始 rosbag/mcap 转换成 `data.hdf5` 的一段数据。
    这个抽象类规定每个序列必须提供三类内容:
    - feature: 网络输入。
    - target: 训练监督目标。
    - aux: 测试/可视化用的辅助数据。
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        """从指定目录读取序列数据。"""
        pass

    @abstractmethod
    def get_feature(self):
        """返回网络输入特征。"""
        pass

    @abstractmethod
    def get_target(self):
        """返回监督目标和轨迹真值。"""
        pass

    @abstractmethod
    def get_aux(self):
        """返回训练不直接使用、但测试/可视化需要的辅助量。"""
        pass


class ModelSequence(CompiledSequence):
    """
    读取并持有一个 `data.hdf5` 序列。

    一个 `ModelSequence` 对应一个具体 split 下的单段数据，例如:
    `.../indoor/manual/high/seq_1/processed_data/train/data.hdf5`

    HDF5 中当前使用的关键字段:
    - `ts`: 统一采样后的时间戳。
    - `gyro_raw`: 原始角速度测量，形状 `[N, 3]`。
    - `accel_raw`: 原始加速度测量，形状 `[N, 3]`。
    - `gyro_calib`: 已减去离线 bias 的角速度，形状 `[N, 3]`。
    - `accel_calib`: 已减去离线 bias 的加速度，形状 `[N, 3]`。
    - `rotor_spd`: 四个电机转速，形状 `[N, 4]`。
    - `traj_target`: 真值轨迹，当前约定为
      `[x, y, z, qx, qy, qz, qw, vx, vy, vz]`，形状 `[N, 10]`。

    训练输入:
    `feat = [accel_calib(3), gyro_calib(3), rotor_spd(4)]`，总计 10 维。

    训练目标:
    `targ_vb`，即机体系速度。它由 `traj_target` 中的世界系速度通过
    `R_world_body.T @ v_world` 转换得到。
    """

    def __init__(self, seq_path, args, **kwargs):
        super().__init__(**kwargs)

        # 统一初始化为 None，便于 load() 之后检查哪些成员已经被填充。
        (
            self.ts,
            self.features,
            self.targets,
            self.gyro_raw,
            self.accel_raw,
            self.feat,
            self.traj_target,
        ) = (None, None, None, None, None, None, None)

        # mode 影响数据增强逻辑。当前只有 train 模式可能扰动 accel。
        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        if seq_path is not None:
            self.load(seq_path)

    def load(self, data_path):
        """
        读取单个序列的 `data.hdf5` 并构造 feature/target。

        参数:
        - data_path: split 目录，例如 `.../processed_data/train`。

        输出成员:
        - self.feat: `[N, 10]`，后续窗口采样前的逐时刻输入特征。
        - self.targ_vb: `[N, 3]`，逐时刻机体系速度标签。
        - self.traj_target: `[N, 10]`，保留原始真值轨迹，供评估/可视化使用。
        """
        with h5py.File(os.path.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            accel_raw = np.copy(f["accel_raw"])
            accel_calib = np.copy(f["accel_calib"])
            traj_target = np.copy(f["traj_target"])
            rotor_spd = np.copy(f["rotor_spd"])

        self.ts = ts
        self.gyro_raw = gyro_raw
        self.accel_raw = accel_raw

        # 训练模式下可对加速度做轻微随机扰动，增强模型对 IMU 噪声/偏差的鲁棒性。
        # 验证和测试模式不做扰动，保证评估结果可复现。
        if self.mode == "train":
            if self.perturb_accel:
                accel_rand = (
                    np.random.uniform(-1, 1, accel_calib.shape)
                    * self.perturb_accel_range
                )
                accel_calib += accel_rand

        # 当前 AI-IO 原始模型的输入通道约定:
        #   0:3  -> accel_calib
        #   3:6  -> gyro_calib
        #   6:10 -> rotor_spd
        #
        # 后续如果要增加 normalized_propulsion、battery_voltage 等特征，
        # 应该从这里扩展特征拼接，同时同步修改网络 forward() 和 filter 推理输入。
        self.feat = np.concatenate([accel_calib, gyro_calib, rotor_spd], axis=1)

        # traj_target[:, 3:7] 是 [qx, qy, qz, qw] 四元数。
        # traj_target[:, 7:10] 是世界系速度 v_world。
        # 网络监督目标使用机体系速度 v_body:
        #   v_body = R_body_world * v_world = R_world_body.T @ v_world
        self.targ_vb = np.zeros((traj_target.shape[0], 3))
        for i in range(traj_target.shape[0]):
            self.targ_vb[i, :] = (
                pose.xyzwQuatToMat(traj_target[i, 3:7]).T @ traj_target[i, 7:10]
            )
        self.traj_target = traj_target

    def get_feature(self):
        """返回逐时刻网络输入特征，形状 `[N, 10]`。"""
        return self.feat

    def get_target(self):
        """
        返回训练目标和轨迹真值。

        - `targ_vb`: `[N, 3]`，机体系速度，用于 loss。
        - `traj_target`: `[N, 10]`，世界系位置/姿态/速度，用于评估和可视化。
        """
        return self.targ_vb, self.traj_target

    def get_aux(self):
        """
        返回辅助数据。

        这些量不直接参与训练 loss，但在 test/eval 中会用于:
        - 输出时间戳。
        - 与原始 IMU 对齐。
        - 画图或保存中间结果。
        """
        return self.ts, self.gyro_raw, self.accel_raw


class ModelOur2Dataset(Dataset):
    """
    AI-IO `our2` 数据集的 PyTorch Dataset 封装。

    该类把多个 `ModelSequence` 拼成一个 Dataset，并把每个序列切成滑动窗口样本。

    DataLoader 每次取出的一个样本包含:
    - `feat`: `[C, T]`，网络输入窗口。当前 C=10，T=window_size。
    - `targ`: `[3]`，窗口末端时刻的机体系速度标签。
    - `gt_traj`: `[T, 10]`，窗口内真值轨迹。
    - `feat_ts`: `[T]`，窗口内时间戳。
    - `raw_gyro_meas_i`: 测试模式下的原始 gyro 窗口，训练/验证模式为零占位。
    - `raw_accel_meas_i`: 测试模式下的原始 accel 窗口，训练/验证模式为零占位。
    """

    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelOur2Dataset, self).__init__()

        # sampling_factor 表示在已预处理数据上隔几个点采一个点。
        # window_size 表示网络输入窗口包含多少个采样点。
        # window_shift_size 表示相邻训练样本窗口末端相隔多少个点。
        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]

        # 当前类里没有直接使用 g，但保留该成员与原项目结构一致。
        self.g = np.array([0.0, 0.0, 9.7946])

        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        # 训练/验证阶段打乱窗口顺序，测试阶段保持时间顺序，便于按时间恢复结果。
        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        # index_map 保存“全局样本编号 -> 具体序列和窗口末端帧”的映射:
        #   [seq_id, frame_id]
        #
        # frame_id 是窗口右端点，即 target 所在时刻。窗口输入覆盖:
        #   [frame_id - window_size * sampling_factor, frame_id)
        self.index_map = []

        # 以下列表按 seq_id 保存各序列数据，避免把不同序列直接拼成一个大数组后丢失边界。
        self.ts, self.features, self.targets, self.gt_traj = [], [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []

        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], args, **kwargs)

            feat = seq.get_feature()
            targ, traj = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            self.gt_traj.append(traj)

            # 对当前序列生成所有可用窗口的右端点。
            # 第一个可用 frame_id 必须保证向前能取到完整 window_size 个点。
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor),
                    N,
                    self.window_shift_size,
                )
            ]

            times, raw_gyro_meas, raw_accel_meas = seq.get_aux()
            self.ts.append(times)

            # 只有测试模式才保留原始 IMU 窗口，训练/验证时不需要这些辅助量。
            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.raw_accel_meas.append(raw_accel_meas)

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        """
        返回一个滑动窗口样本。

        返回值的形状:
        - feat: `[C, T]`，当前 C=10。
        - targ: `[3]`，窗口末端时刻的机体系速度。
        - gt_traj: `[T, 10]`。
        - feat_ts: `[T]`。
        - raw_gyro_meas_i: test 模式下为 `[3, T]`，否则为 `[3]` 零向量。
        - raw_accel_meas_i: test 模式下为 `[3, T]`，否则为 `[3]` 零向量。

        注意:
        PyTorch Conv1d 通常要求输入为 `[B, C, T]`，所以这里单样本返回 `[C, T]`；
        DataLoader 自动 batch 后就变为 `[B, C, T]`。
        """
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        # 构造窗口索引。idxe 不包含在 range 内；indices[-1] 是实际输入窗口最后一帧。
        # target 使用窗口最后一帧对应时刻的速度标签。
        idxs = frame_id - self.window_size * self.sampling_factor
        idxe = frame_id
        indices = range(idxs, idxe, self.sampling_factor)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]

        # 目标是窗口末端时刻的机体系速度，而不是整个窗口的速度序列。
        targ = self.targets[seq_id][idxe, :]

        # 保留窗口内的完整真值轨迹，用于测试/可视化。
        gt_traj = self.gt_traj[seq_id][indices]

        # 窗口内的时间戳。
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]

        return (
            feat.astype(np.float32).T,
            targ.astype(np.float32),
            gt_traj.astype(np.float32),
            feat_ts,
            raw_gyro_meas_i.astype(np.float32).T,
            raw_accel_meas_i.astype(np.float32).T,
        )

    def __len__(self):
        """返回整个 Dataset 中可采样窗口的数量。"""
        return len(self.index_map)


class ModelDIDODataset(Dataset):
    """
    DIDO 数据集的 PyTorch Dataset 封装。

    当前实现与 `ModelOur2Dataset` 基本相同，仍然复用 `ModelSequence`
    读取 `data.hdf5`。保留单独类名主要是为了与训练入口中的 dataset 选择逻辑兼容:

    ```python
    if args.dataset == "DIDO":
        ModelDIDODataset(...)
    else:
        ModelOur2Dataset(...)
    ```

    如果未来 DIDO 的 HDF5 字段、坐标系或 target 定义不同，应在这里单独实现。
    """

    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelDIDODataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0.0, 0.0, 9.7946])

        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        # index_map = [[seq_id, index of the last datapoint in the window], ...]
        self.index_map = []
        self.ts, self.features, self.targets, self.gt_traj = [], [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []

        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], args, **kwargs)

            feat = seq.get_feature()
            targ, traj = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            self.gt_traj.append(traj)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor),
                    N,
                    self.window_shift_size,
                )
            ]

            times, raw_gyro_meas, raw_accel_meas = seq.get_aux()
            self.ts.append(times)

            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.raw_accel_meas.append(raw_accel_meas)

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        """
        返回一个 DIDO 滑动窗口样本。

        当前返回格式与 `ModelOur2Dataset.__getitem__()` 保持一致，便于训练和测试代码
        对不同 dataset 使用同一套 unpack 逻辑。
        """
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        idxs = frame_id - self.window_size * self.sampling_factor
        idxe = frame_id
        indices = range(idxs, idxe, self.sampling_factor)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]

        # target velocity
        targ = self.targets[seq_id][idxe, :]

        gt_traj = self.gt_traj[seq_id][indices]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]

        return (
            feat.astype(np.float32).T,
            targ.astype(np.float32),
            gt_traj.astype(np.float32),
            feat_ts,
            raw_gyro_meas_i.astype(np.float32).T,
            raw_accel_meas_i.astype(np.float32).T,
        )

    def __len__(self):
        """返回 DIDO Dataset 中可采样窗口的数量。"""
        return len(self.index_map)
