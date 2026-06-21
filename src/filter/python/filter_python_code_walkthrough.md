# AI-IO `src/filter/python` 代码讲解报告

本文档说明 `src/filter/python` 目录下每个 Python 文件的职责、主数据流、关键代码行，以及后续接入 Sunray 或扩展输入特征时最应该关注的位置。

## 1. 总体架构

AI-IO 的 filter 侧不是直接让神经网络输出完整里程计，而是采用“IMU propagation + learned velocity measurement update”的结构:

1. `DataIO` 从离线数据集读取原始 IMU、电机转速、ground truth。
2. `FilterManager` 遍历一个序列，把每帧数据送入 `FilterRunner`，并负责保存日志。
3. `FilterRunner` 维护网络输入缓冲，按固定窗口调用神经网络。
4. `MeasSourceNetwork` 把 `[acc, gyro, rotor]` 窗口送入模型，输出机体系速度 `v_body` 和测量协方差。
5. `ImuMSCKF` 用 IMU 推进状态，用网络速度观测做 EKF 更新。
6. `plot_filter_output.py` 读取输出日志，与 ground truth 对齐，生成图和误差指标。

关键链路可以按下面几行跟读:

- `src/filter/python/src/data_io.py:44`：`DataIO.load()` 读取 `data.hdf5`。
- `src/filter/python/src/filter_manager.py:219`：`FilterManager.run()` 遍历序列。
- `src/filter/python/src/filter_runner.py:135`：`FilterRunner.on_imu_measurement()` 是单帧入口。
- `src/filter/python/src/filter_runner.py:212`：每帧调用 `self.filter.propagate(...)`。
- `src/filter/python/src/filter_runner.py:261`：到达 update 时间时调用 `learnt_model_update()`。
- `src/filter/python/src/scekf.py:470`：`ImuMSCKF.propagate()` 执行 IMU 状态传播。
- `src/filter/python/src/scekf.py:578`：`ImuMSCKF.learnt_model_update()` 把网络速度转为 EKF 观测。
- `src/filter/python/src/scekf.py:702`：`ImuMSCKF.apply_update()` 执行 Kalman 更新。

## 2. 核心文件

### `data_io.py`

作用: 读取单条离线序列的 `processed_data/<mode>/data.hdf5`，并把数据整理成 filter 能逐帧消费的格式。

关键代码行:

- `src/filter/python/src/data_io.py:21`：`class DataIO` 定义单序列数据读取器。
- `src/filter/python/src/data_io.py:44`：`load()` 打开 `data.hdf5`，读取 `gyro_raw`、`accel_raw`、`rotor_spd`、`traj_target`、bias 等字段。
- `src/filter/python/src/data_io.py:93`：`get_datai()` 返回第 `idx` 帧 `[ts, acc, gyr, rotor]`。
- `src/filter/python/src/data_io.py:109`：`get_imu_calibration()` 返回离线 gyro/accel bias。
- `src/filter/python/src/data_io.py:116`：`get_groundtruth_pose()` 提供 ground truth pose 插值接口。

理解重点:

- filter propagation 使用的是 `accel_raw` 和 `gyro_raw`。
- 网络输入在 `FilterRunner` 里会通过 `ImuCalib.calibrate_raw()` 做 bias 补偿。
- `rotor_spd` 是 AI-IO 区别于纯惯性里程计的重要输入。

### `filter_manager.py`

作用: 管理单个测试序列的完整 filter 运行流程，负责初始化、逐帧喂数据、收集输出、保存日志。

关键代码行:

- `src/filter/python/src/filter_manager.py:27`：`class FilterManager` 是离线序列调度器。
- `src/filter/python/src/filter_manager.py:36`：构造函数加载数据、准备输出文件、创建 `FilterRunner`。
- `src/filter/python/src/filter_manager.py:130`：`add_data_to_be_logged()` 把当前 filter 状态转成轨迹、速度、bias 日志行。
- `src/filter/python/src/filter_manager.py:197`：`save_logs()` 保存 `stamped_traj_estimate.txt`、`stamped_vel_estimate.txt`、`stamped_bias_estimate.txt`。
- `src/filter/python/src/filter_manager.py:219`：`run()` 是主循环，顺序读取每帧 IMU 和电机转速。
- `src/filter/python/src/filter_manager.py:275`：`reset_filter_state_from_groundtruth()` 可用 ground truth 重置状态，主要用于离线评估。

理解重点:

- 如果 `initialize_with_gt=True`，第一帧会用 ground truth 的位置、速度、姿态初始化。这适合离线评估，不适合真实飞行。
- 输出的 `stamped_traj_estimate.txt` 格式是 `ts x y z qx qy qz qw`。
- `R_ib` 和 `p_ib` 目前是单位外参。如果实际无人机 IMU 与机体系不重合，后续应在这里接入外参。

### `filter_runner.py`

作用: 处理单帧 IMU/rotor 输入，维护网络输入窗口，按频率触发神经网络测量，并驱动 EKF。

关键代码行:

- `src/filter/python/src/filter_runner.py:27`：`class FilterRunner` 管理一条数据流上的 filter。
- `src/filter/python/src/filter_runner.py:39`：构造函数读取 `model_net_parameters.json`，获得网络采样频率和窗口长度。
- `src/filter/python/src/filter_runner.py:112`：`_get_inputs_samples_for_network()` 从缓冲区取固定长度窗口。
- `src/filter/python/src/filter_runner.py:135`：`on_imu_measurement()` 是单帧入口。
- `src/filter/python/src/filter_runner.py:174`：`_on_imu_measurement_after_init()` 处理初始化后的常规帧。
- `src/filter/python/src/filter_runner.py:212`：`self.filter.propagate(...)` 每帧执行 IMU propagation。
- `src/filter/python/src/filter_runner.py:229`：`_process_update()` 执行一次 learned velocity update。
- `src/filter/python/src/filter_runner.py:271`：`_add_interpolated_inputs_to_buffer()` 将测量插值到网络固定时间轴。

理解重点:

- 网络输入使用 `calibrate_raw()`，即减去离线 bias。
- EKF propagation 使用 `scale_raw()`，即不减 bias，让滤波器状态中的 `ba/bg` 参与传播。
- `update_freq` 控制神经网络速度观测频率，`imu_freq_net` 控制网络窗口采样频率。
- 当前初始化分支中有一个值得注意的旧代码点: `on_imu_measurement()` 初始化时调用 `_add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, t_us)`，而函数签名需要 `rotor_spd`。离线 smoke test 使用 ground truth 初始化时绕过了这个分支；若后续关闭 ground truth 初始化，应优先检查这里。

### `meas_source_network.py`

作用: 加载训练好的 AI-IO 网络，执行前向推理，输出机体系速度测量和协方差。

关键代码行:

- `src/filter/python/src/meas_source_network.py:21`：`class MeasSourceNetwork` 定义网络测量源。
- `src/filter/python/src/meas_source_network.py:34`：构造函数加载 checkpoint，并选择 CPU/GPU。
- `src/filter/python/src/meas_source_network.py:52`：`get_measurement()` 是统一测量接口。
- `src/filter/python/src/meas_source_network.py:58`：`get_vb_measurement_model_net()` 执行模型推理。
- `src/filter/python/src/meas_source_network.py:67`：`features = np.concatenate([net_accl_b, net_gyr_b, net_rotor], axis=1)`，当前网络输入为 10 维。

理解重点:

- 当前输入通道是 `acc[3] + gyro[3] + rotor[4] = 10`。
- 如果后续加入归一化油门/归一化推力，需要从这一行扩展特征维度，并同步修改模型定义、训练数据、checkpoint。
- `DiagonalParam.vec2Cov()` 把网络输出的协方差参数转成 3x3 测量协方差。

### `net_input_utils.py`

作用: 提供 IMU 标定工具和网络输入缓冲区。

关键代码行:

- `src/filter/python/src/net_input_utils.py:15`：`class ImuCalib` 保存离线 bias 和 scale 参数。
- `src/filter/python/src/net_input_utils.py:38`：`calibrate_raw()` 给网络输入减去 bias。
- `src/filter/python/src/net_input_utils.py:54`：`scale_raw()` 给 EKF propagation 使用，不减 bias。
- `src/filter/python/src/net_input_utils.py:66`：`class NetInputBuffer` 保存插值后的网络输入。
- `src/filter/python/src/net_input_utils.py:83`：`add_data_interpolated()` 把相邻两帧测量插值到网络时间轴。
- `src/filter/python/src/net_input_utils.py:140`：`get_data_from_to()` 按起止时间取出一个完整窗口。
- `src/filter/python/src/net_input_utils.py:163`：`throw_data_before()` 丢弃旧缓存。

理解重点:

- 实际 IMU/rotor 时间戳不一定严格等间隔，网络需要固定频率输入，所以这里做线性插值。
- `get_data_from_to()` 对起止时间有 1ms 以内误差检查，用于防止窗口错位。

### `scekf.py`

作用: AI-IO 的滤波器数学核心，维护状态、协方差、IMU propagation、网络速度 update、Kalman 修正。

关键代码行:

- `src/filter/python/src/scekf.py:20`：`class State` 保存名义状态和历史 clone。
- `src/filter/python/src/scekf.py:92`：`State.apply_correction()` 把 Kalman update 得到的误差状态注入名义状态。
- `src/filter/python/src/scekf.py:158`：`propagate_rvt_and_jac()` 单步 IMU 积分并返回状态转移 Jacobian。
- `src/filter/python/src/scekf.py:204`：`class ImuMSCKF` 是 filter 主类。
- `src/filter/python/src/scekf.py:306`：`reset_covariance()` 初始化完整协方差矩阵。
- `src/filter/python/src/scekf.py:470`：`propagate()` 用 IMU 推进当前状态和协方差。
- `src/filter/python/src/scekf.py:578`：`learnt_model_update()` 将网络速度测量转成 `innovation/H/R`。
- `src/filter/python/src/scekf.py:600`：`pred = self.state.s_R.T @ self.state.s_v`，把世界系速度投影成机体系速度预测。
- `src/filter/python/src/scekf.py:702`：`apply_update()` 执行标准 EKF 更新。
- `src/filter/python/src/scekf.py:720`：`K = np.linalg.multi_dot([self.Sigma, stacked_H.T, Sinv])` 计算 Kalman gain。
- `src/filter/python/src/scekf.py:754`：`propagate_covariance()` 按离散线性系统传播协方差。

理解重点:

- 当前 evolving state 是 15 维: 姿态、速度、位置、gyro bias、accel bias。
- 历史 clone 每个 9 维: 姿态、速度、位置。当前主路径主要用当前 15 维网络速度 update。
- 网络并不直接改位置，而是通过速度测量修正 `v` 和姿态，再经 propagation 影响位置。
- `learnt_model_update()` 的观测模型是 `v_body = R_wi.T @ v_world`。
- Mahalanobis gating 用于拒绝明显离群的网络速度观测，防止一次错误网络输出把滤波器拉偏。

## 3. 工具文件

### `utils/math_utils.py`

作用: 提供 SO(3) 旋转群相关工具。

关键代码行:

- `src/filter/python/src/utils/math_utils.py:21`：`hat()` 生成反对称矩阵。
- `src/filter/python/src/utils/math_utils.py:29`：`rot_2vec()` 计算从一个方向旋到另一个方向的旋转矩阵。
- `src/filter/python/src/utils/math_utils.py:56`：`mat_exp()` 实现 SO(3) 指数映射。
- `src/filter/python/src/utils/math_utils.py:87`：`mat_log()` 实现 SO(3) 对数映射。
- `src/filter/python/src/utils/math_utils.py:150`：`Jr_exp()` 计算 SO(3) exp 右雅可比。
- `src/filter/python/src/utils/math_utils.py:189`：`unwrap_rpy()` 展开欧拉角曲线，避免绘图跳变。

### `utils/from_scipy.py`

作用: 从 SciPy Rotation 移植的旋转表示转换函数。

关键代码行:

- `src/filter/python/src/utils/from_scipy.py:25`：`compute_euler_from_matrix()` 从旋转矩阵计算欧拉角。
- `src/filter/python/src/utils/from_scipy.py:170`：`compute_q_from_matrix()` 从旋转矩阵计算四元数。

### `utils/plotting.py`

作用: 绘制轨迹、速度、bias、姿态对比图。

关键代码行:

- `src/filter/python/src/utils/plotting.py:13`：`xyPlot()` 绘制二维曲线。
- `src/filter/python/src/utils/plotting.py:38`：`xyztPlot()` 绘制 x/y/z 随时间变化。
- `src/filter/python/src/utils/plotting.py:89`：`plotBiases()` 绘制 IMU bias。
- `src/filter/python/src/utils/plotting.py:121`：`make_position_plots()` 绘制 XY/XZ/YZ 轨迹。
- `src/filter/python/src/utils/plotting.py:144`：`make_velocity_plots()` 绘制速度。
- `src/filter/python/src/utils/plotting.py:174`：`make_ori_euler_plots()` 绘制姿态欧拉角。

### `utils/misc.py`

作用: 时间单位转换和轨迹数组整理。

关键代码行:

- `src/filter/python/src/utils/misc.py:5`：`from_sec_to_usec()` 秒转微秒整数。
- `src/filter/python/src/utils/misc.py:10`：`from_usec_to_sec()` 微秒转秒。
- `src/filter/python/src/utils/misc.py:15`：`getNumpyTraj()` 输出 `[ts, x, y, z, qx, qy, qz, qw]` 轨迹数组。

### `utils/argparse_utils.py`

作用: 为命令行参数添加成对布尔开关。

关键代码行:

- `src/filter/python/src/utils/argparse_utils.py:4`：`add_bool_arg()` 同时注册 `--name` 和 `--no-name`。

### `utils/dotdict.py`

作用: 让字典支持 `cfg.xxx` 风格访问。

关键代码行:

- `src/filter/python/src/utils/dotdict.py:1`：`class dotdict(dict)`。
- `src/filter/python/src/utils/dotdict.py:14`：`__getattr__()` 把属性访问转成 key 查询。

### `utils/logging.py`

作用: 统一配置日志格式和输出级别。

关键代码行:

- `src/filter/python/src/utils/logging.py:19`：`logging.basicConfig(...)` 设置日志输出格式、输出流和等级。

### `utils/profile.py`

作用: 提供 cProfile 上下文管理器。

关键代码行:

- `src/filter/python/src/utils/profile.py:18`：`profile()` 在 `with` 代码块内开启/关闭 cProfile。

## 4. 输出分析脚本

### `plot_filter_output.py`

作用: 读取 filter 输出，和 ground truth 对齐，生成图和误差 CSV。

关键代码行:

- `src/filter/python/plot_filter_output.py:31`：`process_sequence()` 处理单个序列。
- `src/filter/python/plot_filter_output.py:83`：按估计轨迹时间戳插值 ground truth 位置。
- `src/filter/python/plot_filter_output.py:87`：组装插值后的 ground truth 轨迹。
- `src/filter/python/plot_filter_output.py:129`：调用 `compute_position_velocity_orientation_errors()` 计算误差。

理解重点:

- 该脚本不是 filter 主流程，只是结果可视化和误差统计。
- `ekf_metrics.csv` 默认写到当前运行目录，不一定写到序列结果目录。

## 5. 后续扩展输入特征时的修改入口

如果要把输入扩展为“IMU + 电机转速 + 归一化油门/推力 + 外部里程计状态”，至少需要检查以下位置:

1. 数据读取: `DataIO.load()` 和 `DataIO.get_datai()`，需要从 hdf5 中读取并返回新增字段。
2. 输入缓冲: `NetInputBuffer` 需要缓存新增时间序列，并在 `get_data_from_to()` 返回完整窗口。
3. 单帧入口: `FilterRunner.on_imu_measurement()` 和 `_on_imu_measurement_after_init()` 需要接收并传递新增输入。
4. 网络特征拼接: `MeasSourceNetwork.get_vb_measurement_model_net()` 的 `np.concatenate(...)` 必须扩展通道。
5. 模型结构: `learning.network.model_factory.get_model()` 及模型第一层输入通道数需要同步改。
6. 训练数据: 训练集 hdf5/数据集类也要写入同样字段，否则推理和训练输入维度不一致。
7. EKF update: 如果外部里程计不是网络输入特征，而是独立观测，应在 `scekf.py` 中增加对应 measurement update，而不是只拼进网络。

## 6. 本次注释修改范围

本次只补充中文模块说明、类说明、函数说明和关键逻辑注释，未主动改变 filter 行为、接口和数学公式。

已覆盖文件:

- `plot_filter_output.py`
- `src/data_io.py`
- `src/filter_manager.py`
- `src/filter_runner.py`
- `src/meas_source_network.py`
- `src/net_input_utils.py`
- `src/scekf.py`
- `src/utils/argparse_utils.py`
- `src/utils/dotdict.py`
- `src/utils/from_scipy.py`
- `src/utils/logging.py`
- `src/utils/math_utils.py`
- `src/utils/misc.py`
- `src/utils/plotting.py`
- `src/utils/profile.py`
