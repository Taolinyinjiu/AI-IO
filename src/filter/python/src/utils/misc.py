import numpy as np
from scipy.spatial.transform import Rotation


def from_sec_to_usec(t_sec):
    """秒转微秒整数时间戳。滤波器内部统一使用 us 级整数时间。"""
    return int(t_sec * 1e6)


def from_usec_to_sec(t_usec):
    """微秒整数时间戳转秒，用于插值、日志和绘图。"""
    return t_usec * 1e-6


def getNumpyTraj(ts_list, ori_list, pos_list):
    """
    把时间戳、旋转矩阵、位置列表整理成可保存的轨迹数组。

    输入:
    - `ts_list`: 时间戳列表。
    - `ori_list`: 世界到机体/IMU 的旋转矩阵列表。
    - `pos_list`: 位置向量列表。

    输出:
    - `[ts, x, y, z, qx, qy, qz, qw]` 格式的 numpy 数组。
    """
    assert len(ts_list) == len(ori_list) == len(pos_list)

    data = []
    for i, ts in enumerate(ts_list):
        R = ori_list[i]
        # scipy 的 as_quat() 输出顺序是 [qx, qy, qz, qw]。
        q = Rotation.from_matrix(R).as_quat()
        p = pos_list[i]
        datapoint = np.array([
            ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
        data.append(datapoint)

    return np.asarray(data)
