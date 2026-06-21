"""
SO(3) / 姿态相关数学工具。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/math_utils.py

本文件提供滤波器中反复使用的旋转群运算:
- `hat()`: 向量到反对称矩阵。
- `mat_exp()`: so(3) 切向量到 SO(3) 旋转矩阵。
- `mat_log()`: SO(3) 旋转矩阵到 so(3) 切向量。
- `Jr_exp()/Jr_log()`: SO(3) 右雅可比。
"""

import warnings

import numpy as np
from numba import jit

from .from_scipy import compute_q_from_matrix


def hat(v):
    """把 3 维向量转换为反对称矩阵，使 `hat(v) @ x == cross(v, x)`。"""
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R


@jit(nopython=True, parallel=False, cache=True)
def rot_2vec(a, b):
    """
    计算把向量 `a` 旋转到向量 `b` 的旋转矩阵。

    初始化时会用它把第一帧加速度方向对齐到世界系重力方向。
    """
    assert a.shape == (3, 1)
    assert b.shape == (3, 1)

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    # 归一化后只关心方向，不关心向量模长。
    a_hat = a / a_n
    b_hat = b / b_n
    # Rodrigues 公式中的旋转轴相关量。
    omega = np.cross(a_hat.T, b_hat.T).T
    c = 1.0 / (1 + np.dot(a_hat.T, b_hat))
    R_ba = np.eye(3) + hat(omega) + c * hat(omega) @ hat(omega)
    return R_ba


@jit(nopython=True, parallel=False, cache=True)
def mat_exp(omega):
    """
    SO(3) 指数映射: 把 3 维旋转向量转换成 3x3 旋转矩阵。

    在 filter 中，姿态误差 `dtheta` 通过该函数转换为小旋转矩阵后注入名义姿态。
    """
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    # 角度非常小时，使用一阶泰勒展开避免除以接近 0 的数。
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    # Rodrigues 公式。
    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


mat_exp_vec = np.vectorize(mat_exp, signature="(3)->(3,3)")


def mat_log(R):
    """SO(3) 对数映射: 把单个旋转矩阵转换成 3 维旋转向量。"""
    q = compute_q_from_matrix(R)
    w = q[3]
    vec = q[0:3]
    n = np.linalg.norm(vec)
    epsilon = 1e-7

    if n < epsilon:
        # 四元数向量部接近 0，对应接近单位旋转，使用稳定近似。
        w2 = w * w
        n2 = n * n
        atn = 2.0 / w - (2.0 * n2) / (w * w2)
    else:
        if np.absolute(w) < epsilon:
            # 接近 180 度旋转时，atan 形式容易数值不稳定，单独处理。
            if w > 0:
                atn = np.pi / n
            else:
                atn = -np.pi / n
        else:
            atn = 2.0 * np.arctan(n / w) / n
    tangent = atn * vec
    return tangent


def mat_log_vec(R):
    """
    批量 SO(3) 对数映射。

    Args:
        R [n x 3 x 3]
    """

    q = compute_q_from_matrix(R)
    w = q[:, 3]
    vec = q[:, 0:3]
    n = np.linalg.norm(vec, axis=1)
    epsilon = 1e-7

    mask = n < epsilon
    atn_small = 2.0 / w - (2.0 * n * n) / (w * w * w)

    mask2 = np.absolute(w) < epsilon
    atn_normal_small = np.sign(w) * np.pi / n
    atn_normal_normal = 2.0 * np.arctan(n / w) / n

    atn = mask2 * atn_normal_small + (1 - mask2) * atn_normal_normal
    atn = mask * atn_small + (1 - mask2) * atn

    tangent = atn[0, np.newaxis] * vec
    return tangent


def mat_to_rot_ang(R):
    """由旋转矩阵 trace 计算旋转角。"""
    return np.arccos((np.trace(R) - 1) / 2)


"""SO(3) 指数映射的右雅可比。"""


@jit(nopython=True, parallel=False, cache=True)
def Jr_exp(phi):
    """计算 SO(3) exp 右雅可比，propagation Jacobian 会用到。"""
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        # 小角度下使用泰勒展开，避免三角函数表达式的数值问题。
        J = np.eye(3) - 0.5 * hat(phi) + 1.0 / 6.0 * (hat(phi) @ hat(phi))
    else:
        J = (
            np.eye(3)
            - (1 - np.cos(theta)) / np.power(theta, 2.0) * hat(phi)
            + (theta - np.sin(theta)) / np.power(theta, 3.0) * (hat(phi) @ hat(phi))
        )
    return J


def Jr_log(phi):
    """计算 SO(3) log 右雅可比。"""
    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) + 0.5 * hat(phi)
    else:
        J = (
            np.eye(3)
            + 0.5 * hat(phi)
            + (
                1 / np.power(theta, 2.0)
                + (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
            )
            * hat(phi)
            * hat(phi)
        )
    return J


def unwrap_rpy(rpys):
    """将角度序列从 [-180, 180) 展开为连续曲线，避免绘图时出现跳变。"""
    diff = rpys[1:, :] - rpys[0:-1, :]
    uw_rpys = np.zeros(rpys.shape)
    uw_rpys[0, :] = rpys[0, :]
    diff[diff > 300] = diff[diff > 300] - 360
    diff[diff < -300] = diff[diff < -300] + 360
    uw_rpys[1:, :] = uw_rpys[0, :] + np.cumsum(diff, axis=0)
    return uw_rpys


def wrap_rpy(uw_rpys):
    """把展开后的欧拉角重新包装回 [-180, 180) 区间。"""
    rpys = uw_rpys
    while rpys.min() < -180:
        rpys[rpys < -180] = rpys[rpys < -180] + 360
    while rpys.max() >= 180:
        rpys[rpys >= 180] = rpys[rpys >= 180] - 360
    return rpys
