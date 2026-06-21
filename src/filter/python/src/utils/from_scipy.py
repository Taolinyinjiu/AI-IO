"""
从 SciPy Rotation 中移植出来的旋转表示转换工具。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/from_scipy.py

保留这份实现的原因通常是:
- 避免不同 SciPy 版本在接口/行为上的细微差异。
- 在 numba 或批量处理场景中拥有更可控的 numpy 实现。
"""

import warnings

import numpy as np

_AXIS_TO_IND = {"x": 0, "y": 1, "z": 2}


def _elementary_basis_vector(axis):
    """返回指定坐标轴的单位基向量，例如 axis='x' 得到 [1, 0, 0]。"""
    b = np.zeros(3)
    b[_AXIS_TO_IND[axis]] = 1
    return b


def compute_euler_from_matrix(matrix, seq, extrinsic=False):
    """
    从旋转矩阵计算欧拉角。

    参数:
    - `matrix`: 单个 3x3 矩阵或 N 个 3x3 矩阵。
    - `seq`: 欧拉角轴序列，例如 "xyz"。
    - `extrinsic`: 是否按外旋解释。

    返回值单位是弧度，形状为 `[N, 3]`。
    """
    # The algorithm assumes intrinsic frame transformations. The algorithm
    # in the paper is formulated for rotation matrices which are transposition
    # rotation matrices used within Rotation.
    # Adapt the algorithm for our case by
    # 1. Instead of transposing our representation, use the transpose of the
    #    O matrix as defined in the paper, and be careful to swap indices
    # 2. Reversing both axis sequence and angles for extrinsic rotations

    if extrinsic:
        # 外旋可以通过反转轴序列转成内旋处理，最后再反转角顺序。
        seq = seq[::-1]

    if matrix.ndim == 2:
        # 统一成批量维度，便于后续向量化计算。
        matrix = matrix[None, :, :]
    num_rotations = matrix.shape[0]

    # Step 0
    # Algorithm assumes axes as column vectors, here we use 1D vectors
    n1 = _elementary_basis_vector(seq[0])
    n2 = _elementary_basis_vector(seq[1])
    n3 = _elementary_basis_vector(seq[2])

    # Step 2
    # sl/cl 决定论文算法里的 lambda offset。
    sl = np.dot(np.cross(n1, n2), n3)
    cl = np.dot(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    offset = np.arctan2(sl, cl)
    c = np.vstack((n2, np.cross(n1, n2), n1))

    # Step 3
    # 把输入矩阵变换到算法所需的中间坐标系。
    rot = np.array([[1, 0, 0], [0, cl, sl], [0, -sl, cl]])
    res = np.einsum("...ij,...jk->...ik", c, matrix)
    matrix_transformed = np.einsum("...ij,...jk->...ik", res, c.T.dot(rot))

    # Step 4
    angles = np.empty((num_rotations, 3))
    # arccos 输入必须限制在 [-1, 1]，避免浮点误差导致 NaN。
    positive_unity = matrix_transformed[:, 2, 2] > 1
    negative_unity = matrix_transformed[:, 2, 2] < -1
    matrix_transformed[positive_unity, 2, 2] = 1
    matrix_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = np.arccos(matrix_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    # safe1/safe2 用来判断是否接近万向节锁。
    safe1 = np.abs(angles[:, 1]) >= eps
    safe2 = np.abs(angles[:, 1] - np.pi) >= eps

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = np.logical_and(safe1, safe2)
    angles[safe_mask, 0] = np.arctan2(
        matrix_transformed[safe_mask, 0, 2], -matrix_transformed[safe_mask, 1, 2]
    )
    angles[safe_mask, 2] = np.arctan2(
        matrix_transformed[safe_mask, 2, 0], matrix_transformed[safe_mask, 2, 1]
    )

    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = np.arctan2(
            matrix_transformed[~safe1, 1, 0] - matrix_transformed[~safe1, 0, 1],
            matrix_transformed[~safe1, 0, 0] + matrix_transformed[~safe1, 1, 1],
        )
        # 6c
        angles[~safe2, 2] = -(
            np.arctan2(
                matrix_transformed[~safe2, 1, 0] + matrix_transformed[~safe2, 0, 1],
                matrix_transformed[~safe2, 0, 0] - matrix_transformed[~safe2, 1, 1],
            )
        )
    else:
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = np.arctan2(
            matrix_transformed[~safe1, 1, 0] - matrix_transformed[~safe1, 0, 1],
            matrix_transformed[~safe1, 0, 0] + matrix_transformed[~safe1, 1, 1],
        )
        # 6c
        angles[~safe2, 0] = np.arctan2(
            matrix_transformed[~safe2, 1, 0] + matrix_transformed[~safe2, 0, 1],
            matrix_transformed[~safe2, 0, 0] - matrix_transformed[~safe2, 1, 1],
        )

    # Step 7
    if seq[0] == seq[2]:
        # proper Euler angle，第二个角限制到 [0, pi]。
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:
        # Tait-Bryan angle，第二个角限制到 [-pi/2, pi/2]。
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = np.logical_or(angles[:, 1] < -np.pi / 2, angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = np.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not np.all(safe_mask):
        # 万向节锁时无法唯一确定三个角，和 SciPy 行为一致地发出 warning。
        warnings.warn(
            "Gimbal lock detected. Setting third angle to zero since"
            " it is not possible to uniquely determine all angles."
        )

    # Reverse role of extrinsic and intrinsic rotations, but let third angle be
    # zero for gimbal locked cases
    if extrinsic:
        # 外旋输出角顺序恢复到用户传入的轴序列。
        angles = angles[:, ::-1]
    return angles


def compute_q_from_matrix(matrix):
    """
    从旋转矩阵计算四元数。

    输出顺序为 `[qx, qy, qz, qw]`，与 SciPy `Rotation.as_quat()` 一致。
    支持单个 3x3 矩阵或 `[N, 3, 3]` 批量矩阵。
    """
    is_single = False
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected `matrix` to have shape (3, 3) or "
            "(N, 3, 3), got {}".format(matrix.shape)
        )

    # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
    # set self._single to True so that we can return appropriate objects in
    # the `to_...` methods
    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
        is_single = True

    num_rotations = matrix.shape[0]

    # 根据 trace 和对角元素选择数值最稳定的四元数分支。
    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    # 选择 x/y/z 分量最大的分支。
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    # trace 最大时使用 w 分量分支。
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    # 归一化消除数值误差。
    quat /= np.linalg.norm(quat, axis=1)[:, None]

    if is_single:
        return quat[0]
    else:
        return quat
