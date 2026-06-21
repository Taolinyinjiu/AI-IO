"""
统一日志配置。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/logging.py

其他模块通过 `from filter.python.src.utils.logging import logging` 复用这里的
全局 logging 设置，保证 filter 运行时的时间、文件名、行号和等级格式一致。
"""

# Copyright 2004-present Facebook. All Rights Reserved.

# 如果安装了 coloredlogs，则终端日志会带颜色；没有安装也不影响运行。
import logging
import sys


try:
    import coloredlogs

    coloredlogs.install()
except BaseException:
    # coloredlogs 是可选依赖，导入失败时退回 Python 标准 logging。
    pass

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO, # 可改为 logging.WARNING 或 logging.DEBUG 控制输出详细程度。
)
