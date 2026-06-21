"""
性能分析上下文管理器。

Reference: https://github.com/CathIAS/TLIO/blob/master/src/utils/profile.py

用法示例:
    with profile("run.prof", enabled=True):
        run_filter()

启用后会把 cProfile 结果写到指定文件，便于后续用 snakeviz/pstats 分析耗时。
"""

import contextlib
import cProfile


@contextlib.contextmanager
def profile(filename, enabled=True):
    """
    在 `with` 代码块内开启 cProfile。

    参数:
    - `filename`: profile 结果保存路径。
    - `enabled`: False 时该 context manager 不产生任何性能分析开销。
    """
    if enabled:
        # 创建并启动 profiler。
        profile = cProfile.Profile()
        profile.enable()
    try:
        yield
    finally:
        if enabled:
            # 无论代码块正常结束还是抛异常，都会关闭 profiler 并保存结果。
            profile.disable()
            profile.dump_stats(filename)

