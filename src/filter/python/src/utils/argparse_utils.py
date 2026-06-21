import argparse


def add_bool_arg(parser, name, default=False, **kwargs):
    """
    为 argparse 增加一组成对的布尔开关。

    例如 `name="log_full_state"` 时，会同时注册：
    - `--log_full_state`：显式打开。
    - `--no-log_full_state`：显式关闭。

    这样命令行可以清楚地区分“使用默认值”和“用户主动覆盖默认值”。
    """
    # mutually_exclusive_group 保证同一个布尔选项不能同时打开和关闭。
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name,
        dest=name,
        action="store_true",
        help="Default: " + ("Enabled" if default else "Disabled"),
    )
    # `kwargs` 会传给关闭选项，调用方可补充 help 等信息。
    group.add_argument("--no-" + name, dest=name, action="store_false", **kwargs)
    # 如果命令行没有出现该选项，则使用调用方给定的默认值。
    parser.set_defaults(**{name: default})
