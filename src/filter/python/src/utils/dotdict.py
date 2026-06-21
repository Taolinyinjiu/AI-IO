class dotdict(dict):
    """
    支持点号访问的 dict。

    普通字典访问方式:
        cfg["sigma_na"]

    使用 dotdict 后可以写成:
        cfg.sigma_na

    这个项目中常用它把 argparse/config 里的滤波参数传给 `ImuMSCKF`。
    """

    def __getattr__(self, name):
        """把属性访问转发成字典 key 查询。"""
        try:
            return self[name]
        except KeyError:
            # 保持 Python 属性访问语义：不存在的属性抛 AttributeError。
            raise AttributeError
