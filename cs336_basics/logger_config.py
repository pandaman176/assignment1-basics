import logging

def setup_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # 防止重复添加
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)

        logger.addHandler(console)

    return logger
