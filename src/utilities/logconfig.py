import logging

def setup_logging(name, level=logging.INFO, log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    formatter = logging.Formatter(log_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger