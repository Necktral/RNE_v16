import logging
import sys

def get_logger(name=__name__, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def log_metrics(logger, metrics: dict, step: int = None):
    msg = f"[METRICS]{' [step %d]' % step if step is not None else ''} " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(msg)
