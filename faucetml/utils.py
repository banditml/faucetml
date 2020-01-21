import logging


def get_logger(name: str):
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-3s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
