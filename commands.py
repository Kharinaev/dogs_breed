import fire

from dltoolkit.infer import inference
from dltoolkit.train import train_model


def train():
    train_model()


def infer():
    inference()


if __name__ == "__main__":
    fire.Fire()
