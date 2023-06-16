from config import Config
from train import Trainer


if __name__ == '__main__':

    c = Config()
    trainer = Trainer(config = c)
    trainer.train_loop()
