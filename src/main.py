import os
import sys
sys.path.append(os.path.abspath('.'))
from src.trainer.training import Experiment
from config import Config



if __name__ == "__main__":
    config = Config()
    experiment = Experiment(config)
    experiment.run()
