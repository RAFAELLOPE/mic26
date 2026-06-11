import os
import sys
from sklearn.metrics import classification_report
sys.path.append(os.path.abspath('.'))
from src.trainer.training import Experiment
from config import Config
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    config = Config()
    experiment = Experiment(config)
    results =experiment.run()
    out_results = experiment.evaluate()
    print("\n" + classification_report(
        out_results['true'], 
        out_results['prediction'], 
        target_names=['Control (Spec)', 'PD (Sens)'])
    )
    # Save results to file
    with open(os.path.join(experiment.out_dir, "test_results.txt"), "w") as f:
        f.write(classification_report(
            out_results['true'], 
            out_results['prediction'], 
            target_names=['Control (Spec)', 'PD (Sens)'])
        )
