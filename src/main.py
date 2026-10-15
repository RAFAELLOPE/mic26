import os
import sys
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
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
        target_names=[config.labels[0], config.labels[1]])
    )
    # Save results to file
    with open(os.path.join(experiment.out_dir, "test_results.txt"), "w") as f:
        f.write(classification_report(
            out_results['true'], 
            out_results['prediction'], 
            target_names=[config.labels[0], config.labels[1]])
        )

    # Save results in test as a dictionary 
    with open(os.path.join(experiment.out_dir, "model_results.json"), 'w') as fp:
        json.dump(out_results, fp)

    # Save ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(
        out_results['true'], 
        out_results['probabilities'])
    plt.savefig(os.path.join(experiment.out_dir, "ROC_curve.png"))
    plt.close()

    # Save Confusion Matrix
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_true=out_results['true'],
        y_pred=out_results['prediction']
    )
    plt.savefig(os.path.join(experiment.out_dir, "confusion_matrix.png"))
    plt.close()