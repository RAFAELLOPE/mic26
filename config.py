# A Configuration class for image and model parameters 
class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.root_dir = '/lustre/home/rlopez/adni2'
    self.model = 'ST'
    self.preprocess_type = 'native'
    self.tissue_type = 'GM'
    self.labels = ['Normal', 'AD']
    self.test_size = 0.2
    self.val_size = 0.2
    self.force_balanced_data = False
    self.n_epochs = 100
    self.learning_rate = 0.0005
    self.batch_size = 10
    self.classification_threshold= 0.5
    self.test_results_dir = './data'
    if self.force_balanced_data:
      self.name = f"ADNI_{self.model}_DenseNet_B_{self.labels[0]}_vs_{self.labels[1]}_{self.preprocess_type}_{self.tissue_type}"
    else:
      self.name = f"ADNI_{self.model}_DenseNet_NB_{self.labels[0]}_vs_{self.labels[1]}_{self.preprocess_type}_{self.tissue_type}"