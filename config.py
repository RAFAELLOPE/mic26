# A Configuration class for image and model parameters 
class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.name = "Basic_DenseNet"
    self.root_dir = '/lustre/home/rlopez/neuro_db_T1_seg'
    self.labels = ['Control', 'PD']
    self.test_size = 0.15
    self.val_size = 0.15
    self.force_balanced_data = True
    self.n_epochs = 50
    self.learning_rate = 0.0001
    self.batch_size = 16
    self.classification_threshold= 0.5
    self.test_results_dir = './data'