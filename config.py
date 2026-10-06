# A Configuration class for image and model parameters 
class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.name = "ADNI_Basic_DenseNet_normal_vs_ad_native_GM"
    self.root_dir = '/lustre/home/rlopez/adni2'
    self.preprocess_type = 'native'
    self.tissue_type = 'GM'
    self.labels = ['Normal', 'AD']
    self.test_size = 0.15
    self.val_size = 0.15
    self.force_balanced_data = True
    self.n_epochs = 1
    self.learning_rate = 0.0005
    self.batch_size = 10
    self.classification_threshold= 0.5
    self.test_results_dir = './data'