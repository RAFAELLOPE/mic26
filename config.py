# A Configuration class for image and model parameters 
class Config:
  """
  Holds configuration parameters
  """
  def __init__(self):
    self.name = "ADNI_Basic_DenseNet_NB_mci_vs_ad_warped_GM"
    self.root_dir = '/lustre/home/rlopez/adni2'
    self.preprocess_type = 'warped'
    self.tissue_type = 'GM'
    self.labels = ['MCI', 'AD']
    self.test_size = 0.2
    self.val_size = 0.2
    self.force_balanced_data = False
    self.n_epochs = 100
    self.learning_rate = 0.0005
    self.batch_size = 10
    self.classification_threshold= 0.5
    self.test_results_dir = './data'