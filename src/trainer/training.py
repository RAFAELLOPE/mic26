
import time
import os
import torch
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Orientationd,
    NormalizeIntensityd, 
    CropForegroundd, 
    Resized, 
    ToTensord, 
    Lambdad
)
from monai.data import Dataset, DataLoader


class Experiment:
    def __init__(self, config, data):
        self.n_epochs = config.n_epochs
        self.name = config.name
        self.time_start = ""
        self.time_end = ""
        self.epoch = 0

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data loaders
        transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys=["image"], func=lambda x: torch.nan_to_num(x, nan=0.0)),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"]),
            # CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0, margin=5),
            # # Slicing: Mitad superior del cerebro (Eje Z)
            # lambda data: {
            #     **data,
            #     "image": data["image"][:, :, :, data["image"].shape[3]//2 :]
            # },
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            ToTensord(keys=["image", "label"]),
        ])

        self.train_loader = DataLoader(
            train_ds, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=config.batch_size
        )

        # Check CUDA is available
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
    
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    def train(self):
        pass

    def validate(self):
        pass

    def save_model_parameters(self):
        pass

    def load_model_parameters(self):
        pass

    def run_test(self):
        pass

    def run(self):
        pass



