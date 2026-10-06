from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np
import torch
from monai.data import Dataset
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


class PDLoader:
    def __init__(self, config):
        
        data_dicts = []
        tissue_prefix = 1      # By default, always consider Grey Matter (GM)
        process_prefix = 'c'   # By default, always consider native space images
        if config.tissue_type == 'WM':
            tissue_prefix = 2

        if config.preprocess_type == 'warped':
            process_prefix = 'wc'
        elif config.preprocess_type == 'modulated':
            process_prefix = 'mwc'
    
        for i, l in enumerate(config.labels):
            data_paths = sorted(glob.glob(f'{config.root_dir}/{l}/**/{process_prefix}{tissue_prefix}*.nii'))
            for img in data_paths:
                data_dicts.append(
                    {
                        "image" : img,
                        "label": i
                    }
                )
                
        if config.force_balanced_data:
            data_dicts = self._balance_data(data_dicts)


        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys=["image"], func=lambda x: torch.nan_to_num(x, nan=0.0)),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            ToTensord(keys=["image", "label"]),
        ])
    

        self.train_val_files, self.test_files = train_test_split(
            data_dicts, 
            test_size=config.test_size, 
            stratify=[x["label"] for x in data_dicts], 
            random_state=42
        )
        

        self.train_files, self.val_files = train_test_split(
            self.train_val_files, 
            test_size=(len(self.train_val_files) / len(data_dicts)) * config.val_size, 
            stratify=[x["label"] for x in self.train_val_files], 
            random_state=42
        )

        self.train_ds = Dataset(data=self.train_files, transform=self.transforms)
        self.val_ds = Dataset(data=self.val_files, transform=self.transforms)
        self.test_ds = Dataset(data=self.test_files, transform=self.transforms)
    

    def _balance_data(self, data):
        df_data_dicts = pd.DataFrame(data)
        min_samples = np.inf
        for l in list(set(df_data_dicts['label'])):
            if len(df_data_dicts[df_data_dicts['label'] == l]) < min_samples:
                min_samples = len(df_data_dicts[df_data_dicts['label'] == l])
            
        df_data_dicts_s = df_data_dicts.groupby(
            'label'
        ).sample(
            n=min_samples,
            random_state=42
        )

        data_dicts_r = df_data_dicts_s.to_dict(orient='records')
        return data_dicts_r


        


        