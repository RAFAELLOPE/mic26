from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np


class PDLoader:
    def __init__(self, config):
        
        data_dicts = []
        for i, l in enumerate(config.labels):
            data_paths = sorted(glob.glob(f'{config.root_dir}\\{l}\\**\\mwp1*.nii'))
            for img in data_paths:
                data_dicts.append(
                    {
                        "image" : img,
                        "label": i
                    }
                )
        
        if config.force_balanced_data:
            data_dicts = self._balance_data(data_dicts)


        self.train_val, self.test_files = train_test_split(
            data_dicts, 
            test_size=config.test_size, 
            stratify=[x["label"] for x in data_dicts], 
            random_state=42
        )
        

        self.train_files, self.val_files = train_test_split(
            self.train_val, 
            test_size=(len(self.train_val) / len(data_dicts)) * config.val_size, 
            stratify=[x["label"] for x in self.train_val], 
            random_state=42
        )
    

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


        


        