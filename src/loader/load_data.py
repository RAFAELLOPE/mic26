from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np

class PDLoader:
    def __init__(
            self, 
            data_path,
            labels, 
            test_size, 
            val_size,
            force_balanced_data = False
        ):
        
        data_dicts = []
        for i, l in enumerate(labels):
            data_paths = sorted(glob.glob(f'{data_path}/{l}/**/mwp1*.nii'))
            for img in data_paths:
                data_dicts.append(
                    {
                        "image" : img,
                        "label": i
                    }
                )
        
        if force_balanced_data:
            df_data_dicts = pd.DataFrame(data_dicts)
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

            data_dicts = df_data_dicts_s.to_dict(orient='records')


        self.train_val, self.test_files = train_test_split(
            data_dicts, 
            test_size=test_size, 
            stratify=[x["label"] for x in data_dicts], 
            random_state=42
        )
        

        self.train_files, self.val_files = train_test_split(
            self.train_val, 
            test_size=(len(self.train_val) / len(data_dicts)) * val_size, 
            stratify=[x["label"] for x in self.train_val], 
            random_state=42
        )


if __name__ == "__main__":
    pd_loader = PDLoader(
        data_path='C:\\Users\\34616\\Documents\\MATLAB Drive\\neuro_db_T1_seg',
        labels=['Control', 'PD'],
        test_size=0.15,
        val_size=0.15,
        force_balanced_data = False
    )
    print('loader created !!')
        


        