
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
# import sys
# sys.path.append(os.path.abspath('.'))
from src.loader.load_data import PDLoader
from src.model.DenseNet_model import DenseNetModel
from src.trainer.inference import InferenceAgent
import numpy as np


class Experiment:
    def __init__(self, config):
        self.n_epochs = config.n_epochs
        self.name = config.name
        self.time_start = ""
        self.time_end = ""
        self.epoch = 0
        self.classification_threshold = config.classification_threshold

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
            Resized(keys=["image"], spatial_size=(128, 128, 128)),
            ToTensord(keys=["image", "label"]),
        ])

        pd_loader = PDLoader(config)

        self.train_loader = DataLoader(
            Dataset(
                pd_loader.train_files,
                transform=transforms
            ), 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            Dataset(
                pd_loader.val_files,
                transform=transforms
            ), 
            batch_size=config.batch_size
        )

        self.test_loader = DataLoader(
            Dataset(
                pd_loader.test_files,
                transform=transforms
            ), 
            batch_size=1
        )


        # Check CUDA is available
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
    
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Load the model
        self.model = DenseNetModel(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=1
        ).to(self.device)

        # Set loss function
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        # Set Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )

        # Scheduler to help update learning rate automatically
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    
    def train(self):
        """
        This method is executed once per epoch and takes care of 
        model weight update cycle
        """

        print(f"Training epoch {self.epoch} ...")
        self.model.train()
        loss_list = []

        # Loop over minibatches
        for i, batch in enumerate(self.train_loader):
            inputs = batch["image"].to(self.device).type(torch.float)
            targets = batch["label"].to(self.device).unsqueeze(1).type(torch.long)
            predictions = self.model(inputs)
            loss = self.loss_function(predictions, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_list.append(loss.item())

            if (i % 10) == 0:
                # Output to the console every 10th batch
                print(f"Epoch: {self.epoch} Train loss:{loss/len(self.train_loader)}, {100*(i+1)/len(self.train_loader):.1}% complete")
        
        print("\nTraining comple")
        return loss_list


    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval mode 
        and no_grad needs to be called so that gradients do not 
        propagate.
        """
        
        print(f"Validating epoch {self.epoch}...")
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                inputs = batch["image"].to(self.device).type(torch.float)
                targets = batch["label"].to(self.device).unsqueeze(1).type(torch.long)
                predictions = self.model(inputs)

                loss = self.loss_function(predictions, targets)
                loss_list.append(loss)
                
                if (i % 10) == 0:
                    # Output to the console every 10th batch
                    print(f"Epoch: {self.epoch} Val loss:{loss/len(self.val_loader)}, {100*(i+1)/len(self.val_loader):.1}% complete")
        
        self.scheduler.step(np.mean(loss_list))
        print(f"\nValidation complete")
        return loss_list


    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find model path {model_path}")


    def evaluate(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different.
        Here we are computing a lot more metrics and returning a 
        dictionary that could later be persisted on a JSON.
        """
        print("Testing...")
        self.model.eval()

        inference_agent = InferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict['probabilities'] = []
        out_dict['prediction'] = []
        out_dict['true'] = []

        for x in self.test_loader:
            pred_prob, pred_label = inference_agent.single_image_inference(
                x["image"], 
                threshold=self.classification_threshold
            )
            true_label = x["label"].item()
            out_dict['probabilities'].append(pred_prob)
            out_dict['prediction'].append(pred_label)
            out_dict['true'].append(true_label)
        
        print("\nTesting complete.")
        return out_dict



    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self.time_start = time.time()
        print("Experiment started")
        result = dict()
        result['epoch'] = []
        result['train_loss'] = []
        result['val_loss'] = []

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            train_loss_list = self.train()
            val_loss_list = self.validate()
            result['epoch'].append(self.epoch)
            result['train_loss'].append(np.mean(train_loss_list))
            result['val_loss'].append(np.mean(val_loss_list))
            print(f"Epoch: {self.epoch + 1}, Train error: {np.mean(train_loss_list)}, Validation loss: {np.mean(val_loss_list)}")

        # Save model for inferencing
        self.save_model_parameters()

        self.time_end = time.time()
        print(f"Run complete. Total tiem:{time.strftime('%H:%M:%S', time.gmtime(self.time_end - self.time_start))}")
        return result




