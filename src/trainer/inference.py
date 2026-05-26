import torch
import os
from src.model.DenseNet_model import DenseNetModel

class InferenceAgent:
    def __init__(self, 
                 parameter_file_path = '', 
                 model=None, 
                 device='cpu'):
        
        self.model = model
        self.device = device

        if model == None:
            self.model = DenseNetModel(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=1
            )

        if os.path.exists(parameter_file_path):
            self.model.load_state_dict(
                torch.load(
                    parameter_file_path,
                    map_location=self.device
                )
            )
        
        self.model.to(device)
    
    def single_image_inference(self, img, threshold=0.5):
        """
        Runs inference on a single volume of conformant patch size
        """
        self.model.eval()
        input = img.to(self.device)
        prob = torch.sigmoid(self.model(input)).item()
        y_pred = 1 if prob > threshold else 0
        return prob, y_pred




