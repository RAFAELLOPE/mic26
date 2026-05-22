import os
import glob
import torch
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# MONAI
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, CropForegroundd, Resized, ToTensord, Lambdad
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121

# 1. CONFIGURACIÓN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_model = "model_PDw.pth"

# 2. DATOS
# path_control = sorted(glob.glob('./data/pd_norm/CTL/**/ws*.nii'))
# path_pd = sorted(glob.glob('./data/pd_norm/PD/**/ws*.nii'))
path_control = sorted(glob.glob('C:\\Users\\34616\\Documents\\MATLAB Drive\\neuro_db_T1_seg\\Control\\**\\mwp1*.nii'))
path_pd = sorted(glob.glob('C:\\Users\\34616\\Documents\\MATLAB Drive\\neuro_db_T1_seg\\PD\\**\\mwp1*.nii'))

# Forzamos balanceo exacto
min_samples = min(len(path_control), len(path_pd))
path_control = path_control[:min_samples]
path_pd = path_pd[:min_samples]
print(f"****  Control: {len(path_control)} | PD: {len(path_pd)}")

images = path_control + path_pd
labels = [0] * len(path_control) + [1] * len(path_pd)
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# Split 70-15-15
train_val, test_files = train_test_split(data_dicts, test_size=0.15, stratify=labels, random_state=42)
train_files, val_files = train_test_split(train_val, test_size=0.176, stratify=[x["label"] for x in train_val], random_state=42)

#3. TRANSFORMACIONES
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

train_ds = Dataset(data=train_files, transform=transforms)
val_ds = Dataset(data=val_files, transform=transforms)
test_ds = Dataset(data=test_files, transform=transforms)

# DataLoader SIN Sampler (shuffle basta para clases balanceadas)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=1)

# 4. MODELO
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 5. ENTRENAMIENTO
def train(epochs):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch["image"].to(device), batch["label"].to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
        
        # Validación rápida
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for b in val_loader:
                out = model(b["image"].to(device))
                v_loss += loss_function(out, b["label"].to(device).float().unsqueeze(1)).item()
        
        avg_v = v_loss / len(val_loader)
        print(f"E{epoch+1} Val Loss: {avg_v:.4f}")
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), file_model)

# 6. EVALUACIÓN
def evaluate(threshold):
    model.load_state_dict(torch.load(file_model, weights_only=True))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in test_loader:
            prob = torch.sigmoid(model(b["image"].to(device))).item()
            y_pred.append(1 if prob > threshold else 0)
            y_true.append(b["label"].item())
    
    print("\n" + classification_report(y_true, y_pred, target_names=['Control (Spec)', 'PD (Sens)']))

# EJECUCIÓN
if not os.path.exists(file_model):
    train(20)
evaluate(0.5)

