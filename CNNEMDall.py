#%%
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
# 1. Dataset personalizado
class TabularImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.tabular_data = df.drop(columns=[ 'TAG']).values.astype('float32')
        self.labels = df['TAG'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Imagen'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tabular = torch.tensor(self.tabular_data[idx])
        label = torch.tensor(self.labels[idx])
        return image, tabular, label

#%%
# 2. Modelo CNN + MLP
class CNNWithTabular(nn.Module):
    def __init__(self, tabular_input_dim):
        super(CNNWithTabular, self).__init__()
        
        # CNN para imagen
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # input: 3x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x32x32
            nn.Flatten()
        )

        self._to_linear = 32 * 32 * 32  # calcular según la salida real del CNN
        
        # MLP para datos tabulares
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_dim, 32),
            nn.ReLU(),
        )

        # Fusión
        self.combined = nn.Sequential(
            nn.Linear(self._to_linear + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, tabular):
        x1 = self.cnn_branch(image)
        x2 = self.tabular_branch(tabular)
        x = torch.cat((x1, x2), dim=1)
        return self.combined(x).squeeze()

#%%
# 3. Preprocesamiento y carga de datos
df = pd.read_csv("data.csv")
scaler = StandardScaler()
features = df.drop(columns=['Imagen', 'TAG'])
df[features.columns] = scaler.fit_transform(features)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = TabularImageDataset(train_df, image_dir='SlowWeb', transform=transform)
val_dataset = TabularImageDataset(val_df, image_dir='SlowWeb', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

#%%
# 4. Entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithTabular(tabular_input_dim=len(y)).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, tabular, labels in train_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
