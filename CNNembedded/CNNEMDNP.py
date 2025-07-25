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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
# 1. Dataset 
class TabularImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.tabular_data = df.drop(columns=[ 'Imagen','TAG']).values.astype('float32')
        self.labels = df['TAG'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Imagen']+'.png')
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
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self._to_linear = 32 * 32 * 32  

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
            nn.Sigmoid()  # Usamos sigmoide para clasificación binaria
        )

    def forward(self, image, tabular):
        x1 = self.cnn_branch(image) 
        x2 = self.tabular_branch(tabular)  

        x = torch.cat((x1, x2), dim=1)
        out = self.combined(x).squeeze(-1)  
        
        return out


#%%
# 3. Preprocesamiento y carga de datos
df = pd.read_csv("datamodelohenpk2.csv")
scaler = StandardScaler()

# Normalización de las features
features = df.drop(columns=['Imagen', 'TAG'])
df[features.columns] = scaler.fit_transform(features)

# Primero, separa 10% para test (estratificado por 'TAG')
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['TAG'])

# Luego, divide el 90% restante en 80% para entrenamiento y 10% para validación (estratificado por 'TAG')
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['TAG'])

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Crear los datasets para cada conjunto de datos
train_dataset = TabularImageDataset(train_df, image_dir='SlowWeb', transform=transform)
val_dataset = TabularImageDataset(val_df, image_dir='SlowWeb', transform=transform)
test_dataset = TabularImageDataset(test_df, image_dir='SlowWeb', transform=transform)

# Crear los DataLoaders para cada conjunto de datos
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Distribución en Train:")
print(train_df['TAG'].value_counts(normalize=True))
print("Distribución en Validation:")
print(val_df['TAG'].value_counts(normalize=True))
print("Distribución en Test:")
print(test_df['TAG'].value_counts(normalize=True))

#%%
# 4. Entrenamiento

# Entrenamiento del modelo con evaluación en validación y prueba
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithTabular(tabular_input_dim=len(features.columns)).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Número de épocas
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Entrenamiento
    for images, tabular, labels in train_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images, tabular)
       
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcular el accuracy
        preds = (outputs > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    epoch_accuracy = 100 * correct_preds / total_preds
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

#%%


# Función para evaluar el modelo en el conjunto de test
def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Establecer el modelo en modo evaluación
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No calcular gradientes
        for images, tabular, labels in test_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            
            # Hacer las predicciones
            outputs = model(images, tabular)
            preds = (outputs > 0.5).float()  # Predicciones binarizadas
            
            # Almacenar las predicciones y las etiquetas verdaderas
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convertir las listas a arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Generar el Classification Report
    report = classification_report(all_labels, all_preds)
    print("Classification Report:\n", report)

    # Generar la Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Evaluación final en el conjunto de test
print("Evaluating on Test Data after Training")
evaluate_model(model, test_loader, criterion, device)  # Evaluar en conjunto de test

# %%
