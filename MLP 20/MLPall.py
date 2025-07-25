#%%

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#%%
# 1. Dataset personalizado
class TabularDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.tabular_data = df.drop(columns=['TAG','Imagen']).values.astype('float32')
        self.labels = df['TAG'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tabular = torch.tensor(self.tabular_data[idx])
        label = torch.tensor(self.labels[idx])
        return tabular, label
#%%
# 2. Definición del modelo MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(input_dim, 30)  # Capa de 30 neuronas
        self.layer2 = nn.Linear(30, 10)  # Capa de 10 neuronas
        self.layer3 = nn.Linear(10, 10)  # Capa de 10 neuronas
        self.output_layer = nn.Linear(10, 1)  # Capa de salida para clasificación binaria

        # Funciones de activación
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))  # Probabilidad para clasificación binaria
        return x.squeeze(-1) 

#%%
# 3. Preprocesamiento de datos
df = pd.read_csv("..\datamodelohecnps2.csv")

scaler = StandardScaler()
features = df.drop(columns=['TAG','Imagen'])
df[features.columns] = scaler.fit_transform(features)

# División de datos (80% entrenamiento, 10% validación, 10% test)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['TAG'])
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['TAG'])

# Crear los datasets para cada conjunto de datos
train_dataset = TabularDataset(train_df)
val_dataset = TabularDataset(val_df)
test_dataset = TabularDataset(test_df)

# Crear los DataLoaders para cada conjunto de datos
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
#%%
# 4. Entrenamiento del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(features.columns)  # Número de características en los datos tabulares
model = MLP(input_dim).to(device)

criterion = nn.BCELoss()  # Para clasificación binaria
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Entrenamiento
    for tabular, labels in train_loader:
        tabular, labels = tabular.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(tabular)

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

    # Evaluación en el conjunto de validación
    model.eval()  # Poner el modelo en modo evaluación
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for tabular, labels in val_loader:
            tabular, labels = tabular.to(device), labels.float().to(device)

            outputs = model(tabular)
            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    val_accuracy = 100 * correct_preds / total_preds
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Evaluación final en el conjunto de test usando la función creada

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Establecer el modelo en modo evaluación
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No calcular gradientes
        for tabular, labels in test_loader:
            tabular, labels = tabular.to(device), labels.float().to(device)
            
            # Hacer las predicciones
            outputs = model(tabular)
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

print("Evaluating on Test Data after Training")
evaluate_model(model, test_loader, criterion, device)  # Evaluar en conjunto de test
