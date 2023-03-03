import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split  


bin_data = pd.read_csv('datasets/bin_data.csv')
bin_data.drop(bin_data.columns[0], axis=1, inplace=True)
multi_data = pd.read_csv('datasets/multi_data.csv')
multi_data.drop(multi_data.columns[0], axis=1, inplace=True)
le1_classes_ = np.load('labels/le1_classes.npy', allow_pickle=True)
le2_classes_ = np.load('labels/le2_classes.npy', allow_pickle=True)

# splitting the dataset 75% for training and 25% testing
X_train, X_test = train_test_split(bin_data, test_size=0.25, random_state=42)

# dataset excluding target attribute (encoded, one-hot-encoded,original)
X_train = X_train.drop(['intrusion', 'abnormal', 'normal', 'label'], axis=1)

y_test = X_test['intrusion']  # target attribute

# dataset excluding target attribute (encoded, one-hot-encoded,original)
X_val = X_test.drop(['intrusion', 'abnormal', 'normal', 'label'], axis=1)

# Convert datasets to numpy arrays
X_train = X_train.values
X_val = X_val.values


# Định nghĩa AE
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), int(1 + torch.sqrt(torch.Tensor([n_features])))),
            nn.Tanh(),
            nn.Linear(int(1 + torch.sqrt(torch.Tensor([n_features]))), n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Train the model
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    train_loss = []
    early_stop_counter = 0
    min_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data_tensor = torch.stack(data)
            inputs = data_tensor.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))

        # Early stopping: check convergence every 5 epochs
        if (epoch + 1) % 5 == 0:
            if train_loss[-1] < min_loss:
                min_loss = train_loss[-1]
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= 5:
                print(f"Training stopped early at epoch {epoch+1} due to convergence.")
                break

        print(f"Epoch {epoch+1} Loss: {train_loss[-1]:.6f}")

    return model, train_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_tensor = torch.stack(batch)
            batch = batch_tensor.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


# Khởi tạo mô hình
n_features = X_train.shape[1]
model = Autoencoder(n_features)


# Khởi tạo trọng số theo phương pháp Xavier
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


model.apply(weights_init)

# Hàm mất mát và tối ưu
criterion = nn.MSELoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.1)

# Chuyển dữ liệu thành đối tượng torch.Tensor
train_data = torch.Tensor(X_train)
val_data = torch.Tensor(X_val)

# Sử dụng DataLoader để tạo các batch dữ liệu
batch_size = 100
train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)

# Đưa mô hình lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Huấn luyện mô hình
num_epochs = 100
model, train_loss = train(model, train_loader, criterion, optimizer, num_epochs, device)

# Đánh giá mô hình trên tập dữ liệu kiểm tra
val_loss = evaluate(model, val_loader, criterion, device)
print(f"Validation Loss: {val_loss:.6f}")

# Lưu trọng số của mô hình sau khi huấn luyện
if not os.path.exists('models'):
    os.makedirs('models')
model_path = 'models/autoencoder.pt'
torch.save(model.state_dict(), model_path)
print(f"Trained model saved at {model_path}")


plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
