import copy

import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)

X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

y_train, y_test = (
    torch.FloatTensor(y_train).reshape((-1, 1)),
    torch.FloatTensor(y_test).reshape((-1, 1)),
)

train_loader = TensorDataset(X_train, y_train)
dataloader = DataLoader(train_loader, batch_size=32, shuffle=True)

device = torch.device("cpu")
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    nn.Linear(50, 1),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.01)
criterion = nn.MSELoss()


for module in model:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


for _ in range(20):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

median = y_train.median()

y_train_bin = (y_train > median).float()
y_test_bin = (y_test > median).float()

hidden_layer = copy.deepcopy(model[:-1])
if isinstance(model[-1], nn.Linear):
    number_of_hidden_features = model[-1].in_features
else:
    number_of_hidden_features = 50
bin_model = nn.Sequential(
    hidden_layer,
    nn.Linear(number_of_hidden_features, 1),
).to(device)

bin_creterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(bin_model[1].parameters(), lr=0.001)

binary_loader = DataLoader(
    TensorDataset(X_train, y_train_bin),
    32,
    True,
)

for params in bin_model[0].parameters():
    params.requires_grad = False

for interation in range(20):
    bin_model.train()
    bin_model[0].eval()
    for x_batch, y_batch in binary_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = bin_model(x_batch)
        loss = bin_creterion(y_batch, pred)
        loss.backward()
        nn.utils.clip_grad_norm_(bin_model.parameters(), 1.0)

        optimizer.step()

for params in bin_model[0].parameters():
    params.requires_grad = True

optimizer = torch.optim.Adam(bin_model.parameters(), lr=0.001)

for interation in range(20):
    bin_model.train()
    for x_batch, y_batch in binary_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = bin_model(x_batch)
        loss = bin_creterion(y_batch, pred)
        loss.backward()
        nn.utils.clip_grad_norm_(bin_model.parameters(), 1.0)
        optimizer.step()
bin_model.eval()
with torch.no_grad():
    predictions = (bin_model(X_test.to(device)) >= 0).float()
    accuracy = (predictions == y_test_bin.to(device)).float().mean()
    print("Bin accurace ", accuracy.item())

model.eval()

with torch.no_grad():
    predictions = model(X_test.to(device))
    test_loss = criterion(predictions, y_test.to(device))

print(test_loss.item())
