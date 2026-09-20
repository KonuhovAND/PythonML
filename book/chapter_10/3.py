import optuna
import torch
import torchmetrics
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

y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def objective(trial, loader=train_dataset):

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    n_size = trial.suggest_int("n_size", 20, 100)
    train_loader = DataLoader(loader, batch_size=n_size, shuffle=True)
    model = nn.Linear(X_train.shape[1], 1)

    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(20):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "as1.pt")
    model.eval()
    metric = torchmetrics.MeanSquaredError(False)
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            metric.update(y_pred, y_batch)
    return metric.compute().item()


torch.manual_seed(42)
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
