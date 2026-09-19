import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

torch.manual_seed(42)
n_features = X_train.shape[1]

model = nn.Linear(n_features, 1)
optimizer = torch.optim.SGD(model.parameters(), 0.2)
mse = nn.MSELoss()


def test(
    model=model,
    optimizer=optimizer,
    criterion=mse,
    X_train=X_train,
    y_train=y_train,
    epochs=20,
):
    for epoch in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"{epoch}/{epochs} Loss: {loss.item()}")


test()
