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
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
pytorch_nn_model = nn.Linear(in_features=n_features, out_features=1)
optimizer = torch.optim.SGD(pytorch_nn_model.parameters(), lr=0.2)
mse = nn.MSELoss()
print(pytorch_nn_model.bias)
print(pytorch_nn_model.weight)

# learning_rate = 0.4
# for index in range(20):
#     y_pred = X_train @ w + b
#     loss = ((y_pred - y_train) ** 2).mean()
#     loss.backward()
#     with torch.no_grad():
#         assert w.grad is not None and b.grad is not None
#         b -= learning_rate * b.grad
#         w -= learning_rate * w.grad
#         b.grad.zero_()
#         w.grad.zero_()
#     print(f"Number is {index} Loss:{loss.item()}")
