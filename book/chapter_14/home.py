from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
import tokenizers
from datasets import load_dataset

DEVICE = torch.device("cpu")
data = load_dataset(path="stanfordnlp/imdb")
split = data["train"].train_test_split(train_size=0.8, seed=42)
data_train = split["train"]
data_valid = split["test"]

bpe_model = tokenizers.models.BPE(unk_token="<unk>")
bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()

bpe_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=[
        "<pad>",
        "<unk>",
    ],
)
reviews = [review["text"].lower() for review in data_train]
bpe_tokenizer.train_from_iterator(reviews, bpe_trainer)
bpe_encoding = bpe_tokenizer.encode("nigga")


# ..............................

iris = load_iris()
iris_data, iris_target = iris.data, iris.target
iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(
    iris_data, iris_target, train_size=0.8, random_state=42
)
iris_data_train, iris_data_test = (
    torch.FloatTensor(iris_data_train),
    torch.FloatTensor(iris_data_test),
)
iris_target_train = torch.tensor(iris_target_train, dtype=torch.long)
iris_target_test = torch.tensor(iris_target_test, dtype=torch.long)

train_data_loader = TensorDataset(iris_data_train, iris_target_train)
data = DataLoader(
    train_data_loader,
    32,
    True,
)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(50):
    model.train()
    for x_batch, y_batch in data:
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(model(iris_data_test[:1]).argmax(dim=1), iris_target_test)


gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_tokenizer
