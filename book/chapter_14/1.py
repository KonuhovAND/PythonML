from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import tokenizers as tk
import torch
import torch.nn as nn

imdb_dataset = load_dataset("stanfordnlp/imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]
train_reviews = [review["text"].lower() for review in imdb_train_set]
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch, tokenizer=bert_tokenizer):
    reviews = [review["text"] for review in batch]
    labels = [review["label"] for review in batch]
    if tokenizer is None:
        raise RuntimeError("model doesnt load")
    else:
        encodings = tokenizer(
            reviews,
            padding=True,
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
    labels = torch.tensor(labels, torch.float32)
    return encodings, labels


batch_size = 256
imdb_train_set = DataLoader(
    imdb_train_set,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
)

imdb_valid_loader = DataLoader(
    imdb_valid_set,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

imdb_test_loader = DataLoader(
    imdb_test_set,
    batch_size=32,
    collate_fn=collate_fn,
)


class SentimentAnalysisModel(nn.Module):
    def __init__(
        self, vocab_size, n_layers=2, embed_dim=128, hidden_di=64, pad_id=0, dropout=0.2
    ):
        super().__init__()
        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            pad_id,
        )
        self.gru = nn.GRU(
            embed_dim,
            hidden_di,
            n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Linear(hidden_di, 1)

    def forward(self, encodings):
        embeddings = self.embed(encodings["input_ids"])
        _outputs, hidden_states = self.gru(embeddings)
        return self.output(hidden_states[-1])
