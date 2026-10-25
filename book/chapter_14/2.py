from transformers import BertForSequenceClassification
import torch

torch.manual_seed(42)
device = torch.device("cpu")

best_for_binary_clf = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    dtype=torch.float16,
).to(device)
