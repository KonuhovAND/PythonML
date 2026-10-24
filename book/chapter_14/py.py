from datasets import load_dataset
import tokenizers as tk
from transformers import AutoTokenizer


imdb_dataset = load_dataset("stanfordnlp/imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]


print(imdb_train_set[1]["label"])
bpe_model = tk.models.BPE(unk_totek="<unk>")
bpe_tokenizer = tk.Tokenizer(bpe_model)
bpe_tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel()

special_tokens = [
    "<pad>",
    "<unk>",
]
bpe_trainer = tk.trainers.BpeTrainer(
    vocab_size=10**3,
    special_tokens=special_tokens,
)
print(len(imdb_train_set))
train_reviews = [review["text"].lower() for review in imdb_train_set]
bpe_tokenizer.train_from_iterator(train_reviews, bpe_trainer)


some_review = "what an awesome movie!"
bpe_encoding = bpe_tokenizer.encode(some_review)
print(bpe_encoding)
print(bpe_encoding.tokens)

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
if gpt2_tokenizer is None:
    raise RuntimeError("model doesnot load")
else:
    gpt2_encoding = gpt2_tokenizer(train_reviews[:3], trancation=True, max_length=500)
gpt2_tokens_ids = gpt2_encoding["input_ids"][0][:10]
print(gpt2_tokens_ids)
print(gpt2_tokenizer.decode(gpt2_tokens_ids))
