import tokenizers
from datasets import load_dataset


def train_eng_spa():
    for pair in nmt_train_set:
        yield pair["source_text"]
        yield pair["target_text"]


nmt_original_valid_set, nmt_test_set = load_dataset(
    path="ageron/tatoeba_mt_train",
    name="eng-spa",
    split=["validation", "test"],
)


split = nmt_original_valid_set.train_test_split(train_size=0.8, seed=42)
nmt_train_set, nmt_valid_set = split["train"], split["test"]
print(nmt_train_set[0])


max_length = 256
vocab_size = 10_000
nmt_tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
nmt_tokenizer = tokenizers.Tokenizer(nmt_tokenizer_model)
nmt_tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
nmt_tokenizer.enable_truncation(max_length)
nmt_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
nmt_tokenizer_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
)
nmt_tokenizer.train_from_iterator(
    train_eng_spa(),
    nmt_tokenizer_trainer,
)
