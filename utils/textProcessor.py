import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length, (
            "Number of tokenized inputs must at least be equal to max_length+1"
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def get_corpus_stats(text):
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True
    )

    print("pretrain corpus length: ", len(text))
    print("pretrain tokens: ", len(tokenizer.encode(text)))
