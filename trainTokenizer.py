import os

import tokenizers
from transformers import GPT2TokenizerFast


def train_tokenizer():
    tokenizer = tokenizers.ByteLevelBPETokenizer()

    tokenizer.train(
        files="dataset/marathi_pretrain.txt",
        vocab_size=32000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            "### सूचना:",
            "### उत्तर:",
            "### इनपुट:",  # custom tokens for instruction format
        ],
    )

    os.makedirs("./marathi_tokenizer", exist_ok=True)

    tokenizer.save_model("./marathi_tokenizer")


tok = GPT2TokenizerFast.from_pretrained(
    "marathi_tokenizer",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

tok.add_special_tokens(
    {"additional_special_tokens": ["### सूचना:", "### उत्तर:", "### इनपुट:"]}
)


print(tok.decode(tok.encode("खरे तर मी मूळचा मुंबईचाच, तोही गिरगावातला.")))
print(tok)
