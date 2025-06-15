import os

import torch

from languageModel import GPT_CONFIG_124M, GPTModel, tokenizer
from utils.modelTrainer import sample_batch


# get the best checkpoint from the checkpoint folder
def load_checkpoint():
    ckp_path = "checkpoints/" + str(sorted(os.listdir("checkpoints"))[-1])
    print("latest checkpoint :", ckp_path)
    model = GPTModel(GPT_CONFIG_124M)
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp["model_state_dict"])
    print(model.load_state_dict(ckp["model_state_dict"]))
    return model


device = "mps" if torch.mps.is_available() else "cpu"
model = load_checkpoint().to(device)
sample_batch(
    model=model,
    tokenizer=tokenizer,
    start_context="मुंबई ही दक्षिण आशियाची आर्थिक",
    device=device,
    new_tokens=50,
    temperature=0.5,
    k=3,
    cfg=GPT_CONFIG_124M,
)
