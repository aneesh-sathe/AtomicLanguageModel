import torch


def text_to_token(text, tokenizer):
    tokens = torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0)
    return tokens


def token_to_text(tokens, tokenizer):
    text = tokenizer.decode(tokens.squeeze(0).tolist())
    return text
