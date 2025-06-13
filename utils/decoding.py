import torch


def text_to_token(text, tokenizer):
    tokens = torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0)
    return tokens


def token_to_text(tokens, tokenizer):
    text = tokenizer.decode(tokens.squeeze(0).tolist())
    return text


def generate_topk_out_tokens(
    model, idx, context_length, new_tokens, k=None, temperature=0.0, eos_id=None
):
    for _ in range(new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if k is not None:
            top_logits, _ = torch.topk(logits, k)
            min_logit = top_logits[:, -1]
            logits = torch.where(
                logits < min_logit,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(torch.softmax(logits, dim=-1))

        if idx_next == eos_id:
            break  # early stopping

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
