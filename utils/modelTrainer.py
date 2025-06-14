import torch
import torch.nn.functional as F

from utils.decoder import generate_topk_out_tokens, text_to_token, token_to_text


def get_train_test_split(text, split):
    assert split <= 1, "split cannot be greater than 1"
    split_idx = int(len(text) * split)
    train = text[:split_idx]
    test = text[split_idx:]

    return train, test


def cross_entropy(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss


def batch_loss(data_loader, model, device, num_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cross_entropy(input_batch, target_batch, model, device)
            total_loss += loss.item()

        else:
            break

    return total_loss / num_batches


def sample_batch(
    model, tokenizer, start_context, device, new_tokens, temperature, k, cfg
):
    model.eval()
    context_length = cfg["context_length"]
    tokens = text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_topk_out_tokens(
            model=model,
            idx=tokens,
            context_length=context_length,
            new_tokens=new_tokens,
            temperature=temperature,
            k=k,
        )
    text = token_to_text(token_ids, tokenizer)
    print(text.replace("/n", " "))
    model.train()


def eval_model(model, train_data_loader, val_data_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = batch_loss(train_data_loader, model, device, num_batches=eval_iter)
        val_loss = batch_loss(val_data_loader, model, device, num_batches=eval_iter)
        model.train()

    return train_loss, val_loss


def gpt_trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
    device,
    start_context,
    tokenizer,
    cfg,
    new_tokens,
    temperature,
    k,
):
    train_losses, val_losses, track_tokens = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cross_entropy(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                track_tokens.append(tokens_seen)

                print(f" epoch : {epoch}    step: {global_step}")
                print(f" train loss: {train_loss}   val loss: {val_loss}")

                if val_losses and val_loss < val_losses[-1]:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                        },
                        f"checkpoints/checkpoint_{epoch}.pt",
                    )
                    print("saving the model checkpoint...")

                train_losses.append(train_loss)
                val_losses.append(val_loss)

        sample_batch(
            model, tokenizer, start_context, device, new_tokens, temperature, k, cfg
        )

    return train_losses, val_losses, track_tokens
