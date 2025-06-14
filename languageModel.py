import warnings

import torch
from torch import nn
from transformers import GPT2TokenizerFast

from utils.makePlots import plot_metrics
from utils.modelTrainer import get_train_test_split, gpt_trainer
from utils.textProcessor import create_dataloader, get_corpus_stats

warnings.filterwarnings("ignore")

# Initialize tokenizer first to get actual vocab size
""" tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True
) """

tokenizer = GPT2TokenizerFast.from_pretrained("marathi_tokenizer")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get actual vocab size from tokenizer
ACTUAL_VOCAB_SIZE = len(tokenizer)
print(f"Actual tokenizer vocab size: {ACTUAL_VOCAB_SIZE}")

# Updated config with correct vocab size
GPT_CONFIG_124M = {
    "vocab_size": ACTUAL_VOCAB_SIZE,  # Use actual vocab size
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

PRETRAIN_CORPUS = "dataset/marathi_pretrain.txt"
INSTRUCTION_CORPUS = "dataset/instructions.json"


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        # Splitting the matrix for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        attn_scores = queries @ keys.transpose(2, 3)

        # Create causal mask dynamically
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1
        ).bool()

        # Apply mask
        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute context vector
        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForwardNetwork(cfg)
        self.pre_norm = LayerNorm(cfg["emb_dim"])
        self.post_norm = LayerNorm(cfg["emb_dim"])
        self.drop_skip_connection = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm architecture
        skip_connection = x
        x = self.pre_norm(x)
        x = self.mha(x)
        x = self.drop_skip_connection(x)
        x = x + skip_connection

        skip_connection = x
        x = self.post_norm(x)
        x = self.ff(x)
        x = self.drop_skip_connection(x)
        x = x + skip_connection

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Token embeddings
        tok_embs = self.token_embedding(in_idx)

        # Position embeddings
        pos_embs = self.position_embedding(torch.arange(seq_len, device=in_idx.device))

        # Combine embeddings
        x = tok_embs + pos_embs
        x = self.drop_emb(x)

        # Pass through transformer blocks
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Output projection
        logits = self.out_head(x)
        return logits


def get_model_stats(model):
    """Calculate and print model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Calculate model size
    total_size_bytes = total_params * 4  # 4 bytes per float32
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Model size: {total_size_mb:.2f} MB")

    return total_params, trainable_params, total_size_mb


def main():
    """Main training function"""
    print("Loading corpus...")
    try:
        with open(PRETRAIN_CORPUS, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: Could not find {PRETRAIN_CORPUS}")
        print("Please make sure the corpus file exists.")
        return
    except UnicodeDecodeError:
        print("Error: Could not decode the corpus file. Trying different encoding...")
        try:
            with open(PRETRAIN_CORPUS, "r", encoding="latin-1") as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

    # Limit text size for faster training/testing
    text = text[:300000]
    print(f"Using {len(text)} characters from corpus")

    # Get corpus statistics
    get_corpus_stats(text)

    # Split data
    train_data, val_data = get_train_test_split(text, 0.75)
    print(f"Train data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")

    # Create data loaders
    print("Creating data loaders...")
    train_data_loader = create_dataloader(
        txt=train_data,
        batch_size=4,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        tokenizer=tokenizer,  # Make sure to pass tokenizer
    )

    val_data_loader = create_dataloader(
        txt=val_data,
        batch_size=4,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,  # Don't shuffle validation data
        drop_last=False,
        num_workers=0,
        tokenizer=tokenizer,  # Make sure to pass tokenizer
    )

    # Print data loader info
    print("\n----- Training set -----")
    for i, (x, y) in enumerate(train_data_loader):
        if i < 3:  # Print first 3 batches
            print(f"Batch {i}: Input shape {x.shape}, Target shape {y.shape}")
        if i >= 2:
            break

    print("\n----- Validation set -----")
    for i, (x, y) in enumerate(val_data_loader):
        if i < 3:  # Print first 3 batches
            print(f"Batch {i}: Input shape {x.shape}, Target shape {y.shape}")
        if i >= 2:
            break

    # Initialize model
    print("\nInitializing model...")
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    gpt = GPTModel(GPT_CONFIG_124M).to(device)

    # Print model statistics
    get_model_stats(gpt)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        gpt.parameters(),
        lr=0.0004,
        weight_decay=0.1,
        betas=(0.9, 0.95),  # Better defaults for transformer training
    )

    # Training parameters
    num_epochs = 5
    start_context = "हॅप्पी बर्थडे भाऊ!"

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Start context: '{start_context}'")

    # Train the model
    try:
        train_loss, val_loss, tokens_seen = gpt_trainer(
            model=gpt,
            train_loader=train_data_loader,
            val_loader=val_data_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            eval_freq=10,
            eval_iter=10,
            start_context=start_context,
            tokenizer=tokenizer,
            device=device,
            cfg=GPT_CONFIG_124M,
            new_tokens=50,
            temperature=1.0,
            k=3,
        )

        # Plot training metrics
        print("\nGenerating training plots...")
        plot_metrics(
            tokens_seen=tokens_seen,
            train_loss=train_loss,
            val_loss=val_loss,
            num_epochs=num_epochs,
        )

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
