import torch
from torch import nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


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
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # splitting the matrix in order to
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # generate (B, NUM_HEADS) independent
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # matrices of size (NUM_TOKENS,
        # HEAD_DIM) for parallel computation

        attn_scores = queries @ keys.transpose(2, 3)
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device), diagonal=1
        ).bool()  # causal mask
        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector


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
                    torch.sqrt(torch.tensor(2) / torch.pi)
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )  # as per the paper


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
        (
            batch_size,
            seq_len,
        ) = in_idx.shape
        tok_embs = self.token_embedding(
            in_idx
        )  # picks the in_idx^th row from the tok_embedding table
        pos_embs = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )  # picks row from the pos_embedding table
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
