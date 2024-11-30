import torch
import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim)


class PositionalLayer(nn.Module):
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)


class LayerNormalization(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps)


class FeedForwardBlock(nn.Module):

    def __init__(self, dim: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, d_ff)
        self.linear2 = nn.Linear(d_ff, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class ResidualConnection(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class MultiHeadAttention(nn.Module):

    def __init__(self, dim: int, h: int) -> None:
        super().__init__()
        assert dim % h == 0, "dim must be divisible by h"
        self.dim = dim
        self.h = h
        self.d_k = dim // h

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

    def attention(query, key, value):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)

        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection() for _ in range(2)]
        )

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        src_embed: EmbeddingLayer,
        src_pos: PositionalLayer,
        generator: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.generator = generator

    def encode(self, src):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src)

    def generate(self, x):
        return self.generator(x)


def BuildTransformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    dim: int = 512,
    N: int = 6,
    h: int = 2,
    d_ff: int = 2048,
) -> Transformer:

    src_embed = EmbeddingLayer(dim, src_vocab_size)

    src_pos = PositionalLayer(dim, src_seq_len)

    encoder_blocks = nn.ModuleList(
        [
            EncoderBlock(MultiHeadAttention(dim, h), FeedForwardBlock(dim, d_ff))
            for _ in range(N)
        ]
    )

    encoder = Encoder(dim, encoder_blocks)

    generator = ProjectionLayer(dim, tgt_vocab_size)

    transformer = Transformer(encoder, src_embed, src_pos, generator)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
