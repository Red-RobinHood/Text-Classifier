import pandas as pd
import math
import io
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T

model_name = "TextClassifier"
learning_rate = 1e-4
nepochs = 20
batch_size = 128
max_len = 128

class CreateDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna("", inplace=True)
        self.df["Article"] = self.df["Title"] + " : " + self.df["Description"]
        self.df.drop(["Title", "Description"], axis=1, inplace=True)
        self.df["Article"] = self.df["Article"].str.replace(
            r'\\n|\\|\\r|\\r\\n|\n|"', " ", regex=True
        )
        self.df["Article"] = self.df["Article"].replace(
            {" #39;": "'", " #38;": "&", " #36;": "$", " #151;": "-"}, regex=True
        )

    def __getitem__(self, index):
        text = self.df.loc[index]["Article"].lower()
        class_index = int(self.df.loc[index]["Class Index"]) - 1
        return class_index, text

    def __len__(self):
        return len(self.df)

print(os.getcwd())

current_dir = Path(__file__).parent.resolve()
dataset_path = current_dir.parent / "Datasets" / model_name

dataset_train = CreateDataset(str(dataset_path / "train.csv"))
dataset_test = CreateDataset(str(dataset_path / "test.csv"))
dataset_val = CreateDataset(str(dataset_path / "val.csv"))

data_loader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
)
data_loader_test = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True
)
data_loader_val = DataLoader(
    dataset_val, batch_size=256, shuffle=True
)


def yield_tokens(file_path):
    with io.open(file_path, encoding="utf-8") as f:
        for line in f:
            yield [line.split("\t")[0]]

tokenizer_path = current_dir.parent / "Tokenizer" / model_name

vocab = build_vocab_from_iterator(
    yield_tokens(str(tokenizer_path / "spm_ag_news.vocab")),
    specials=["<pad>", "<sos>", "<eos>", "<unk>"],
    special_first=True,
)
vocab.set_default_index(vocab["<unk>"])

text_tranform = T.Sequential(
    T.SentencePieceTokenizer(str(tokenizer_path / "spm_ag_news.model")),
    T.VocabTransform(vocab=vocab),
    T.AddToken(1, begin=True),
    T.Truncate(max_seq_len=max_len),
    T.AddToken(2, begin=False),
    T.ToTensor(padding_value=0),
)


def weights_path(model_name: str, epoch: int) -> str:
    current_dir = Path(__file__).parent.resolve()
    weights_folder = current_dir.parent / "Weights" / model_name
    weights_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{model_name}_epoch_{epoch}.pt"
    return str(weights_folder / model_filename)

def latest_weights(model_name: str) -> str:
    current_dir = Path(__file__).parent.resolve()
    weights_folder = current_dir.parent / "Weights" / model_name
    model_filename_pattern = f"{model_name}_epoch_*.pt"
    weights_files = list(weights_folder.glob(model_filename_pattern))
    if len(weights_files) == 0:
        return None
    weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(weights_files[-1])


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
        seq_len = x.shape[1]
        return x + (self.pe[:, :seq_len, :]).requires_grad_(False)


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
    src_seq_len: int,
    dim: int = 128,
    N: int = 3,
    h: int = 2,
    d_ff: int = 256,
) -> Transformer:

    src_embed = EmbeddingLayer(src_vocab_size, dim)
    src_pos = PositionalLayer(src_seq_len, dim)
    encoder_blocks = nn.ModuleList(
        [
            EncoderBlock(MultiHeadAttention(dim, h), FeedForwardBlock(dim, d_ff))
            for _ in range(N)
        ]
    )
    encoder = Encoder(encoder_blocks)
    generator = ProjectionLayer(dim, 4)
    transformer = Transformer(encoder, src_embed, src_pos, generator)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


device = torch.device(0 if torch.cuda.is_available() else "cpu")
tf_classifier = BuildTransformer(len(vocab), max_len + 1).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(tf_classifier.parameters(), lr=learning_rate)

initial_epoch = 0
train_acc = 0
test_acc = 0

model_filename = latest_weights(model_name)
if model_filename:
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    tf_classifier.load_state_dict(state["model_state_dict"])
    initial_epoch = state["epoch"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
else:
    print("No model to preload, starting from scratch")

for epoch in range(initial_epoch, 20):
    train_acc_count = 0
    test_acc_count = 0
    batch_iterator = tqdm(data_loader_train, desc=f"Processing Epoch {epoch:02d}")
    batch_iterator.set_postfix({})
    tf_classifier.train()
    steps = 0

    for label, text in tqdm(data_loader_train, desc="Training", leave=False):
        bs = label.shape[0]
        text_tokens = text_tranform(list(text)).to(device)
        label = label.to(device)
        pred = tf_classifier.encode(text_tokens)
        loss = loss_fn(pred[:, 0, :], label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc_count += (pred[:, 0, :].argmax(1) == label).sum()
        steps += bs

    train_acc = (train_acc_count / steps).item()
    tf_classifier.eval()
    steps = 0

    with torch.no_grad():
        for label, text in tqdm(data_loader_test, desc="Testing", leave=False):
            bs = label.shape[0]
            text_tokens = text_tranform(list(text)).to(device)
            label = label.to(device)
            pred = tf_classifier.encode(text_tokens)
            loss = loss_fn(pred[:, 0, :], label)

    model_filename = weights_path(model_name,f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": tf_classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_filename,
    )

ag_news_classes = ["World", "Sports", "Business", "Science/Technology"]

custominput = False

if custominput:
    while True:
        with torch.no_grad():
            text = input("Enter a news article: ")
            if text == "exit":
                break
            text = (text,)
            text_tokens = text_tranform(list(text)).to(device)
            pred = tf_classifier.encode(text_tokens)

        count = 0

        pred_class = ag_news_classes[pred[0, -1].argmax().item()]
        print("\nPredicted label:")
        print(pred_class)
else:
    with torch.no_grad():
        label, text = next(iter(data_loader_val))
        text_tokens = text_tranform(list(text)).to(device)
        pred = tf_classifier.encode(text_tokens)

    count = 0

    for test_index in range(0, label.shape[0]):
        assert test_index < label.shape[0]
        pred_class = ag_news_classes[pred[test_index, -1].argmax().item()]
        print("Article:")
        print(text[test_index])
        print("\nPredicted label:")
        print(pred_class)
        print("True label:")
        print(ag_news_classes[label[test_index].item()])
        if pred_class == ag_news_classes[label[test_index].item()]:
            count += 1

    print(f"Accuracy: {count / label.shape[0] * 100:.2f}%")