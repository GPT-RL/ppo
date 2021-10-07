from __future__ import print_function

import re
import zipfile
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas._typing import FilePathOrBuffer
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

GPTSize = Literal["small", "medium", "large", "xl"]


def build_gpt(gpt_size: GPTSize, randomize_parameters: bool):
    gpt_size = get_gpt_size(gpt_size)
    return (
        GPT2Model(
            GPT2Config.from_pretrained(
                gpt_size,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        if randomize_parameters
        else GPT2Model.from_pretrained(
            gpt_size,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    )


class GPTEmbed(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        self.gpt = build_gpt(embedding_size, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x, **_):
        return self.gpt.forward(x).last_hidden_state[:, :, -1]


class Net(nn.Module):
    def __init__(self, embedding_size: GPTSize, hidden_size: int, **kwargs):
        super(Net, self).__init__()
        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size)
        ).n_embd
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.gpt = nn.Sequential(
            GPTEmbed(embedding_size=embedding_size, **kwargs),
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        embedded = self.gpt(x)
        n_classes = x.size(1) - 1
        lemma, choices = torch.split(embedded, [1, n_classes], dim=1)
        lemma = self.K(lemma)
        choices = self.Q(choices)
        weights = (lemma * choices).sum(-1)
        output = F.log_softmax(weights, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def get_gpt_size(gpt_size: GPTSize):
    gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
    gpt_size = f"gpt2{gpt_size}"
    return gpt_size


class Antonyms(Dataset):
    def __init__(
        self,
        csv_file: FilePathOrBuffer,
        gpt_size: GPTSize,
        n_classes: int,
        seed: int,
    ):
        ANTONYMS = "antonyms"
        LEMMA = "lemma"
        TARGET = "target"

        data: pd.DataFrame = pd.read_csv(csv_file)

        # split antonyms into separate rows
        data[ANTONYMS] = data.apply(
            func=lambda x: re.split("[;|]", x.antonyms),
            axis=1,
        )
        data = data.explode(ANTONYMS)

        def shuffle(df, **kwargs):
            return df.sample(frac=1, **kwargs).reset_index(drop=True)

        data = shuffle(data, random_state=seed)  # shuffle data

        data = data.rename(columns=dict(antonyms=0))  # correct answer goes to column 0
        assert n_classes >= 2
        for i in range(1, n_classes):
            # classes 1...n_classes contain randomly chosen wrong choices
            data[i] = shuffle(data[LEMMA], random_state=seed + i)

        # permute choices (otherwise correct answer is always 0)
        input_columns = list(range(n_classes))  # inputs will be columns 0...n_classes
        N = len(data)
        ii = np.tile(np.expand_dims(np.arange(N), 1), (1, n_classes))
        jj = np.tile(np.arange(n_classes), (N, 1))
        jj = np.random.default_rng(seed).permuted(
            jj, axis=1
        )  # shuffle indices along y-axis
        permuted_inputs = data[input_columns].to_numpy()[
            ii, jj
        ]  # shuffle data using indices
        data[input_columns] = permuted_inputs
        _, data[TARGET] = (
            jj == 0
        ).nonzero()  # identify new targets (where 0-index was shuffled to)

        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(gpt_size))

        def encode(s: str):
            return tokenizer.encode(s, return_tensors="pt").squeeze(0)

        def generate_tensors():
            for col in [LEMMA, *input_columns]:
                tensors = data[col].apply(encode)
                yield pad_sequence(list(tensors), padding_value=tokenizer.eos_token_id)

        self.data = pad_sequence(
            list(generate_tensors()), padding_value=tokenizer.eos_token_id
        )
        self.data = self.data.transpose(0, 2)
        self.targets = torch.tensor(data[TARGET].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class Args(Tap):
    batch_size: int = 32
    data_path: str = "antonyms.zip"
    dry_run: bool = False
    embedding_size: GPTSize = "small"
    epochs: int = 14
    gamma: float = 0.7
    hidden_size: int = 512
    log_interval: int = 10
    lr: float = 1.0
    n_classes: int = 3
    n_train: int = 10000
    n_test: int = 1000
    no_cuda: bool = False
    randomize_parameters: bool = False
    save_model: bool = False
    seed: int = 1
    test_batch_size: int = 1000
    train_ln: bool = False
    train_wpe: bool = False


def main(args: Args):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    rng = torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    with zipfile.ZipFile(args.data_path) as zip_file:
        with zip_file.open("antonyms.csv") as file:
            dataset = Antonyms(
                file,
                gpt_size=args.embedding_size,
                n_classes=args.n_classes,
                seed=0,
            )
    dataset = torch.utils.data.Subset(dataset, range(args.n_train + args.n_test))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [args.n_train, args.n_test], generator=rng
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        randomize_parameters=args.randomize_parameters,
        train_wpe=args.train_wpe,
        train_ln=args.train_ln,
    ).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "antonyms_model.pt")


if __name__ == "__main__":
    main(Args().parse_args())
