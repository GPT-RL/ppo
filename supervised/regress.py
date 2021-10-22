from __future__ import print_function

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional, cast, get_args

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gql import gql
from run_logger import HasuraLogger
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from spec import spec

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


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class GPTEmbed(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
        tokenized: torch.Tensor,
    ):
        super().__init__()
        gpt = build_gpt(embedding_size, randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

        gpt = nn.Sequential(
            Lambda(lambda x: x.long()),
            gpt,
            Lambda(lambda x: x.last_hidden_state[:, -1]),
        )
        if train_ln or train_wpe:
            self.net = gpt
        else:
            dummy_tokens = torch.arange(tokenized.max() + 1).unsqueeze(-1)
            embeddings = gpt(dummy_tokens)
            self.net = nn.Sequential(
                Lambda(lambda x: x.long()),
                nn.Embedding.from_pretrained(embeddings),
                Lambda(lambda x: x[:, -1]),
            )

    def forward(self, x, **_):
        return self.net(x)


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: GPTSize,
        hidden_size: int,
        max_int: int,
        n_layers: int,
        **kwargs,
    ):
        super(Net, self).__init__()
        self.max_int = max_int
        self.embedding_size = GPT2Config.from_pretrained(
            get_gpt_size(embedding_size)
        ).n_embd
        self.gpt = GPTEmbed(embedding_size=embedding_size, **kwargs)
        self.embed = nn.Linear(max_int, self.embedding_size)

        def inner_layers():
            in_size = 2 * self.embedding_size
            for _ in range(n_layers):
                yield nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU())
                in_size = hidden_size
            yield nn.Sequential(nn.Linear(in_size, 1), nn.ReLU())

        self.net = nn.Sequential(*inner_layers())

    def forward(self, x):
        x1, x2 = torch.split(x, [self.max_int, 1], dim=-1)
        embedded1 = self.embed(x1)
        embedded2 = self.gpt(x2)
        cat = torch.cat([embedded1, embedded2], dim=-1).squeeze(1)
        return self.net(cat).squeeze(-1)


def get_gpt_size(gpt_size: GPTSize):
    gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
    gpt_size = f"gpt2{gpt_size}"
    return gpt_size


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)


ANTONYMS = "antonyms"
SYNONYMS = "synonyms"
LEMMA = "lemma"
TARGET = "target"
CATEGORY = "category"
WORD1 = "word1"
WORD2 = "word2"


@dataclass
class _Dataset(Dataset):
    inputs: torch.tensor
    targets: torch.tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


RUN_OR_SWEEP = Literal["run", "sweep"]


class Run(Tap):
    name: str

    def configure(self) -> None:
        self.add_argument("name", type=str)  # positional


class Sweep(Tap):
    sweep_id: int = None


def configure_logger_args(args: Tap):
    args.add_subparser("run", Run)
    args.add_subparser("sweep", Sweep)


class Args(Tap):
    batch_size: int = 32
    config: Optional[str] = None  # If given, yaml config from which to load params
    data_path: str = "data.zip"
    dry_run: bool = False
    embedding_size: GPTSize = "small"
    epochs: int = 14
    gamma: float = 0.99
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 512
    host_machine: str = os.getenv("HOST_MACHINE")
    load_id: int = None  # path to load parameters from if at all
    log_interval: int = 100
    log_level: str = "INFO"
    lr: float = 1.0
    max_integer: int = 20
    n_layers: int = 1
    no_cuda: bool = False
    randomize_parameters: bool = False
    save_model: bool = False
    seed: int = 1
    test_batch_size: int = 1000
    test_integer: int = 2
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class ArgsType(Args):
    logger_args: Optional[RUN_OR_SWEEP]


def get_save_path(run_id: Optional[int]):
    return (
        Path("/tmp/logs/checkpoint.pkl")
        if run_id is None
        else Path("/tmp/logs", str(run_id), "checkpoint.pkl")
    )


def max_agreement(
    goals: torch.tensor,
    targets: torch.tensor,
    outputs: torch.tensor,
):
    goals = goals.argmax(-1)
    goals = goals.cpu().numpy()
    targets = targets.cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    df = pd.DataFrame(data=dict(goals=goals, targets=targets, outputs=outputs))

    def pair_inequalities(s: pd.Series):
        a = s.to_numpy()
        return np.expand_dims(a, axis=0) >= np.expand_dims(a, axis=1)

    def matching_inequalities(s1: pd.Series, s2: pd.Series):
        return np.mean(pair_inequalities(s1) == pair_inequalities(s2))

    return np.mean(
        [
            matching_inequalities(gdf["targets"], gdf["outputs"])
            for _, gdf in df.groupby("goals")
        ]
    )


def compute_targets(inputs, goals):
    return 0.99 ** abs(goals - inputs.argmax(-1))


def train(args: Args, logger: HasuraLogger):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    obs = np.tile(np.eye(args.max_integer), (args.max_integer, 1))
    goal = np.repeat(np.arange(args.max_integer), args.max_integer)

    def get_divisors():
        divisor = 1
        while divisor <= args.max_integer:
            divisor *= 10
            yield divisor

    tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))

    def tokenize():
        for n in tqdm(goal, desc="Tokenizing data"):
            encode = tokenizer.encode(str(n), return_tensors="pt")
            yield encode.squeeze(0)

    tokenized = list(tokenize())
    tokenized = pad_sequence(tokenized, padding_value=tokenizer.eos_token_id).T
    targets = compute_targets(inputs=obs, goals=goal)
    is_test = [
        goal == args.test_integer,
        *((goal % d) == args.test_integer for d in get_divisors()),
    ]
    is_test = np.stack(is_test)
    is_test = is_test.any(axis=0)
    data = np.stack(
        [targets, is_test],
        axis=1,
    )
    data = np.concatenate([obs, tokenized, data], axis=1)
    _inputs = data[:, : args.max_integer + 1]
    _inputs = torch.tensor(_inputs, dtype=torch.float32).to(device)
    _targets = torch.tensor(targets).to(device)
    _is_test = torch.tensor(is_test).to(device)
    _goal = torch.sort(torch.tensor(goal).to(device)).values

    def repeat_data(in_dataset, batch_size):
        tiles = int(np.ceil(batch_size / sum(in_dataset)))
        return np.tile(data[in_dataset], (tiles, 1))

    data = np.concatenate(
        [
            repeat_data(~is_test, args.batch_size),
            repeat_data(is_test, args.test_batch_size),
        ],
        axis=0,
    )

    rng = np.random.default_rng(seed=args.seed)
    rng.shuffle(data, axis=0)
    data = torch.tensor(data, dtype=torch.float32)

    inputs, targets, is_test = torch.split(data, [args.max_integer + 1, 1, 1], dim=-1)
    is_test = is_test.bool().squeeze(-1)

    train_dataset = _Dataset(inputs=inputs[~is_test], targets=targets[~is_test])
    test_dataset = _Dataset(inputs=inputs[is_test], targets=targets[is_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        randomize_parameters=args.randomize_parameters,
        train_wpe=args.train_wpe,
        train_ln=args.train_ln,
        max_int=args.max_integer,
        tokenized=tokenized,
        n_layers=args.n_layers,
    ).to(device)

    save_path = get_save_path(logger.run_id)
    if args.load_id is not None:
        load_path = get_save_path(args.load_id)
        logging.info(f"Loading checkpoint from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    if args.save_model:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    start = time.time()

    save_count = 0

    def get_metric(f):
        with torch.no_grad():
            _outputs = model(_inputs)
            return torch.mean(f(_outputs, _targets).float()).item()

    def get_accuracy(is_dataset: torch.Tensor):
        def f(_outputs: torch.Tensor, _targets: torch.Tensor):
            distances = torch.abs(_outputs - _targets.unsqueeze(-1))
            correct_target = _targets[distances.argmin(0)] == _targets
            return correct_target[is_dataset]

        return get_metric(f)

    def get_expected_return(is_dataset: torch.Tensor):
        def sequential_order(t: torch.Tensor):
            return F.pad(cast(torch.Tensor, t[:-1] < t[1:]), (0, 1), value=True)

        def f(_outputs: torch.Tensor, _targets: torch.Tensor):
            orderings = []
            unique_goals, goals_count = _goal[is_dataset].unique(return_counts=True)
            out = torch.split(_outputs[is_dataset], list(goals_count))
            tgt = torch.split(_targets[is_dataset], list(goals_count))
            for g, o, t in zip(unique_goals, out, tgt):
                correct_ordering = sequential_order(o) == sequential_order(t)
                correct_ordering = cast(torch.Tensor, correct_ordering)
                orderings.extend([correct_ordering[g:], correct_ordering[:g].flip(-1)])
            orderings = pad_sequence(orderings)
            orderings = torch.cumprod(orderings, dim=0)
            return orderings.sum() / is_dataset.sum()

        return get_metric(f)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):

        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.mse_loss(output.flatten(), target.flatten()).item()

        now = time.time()
        log = {
            EPOCH: epoch,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: get_accuracy(_is_test),
            TEST_EXPECTED_RETURN: get_expected_return(_is_test),
            RUN_ID: logger.run_id,
            HOURS: (now - start) / 3600,
        }
        pprint(log)
        if logger.run_id is not None:
            logger.log(log)

        frames = 0
        tick = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            frames += len(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                log = {
                    EPOCH: epoch,
                    LOSS: loss.item(),
                    RUN_ID: logger.run_id,
                    HOURS: (time.time() - start) / 3600,
                    ACCURACY: get_accuracy(~_is_test),
                    EXPECTED_RETURN: get_expected_return(~_is_test),
                    SAVE_COUNT: save_count,
                }
                pprint(log)
                if logger.run_id is not None:
                    logger.log(log)

                if args.dry_run:
                    break

        now = time.time()
        log = {
            RUN_ID: logger.run_id,
            EPOCH: epoch,
            HOURS: (now - start) / 3600,
            FPS: frames / (now - tick),
        }
        pprint(log)
        if logger.run_id is not None:
            logger.log(log)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), str(save_path))
            save_count += 1


EXCLUDED = {
    "config",
    "name",
    "sync_envs",
    "render",
    "render_test",
    "subcommand",
    "sweep_id",
    "load_id",
    "logger_args",
}

FPS = "FPS"
GRADIENT_NORM = "gradient norm"
TIME = "time"
HOURS = "hours"
EPOCH = "epoch"
SAVE_COUNT = "save count"
LOSS = "loss"
TEST_LOSS = "test loss"
ACCURACY = "accuracy"
TEST_ACCURACY = "test accuracy"
EXPECTED_RETURN = "expected return"
TEST_EXPECTED_RETURN = "test expected return"
RUN_ID = "run ID"


def update_args(args, parameters, check_hasattr=True):
    for k, v in parameters.items():
        if k not in EXCLUDED:
            if check_hasattr:
                assert hasattr(args, k), k
            setattr(args, k, v)


def main(args: ArgsType):
    logging.getLogger().setLevel(args.log_level)
    if args.config is not None:
        with Path(args.config).open() as f:
            config = yaml.load(f, yaml.FullLoader)
            args = args.from_dict(
                {k: v for k, v in config.items() if k not in EXCLUDED}
            )

    metadata = dict(reproducibility_info=args.get_reproducibility_info())
    if args.host_machine:
        metadata.update(host_machine=args.host_machine)
    if name := getattr(args, "name", None):
        metadata.update(name=name)

    logger: HasuraLogger
    with HasuraLogger(args.graphql_endpoint) as logger:
        valid = (*get_args(RUN_OR_SWEEP), None)
        assert args.logger_args in valid, f"{args.logger_args} is not in {valid}."

        if args.logger_args is not None:
            charts = [
                spec(x=x, y=y, scale_type="log" if LOSS in y else "linear")
                for y in (
                    LOSS,
                    ACCURACY,
                    TEST_ACCURACY,
                    EXPECTED_RETURN,
                    TEST_EXPECTED_RETURN,
                    TEST_LOSS,
                    SAVE_COUNT,
                    FPS,
                )
                for x in (HOURS, EPOCH)
            ]
            sweep_id = getattr(args, "sweep_id", None)
            parameters = logger.create_run(
                metadata=metadata,
                sweep_id=sweep_id,
                charts=charts,
            )

            if parameters is not None:
                update_args(args, parameters)
            logger.update_metadata(
                dict(parameters=args.as_dict(), run_id=logger.run_id)
            )

        if args.load_id is not None:
            parameters = logger.execute(
                gql(
                    """
query GetParameters($id: Int!) {
  run_by_pk(id: $id) {
    metadata(path: "parameters")
  }
}"""
                ),
                variable_values=dict(id=args.load_id),
            )["run_by_pk"]["metadata"]
            update_args(args, parameters, check_hasattr=False)
        return train(args=args, logger=logger)


if __name__ == "__main__":
    main(cast(ArgsType, Args().parse_args()))
