from __future__ import print_function

import logging
import os
import time
from pathlib import Path
from pprint import pformat
from typing import List, Literal, Optional

import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sweep_logger import HasuraLogger, Logger
from tap import Tap
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from transformers import GPT2Config, GPT2Model

from spec import spec


class LoggerArgs(Tap):
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    host_machine: str = os.getenv("HOST_MACHINE")


class Run(LoggerArgs):
    name: str

    def configure(self) -> None:
        self.add_argument("name", type=str)  # positional


class Sweep(LoggerArgs):
    sweep_id: int = None


class Args(Tap):
    batch_size: int = 16
    config: Optional[str] = None
    cuda: bool = True
    dry_run: bool = False  # quickly check a single pass
    dataset: Literal["mnist", "xor"] = "mnist"
    epochs: int = 14
    gamma: Optional[float] = None  # Learning rate step gamma (default: 0.7)
    gpt_size: Literal["small", "medium", "large", "xl"] = None
    log_interval: int = 100000
    log_level: str = "INFO"
    lr: float = 1e-3
    architecture: Literal["1conv", "2conv", "1linear"] = "1conv"
    randomize_parameters: bool = False
    save_model: bool = False
    seed: int = 1
    test_batch_size: int = 1000
    train_ln: bool = True
    train_wpe: bool = True

    def configure(self) -> None:
        self.add_subparsers(dest="subcommand")
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class GPTNet(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        num_outputs: int,
        gpt_size: str,
        randomize_parameters: bool,
        train_ln: bool,
        train_wpe: bool,
        architecture: str,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.input_shape = input_shape
        self.architecture = architecture
        gpt_size = "" if gpt_size == "small" else f"-{gpt_size}"
        gpt_size = f"gpt2{gpt_size}"
        self.gpt = (
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
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)
        self.n_embd = self.gpt.config.n_embd
        self.out = nn.Linear(self.gpt.config.n_embd, self.num_outputs)

    def forward(self, x):
        if self.architecture == "1linear":
            x = x.reshape(x.size(0), -1)
        x = self.net(x)
        x = x.reshape(x.size(0), self.n_embd, -1).transpose(2, 1)
        x = self.gpt(inputs_embeds=x).last_hidden_state[:, -1]
        return x


class GPTNetMNIST(GPTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.architecture == "1conv":
            self.net = nn.Conv2d(1, self.n_embd, 4, 4)
        elif self.architecture == "2conv":
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, self.n_embd, 4, 4),
            )
        elif self.architecture == "1linear":
            _, *dims = self.input_shape
            self.net = nn.Linear(int(np.prod(dims)), self.n_embd)
        else:
            raise InvalidArchitectureError()

    def forward(self, x):
        x = super().forward(x)
        x = self.out(x)
        return x


class GPTNetXOR(GPTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.architecture == "1conv":
            self.net = nn.Conv1d(1, self.n_embd, 1, 1)
        elif self.architecture == "2conv":
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 1, 1),
                nn.ReLU(),
                nn.Conv1d(32, self.n_embd, 1, 1),
            )
        elif self.architecture == "1linear":
            _, *dims = self.input_shape
            self.net = nn.Linear(int(np.prod(dims)), self.n_embd)
        else:
            raise InvalidArchitectureError()

    def forward(self, x):
        x = super().forward(x.unsqueeze(1))
        x = self.out(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        num_outputs: int,
    ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
        # output = F.log_softmax(x, dim=1)
        # return output


class XOR(VisionDataset):
    def __init__(
        self,
        root: str,
        seed: int,
    ):
        super().__init__(root)
        random = np.random.RandomState(seed=seed)
        self.inputs1 = random.choice(2, size=(len(self), 5))
        self.inputs2 = random.choice(2, size=(len(self), 5))
        self.targets = np.bitwise_xor(self.inputs1, self.inputs2).astype(np.float32)
        self.inputs = np.concatenate([self.inputs1, self.inputs2], axis=-1).astype(
            np.float32
        )

    def __len__(self) -> int:
        return 70000

    def __getitem__(self, index) -> T_co:
        return self.inputs[index], self.targets[index]


EPOCH = "epoch"
TRAIN_LOSS = "train loss"
TEST_LOSS = "test loss"
TEST_ACCURACY = "test accuracy"
FPS = "fps"
HOURS = "hours"


class InvalidArchitectureError(RuntimeError):
    pass


class InvalidDatasetError(RuntimeError):
    pass


def main(args: Args):
    logging.getLogger().setLevel(args.log_level)
    excluded = {
        "subcommand",
        "sweep_id",
        "config",
        "name",
    }
    if args.config is not None:
        with Path(args.config).open() as f:
            config = yaml.load(f, yaml.FullLoader)
            args = args.from_dict(
                {k: v for k, v in config.items() if k not in excluded}
            )
    if args.subcommand is None:
        return run(args)
    metadata = dict(reproducibility_info=args.get_reproducibility_info())
    if args.host_machine:
        metadata.update(host_machine=args.host_machine)
    if name := getattr(args, "name", None):
        metadata.update(name=name)

    logger: Logger
    with HasuraLogger(args.graphql_endpoint) as logger:
        charts = [
            *[
                spec(x=x, y=y)
                for x in [EPOCH, HOURS]
                for y in (
                    TRAIN_LOSS,
                    TEST_LOSS,
                    TEST_ACCURACY,
                    FPS,
                )
            ],
        ]
        sweep_id = getattr(args, "sweep_id", None)
        parameters = logger.create_run(
            metadata=metadata,
            sweep_id=sweep_id,
            charts=charts,
        )
        if parameters is not None:
            for k, v in parameters.items():
                if k not in excluded:
                    assert hasattr(args, k), k
                    setattr(args, k, v)
        logger.update_metadata(dict(parameters=args.as_dict(), run_id=logger.run_id))
        logging.info(pformat(args.as_dict()))
        return run(args=args, logger=logger)


def run(args, logger: Logger = None):
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if args.dataset == "mnist":
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
        num_outputs = 10
        guesses_per_item = 1

        def get_loss(out, tgt, reduction="mean"):
            return F.nll_loss(F.log_softmax(out, dim=1), tgt, reduction=reduction)

        def prediction(out):
            return F.log_softmax(out, dim=1).argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability

    elif args.dataset == "xor":
        dataset = XOR("../data", args.seed)
        dataset1, dataset2 = torch.utils.data.random_split(dataset, [60000, 10000])
        num_outputs = 5
        guesses_per_item = 5

        def get_loss(out, tgt, reduction="mean"):
            return F.mse_loss(torch.sigmoid(out), tgt, reduction=reduction)

        def prediction(out):
            return torch.sigmoid(out).round()

    else:
        raise InvalidDatasetError()

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    sample_input, _ = dataset1[0]

    if args.gpt_size is None:
        model = Net(input_shape=sample_input.shape, num_outputs=num_outputs)
    else:
        kwargs = dict(
            input_shape=sample_input.shape,
            num_outputs=num_outputs,
            gpt_size=args.gpt_size,
            randomize_parameters=args.randomize_parameters,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
            architecture=args.architecture,
        )
        if args.dataset == "mnist":
            model = GPTNetMNIST(**kwargs)
        elif args.dataset == "xor":
            model = GPTNetXOR(**kwargs)
        else:
            raise InvalidDatasetError()
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = (
        None if args.gamma is None else StepLR(optimizer, step_size=1, gamma=args.gamma)
    )
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += get_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = prediction(output)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        log = {
            EPOCH: epoch,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: 100.0
            * correct
            / (len(test_loader.dataset) * guesses_per_item),
            HOURS: (time.time() - start) / 3600,
        }
        if logger is not None:
            log.update({"run ID": logger.run_id})
            logger.log(log)
        logging.info(pformat(log))
        if logger is not None:
            logger.log(log)

        model.train()
        tick = None
        frames = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = get_loss(output, target)
            loss.backward()
            optimizer.step()
            frames += len(data)
            if batch_idx % args.log_interval == 0:
                now = time.time()
                log = {
                    EPOCH: epoch,
                    TRAIN_LOSS: loss.item(),
                    HOURS: (now - start) / 3600,
                }
                if tick is not None:
                    fps = frames / (now - tick)
                    log.update({FPS: fps})
                tick = now
                frames = 0
                logging.info(pformat(log))
                if logger is not None:
                    log.update({"run ID": logger.run_id})
                    logger.log(log)
                if args.dry_run:
                    break
        if scheduler is not None:
            scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main(Args(explicit_bool=True).parse_args())
