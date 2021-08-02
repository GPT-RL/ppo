#!/usr/bin/env python3
import argparse
import copy
import logging
from pathlib import Path
from typing import Optional
from pprint import pprint, pformat

import yaml
from redis import Redis

from sweep_logger import HasuraLogger
from sweep_logger.params import ParamChoice, SweepMethod
from sweep_logger.reproducibility_info import get_reproducibility_info


def run(
    config: Path,
    log_level: str,
    method: str,
    name: Optional[str],
    project: Optional[str],
    graphql_endpoint: str,
    remaining_runs: Optional[int],
) -> int:
    with config.open() as f:
        config = yaml.load(f, yaml.FullLoader)

    assert isinstance(config, (dict, list)), pformat(config)
    logging.getLogger().setLevel(log_level)
    metadata = dict(
        config=config,
        **get_reproducibility_info(),
    )
    if name is not None:
        metadata.update(name=name)
    if project is not None:
        metadata.update(project=project)
    with HasuraLogger(graphql_endpoint=graphql_endpoint) as logger:
        if isinstance(config, list):
            config = {"": config}
        choices = [ParamChoice(k, v) for k, v in config.items()]
        logging.info(f"Remaining runs: {remaining_runs}")

        sweep_id = logger.create_sweep(
            method=SweepMethod[method],
            metadata=metadata,
            choices=choices,
            remaining_runs=remaining_runs,
        )
    logging.info(f"Sweep ID: {sweep_id}")
    return sweep_id


def run_redis(host, port, **kwargs):
    sweep_id = run(**kwargs)
    redis = Redis(host=host, port=port)
    redis.set("sweep_id", sweep_id)


log_levels = [
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARN",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to sweep config yaml file.",
        type=Path,
        default=Path("config.yml"),
    )
    parser.add_argument("--log-level", "-ll", choices=log_levels, default="INFO")
    parser.add_argument(
        "--method",
        "-m",
        choices=["grid", "random"],
        default="random",
        help="Whether to perform grid-search on parameters in config.yml or randomly sample.",
    )
    parser.add_argument(
        "--name", "-n", help="Name of sweep (logged in metadata).", default=None
    )
    parser.add_argument(
        "--project", "-p", help="Name of project (logged in metadata).", default=None
    )
    parser.add_argument("--graphql-endpoint", "-g", help="Endpoint to use for hasura.")
    parser.add_argument(
        "--remaining-runs",
        "-r",
        help="Set a limit on the number of runs to launch for this sweep. If None or '', an unlimited number of runs "
        "will be launched.",
        type=lambda string: int(string)
        if string
        else None,  # handle case where string == ''
    )
    parser.set_defaults(func=run)
    subparsers = parser.add_subparsers()
    redis_parser = subparsers.add_parser("redis")
    redis_parser.set_defaults(func=run_redis)
    redis_parser.add_argument("--host", default="redis")
    redis_parser.add_argument("--port", default=6379)
    args = parser.parse_args()
    _args = vars(copy.deepcopy(args))
    del _args["func"]
    args.func(**_args)


if __name__ == "__main__":
    main()
