import logging
import os
import time
from collections import deque
from pathlib import Path
from pprint import pformat
from typing import Optional

import numpy as np
import torch
import utils
from envs import make_vec_envs
from evaluation import evaluate
from model import Policy
from ppo import PPO
from run_logger import Logger, get_logger
from spec import spec
from storage import RolloutStorage
from tap import Tap

EPISODE_RETURN = "episode return"
ACTION_LOSS = "action loss"
VALUE_LOSS = "value loss"
FPS = "fps"
TIME = "time"
STEP = "step"
ENTROPY = "entropy"


class Run(Tap):
    name: str

    def configure(self) -> None:
        self.add_argument("name", type=str)  # positional
        self.set_defaults(func=Trainer.train_with_logger)


class Sweep(Tap):
    sweep_id: int = None

    def configure(self) -> None:
        self.add_argument("sweep_id", type=int)
        self.set_defaults(func=Trainer.train_with_logger)


class Args(Tap):
    lr: float = 7e-4  # learning rate
    eps: float = 1e-5  # RMSProp epsilon
    alpha: float = 0.99  # RMSProp alpha
    gamma: float = 0.99  # discount factor
    gae: bool = False  # use Generalized Advantage Estimation
    gae_lambda: float = 0.94  # GAE lambda parameter
    entropy_coef: float = 0.01  # auxiliary entropy objective coefficient
    value_coef: float = 0.5  # value loss coefficient
    max_grad_norm: float = 0.5  # clip gradient norms
    seed: int = 0  # clip gradient norms
    num_processes: int = 16  # number of parallel environments
    num_steps: int = 5  # number of forward steps in A2C
    ppo_epoch: int = 5  # number of PPO updates
    num_mini_batch: int = 5  # number of mini-batches per update
    clip_param: float = 0.2  # PPO clip parameter
    log_interval: int = 10  # how many updates to log between
    save_interval: int = 100  # how many updates to save between
    eval_interval: Optional[int] = None  # how many updates to evaluate between
    num_env_steps: int = 1e9  # total number of environment steps
    env_name: str = "PongNoFrameskip-v4"  # env ID for gym
    save_path: Optional[str] = None  # path to save parameters if saving locally
    cuda: bool = True  # enable CUDA
    use_proper_time_limits: bool = False  # compute returns with time limits
    recurrent_policy: bool = False  # use recurrence in the policy
    linear_lr_decay: bool = False  # anneal the learning rate
    log_level: str = "INFO"

    def configure(self) -> None:
        self.add_subparsers(dest="subcommand")
        self.set_defaults(func=Trainer.train)
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class Trainer:
    @staticmethod
    def train(args: Args, logger: Optional[Logger] = None):
        logging.getLogger().setLevel(args.log_level)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.cuda and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)
        device = torch.device("cuda:0" if args.cuda else "cpu")

        envs = make_vec_envs(
            env_name=args.env_name,
            seed=args.seed,
            num_processes=args.num_processes,
            gamma=args.gamma,
            device=device,
            allow_early_resets=False,
        )

        actor_critic = Policy(
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            base_kwargs=dict(recurrent=args.recurrent_policy),
        )
        actor_critic.to(device)

        agent = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

        rollouts = RolloutStorage(
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
        for j in range(num_updates):

            if args.linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = actor_critic.act(
                        inputs=rollouts.obs[step],
                        rnn_hxs=rollouts.recurrent_hidden_states[step],
                        masks=rollouts.masks[step],
                    )

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if "episode" in info.keys():
                        episode_rewards.append(info["episode"]["r"])

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if "bad_transition" in info.keys() else [1.0]
                        for info in infos
                    ]
                )
                rollouts.insert(
                    obs=obs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    actions=action,
                    action_log_probs=action_log_prob,
                    value_preds=value,
                    rewards=reward,
                    masks=masks,
                    bad_masks=bad_masks,
                )

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    inputs=rollouts.obs[-1],
                    rnn_hxs=rollouts.recurrent_hidden_states[-1],
                    masks=rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value=next_value,
                use_gae=args.gae,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                use_proper_time_limits=args.use_proper_time_limits,
            )

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if j % args.save_interval == 0 or j == num_updates - 1:
                if args.save_path:
                    try:
                        os.makedirs(args.save_path)
                    except OSError:
                        pass

                    torch.save(
                        [
                            actor_critic,
                            getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                        ],
                        str(Path(args.save_path, args.env_name + ".pt")),
                    )

            if j % args.log_interval == 0:  # and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                now = time.time()
                fps = int(total_num_steps / (now - start))
                log = {
                    EPISODE_RETURN: np.mean(episode_rewards),
                    ACTION_LOSS: action_loss,
                    VALUE_LOSS: value_loss,
                    FPS: fps,
                    TIME: now * 1000000,
                    STEP: total_num_steps,
                    ENTROPY: dist_entropy,
                }
                logging.info(pformat(log))
                if logger is not None:
                    log.update({"run ID": logger.run_id})
                logging.info(pformat(log))
                if logger is not None:
                    logger.log(log)

            if (
                args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0
            ):
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(
                    actor_critic=actor_critic,
                    obs_rms=obs_rms,
                    env_name=args.env_name,
                    seed=args.seed,
                    num_processes=args.num_processes,
                    device=device,
                )

    @classmethod
    def train_with_logger(cls, args: Args):
        metadata = dict(reproducibility_info=args.get_reproducibility_info())
        if host_machine := os.getenv("HOST_MACHINE"):
            metadata.update(host_machine=host_machine)
        if name := getattr(args, "name", None):
            metadata.update(name=name)

        logger: Logger
        with get_logger("hasura") as logger:
            parameters = logger.create_run(
                metadata=metadata,
                sweep_id=getattr(args, "sweep_id", None),
                charts=[
                    spec(x="step", y=y)
                    for y in (
                        EPISODE_RETURN,
                        ACTION_LOSS,
                        VALUE_LOSS,
                        FPS,
                        TIME,
                        STEP,
                        ENTROPY,
                    )
                ],
            )
            if parameters is not None:
                for k, v in parameters.items():
                    if k not in ["subcommand", "log_level", "logger"]:
                        setattr(args, k, v)
            logger.update_metadata(
                dict(parameters=args.as_dict(), run_id=logger.run_id)
            )
            logging.info(pformat(args.as_dict()))
            return cls.train(args=args, logger=logger)


if __name__ == "__main__":
    ARGS = Args(explicit_bool=True).parse_args()
    ARGS.func(ARGS)