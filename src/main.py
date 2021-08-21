import logging
import os
import time
from collections import deque
from pathlib import Path
from pprint import pformat
from typing import Optional

import gym
import numpy as np
import torch
import yaml
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sweep_logger import HasuraLogger, Logger
from tap import Tap

import utils
from agent import Agent
from babyai_env import Env
from envs import (
    TimeLimitMask,
    TransposeImage,
    VecPyTorch,
    VecPyTorchFrameStack,
)
from ppo import PPO
from rollouts import Rollouts
from spec import spec

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


class InvalidEnvId(RuntimeError):
    pass


EPISODE_RETURN = "episode return"
TEST_EPISODE_RETURN = "test episode return"
ACTION_LOSS = "action loss"
VALUE_LOSS = "value loss"
FPS = "fps"
ENTROPY = "entropy"
GRADIENT_NORM = "gradient norm"
TIME = "time"
HOURS = "hours"
STEP = "step"


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
    alpha: float = 0.99  # Adam alpha
    clip_param: float = 0.1  # PPO clip parameter
    cuda: bool = True  # enable CUDA
    entropy_coef: float = 0.01  # auxiliary entropy objective coefficient
    env: str = "BreakoutNoFrameskip-v4"  # env ID for gym
    test_interval: Optional[int] = None  # how many updates to evaluate between
    eps: float = 1e-5  # RMSProp epsilon
    gae: bool = True  # use Generalized Advantage Estimation
    gae_lambda: float = 0.95  # GAE lambda parameter
    gamma: float = 0.99  # discount factor
    hidden_size: int = 512
    log_interval: int = 100  # how many updates to log between
    linear_lr_decay: bool = False  # anneal the learning rate
    log_level: str = "INFO"
    lr: float = 2.5e-4  # learning rate
    max_grad_norm: float = 0.5  # clip gradient norms
    num_env_steps: int = 1e9  # total number of environment steps
    num_mini_batch: int = 4  # number of mini-batches per update
    num_processes: int = 8  # number of parallel environments
    num_steps: int = 128  # number of forward steps in A2C
    ppo_epoch: int = 3  # number of PPO updates
    recurrent_policy: bool = False  # use recurrence in the policy
    save_interval: Optional[int] = None  # how many updates to save between
    save_dir: str = "/tmp/logs"  # path to save parameters if saving locally
    seed: int = 0  # random seed
    use_proper_time_limits: bool = False  # compute returns with time limits
    value_coef: float = 1  # value loss coefficient
    config: Optional[str] = None  # If given, yaml config from which to load params

    def configure(self) -> None:
        self.add_subparsers(dest="subcommand")
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class Trainer:
    @classmethod
    def train(cls, args: Args, logger: Optional[Logger] = None):
        logging.getLogger().setLevel(args.log_level)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        if args.cuda and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if logger is not None:
            args.save_dir = Path(args.save_dir, str(logger.run_id))

        torch.set_num_threads(1)
        device = torch.device("cuda:0" if args.cuda else "cpu")

        envs = cls.make_vec_envs(args, device, test=False)

        agent = cls.make_agent(envs=envs, args=args)
        agent.to(device)

        ppo = PPO(
            agent=agent,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

        rollouts = Rollouts(
            num_steps=args.num_steps,
            num_processes=args.num_processes,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
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
                utils.update_linear_schedule(ppo.optimizer, j, num_updates, args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = agent.forward(
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
                next_value = agent.get_value(
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

            value_loss, action_loss, dist_entropy, gradient_norm = ppo.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (
                args.save_interval is not None
                and j % args.save_interval == 0
                or j == num_updates - 1
            ):
                if args.save_dir:
                    args.save_dir.mkdir(parents=True, exist_ok=True)
                    cls.save(agent, args, envs)

            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            if j % args.log_interval == 0:  # and len(episode_rewards) > 1:
                now = time.time()
                fps = int(total_num_steps / (now - start))
                log = {
                    EPISODE_RETURN: np.mean(episode_rewards),
                    ACTION_LOSS: action_loss,
                    VALUE_LOSS: value_loss,
                    FPS: fps,
                    TIME: now * 1000000,
                    HOURS: (now - start) / 3600,
                    GRADIENT_NORM: gradient_norm,
                    STEP: total_num_steps,
                    ENTROPY: dist_entropy,
                }
                logging.info(pformat(log))
                if logger is not None:
                    log.update({"run ID": logger.run_id})
                logging.info(pformat(log))
                if logger is not None:
                    logger.log(log)

            if args.test_interval is not None and j % args.test_interval == 0:
                cls.test(
                    agent=agent,
                    envs=cls.make_vec_envs(args, device, test=True),
                    num_processes=args.num_processes,
                    device=device,
                    start=start,
                    total_num_steps=total_num_steps,
                    logger=logger,
                )

    @classmethod
    def test(cls, agent, envs, num_processes, device, start, total_num_steps, logger):

        episode_rewards = []

        obs = envs.reset()
        recurrent_hidden_states = torch.zeros(
            num_processes, agent.recurrent_hidden_state_size, device=device
        )
        masks = torch.zeros(num_processes, 1, device=device)

        while len(episode_rewards) < 10:
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = agent.forward(
                    obs, recurrent_hidden_states, masks, deterministic=True
                )

            # Obser reward and next obs
            obs, _, done, infos = envs.step(action)

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

        envs.close()
        now = time.time()
        log = {
            TEST_EPISODE_RETURN: np.mean(episode_rewards),
            TIME: now * 1000000,
            HOURS: (now - start) / 3600,
            STEP: total_num_steps,
        }
        logging.info(pformat(log))
        if logger is not None:
            log.update({"run ID": logger.run_id})
        logging.info(pformat(log))
        if logger is not None:
            logger.log(log)

        print(
            " Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(episode_rewards), np.mean(episode_rewards)
            )
        )

    @staticmethod
    def make_env(env_id, seed, rank, allow_early_resets, *args, **kwargs):
        def _thunk():
            if env_id == "GoToLocal":
                env = Env(*args, **kwargs)
            elif env_id.startswith("dm"):
                _, domain, task = env_id.split(".")
                env = dmc2gym.make(domain_name=domain, task_name=task)
                env = ClipAction(env)
            else:
                env = gym.make(env_id)

            is_atari = hasattr(gym.envs, "atari") and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv
            )
            if is_atari:
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)

            env.seed(seed + rank)

            if str(env.__class__.__name__).find("TimeLimit") >= 0:
                env = TimeLimitMask(env)

            env = Monitor(env, allow_early_resets=allow_early_resets)

            if is_atari:
                if len(env.observation_space.shape) == 3:
                    env = EpisodicLifeEnv(env)
                    if "FIRE" in env.unwrapped.get_action_meanings():
                        env = FireResetEnv(env)
                    env = WarpFrame(env, width=84, height=84)
                    env = ClipRewardEnv(env)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example."
                )

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return _thunk

    @classmethod
    def make_vec_envs(cls, args, device, num_frame_stack=None, **kwargs):
        envs = [
            cls.make_env(args.env, args.seed, i, False, **kwargs)
            for i in range(args.num_processes)
        ]

        if len(envs) > 1:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)

        envs = VecPyTorch(envs, device)

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        elif len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

        return envs

    @staticmethod
    def save(agent, args, envs):
        torch.save(agent, Path(args.save_dir, f"checkpoint.pkl"))

    @staticmethod
    def make_agent(envs: VecPyTorch, args) -> Agent:
        obs_shape = envs.observation_space.shape
        action_space = envs.action_space
        return Agent(
            obs_shape=obs_shape,
            action_space=action_space,
            recurrent=args.recurrent_policy,
        )

    @classmethod
    def main(cls, args: Args):
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
            return cls.train(args)
        metadata = dict(reproducibility_info=args.get_reproducibility_info())
        if args.host_machine:
            metadata.update(host_machine=args.host_machine)
        if name := getattr(args, "name", None):
            metadata.update(name=name)

        logger: Logger
        with HasuraLogger(args.graphql_endpoint) as logger:
            charts = [
                *[
                    spec(x=HOURS, y=y)
                    for y in (
                        TEST_EPISODE_RETURN,
                        EPISODE_RETURN,
                    )
                ],
                *[
                    spec(x=STEP, y=y)
                    for y in (
                        TEST_EPISODE_RETURN,
                        EPISODE_RETURN,
                        FPS,
                        ENTROPY,
                        GRADIENT_NORM,
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
            logger.update_metadata(
                dict(parameters=args.as_dict(), run_id=logger.run_id)
            )
            logging.info(pformat(args.as_dict()))
            return cls.train(args=args, logger=logger)


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())
