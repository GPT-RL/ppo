import gym
from gym.spaces import Dict, MultiDiscrete
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import babyai_main
import gpt_main
from babyai_env import GoToLocalEnv, ObsSpaceWrapper
from babyai_gpt_agent import Agent
from babyai_main import RolloutsWrapper
from envs import VecPyTorch
from gpt_agent import get_gpt_size


class Args(babyai_main.Args, gpt_main.Args):
    pass


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer: GPT2Tokenizer, longest_mission: str):
        self.tokenizer: GPT2Tokenizer = tokenizer
        encoded = tokenizer.encode(longest_mission)
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.observation_space = Dict(
            spaces=dict(**spaces, mission=MultiDiscrete([50257 for _ in encoded]))
        )

    def observation(self, observation):
        mission = self.tokenizer.encode(observation["mission"])
        observation.update(mission=mission)
        return observation


class Trainer(babyai_main.Trainer, gpt_main.Trainer):
    @staticmethod
    def make_agent(envs: VecPyTorch, args: Args) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            gpt_size=args.gpt_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            randomize_parameters=args.randomize_parameters,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
        )

    @classmethod
    def make_vec_envs(cls, args: Args, device, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.gpt_size))
        return super().make_vec_envs(args, device, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def make_env(env_id, seed, rank, allow_early_resets, *args, **kwargs):
        def _thunk():
            tokenizer = kwargs.pop("tokenizer")
            max_episode_steps = kwargs.pop("max_episode_steps")
            env = GoToLocalEnv(*args, seed=seed + rank, **kwargs)
            env = ObsSpaceWrapper(env)
            env = TokenizerWrapper(
                env, tokenizer=tokenizer, longest_mission="go to a blue ball"
            )
            env = RolloutsWrapper(env)

            # if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
            #
            env = Monitor(env, allow_early_resets=allow_early_resets)

            return env

        return _thunk


if __name__ == "__main__":
    Trainer.main(Args(explicit_bool=True).parse_args())
