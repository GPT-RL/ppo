import codecs
import os
import pickle

import torch
from gql import gql
from run_logger import HasuraLogger

import main
from envs import make_vec_envs
from utils import get_render_func, get_vec_normalize


class Args(main.Args):
    load_id: str = None  # path to load parameters from if at all
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")


def main(args: Args):
    env = make_vec_envs(
        args.env,
        args.seed + 1000,
        1,
        None,
        None,
        device="cpu",
        allow_early_resets=False,
    )

    if args.load_id:
        with HasuraLogger(hasura_uri=args.graphql_endpoint) as logger:
            blob_id = logger.execute(
                gql(
                    """
        query GetMostRecentCheckpointID($id: Int) {
        run_blob_aggregate(where: {run_id: {_eq: $id}, metadata: {_contains: {type: "checkpoint"}}}) {
        aggregate {
        max {
        id
        }
        }
        }
        } 
        """
                ),
                variable_values=dict(id=args.load_id),
            )["run_blob_aggregate"]["aggregate"]["max"]["id"]
            if blob_id is None:
                raise RuntimeError(
                    f"No checkpoint found in database for run {args.load_id}"
                )
            pickled = logger.execute(
                gql(
                    """
        query GetCheckpoint($id: Int!) {
        run_blob_by_pk(id: $id) {
        text
        }
        }
        """
                ),
                variable_values=dict(id=blob_id),
            )["run_blob_by_pk"]["text"]
            data = pickle.loads(codecs.decode(pickled.encode(), "base64"))
            agent.load_state_dict(data)

    torch.save

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic, obs_rms = torch.load(
        os.path.join(args.load_dir, args.env_name + ".pt"), map_location="cpu"
    )

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    if render_func is not None:
        render_func("human")

    if args.env_name.find("Bullet") > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() == "torso":
                torsoId = i

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det
            )

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find("Bullet") > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func("human")


if __name__ == "__main__":
    main()
