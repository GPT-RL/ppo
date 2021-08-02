import argparse
import os
import subprocess
import time
import sys
from gql import gql

from redis import Redis
from run_logger import Client


def execute_sweep(graphql_endpoint: str, command: str):
    redis = Redis(host="redis")
    rank = redis.decr("rank-counter")
    print("rank ==", rank)

    while (sweep_id := redis.get("sweep_id")) is None:
        time.sleep(0.1)
    sweep_id = int(sweep_id)
    print("sweep_id ==", sweep_id)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(rank)

    client = Client(graphql_endpoint)

    def keep_running():
        data = client.execute(
            gql(
                """
mutation decr_remaining_runs($sweep_id: Int!) {
  update_sweep(where: {id: {_eq: $sweep_id}}, _inc: {remaining_runs: -1}) {
    returning {
      remaining_runs
    }
  }
}
        """
            ),
            variable_values=dict(sweep_id=sweep_id),
        )
        remaining_runs = data["update_sweep"]["returning"]
        print("Remaining runs:", remaining_runs)
        return not remaining_runs or (remaining_runs[0]["remaining_runs"] >= 0)

    while keep_running():
        cmd = f"{command} {sweep_id}"
        print(cmd)
        subprocess.run(cmd.split(), env=env)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        required=True,
        help="command execute. `sweep_id` will be given to command as in `script sweep_id`.",
    )
    parser.add_argument(
        "--graphql-endpoint", required=True, help="Endpoint to use for Hasura"
    )
    execute_sweep(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
