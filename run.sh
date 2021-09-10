#! /usr/bin/env bash
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
name=ppo_agent
docker build -t $name .
docker run --rm -it\
  --env-file .env\
  -e HOST_MACHINE="$(hostname -s)"\
  -v "$(pwd)/logs:/tmp/logs"\
  -v "$HOME/.cache/GPT/:/root/.cache/GPT" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
  $name "$@"
