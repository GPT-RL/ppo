#! /usr/bin/env bash
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
name=gpt_agent
docker build -t $name .
docker run --rm -it\
  --env-file .env\
  --gpus "$1"\
  -e HOST_MACHINE="$(hostname -s)"\
  -v "$(pwd):/project"\
  -v "$(pwd)/logs:/tmp/logs"\
  -v "$HOME/.cache/GPT/:/root/.cache/GPT" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
  -v "$HOME/.cache/data/:/data" \
  $name "${@:2}"
