#! /usr/bin/env bash
rm -rf logs
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
name=gpt
docker build -t $name .
docker run --rm -it\
  --env-file .env\
  --gpus all\
  -e HOST_MACHINE="$(hostname -s)"\
  -v "$(pwd)/logs:/tmp/logs"\
  -v "$HOME/.cache/GPT/:/root/.cache/GPT" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
  $name "$@"
