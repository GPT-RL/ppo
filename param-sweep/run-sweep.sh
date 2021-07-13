#!/usr/bin/env bash

while [ -z "$rank" ]
do
  rank=$(redis-cli RPOP rank-queue)
  sleep .1
done

while [ -z "$sweep_id" ]
do
  sweep_id=$(redis-cli GET "sweep_id")
  sleep .1
done

for (( i=1; ; i++ ))
do
  if [ ! -z "$NUM_RUNS" ] && [ -z "$(redis-cli RPOP runs-queue)" ]
  then
    break
  fi
  cmd="CUDA_VISIBLE_DEVICES=$rank python $SCRIPT sweep $sweep_id"
  echo "$cmd"
  eval "$cmd"
  sleep 5
done
