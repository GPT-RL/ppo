#!/usr/bin/env bash

while [ -z "$rank" ]
do
  rank=$(redis-cli -h redis RPOP rank-queue)
  sleep .1
done

while [ -z "$sweep_id" ]
do
  sweep_id=$(redis-cli -h redis GET "sweep_id")
  sleep .1
done

for (( i=1; ; i++ ))
do
  if [ ! -z "$NUM_RUNS" ] && [ -z "$(redis-cli -h redis RPOP runs-queue)" ]
  then
    break
  fi
  cmd="CUDA_VISIBLE_DEVICES=$rank python $SCRIPT sweep $sweep_id"
  echo "$cmd"
  eval "$cmd"
  sleep 5
done
