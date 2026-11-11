#!/usr/bin/env bash
# Submit the two 4096 runs now and watch all three jobs (Martin: "queue all runs at once").
cd /e/scratch/weatherai/maranonpaliz1/latent_chain || exit 1
nohup bash watch_job.sh base2048 > watch_base2048.out 2>&1 < /dev/null &
for spec in global4096=01:00:00 both4096=01:00:00; do
  name=${spec%%=*}; t=${spec#*=}
  bash submit_one.sh "$name" "$t" && nohup bash watch_job.sh "$name" > "watch_$name.out" 2>&1 < /dev/null &
  wait $!
done
