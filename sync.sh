#!/bin/bash

host=cs-199
target=cs-182-project

ssh -MNf -S /tmp/${host} ${host}
echo "Multiplexed SSH connection started (/tmp/${host})"

function cleanup {
  ssh -S /tmp/${host} -O exit ${host}
  echo "Cleaned up SSH connection."
}

trap cleanup EXIT

inotifywait --exclude '(data|logs|params|\.venv|\.git)' -mr --format '%w%f' -e close_write . | while read path; do
  # Maybe `rsync` would be better
  scp -o "ControlPath=/tmp/${host}" -o "ControlMaster=no" ${path} ${host}:${target}
done
