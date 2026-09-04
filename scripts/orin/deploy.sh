#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法：$0 <user@orin> <remote-dir>" >&2
  exit 2
fi

target="$1"
remote_dir="$2"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh "$target" mkdir -p "$remote_dir"
rsync -a \
  --exclude .git \
  --exclude ros2_ws/build \
  --exclude ros2_ws/install \
  --exclude ros2_ws/log \
  "$repository_root/" "$target:$remote_dir/"
