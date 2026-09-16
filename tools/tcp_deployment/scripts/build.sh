#!/usr/bin/env bash
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec bash "$bundle_root/scripts/simulation/build.sh" -DBUILD_TESTING=OFF "$@"
