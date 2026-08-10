#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sudo $0 /absolute/platform_profile.yaml /absolute/interface_profile.yaml" >&2
  exit 2
fi
case "$1:$2" in
  /*:/*) ;;
  *) echo "both source files must use absolute paths" >&2; exit 2 ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
platform_source="$(realpath "$1")"
interface_source="$(realpath "$2")"
test -f "$platform_source"
test -f "$interface_source"

python3 - "$platform_source" "$interface_source" "$repository_root" <<'PY'
from pathlib import Path
import hashlib
import sys
import yaml

platform = Path(sys.argv[1])
interface = Path(sys.argv[2])
root = Path(sys.argv[3])
sys.path.insert(0, str(root / "ros2_ws/src/lunar_external_adapter"))
from lunar_external_adapter.profile import load_interface_profile

document = yaml.safe_load(platform.read_text(encoding="utf-8"))
if not isinstance(document, dict) or document.get("schema_version") != "lunar-platform-profile/v1":
    raise SystemExit("platform profile schema is invalid")
identity = document.get("platform")
if not isinstance(identity, dict) or identity.get("platform_type") not in {"WHEELED", "LEGGED", "HOPPER"}:
    raise SystemExit("platform profile identity is invalid")
approved = {
    "WHEELED": "3a4f87e310cf7721be71818f5e4abc6cc73e78644bcdb0f8465426e8bcef9360",
    "LEGGED": "1f25b2fc4796e50ef7e966473c03cf902b1073291e2d1ccb09983ec90f0b17f0",
    "HOPPER": "1af41026d4c81500ce3351639d1fa7443f0c161b4de67f5da13d643b44dff841",
}
if hashlib.sha256(platform.read_bytes()).hexdigest() != approved[identity["platform_type"]]:
    raise SystemExit("platform profile is not frozen for interface-v1")
load_interface_profile(interface)
print(identity["platform_type"])
PY

# 只替换两个明确固定路径；脚本不启动、停止或重启任何 ROS 进程。
install -d -m 0755 /etc/lunar_navigation
install -m 0644 "$platform_source" /etc/lunar_navigation/platform_profile.yaml
install -m 0644 "$interface_source" /etc/lunar_navigation/interface_profile.yaml
echo "installed; restart the ROS nodes manually to apply the files"
