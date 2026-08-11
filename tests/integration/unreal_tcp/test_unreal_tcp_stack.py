from pathlib import Path
import sys
import types


_launch_file = Path(__file__).with_name("test_unreal_tcp_stack.launch.py")
__path__ = []
_launch_module = types.ModuleType(f"{__name__}.launch")
_launch_module.__file__ = str(_launch_file)
exec(
    compile(_launch_file.read_bytes(), str(_launch_file), "exec"),
    _launch_module.__dict__,
)
sys.modules[_launch_module.__name__] = _launch_module
