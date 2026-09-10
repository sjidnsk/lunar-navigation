"""SIGINT must stop the controller before its ROS context is destroyed."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist


def test_sigint_exits_without_publishing_into_a_closed_context():
    environment = os.environ.copy()
    source = Path(__file__).resolve().parents[1] / 'python'
    environment['PYTHONPATH'] = str(source) + os.pathsep + environment.get('PYTHONPATH', '')
    rclpy.init()
    observer = rclpy.create_node('controller_shutdown_observer')
    messages = []
    observer.create_subscription(Twist, '/lunar_shutdown_test/cmd', messages.append, 10)
    process = subprocess.Popen([
        sys.executable, '-c', 'from lunar_pure_wheeled_controller.node import main; main()',
        '--ros-args', '-p', 'input_mode:=incremental_reference',
        '-p', 'command_topic:=/lunar_shutdown_test/cmd',
    ], env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 8
        while not messages and time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=0.05)
        assert messages, 'controller did not start'
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 0, output
        assert 'context is invalid' not in output
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        observer.destroy_node()
        rclpy.shutdown()
