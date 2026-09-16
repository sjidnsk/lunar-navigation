"""Loopback-only kinematic vehicle for interactive RViz, not UE physics."""
from collections import deque
import math
import select
import socket
import struct
import threading
import time

from .coordinates import world_position, wire_orientation
from .kinematics import FourWheelSteering


class LocalVehicle:
    def __init__(self, position, *, yaw=0., radius=.1595, track=.67, wheelbase=.8175, port=0, steering_options=None,
                 height_at=None, command_timeout=.5, vehicle_id=0):
        if radius <= 0 or track <= 0 or command_timeout <= 0:
            raise ValueError('wheel dimensions and timeout must be positive')
        self.vehicle_id = vehicle_id
        self.position = list(position)
        self.yaw, self.v, self.w = yaw, 0., 0.
        self.radius, self.track = radius, track
        self.model = FourWheelSteering(2*radius, track, wheelbase, **(steering_options or {}))
        self._speeds = [0.]*4
        self._angles = [0.]*4
        self._target_angles = [0.]*4
        self.vy = 0.
        self.height_at = height_at
        self.command_timeout = command_timeout
        self.errors = []
        self.commands = deque(maxlen=10000)
        self.stopping = threading.Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(1)
        self.listener.settimeout(.1)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.run, name='local-tcp-vehicle', daemon=True)
        self.thread.start()

    def decode(self, packet):
        magic, command, length, checksum = struct.unpack('<4I', packet[:16])
        head, *values = struct.unpack('<I8fB', packet[20:])
        payload_sum = sum(struct.unpack('<I', struct.pack('<f', v))[0] for v in values[:8])
        if ((magic, command, length) != (0x4C494441, 3, 41)
                or checksum != magic ^ command ^ length or head != 0xEB92EB92
                or values[-1] != payload_sum & 255
                or not all(math.isfinite(v) for v in values[:8])):
            raise ValueError('invalid local vehicle command packet')
        if struct.unpack_from('<I',packet,16)[0] != self.vehicle_id:
            return None
        for slot,i in enumerate(self.model.wheel_order):
            self._speeds[i] = values[slot]*self.model.wheel_signs[slot]
        for slot,i in enumerate(self.model.steering_order):
            self._target_angles[i] = (values[4+slot]-self.model.zero[slot])*self.model.steering_signs[slot]
        v, _, w = self.body_velocity(self._target_angles)
        return v, w

    def body_velocity(self, angles):
        vectors = [(speed*self.radius*math.cos(a), speed*self.radius*math.sin(a))
                   for speed,a in zip(self._speeds,angles)]
        vx = sum(v[0] for v in vectors)/4
        vy = sum(v[1] for v in vectors)/4
        w = sum(x*v[1]-y*v[0] for (x,y),v in zip(self.model.positions,vectors)) / sum(
            x*x+y*y for x,y in self.model.positions)
        return vx,vy,w

    def run(self):
        connection = None
        try:
            while not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if connection is None:
                return
            connection.settimeout(.2)
            buffer = bytearray()
            target_v = target_w = 0.
            previous = time.monotonic()
            last_command = last_send = 0.
            while not self.stopping.is_set():
                ready, _, _ = select.select([connection], [], [], .01)
                if ready:
                    data = connection.recv(65576)
                    if not data:
                        break
                    buffer.extend(data)
                    while len(buffer) >= 57:
                        command = self.decode(bytes(buffer[:57]))
                        del buffer[:57]
                        if command is None:
                            continue
                        target_v, target_w = command
                        last_command = time.monotonic()
                        self.commands.append((target_v, target_w))
                now = time.monotonic()
                dt = min(now - previous, .05)
                previous = now
                if now - last_command > self.command_timeout:
                    self._speeds = [0.]*4
                for i in range(4):
                    self._angles[i] += max(-1.5*dt, min(1.5*dt, self._target_angles[i]-self._angles[i]))
                target_v, target_vy, target_w = self.body_velocity(self._angles)
                self.vy += max(-.5*dt, min(.5*dt, target_vy-self.vy))
                self.v += max(-.5*dt, min(.5*dt, target_v-self.v))
                self.w += max(-.5*dt, min(.5*dt, target_w-self.w))
                x = self.position[0] + (self.v*math.cos(self.yaw+self.w*dt/2)-self.vy*math.sin(self.yaw+self.w*dt/2))*dt
                y = self.position[1] + (self.v*math.sin(self.yaw+self.w*dt/2)+self.vy*math.cos(self.yaw+self.w*dt/2))*dt
                z = self.height_at(x, y) if self.height_at else self.position[2]
                if not math.isfinite(z):
                    raise ValueError('vehicle left valid OBJ elevation; restart inside the map')
                self.position[:] = [x, y, z]
                self.yaw += self.w*dt
                if now-last_send >= .05:
                    speeds = [((self.v-self.w*y)*math.cos(a)+(self.vy+self.w*x)*math.sin(a))/self.radius
                              for (x,y),a in zip(self.model.positions,self._angles)]
                    values = [speeds[i]*sign for i,sign in zip(self.model.wheel_order,self.model.wheel_signs)]
                    values += [self._angles[i]*sign+zero for i,sign,zero in
                               zip(self.model.steering_order,self.model.steering_signs,self.model.zero)]
                    values += list(world_position(self.position))
                    values += list(wire_orientation((0., 0., math.sin(self.yaw/2), math.cos(self.yaw/2))))
                    values += [self.v, self.vy, 0.]
                    values += [0., self.w, 0.]
                    connection.sendall(struct.pack('<4I',0x4C494441,4,92,0x4C494441^4^92)+struct.pack('<II21f',self.vehicle_id,0xEB93EB93,*values))
                    last_send = now
        except (OSError, ValueError) as error:
            if not self.stopping.is_set():
                self.errors.append(str(error))
        finally:
            self.v = self.w = 0.
            if connection is not None:
                connection.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2.)
        self.listener.close()
