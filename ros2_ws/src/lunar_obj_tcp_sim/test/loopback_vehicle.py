"""Test-only physics stand-in; not packaged as a runtime or Unreal replacement."""
import math
import select
import socket
import struct
import threading
import time


class LoopbackVehicle:
    def __init__(self, position=(0.,0.,0.), yaw=0., slip_mps=0.):
        self.listener=socket.socket(); self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.listener.bind(('127.0.0.1',0)); self.listener.listen(1); self.listener.settimeout(.1)
        self.port=self.listener.getsockname()[1]
        self.slip_mps=slip_mps
        self.position=list(position); self.yaw=yaw; self.v=self.w=0.
        self.steering_commands=[]; self.angles=[0.]*4; self.target_angles=[0.]*4; self.speeds=[0.]*4
        self.commands=[]; self.trajectory=[]; self.errors=[]; self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True); self.thread.start()

    def run(self):
        connection=None
        try:
            while not self.stop.is_set():
                try:
                    connection,_=self.listener.accept(); break
                except socket.timeout:
                    continue
            if connection is None: return
            connection.setblocking(False)
            buffer=bytearray(); target_v=target_w=0.; previous=time.monotonic(); last_send=0.
            while not self.stop.is_set():
                ready,_,_=select.select([connection],[],[],.01)
                if ready:
                    data=connection.recv(65576)
                    if not data: break
                    buffer.extend(data)
                    while len(buffer)>=57:
                        packet=bytes(buffer[:57]);del buffer[:57]
                        magic,cmd,length,checksum=struct.unpack('<4I',packet[:16])
                        assert (magic,cmd,length)==(0x4C494441,3,41)
                        assert checksum==magic^cmd^length
                        head,*values=struct.unpack('<I8fB',packet[20:])
                        assert head==0xEB92EB92
                        assert values[-1]==(sum(struct.unpack('<I',struct.pack('<f',v))[0] for v in values[:8]))&255
                        self.speeds=values[:4]; self.target_angles=[-a for a in values[4:8]]
                        self.steering_commands.append(tuple(self.target_angles))
                        # Independent forward kinematics from four wheel vectors.
                        xy=((.40875,.335),(.40875,-.335),(-.40875,.335),(-.40875,-.335))
                        vx=[.1595*s*math.cos(a) for s,a in zip(self.speeds,self.target_angles)]
                        vy=[.1595*s*math.sin(a) for s,a in zip(self.speeds,self.target_angles)]
                        target_v=sum(vx)/4
                        target_w=sum(x*b-y*a for (x,y),a,b in zip(xy,vx,vy))/sum(x*x+y*y for x,y in xy)
                        self.commands.append((target_v,target_w))
                now=time.monotonic(); dt=min(now-previous,.05); previous=now
                for i in range(4):
                    self.angles[i]+=max(-1.5*dt,min(1.5*dt,self.target_angles[i]-self.angles[i]))
                self.v+=max(-.5*dt,min(.5*dt,target_v-self.v))
                self.w+=max(-.5*dt,min(.5*dt,target_w-self.w))
                self.position[0]+=self.v*math.cos(self.yaw+self.w*dt/2)*dt
                self.position[1]+=self.v*math.sin(self.yaw+self.w*dt/2)*dt
                slip=self.slip_mps if abs(self.w)>.02 and abs(self.v)<.005 else 0.
                self.position[0]+=slip*dt
                self.yaw+=self.w*dt
                self.trajectory.append((*self.position, self.v, self.w))
                if now-last_send>=.05:
                    xy=((.40875,.335),(.40875,-.335),(-.40875,.335),(-.40875,-.335))
                    speeds=[((self.v-self.w*y)*math.cos(a)+self.w*x*math.sin(a))/.1595 for (x,y),a in zip(xy,self.angles)]
                    values=speeds+[-a for a in self.angles]+[self.position[0], -self.position[1], self.position[2]]
                    s, c = math.sin(self.yaw/2)/math.sqrt(2), math.cos(self.yaw/2)/math.sqrt(2)
                    values += [-s,-c,c,s,self.v+slip*math.cos(self.yaw),-slip*math.sin(self.yaw),0.,0.,self.w,0.]
                    connection.sendall(struct.pack('<4I',0x4C494441,4,92,0x4C494441^4^92)+struct.pack('<II21f',0,0xEB93EB93,*values))
                    last_send=now
        except (OSError,AssertionError) as error:
            if not self.stop.is_set():self.errors.append(repr(error))
        finally:
            if connection is not None: connection.close()

    def close(self):
        self.stop.set(); self.thread.join(timeout=2.); self.listener.close()
