"""Bounded duplex transport: protocol owners never perform blocking socket I/O.

Each endpoint has one reader and one writer. A slot can have only one reset or
action outstanding, with ACK before reuse. Its finite window contains HELLO,
SCENE, READY/EPISODE_END, NEED_RESET, RESERVE, DONE, ACK, ABORTED, RECOVERY,
OWNERSHIP and CLOSED (12 entries); six global controls plus the publication
window cover barrier/stop and ceil(max_update_credit / actor_publish_updates)
publications (two by default). Configuration changes resize that finite window.
Each mailbox and the reader's current burst use that window; the writer owns
one additional frame. Only replaceable
STATUS and identical unacknowledged DONE retries coalesce; reliable FIFO order
is preserved. Exceeding the window is a protocol error, never backpressure in
the application event loop. No history or scene archive is retained here.
"""
from collections import deque
import os
import socket
import threading
import time


class Duplex:
    def __init__(self, connection, slots=1, wake=None, publication_window=2):
        self.connection = connection
        self.capacity = 12 * slots + 6 + publication_window
        self.wake = wake or threading.Event()
        self.condition = threading.Condition()
        self.inbox, self.outbox = deque(), deque()
        self.error = None
        self.closed = self.writing = False
        self.threads = [threading.Thread(target=target, name='drl-ipc-' + name)
            for target, name in ((self._read, 'reader'), (self._write, 'writer'))]
        for thread in self.threads: thread.start()

    @staticmethod
    def _key(message):
        kind = message.get('kind')
        if kind == 'STATUS': return kind, message.get('env')
        if kind == 'DONE': return kind, message.get('env'), message['token']
        return None

    def _append(self, mailbox, message):
        key = self._key(message)
        if key is not None:
            for index, old in enumerate(mailbox):
                if self._key(old) == key:
                    if key[0] == 'DONE': return
                    del mailbox[index]
                    break
        if len(mailbox) >= self.capacity:
            raise RuntimeError('IPC protocol outstanding-message window exceeded')
        mailbox.append(message)

    def _failed(self, exc):
        with self.condition:
            if not self.closed: self.error = exc
            self.condition.notify_all()
            self.wake.set()

    def _read(self):
        try:
            while True:
                message = self.connection.recv()
                # Publish a bounded burst atomically so controls already on the
                # socket are visible together before the next Actor call.
                burst = [message]
                ended = None
                while len(burst) < self.capacity and self.connection.poll():
                    try: burst.append(self.connection.recv())
                    except (EOFError, OSError) as exc:
                        ended = exc
                        break
                with self.condition:
                    while len(self.inbox) + len(burst) > self.capacity and not self.closed:
                        self.condition.wait()
                    if self.closed: return
                    for message in burst: self._append(self.inbox, message)
                    self.condition.notify_all()
                    self.wake.set()
                if ended is not None:
                    self._failed(ended)
                    return
        except (EOFError, OSError, RuntimeError) as exc: self._failed(exc)

    def _write(self):
        try:
            while True:
                with self.condition:
                    while not self.outbox and not self.closed: self.condition.wait()
                    if self.closed: return
                    message = self.outbox.popleft()
                    self.writing = True
                self.connection.send(message)
                with self.condition:
                    self.writing = False
                    self.condition.notify_all()
        except (EOFError, OSError, RuntimeError) as exc: self._failed(exc)

    def send(self, message):
        with self.condition:
            if self.closed: raise BrokenPipeError('closed IPC endpoint')
            if self.error is not None: raise self.error
            self._append(self.outbox, message)
            self.condition.notify_all()

    def poll(self, timeout=0):
        with self.condition:
            if not self.inbox and self.error is None and not self.closed and timeout:
                self.condition.wait_for(lambda: self.inbox or self.error or self.closed, timeout)
            return bool(self.inbox) or self.error is not None or self.closed

    def recv(self):
        with self.condition:
            self.condition.wait_for(lambda: self.inbox or self.error or self.closed)
            if self.inbox:
                message = self.inbox.popleft()
                self.condition.notify_all()
                return message
            if self.error is not None: raise self.error
            raise EOFError('closed IPC endpoint')

    def close(self, *, flush=True, timeout=3.):
        deadline = time.monotonic() + timeout
        with self.condition:
            if self.closed: return
            if flush:
                self.condition.wait_for(lambda: (not self.outbox and not self.writing) or self.error,
                    max(0, deadline - time.monotonic()))
            undelivered = bool(self.outbox or self.writing)
            self.closed = True
            self.condition.notify_all()
        # Closing a fd from another thread does not interrupt a blocked send.
        # Shutdown the underlying Unix socket, then join both owners explicitly.
        with socket.socket(fileno=os.dup(self.connection.fileno())) as endpoint:
            endpoint.shutdown(socket.SHUT_RDWR)
        for thread in self.threads: thread.join(max(.1, deadline - time.monotonic()))
        self.connection.close()
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError('IPC I/O owner failed to join after socket shutdown')
        if flush and undelivered: raise BrokenPipeError('IPC closed before reliable messages drained')
