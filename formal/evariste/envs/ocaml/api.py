# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict
import socket
from datetime import datetime
import subprocess
from threading import Thread
import re
import time
import os

from evariste.logger import create_logger, _MixinLoggerFactory


class OCamlError(Exception):
    pass


class OCamlErrorTerminatedProcess(OCamlError):
    pass


class OCamlErrorTimeout(OCamlError):
    pass


class OCamlErrorBadFormat(OCamlError):
    pass


class OCamlErrorIncompleteMsg(OCamlError):
    pass


class OCamlErrorWrongMessage(OCamlError):
    pass


class OCamlErrorFailedRequest(OCamlError):
    pass


def _log_formatter(obj, _, msg):
    return f"OCamlAPI PID#{obj.rank}: {msg}"


class OCamlAPI(_MixinLoggerFactory("debug", _log_formatter)):  # type: ignore  ## mypy unsupported dynamic base class
    def __init__(
        self, checkpoint_path: str, timeout: float, logger=None,
    ):
        logger = logger if logger is not None else create_logger(None)
        self.set_logger(logger)
        self._timeout = timeout
        self._proc = subprocess.Popen(
            [checkpoint_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        self.rank = self._proc.pid
        self.log("OCaml process booted")
        self._server_address = (
            f"/tmp/topsocket-{self.rank}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(
            b'topsocket_interact "' + bytes(self._server_address, "utf8") + b'";;\n',
        )
        self._proc.stdin.flush()

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while True:
            try:
                self.log(f"connecting to socket {self._server_address} ...")
                self._sock.connect(self._server_address)
            except FileNotFoundError as e:
                returncode = self._proc.poll()
                if returncode is not None:
                    self.error(
                        OCamlErrorTerminatedProcess,
                        f"process terminated with returncode {-returncode}",
                    )
                self.log(
                    f"failed to connect to socket {self._server_address} - waiting 3s"
                )
                time.sleep(3)
            else:
                self.log(f"connected to socket {self._server_address}")
                break
        self._id_msg = 0

    def error(self, error_cls, msg):
        self.log(f"{error_cls.__name__}: {msg}")
        exc = error_cls()
        error_cls.__init__(exc, f"PID#{self.rank}: {msg}")
        raise exc

    def send(self, msg: str) -> str:
        request_id = self._id_msg
        self._id_msg += 1
        req = OCamlAPI.wrap_msg(request_id, msg)
        self._sock.sendall(req)
        reply: Dict = {}
        th = Thread(target=self._recv, args=(reply,))
        th.start()
        th.join(timeout=self._timeout)
        if th.is_alive():
            self.error(
                OCamlErrorTimeout,
                f'Request #{request_id}: "{msg}"; got reply "{reply}"',
            )
        if "error" in reply:
            exc = reply["error"]
            raise type(exc).__init__(f"PID#{self.rank}: {str(exc)}")
        if not {"id", "status", "msg"} <= set(
            reply
        ):  # should not happen, but who knows if the update of a dict is atomic...
            self.error(
                OCamlErrorIncompleteMsg,
                f'Request #{request_id}: "{msg}"; got reply "{reply}"',
            )
        if reply["id"] != request_id:
            self.error(
                OCamlErrorWrongMessage,
                f"Expected a reply for request #{request_id} - got #{reply['id']} \"{reply['msg']}\"",
            )
        if reply["status"] != "fine":
            self.error(
                OCamlErrorFailedRequest,
                f"Got a {reply['status']} status for request #{request_id}: \"{reply['msg']}\"",
            )
        if reply["status"] != "fine":
            self.error(
                OCamlErrorFailedRequest,
                f"Got a {reply['status']} status for request #{request_id}: \"{reply['msg']}\"",
            )
        if reply["status"] != "fine":
            self.error(
                OCamlErrorFailedRequest,
                f"Got a {reply['status']} status for request #{request_id}: \"{reply['msg']}\"",
            )
        return reply["msg"]

    def _recv(self, reply: dict) -> None:
        HEADER_SIZE = 27
        reply.update({"header_size": HEADER_SIZE})
        try:
            header_data = b""
            while len(header_data) < HEADER_SIZE:
                header_data += self._sock.recv(HEADER_SIZE - len(header_data))
                reply.update(
                    {
                        "header_data": header_data,
                        "header_size_received": len(header_data),
                    }
                )
            header = OCamlAPI.get_header(header_data, HEADER_SIZE)
            reply.update(
                {
                    "id": header["id"],
                    "status": header["status"],
                    "msg_size": header["size"],
                }
            )
            msg_size = header["size"]
            msg_data = b""
            while len(msg_data) < msg_size:
                msg_data += self._sock.recv(msg_size - len(msg_data))
                reply.update({"msg_data": msg_data, "msg_size_received": len(msg_data)})
            msg = OCamlAPI.get_body(msg_data, msg_size)
        except (OCamlError, OSError) as e:
            reply["error"] = e
        else:
            reply.update({"msg": msg})

    def __del__(self):
        """
        Destructor.
        Kill OCaml instance.
        """
        self.hard_kill()

    def hard_kill(self):
        self.log("Hard kill ...")
        if hasattr(self, "_proc"):
            self.log("Killing OCaml instance ...")
            try:
                self._proc.kill()
            except ProcessLookupError:
                self.log(
                    "You can't kill what's already dead... Ignoring ProcessLookupError",
                    "warning",
                )
            else:
                self.log("OCaml instance killed")
            delattr(self, "_proc")
        else:
            self.log("No OCaml instance to kill")

        if hasattr(self, "_sock"):
            self.log(f"Closing socket {self._server_address} ...")
            self._sock.close()
            self.log(f"Socket {self._server_address} closed")
        else:
            self.log(f"No socket {self._server_address} to close")
        if hasattr(self, "_server_address"):
            self.log(f"Deleting socket file {self._server_address} ...")
            try:
                os.remove(self._server_address)
            except FileNotFoundError:
                self.log(
                    f"Tried to delete socket file {self._server_address} but it did not exist",
                )
            else:
                self.log(f"Deleted socket file {self._server_address}")
        else:
            self.log(f"No file {self._server_address} to delete")

    @staticmethod
    def wrap_msg(id: int, msg: str) -> bytes:
        """
        protocol:
            header:          "req"
            space:           " "
            sequence number: [0-9]+
            space:           " "
            input size:      [0-9]+
            newline:         "\n"
            input:           BYTE x INPUT_SIZE
        """
        bmsg = bytes(msg, "utf8")
        lmsg = len(bmsg)
        return (
            b"req "
            + bytes(str(id), "utf8")
            + b" "
            + bytes(str(lmsg), "utf8")
            + b"\n"
            + bmsg
        )

    @staticmethod
    def get_header(data: bytes, size: int):
        """
        protocol:
            header:          "rsp"
            space:           " "
            sequence number: [0-9a-f] x 8
            space:           " "
            status:          "fail"|"nopa"|"fine"
            space:           " "
            output size:     [0-9a-f] x 8
            newline:         "\n"
            output:          BYTE x OUTPUT_SIZE

            The header line of responses is constant size:
            3 + 1 + 8 + 1 + 4 + 1 + 8 + 1 == 27 bytes
        """
        if len(data) != size:
            raise OCamlErrorBadFormat(
                f"expected header of size {size}, got header of size {len(data)}: {repr(data)}"
            )
        header_pattern = r"rsp (?P<id>[0-9a-f]{8}) (?P<status>fail|nopa|fine) (?P<size>[0-9a-f]{8})\n"
        match = re.match(header_pattern, data.decode("utf8"))
        if not match:
            raise OCamlErrorBadFormat(f"got header: {repr(data)}")
        size = int(match["size"], 16)
        req_id = int(match["id"], 16)
        return {"size": size, "id": req_id, "status": match["status"]}

    @staticmethod
    def get_body(data: bytes, size: int):
        if len(data) != size:
            raise OCamlErrorBadFormat(
                f"expected msg of size {size}, got msg of size {len(data)}: {repr(data)}"
            )
        return data.decode("utf8")
