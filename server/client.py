import socket
from typing import Optional

import time

from common.protocol import (
    MAGIC,
    VERSION,
    MsgType,
    HEADER_SIZE,
    encode_infer_request,
    encode_load_weights_begin,
    encode_load_weights_chunk,
    encode_load_weights_end,
    encode_placement_plan,
    decode_infer_response,
    decode_inventory_reply,
    decode_resident_inventory_reply,
    decode_placement_ack,
    pack_header,
    unpack_header,
)


class ProtocolError(RuntimeError):
    pass


class NodeClient:
    def __init__(self, host: str, control_port: int, timeout_sec: float = 5.0):
        self.host = host
        self.control_port = control_port
        self.timeout_sec = timeout_sec
        self.sock: Optional[socket.socket] = None
        self._next_request_id = 1

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.control_port), timeout=self.timeout_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout_sec)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "NodeClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def next_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _require_sock(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("socket is not connected")
        return self.sock

    def _recv_exact(self, n: int) -> bytes:
        sock = self._require_sock()
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError(
                    f"socket closed while receiving {n} bytes "
                    f"(received {n - remaining} bytes before EOF)"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_all(self, data: bytes) -> None:
        sock = self._require_sock()
        sock.sendall(data)

    def send_message(self, msg_type: MsgType, request_id: int, body: bytes = b"") -> None:
        header = pack_header(msg_type=msg_type, request_id=request_id, body_len=len(body))
        self._send_all(header)
        if body:
            self._send_all(body)

    def recv_message(self):
        header_bytes = self._recv_exact(HEADER_SIZE)
        header = unpack_header(header_bytes)

        if header["magic"] != MAGIC:
            raise ProtocolError(f"bad magic: got {header['magic']:#x}, expected {MAGIC:#x}")
        if header["version"] != VERSION:
            raise ProtocolError(
                f"bad version: got {header['version']}, expected {VERSION}"
            )

        body_len = header["body_len"]
        body = self._recv_exact(body_len) if body_len > 0 else b""

        return {
            "header": header,
            "body": body,
        }

    def request(
        self,
        req_type: MsgType,
        resp_type: MsgType,
        body: bytes = b"",
        request_id: Optional[int] = None,
    ) -> bytes:
        if request_id is None:
            request_id = self.next_request_id()
     
        t0 = time.perf_counter()
        self.send_message(req_type, request_id=request_id, body=body)
        t1 = time.perf_counter()
     
        header_bytes = self._recv_exact(HEADER_SIZE)
        t2 = time.perf_counter()
     
        header = unpack_header(header_bytes)
     
        if header["magic"] != MAGIC:
            raise ProtocolError(f"bad magic: got {header['magic']:#x}, expected {MAGIC:#x}")
        if header["version"] != VERSION:
            raise ProtocolError(
                f"bad version: got {header['version']}, expected {VERSION}"
            )
     
        body_len = header["body_len"]
        resp_body = self._recv_exact(body_len) if body_len > 0 else b""
        t3 = time.perf_counter()
     
        msg_type = header["msg_type"]
        if msg_type != resp_type:
            raise ProtocolError(
                f"expected {resp_type.name}, got {msg_type.name} ({int(msg_type)})"
            )
     
        if header["request_id"] != request_id:
            raise ProtocolError(
                f"request_id mismatch: got {header['request_id']}, expected {request_id}"
            )
     
        if req_type in (
            MsgType.LoadWeightsBegin,
            MsgType.LoadWeightsChunk,
            MsgType.LoadWeightsEnd,
        ):
            print(
                f"[client-request-profile] "
                f"req={req_type.name} rid={request_id} req_body_len={len(body)} resp_body_len={body_len} "
                f"send_ms={(t1-t0)*1000:.3f} "
                f"recv_header_ms={(t2-t1)*1000:.3f} "
                f"recv_body_ms={(t3-t2)*1000:.3f} "
                f"total_ms={(t3-t0)*1000:.3f}"
            )
     
        return resp_body

    def request_inventory(self, request_id: Optional[int] = None):
        body = self.request(
            req_type=MsgType.InventoryRequest,
            resp_type=MsgType.InventoryReply,
            body=b"",
            request_id=request_id,
        )
        return decode_inventory_reply(body)

    def request_resident_inventory(self, request_id: Optional[int] = None):
        body = self.request(
            req_type=MsgType.ResidentInventoryRequest,
            resp_type=MsgType.ResidentInventoryReply,
            body=b"",
            request_id=request_id,
        )
        return decode_resident_inventory_reply(body)

    def send_heartbeat(self, request_id: Optional[int] = None) -> bytes:
        return self.request(
            req_type=MsgType.HeartbeatRequest,
            resp_type=MsgType.HeartbeatReply,
            body=b"",
            request_id=request_id,
        )

    def send_placement_plan(
        self,
        assignments,
        drop_non_target_residents: bool = False,
        request_id=None,
    ) -> bytes:
        body = encode_placement_plan(
            assignments,
            drop_non_target_residents=drop_non_target_residents,
        )
        resp_body = self.request(
            req_type=MsgType.PlacementPlan,
            resp_type=MsgType.PlacementAck,
            body=body,
            request_id=request_id,
        )
        return decode_placement_ack(resp_body)

    def send_load_weights_begin(self, msg, request_id=None) -> bytes:
        body = encode_load_weights_begin(msg)
        return self.request(
            req_type=MsgType.LoadWeightsBegin,
            resp_type=MsgType.LoadWeightsAck,
            body=body,
            request_id=request_id,
        )

    def send_load_weights_chunk(self, msg, request_id=None) -> bytes:
        body = encode_load_weights_chunk(msg)
        return self.request(
            req_type=MsgType.LoadWeightsChunk,
            resp_type=MsgType.LoadWeightsAck,
            body=body,
            request_id=request_id,
        )


    def send_load_weights_end(self, msg, request_id=None) -> bytes:
        body = encode_load_weights_end(msg)
        return self.request(
            req_type=MsgType.LoadWeightsEnd,
            resp_type=MsgType.LoadWeightsAck,
            body=body,
            request_id=request_id,
        )

    def send_infer_request(self, msg, request_id=None):
        body = encode_infer_request(msg)
        resp_body = self.request(
            req_type=MsgType.InferRequest,
            resp_type=MsgType.InferResponse,
            body=body,
            request_id=request_id,
        )
        return decode_infer_response(resp_body)
