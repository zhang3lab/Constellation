import enum
import struct
from typing import Tuple, TypedDict


MAGIC = 0x45585054  # "EXPT"
VERSION = 1

# Wire format is little-endian:
# uint32 magic
# uint16 version
# uint16 msg_type
# uint32 request_id
# uint32 body_len
HEADER_STRUCT = struct.Struct("<IHHII")
HEADER_SIZE = HEADER_STRUCT.size

U32 = struct.Struct("<I")
I32 = struct.Struct("<i")
U64 = struct.Struct("<Q")

MAX_STRING_BYTES = 1 << 20   # 1 MiB, very generous for names/ids
MAX_GPUS_PER_NODE = 1024


class ProtocolError(RuntimeError):
    pass


class MsgType(enum.IntEnum):
    InventoryRequest = 1
    InventoryReply = 2

    PlacementPlan = 10
    PlacementAck = 11

    LoadWeightsBegin = 20
    LoadWeightsChunk = 21
    LoadWeightsEnd = 22
    LoadDone = 23
    LoadWeightsAck = 24

    CommitServing = 30

    InferRequest = 40
    InferResponse = 41

    HeartbeatRequest = 50
    HeartbeatReply = 51

    ErrorReport = 60


class TensorKind(enum.IntEnum):
    WUp = 0
    WGate = 1
    WDown = 2
    WUpScale = 3
    WGateScale = 4
    WDownScale = 5


class ActivationDType(enum.IntEnum):
    Unknown = 0
    FP16 = 1
    BF16 = 2


class HeaderDict(TypedDict):
    magic: int
    version: int
    msg_type: MsgType
    request_id: int
    body_len: int


def pack_header(msg_type: MsgType, request_id: int, body_len: int) -> bytes:
    if request_id < 0 or request_id > 0xFFFFFFFF:
        raise ValueError(f"request_id out of range: {request_id}")
    if body_len < 0 or body_len > 0xFFFFFFFF:
        raise ValueError(f"body_len out of range: {body_len}")

    return HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(msg_type),
        request_id,
        body_len,
    )


def unpack_header(data: bytes) -> HeaderDict:
    if len(data) != HEADER_SIZE:
        raise ProtocolError(
            f"bad header size: got {len(data)} bytes, expected {HEADER_SIZE}"
        )

    magic, version, msg_type_raw, request_id, body_len = HEADER_STRUCT.unpack(data)

    try:
        msg_type = MsgType(msg_type_raw)
    except ValueError as exc:
        raise ProtocolError(f"unknown msg_type: {msg_type_raw}") from exc

    return {
        "magic": magic,
        "version": version,
        "msg_type": msg_type,
        "request_id": request_id,
        "body_len": body_len,
    }


def check_magic_version(header: HeaderDict) -> None:
    if header["magic"] != MAGIC:
        raise ProtocolError(
            f"bad magic: got {header['magic']:#x}, expected {MAGIC:#x}"
        )
    if header["version"] != VERSION:
        raise ProtocolError(
            f"bad version: got {header['version']}, expected {VERSION}"
        )


def pack_u32(x: int) -> bytes:
    if x < 0 or x > 0xFFFFFFFF:
        raise ValueError(f"u32 out of range: {x}")
    return U32.pack(x)


def pack_i32(x: int) -> bytes:
    if x < -0x80000000 or x > 0x7FFFFFFF:
        raise ValueError(f"i32 out of range: {x}")
    return I32.pack(x)


def pack_u64(x: int) -> bytes:
    if x < 0 or x > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"u64 out of range: {x}")
    return U64.pack(x)


def unpack_u32(buf: bytes, offset: int) -> Tuple[int, int]:
    if offset + 4 > len(buf):
        raise ProtocolError("buffer too short while reading u32")
    return U32.unpack_from(buf, offset)[0], offset + 4


def unpack_i32(buf: bytes, offset: int) -> Tuple[int, int]:
    if offset + 4 > len(buf):
        raise ProtocolError("buffer too short while reading i32")
    return I32.unpack_from(buf, offset)[0], offset + 4


def unpack_u64(buf: bytes, offset: int) -> Tuple[int, int]:
    if offset + 8 > len(buf):
        raise ProtocolError("buffer too short while reading u64")
    return U64.unpack_from(buf, offset)[0], offset + 8


def pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    if len(raw) > 0xFFFFFFFF:
        raise ValueError("string too long for u32 length prefix")
    return pack_u32(len(raw)) + raw


def unpack_string(buf: bytes, offset: int) -> Tuple[str, int]:
    n, offset = unpack_u32(buf, offset)
    if n > MAX_STRING_BYTES:
        raise ProtocolError(f"string too long: {n} bytes")
    if offset + n > len(buf):
        raise ProtocolError("buffer too short while reading string")
    raw = buf[offset : offset + n]
    return raw.decode("utf-8"), offset + n


def decode_inventory_reply(body: bytes):
    offset = 0

    node_id, offset = unpack_string(body, offset)
    node_status, offset = unpack_u32(body, offset)
    num_gpus, offset = unpack_u32(body, offset)

    if num_gpus > MAX_GPUS_PER_NODE:
        raise ProtocolError(f"unreasonable num_gpus: {num_gpus}")

    gpus = []
    for _ in range(num_gpus):
        worker_id, offset = unpack_i32(body, offset)
        gpu_name, offset = unpack_string(body, offset)
        total_mem_bytes, offset = unpack_u64(body, offset)
        free_mem_bytes, offset = unpack_u64(body, offset)
        worker_port, offset = unpack_u32(body, offset)
        gpu_status, offset = unpack_u32(body, offset)

        gpu_vendor, offset = unpack_u32(body, offset)
        capability_flags, offset = unpack_u32(body, offset)
        arch_name, offset = unpack_string(body, offset)

        gpus.append(
            {
                "worker_id": worker_id,
                "gpu_name": gpu_name,
                "total_mem_bytes": total_mem_bytes,
                "free_mem_bytes": free_mem_bytes,
                "worker_port": worker_port,
                "gpu_status": gpu_status,
                "gpu_vendor": gpu_vendor,
                "capability_flags": capability_flags,
                "arch_name": arch_name,
            }
        )

    if offset != len(body):
        raise ProtocolError(
            f"inventory reply has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "node_id": node_id,
        "node_status": node_status,
        "num_gpus": num_gpus,
        "gpus": gpus,
    }


def encode_placement_plan(assignments):
    body = bytearray()
    body += pack_u32(len(assignments))
    for a in assignments:
        expert_id = int(a["expert_id"])
        worker_id = int(a["worker_id"])
        body += pack_i32(expert_id)
        body += pack_i32(worker_id)
    return bytes(body)


def decode_placement_plan(body: bytes):
    offset = 0
    num_assignments, offset = unpack_u32(body, offset)

    assignments = []
    for _ in range(num_assignments):
        expert_id, offset = unpack_i32(body, offset)
        worker_id, offset = unpack_i32(body, offset)
        assignments.append(
            {
                "expert_id": expert_id,
                "worker_id": worker_id,
            }
        )

    if offset != len(body):
        raise ProtocolError(
            f"placement plan has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return assignments


def encode_load_weights_begin(msg):
    meta = msg.get("meta", {})
    shape = meta.get("shape", [])
    dtype = str(meta.get("dtype", ""))
    row_block = int(meta.get("row_block", 0))
    col_block = int(meta.get("col_block", 0))

    if not dtype:
        raise ValueError("meta.dtype must be non-empty")
    if row_block <= 0 or col_block <= 0:
        raise ValueError(
            f"meta.row_block/meta.col_block must be > 0, got {row_block}/{col_block}"
        )

    body = bytearray()
    body += pack_i32(int(msg["expert_id"]))
    body += pack_i32(int(msg["worker_id"]))
    body += pack_i32(int(msg["tensor_kind"]))
    body += pack_u64(int(msg["total_bytes"]))

    body += pack_u32(len(shape))
    for d in shape:
        d = int(d)
        if d < 0:
            raise ValueError(f"shape dim must be >= 0, got {d}")
        body += pack_u64(d)

    body += pack_string(dtype)
    body += pack_u32(row_block)
    body += pack_u32(col_block)
    return bytes(body)


def decode_load_weights_begin(body: bytes):
    offset = 0
    expert_id, offset = unpack_i32(body, offset)
    worker_id, offset = unpack_i32(body, offset)

    tensor_kind_raw, offset = unpack_i32(body, offset)
    try:
        tensor_kind = TensorKind(tensor_kind_raw)
    except ValueError as exc:
        raise ProtocolError(f"invalid tensor_kind: {tensor_kind_raw}") from exc

    total_bytes, offset = unpack_u64(body, offset)

    ndim, offset = unpack_u32(body, offset)
    if ndim > 16:
        raise ProtocolError(f"unreasonable ndim: {ndim}")

    shape = []
    for _ in range(ndim):
        d, offset = unpack_u64(body, offset)
        shape.append(d)

    dtype, offset = unpack_string(body, offset)
    if not dtype:
        raise ProtocolError("load_weights_begin has empty dtype")

    row_block, offset = unpack_u32(body, offset)
    col_block, offset = unpack_u32(body, offset)
    if row_block == 0 or col_block == 0:
        raise ProtocolError(
            f"load_weights_begin has invalid block size {row_block}x{col_block}"
        )

    if offset != len(body):
        raise ProtocolError(
            f"load_weights_begin has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "expert_id": expert_id,
        "worker_id": worker_id,
        "tensor_kind": tensor_kind,
        "total_bytes": total_bytes,
        "meta": {
            "shape": shape,
            "dtype": dtype,
            "row_block": row_block,
            "col_block": col_block,
        },
    }


def encode_load_weights_chunk(msg):
    chunk_data = msg["chunk_data"]
    if not isinstance(chunk_data, (bytes, bytearray)):
        raise ValueError("chunk_data must be bytes-like")

    body = bytearray()
    body += pack_i32(int(msg["expert_id"]))
    body += pack_i32(int(msg["worker_id"]))
    body += pack_i32(int(msg["tensor_kind"]))
    body += pack_u64(int(msg["chunk_offset"]))
    body += pack_u32(len(chunk_data))
    body += chunk_data
    return bytes(body)


def decode_load_weights_chunk(body: bytes):
    offset = 0
    expert_id, offset = unpack_i32(body, offset)
    worker_id, offset = unpack_i32(body, offset)

    tensor_kind_raw, offset = unpack_i32(body, offset)
    try:
        tensor_kind = TensorKind(tensor_kind_raw)
    except ValueError as exc:
        raise ProtocolError(f"invalid tensor_kind: {tensor_kind_raw}") from exc

    chunk_offset, offset = unpack_u64(body, offset)
    chunk_size, offset = unpack_u32(body, offset)

    if offset + chunk_size > len(body):
        raise ProtocolError("buffer too short while reading chunk_data")
    chunk_data = body[offset : offset + chunk_size]
    offset += chunk_size

    if offset != len(body):
        raise ProtocolError(
            f"load_weights_chunk has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "expert_id": expert_id,
        "worker_id": worker_id,
        "tensor_kind": tensor_kind,
        "chunk_offset": chunk_offset,
        "chunk_data": chunk_data,
    }


def encode_load_weights_end(msg):
    body = bytearray()
    body += pack_i32(int(msg["expert_id"]))
    body += pack_i32(int(msg["worker_id"]))
    body += pack_i32(int(msg["tensor_kind"]))
    return bytes(body)


def decode_load_weights_end(body: bytes):
    offset = 0
    expert_id, offset = unpack_i32(body, offset)
    worker_id, offset = unpack_i32(body, offset)

    tensor_kind_raw, offset = unpack_i32(body, offset)
    try:
        tensor_kind = TensorKind(tensor_kind_raw)
    except ValueError as exc:
        raise ProtocolError(f"invalid tensor_kind: {tensor_kind_raw}") from exc

    if offset != len(body):
        raise ProtocolError(
            f"load_weights_end has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "expert_id": expert_id,
        "worker_id": worker_id,
        "tensor_kind": tensor_kind,
    }


def encode_infer_request(msg):
    activation = msg["activation"]
    if not isinstance(activation, (bytes, bytearray)):
        raise ValueError("activation must be bytes-like")

    body = bytearray()
    body += pack_i32(int(msg["expert_id"]))
    body += pack_i32(int(msg["batch_size"]))
    body += pack_i32(int(msg["hidden_dim"]))
    body += pack_u32(int(msg["input_dtype"]))
    body += pack_u32(int(msg["output_dtype"]))
    body += pack_u32(len(activation))
    body += activation
    return bytes(body)


def decode_infer_request(body: bytes):
    offset = 0
    expert_id, offset = unpack_i32(body, offset)
    batch_size, offset = unpack_i32(body, offset)
    hidden_dim, offset = unpack_i32(body, offset)
    input_dtype, offset = unpack_u32(body, offset)
    output_dtype, offset = unpack_u32(body, offset)
    activation_nbytes, offset = unpack_u32(body, offset)

    if offset + activation_nbytes > len(body):
        raise ProtocolError("buffer too short while reading activation")
    activation = body[offset : offset + activation_nbytes]
    offset += activation_nbytes

    if offset != len(body):
        raise ProtocolError(
            f"infer_request has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "expert_id": expert_id,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "activation": activation,
    }

def encode_infer_response(msg):
    output = msg["output"]
    if not isinstance(output, (bytes, bytearray)):
        raise ValueError("output must be bytes-like")

    body = bytearray()
    body += pack_i32(int(msg["status_code"]))
    body += pack_i32(int(msg["batch_size"]))
    body += pack_i32(int(msg["hidden_dim"]))
    body += pack_u32(int(msg["output_dtype"]))
    body += pack_u32(len(output))
    body += output
    return bytes(body)


def decode_infer_response(body: bytes):
    offset = 0
    status_code, offset = unpack_i32(body, offset)
    batch_size, offset = unpack_i32(body, offset)
    hidden_dim, offset = unpack_i32(body, offset)
    output_dtype, offset = unpack_u32(body, offset)
    output_nbytes, offset = unpack_u32(body, offset)

    if offset + output_nbytes > len(body):
        raise ProtocolError("buffer too short while reading output")
    output = body[offset : offset + output_nbytes]
    offset += output_nbytes

    if offset != len(body):
        raise ProtocolError(
            f"infer_response has trailing bytes: parsed {offset}, total {len(body)}"
        )

    return {
        "status_code": status_code,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "output_dtype": output_dtype,
        "output": output,
    }
