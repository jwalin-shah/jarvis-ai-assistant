"""Shared serialization utilities for cache tiers."""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Safety limits for deserialization
_MAX_DESERIALIZE_BYTES = 10 * 1024 * 1024  # 10MB
_MAX_JSON_NESTING_DEPTH = 20
_ALLOWED_NUMPY_DTYPES = frozenset({
    "float32", "float64", "int32", "int64", "uint8", "bool",
})


def serialize_value(value: Any) -> tuple[bytes, str]:
    """Serialize a value to bytes for cache storage.

    Args:
        value: Value to serialize.

    Returns:
        Tuple of (serialized bytes, type string).
    """
    if isinstance(value, np.ndarray):
        buffer = value.tobytes()
        shape_bytes = struct.pack(f"{len(value.shape)}I", *value.shape)
        dtype_bytes = str(value.dtype).encode()
        header = struct.pack("II", len(shape_bytes), len(dtype_bytes))
        return header + shape_bytes + dtype_bytes + buffer, "numpy"
    elif isinstance(value, (dict, list)):
        return json.dumps(value).encode(), "json"
    elif isinstance(value, str):
        return value.encode(), "str"
    elif isinstance(value, bytes):
        return value, "bytes"
    else:
        try:
            return json.dumps({"__repr__": repr(value)}).encode(), "json"
        except (TypeError, ValueError):
            return json.dumps({"__repr__": str(value)}).encode(), "json"


def _check_json_nesting_depth(raw: str, max_depth: int = _MAX_JSON_NESTING_DEPTH) -> bool:
    """Check if a JSON string exceeds the maximum nesting depth.

    Scans for bracket/brace depth without fully parsing. Returns True if safe.
    """
    depth = 0
    for ch in raw:
        if ch in ("{", "["):
            depth += 1
            if depth > max_depth:
                return False
        elif ch in ("}", "]"):
            depth -= 1
            if depth < 0:
                return False
    return True


def deserialize_value(data: bytes, value_type: str) -> Any:
    """Deserialize bytes back to a value.

    Args:
        data: Serialized bytes.
        value_type: Type string from serialization.

    Returns:
        Deserialized value for valid types (numpy, json, str, bytes).

    Raises:
        ValueError: If data exceeds the 10MB size limit, numpy dtype is not
            in the allowlist, numpy header is malformed/truncated, JSON nesting
            depth exceeds the limit, or value_type is pickle/unknown.
    """
    if len(data) > _MAX_DESERIALIZE_BYTES:
        msg = (
            f"Refusing to deserialize {len(data)} bytes (limit {_MAX_DESERIALIZE_BYTES})."
        )
        logger.warning(msg)
        raise ValueError(msg)

    if value_type == "numpy":
        if len(data) < 8:
            msg = f"NumPy data too short ({len(data)} bytes), need at least 8."
            logger.warning(msg)
            raise ValueError(msg)
        shape_len, dtype_len = struct.unpack("II", data[:8])
        # Validate dtype before unpacking the full buffer
        min_header = 8 + shape_len + dtype_len
        if min_header > len(data):
            msg = (
                f"NumPy header extends beyond data "
                f"(header {min_header} bytes, data {len(data)} bytes)."
            )
            logger.warning(msg)
            raise ValueError(msg)
        dtype_str = data[8 + shape_len : 8 + shape_len + dtype_len].decode()
        if dtype_str not in _ALLOWED_NUMPY_DTYPES:
            msg = (
                f"Refusing to deserialize numpy array with dtype '{dtype_str}' "
                f"(allowed: {', '.join(sorted(_ALLOWED_NUMPY_DTYPES))})."
            )
            logger.warning(msg)
            raise ValueError(msg)
        shape = struct.unpack(f"{shape_len // 4}I", data[8 : 8 + shape_len])
        buffer = data[8 + shape_len + dtype_len :]
        return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(shape)
    elif value_type == "json":
        raw = data.decode()
        if not _check_json_nesting_depth(raw):
            msg = (
                f"Refusing to deserialize JSON with nesting depth > "
                f"{_MAX_JSON_NESTING_DEPTH}."
            )
            logger.warning(msg)
            raise ValueError(msg)
        return json.loads(raw)
    elif value_type == "str":
        return data.decode()
    elif value_type == "bytes":
        return data
    elif value_type == "pickle":
        msg = "Refusing to deserialize pickle data (security risk)."
        logger.warning(msg)
        raise ValueError(msg)
    else:
        msg = f"Unknown value_type '{value_type}'."
        logger.warning(msg)
        raise ValueError(msg)
