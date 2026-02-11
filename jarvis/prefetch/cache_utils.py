"""Shared serialization utilities for cache tiers."""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


def deserialize_value(data: bytes, value_type: str) -> Any:
    """Deserialize bytes back to a value.

    Args:
        data: Serialized bytes.
        value_type: Type string from serialization.

    Returns:
        Deserialized value.
    """
    if value_type == "numpy":
        shape_len, dtype_len = struct.unpack("II", data[:8])
        shape = struct.unpack(f"{shape_len // 4}I", data[8 : 8 + shape_len])
        dtype_str = data[8 + shape_len : 8 + shape_len + dtype_len].decode()
        buffer = data[8 + shape_len + dtype_len :]
        return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(shape)
    elif value_type == "json":
        return json.loads(data.decode())
    elif value_type == "str":
        return data.decode()
    elif value_type == "bytes":
        return data
    elif value_type == "pickle":
        logger.warning("Refusing to deserialize pickle data (security risk). Returning None.")
        return None
    else:
        logger.warning("Unknown value_type '%s', returning None.", value_type)
        return None
