from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

try:  # Optional dependency
    import torch
except ImportError as e:  # pragma: no cover - exercised through runtime checks
    torch = None  # type: ignore[assignment]
    _torch_import_error = e
else:
    _torch_import_error = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch import Tensor


def _require_torch() -> None:
    """Raise an informative error when torch is missing at runtime."""

    if torch is None:
        message = (
            "VFSStore requires the optional 'torch' dependency to serialize "
            "embeddings. Install torch to enable tensor persistence."
        )
        if _torch_import_error is not None:
            raise ImportError(message) from _torch_import_error
        raise ImportError(message)


class XAttrFile:
    def __init__(self, file_path: Path) -> None:
        """
        Initialize an XAttrFile for managing extended attributes on a file.
        :param file_path: Path to the target file.
        """
        self.file_path = file_path

    def list(self) -> list[str]:
        """
        List all extended attribute names set on the file.
        :return: List of attribute names.
        """
        return os.listxattr(str(self.file_path))

    def write(self, key: str, data: bytes) -> None:
        """
        Write or replace an extended attribute on the file.
        :param key: Name of the attribute (e.g., 'user.comment').
        :param data: Bytes to store in the attribute.
        """
        os.setxattr(str(self.file_path), key, data)

    def read(self, key: str) -> bytes:
        """
        Read the value of an extended attribute from the file.
        :param key: Name of the attribute to read.
        :return: Bytes stored in the attribute.
        """
        return os.getxattr(str(self.file_path), key)

    def remove(self, key: str) -> None:
        """
        Remove an extended attribute from the file.
        :param key: Name of the attribute to remove.
        """
        os.removexattr(str(self.file_path), key)


class VFSStore:
    def __init__(self, xattrfile: XAttrFile) -> None:
        self.xattrfile = xattrfile

    def _tensor_to_bytes(self, tensor: "Tensor") -> bytes:
        _require_torch()
        assert torch is not None  # for type checkers
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def _bytes_to_tensor(self, b: bytes, map_location=None) -> "Tensor":
        _require_torch()
        assert torch is not None  # for type checkers
        buffer = io.BytesIO(b)
        return torch.load(buffer, map_location=map_location, weights_only=True)

    def write_tensor(self, tensor: "Tensor") -> int:
        btensor = self._tensor_to_bytes(tensor)
        self.xattrfile.write("user.vectorvfs", btensor)
        return len(btensor)

    def read_tensor(self) -> "Tensor":
        btensor = self.xattrfile.read("user.vectorvfs")
        tensor = self._bytes_to_tensor(btensor)
        return tensor
