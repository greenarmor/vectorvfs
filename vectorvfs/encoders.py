from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

try:  # Optional heavy dependencies
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
except ImportError as e:  # pragma: no cover - exercised through runtime checks
    pe = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    _encoder_import_error = e
else:
    _encoder_import_error = None

try:  # Optional heavy dependency
    import torch
except ImportError as e:  # pragma: no cover - exercised through runtime checks
    torch = None  # type: ignore[assignment]
    _torch_import_error = e
else:
    _torch_import_error = None

try:  # Optional dependency
    from PIL import Image
except ImportError as e:  # pragma: no cover - exercised through runtime checks
    Image = None  # type: ignore[assignment]
    _pillow_import_error = e
else:
    _pillow_import_error = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch import Tensor


def _require_dependencies() -> None:
    """Ensure optional runtime dependencies are available.

    Raises a descriptive :class:`ImportError` when the optional encoder stack is
    not installed so callers receive an actionable message instead of a vague
    failure at import time.
    """

    missing = []
    if torch is None:
        missing.append("torch")
    if pe is None or transforms is None:
        missing.append("core.vision_encoder")
    if Image is None:
        missing.append("Pillow")

    if missing:
        help_text = (
            "PerceptionEncoder requires optional dependencies: "
            + ", ".join(missing)
            + ". Install them to enable vision/text encoding."
        )
        # Prefer to re-raise the first captured import error for context.
        if torch is None and _torch_import_error is not None:
            raise ImportError(help_text) from _torch_import_error
        if (pe is None or transforms is None) and _encoder_import_error is not None:
            raise ImportError(help_text) from _encoder_import_error
        if Image is None and _pillow_import_error is not None:
            raise ImportError(help_text) from _pillow_import_error
        raise ImportError(help_text)


class DualEncoder(ABC):
    """
    Abstract base class for dual encoders that provide methods to encode vision and text inputs into embeddings.

    This interface defines methods for encoding images and text into a shared embedding space,
    as well as obtaining the scaling factor for logits in similarity computation.
    """
    @abstractmethod
    def encode_vision(self, file: Path) -> "Tensor":
        """
        Encode an image file into a tensor representation.

        :param file: Path to the image file to encode.
        :return: Tensor representing the encoded image.
        """
        ...
    
    @abstractmethod
    def encode_text(self, text: str) -> "Tensor":
        """
        Encode a text string into a tensor representation.

        :param text: The text to encode.
        :return: Tensor representing the encoded text.
        """
        ...

    @abstractmethod
    def logit_scale(self) -> "Tensor":
        """
        Get the scale factor applied to logits for similarity computation.

        :return: Logit scale tensor.
        """
        ...


class PerceptionEncoder(DualEncoder):
    """
    Perception encoder implementation using a CLIP-based model to encode images and text.

    :param model_name: Name of the CLIP model configuration to load (default: "PE-Core-L14-336").
    """
    def __init__(self, model_name: str = "PE-Core-L14-336") -> None:
        _require_dependencies()

        assert torch is not None  # for type checkers
        assert pe is not None and transforms is not None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    def encode_vision(self, file: Path) -> "Tensor":
        """
        Encode an image file into a tensor of image features using the perception model.

        :param file: Path to the image file to encode.
        :return: Tensor of encoded image features.
        """
        pil_image = Image.open(file)
        image = self.preprocess(pil_image).unsqueeze(0)
        image = image.to(self.device)
        with torch.inference_mode():
            image_features, _, _ = self.model(image, None)
        return image_features

    def encode_text(self, text: str) -> "Tensor":
        """
        Encode a text string into a tensor of text features using the perception model.

        :param text: The text to encode.
        :return: Tensor of encoded text features.
        """
        tokenized_text = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            _, text_features, _ = self.model(None, tokenized_text)
        return text_features

    def logit_scale(self) -> "Tensor":
        """
        Get the exponential of the model's logit scale parameter for similarity computation.

        :return: Logit scale tensor.
        """
        return self.model.logit_scale.exp()
