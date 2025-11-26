import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
except ImportError:
    pe = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

try:
    from PIL import Image  # noqa: F401
except ImportError:
    Image = None  # type: ignore[assignment]

from vectorvfs.encoders import PerceptionEncoder


@unittest.skipIf(
    torch is None or pe is None or transforms is None or Image is None,
    "Perception encoder dependencies are not installed",
)
class TestPerceptionEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = Path(__file__).parent / "data"
        self.pe_encoder = PerceptionEncoder()

    def test_vision_encoder(self) -> None:
        cat_image = self.data_path / "cat.jpg"
        features = self.pe_encoder.encode_vision(cat_image)
        self.assertEqual(features.shape, torch.Size([1, 1024]))

    def test_text_encoder(self) -> None:
        features = self.pe_encoder.encode_text("a cat")
        self.assertEqual(features.shape, torch.Size([1, 1024]))
    
    def test_logit_scale(self) -> None:
        scale = self.pe_encoder.logit_scale()
        self.assertTrue(scale.item() <= 100.0)

    def test_similarity(self) -> None:
        cat_image = self.data_path / "cat.jpg"
        cat_features = self.pe_encoder.encode_vision(cat_image)

        dog_image = self.data_path / "dog.jpg"
        dog_features = self.pe_encoder.encode_vision(dog_image)

        vision_features = torch.cat([cat_features, dog_features])

        text_features = self.pe_encoder.encode_text("a cat")
        logit_scale = self.pe_encoder.logit_scale()

        with torch.inference_mode():
            text_probs = (logit_scale * vision_features @ text_features.T).softmax(dim=0)
        
        self.assertEqual(text_probs.argmax().item(), 0)


if __name__ == "__main__":
    unittest.main()
