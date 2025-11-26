import pytest


def test_require_dependencies_provides_descriptive_error(monkeypatch):
    import vectorvfs.encoders as encoders

    # Simulate the optional stack being absent and make sure the error message is
    # explicit enough to guide the user toward the missing extras.
    monkeypatch.setattr(encoders, "torch", None)
    monkeypatch.setattr(encoders, "pe", None)
    monkeypatch.setattr(encoders, "transforms", None)
    monkeypatch.setattr(encoders, "Image", None)
    monkeypatch.setattr(encoders, "_torch_import_error", ImportError("torch"))
    monkeypatch.setattr(encoders, "_encoder_import_error", ImportError("pe"))
    monkeypatch.setattr(encoders, "_pillow_import_error", ImportError("pillow"))

    with pytest.raises(ImportError, match="PerceptionEncoder requires optional dependencies") as excinfo:
        encoders._require_dependencies()

    message = str(excinfo.value)
    # Ensure the guidance lists every missing optional component so users know
    # exactly which extras to install when the perception stack is unavailable.
    assert "torch" in message
    assert "core.vision_encoder" in message
    assert "Pillow" in message


def test_require_torch_provides_descriptive_error(monkeypatch):
    import vectorvfs.vfsstore as vfsstore

    # Force a missing torch dependency and ensure the guidance is actionable.
    monkeypatch.setattr(vfsstore, "torch", None)
    monkeypatch.setattr(vfsstore, "_torch_import_error", ImportError("torch"))

    with pytest.raises(ImportError, match="VFSStore requires the optional 'torch' dependency"):
        vfsstore._require_torch()
