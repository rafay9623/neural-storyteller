"""Tests for Neural Storyteller app: vocab load, model load, caption generation."""
import os
import sys

# Run from project root so vocab.pkl and best_model.pt are found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Patch streamlit before importing streamlit_app
with open(os.path.join(os.path.dirname(__file__), "conftest.py")) as f:
    exec(f.read())

import torch


def test_vocab_exists():
    assert os.path.isfile(os.path.join(ROOT, "vocab.pkl")), "vocab.pkl missing"


def test_model_file_exists():
    assert os.path.isfile(os.path.join(ROOT, "best_model.pt")), "best_model.pt missing"


def test_load_vocab():
    import streamlit_app as app

    vocab = app.load_vocab(os.path.join(ROOT, "vocab.pkl"))
    assert vocab is not None
    assert len(vocab) > 0
    assert hasattr(vocab, "stoi") and hasattr(vocab, "itos")
    assert "<START>" in vocab.stoi and "<END>" in vocab.stoi


def test_load_all_resources():
    import streamlit_app as app

    resources = app.load_all_resources()
    assert resources is not None, "load_all_resources returned None"
    model, vocab, device, resnet, transform, imagenet_fc, imagenet_categories = resources
    assert model is not None
    assert vocab is not None
    assert len(vocab) > 0
    assert resnet is not None
    assert transform is not None


def test_generate_caption_beam():
    import streamlit_app as app

    resources = app.load_all_resources()
    assert resources is not None
    model, vocab, device, resnet, transform, _, _ = resources

    # Fake (1, 2048) features
    features = torch.randn(1, 2048)
    caption = app.generate_caption_beam(
        model, features, vocab, max_len=20, device=device, beam_size=3
    )
    assert isinstance(caption, str)
    assert len(caption) > 0
    assert caption != "A scene captured by the model." or "scene" in caption.lower()


def test_resnet_features_shape():
    import streamlit_app as app

    resources = app.load_all_resources()
    assert resources is not None
    model, vocab, device, resnet, transform, _, _ = resources

    # Dummy image (1, 3, 224, 224)
    img = torch.rand(1, 3, 224, 224).to(device)
    with torch.no_grad():
        features = resnet(img).view(1, -1)
    assert features.shape == (1, 2048)


def test_encoder_accepts_features():
    import streamlit_app as app

    resources = app.load_all_resources()
    assert resources is not None
    model, vocab, device, resnet, transform, _, _ = resources

    features = torch.randn(1, 2048).to(device)
    with torch.no_grad():
        out = model.encoder(features)
    assert out.shape == (1, 512)


if __name__ == "__main__":
    # Run without pytest: python tests/test_app.py
    import traceback
    tests = [
        test_vocab_exists,
        test_model_file_exists,
        test_load_vocab,
        test_load_all_resources,
        test_generate_caption_beam,
        test_resnet_features_shape,
        test_encoder_accepts_features,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
