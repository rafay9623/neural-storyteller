import re
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import time
from PIL import Image
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


class Vocabulary:
    """
    Minimal Vocabulary class used only for loading the pickled vocab.
    The original training script pickled an instance of this class, so we
    recreate it here with the attributes the app expects: `stoi`, `itos`
    and a working `__len__`.
    """
    def __init__(self):
        # These will be populated from the pickle; defaults are placeholders.
        self.stoi = {}
        self.itos = []

    def __len__(self):
        return len(self.itos)


class VocabUnpickler(pickle.Unpickler):
    """
    Custom unpickler that maps any pickled `Vocabulary` reference
    to the local `Vocabulary` class defined above, regardless of
    which module name was used when it was originally pickled.
    """
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(path: str) -> Vocabulary:
    """Load the vocabulary using the custom unpickler."""
    with open(path, "rb") as f:
        return VocabUnpickler(f).load()


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, features):
        # features shape: (batch, 2048)
        x = self.fc(features)
        x = self.bn(x)
        return self.relu(x)

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_size=512, num_layers=1, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, captions, hidden_state, cell_state):
        x = self.embedding(captions)
        x = self.dropout(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        return self.fc(self.dropout(lstm_out)), (hidden_state, cell_state)

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, images_features, captions):
        hidden_state = self.encoder(images_features)
        batch_size = hidden_state.size(0)
        cell_state = torch.zeros(self.decoder.num_layers, batch_size,
                                self.decoder.hidden_size).to(self.device)
        hidden_state = hidden_state.unsqueeze(0).expand(self.decoder.num_layers, -1, -1)
        outputs, _ = self.decoder(captions, hidden_state, cell_state)
        return outputs

# ============================================================================
# LOAD MODELS & VOCAB
# ============================================================================
@st.cache_resource
def load_all_resources():
    """Load model, feature extractor, and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Load Vocabulary
        if not os.path.exists('vocab.pkl'):
            st.error("Missing 'vocab.pkl' file!")
            return None
        vocab = load_vocab('vocab.pkl')
        
        # 2. Build the Captioning Model
        encoder = ImageEncoder(feature_dim=2048, hidden_size=512)
        decoder = CaptionDecoder(vocab_size=len(vocab), embedding_dim=512, 
                                hidden_size=512, num_layers=1, dropout=0.5)
        model = Seq2SeqModel(encoder, decoder, device).to(device)
        
        # 3. Load Trained Weights
        if os.path.exists('best_model.pt'):
            model.load_state_dict(torch.load('best_model.pt', map_location=device))
        model.eval()

        # 4. Load ResNet50 for Feature Extraction + classifier for correction
        resnet_full = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet = nn.Sequential(*list(resnet_full.children())[:-1])
        resnet.to(device)
        resnet.eval()
        imagenet_fc = resnet_full.fc.to(device)
        imagenet_fc.eval()
        _meta = getattr(ResNet50_Weights.DEFAULT, "meta", None)
        imagenet_categories = (_meta.get("categories") or []) if isinstance(_meta, dict) else []

        # 5. Define Image Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return model, vocab, device, resnet, transform, imagenet_fc, imagenet_categories

    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def generate_caption_beam(
    model,
    features,
    vocab,
    max_len: int = 50,
    device: str = "cuda",
    beam_size: int = 5,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.2,
):
    """
    Generate caption using beam search with repetition penalty for better quality.
    """
    model.eval()
    ignore_tokens = {
        vocab.stoi["<START>"],
        vocab.stoi["<PAD>"],
        vocab.stoi["<END>"],
    }
    with torch.no_grad():
        features = features.to(device)
        h0 = model.encoder(features).unsqueeze(0)
        c0 = torch.zeros(1, 1, 512, device=device)

        start_id = vocab.stoi["<START>"]
        end_id = vocab.stoi["<END>"]

        beams = [(0.0, [start_id], h0, c0)]

        for _ in range(max_len - 1):
            new_beams = []
            for log_p, seq, h, c in beams:
                if seq[-1] == end_id:
                    new_beams.append((log_p, seq, h, c))
                    continue

                token = torch.LongTensor([[seq[-1]]]).to(device)
                embedded = model.decoder.embedding(token)
                embedded = model.decoder.dropout(embedded)
                lstm_out, (h_new, c_new) = model.decoder.lstm(embedded, (h, c))
                logits = model.decoder.fc(lstm_out[:, 0, :]).squeeze(0).float()

                log_probs = F.log_softmax(logits, dim=-1).clone()
                # Penalize tokens that already appeared to reduce repetition
                for tid in set(seq):
                    if tid in ignore_tokens:
                        continue
                    if log_probs[tid].item() > -1e9:
                        log_probs[tid] -= repetition_penalty

                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    token_id = top_indices[k].item()
                    score = log_p + top_log_probs[k].item()
                    new_seq = seq + [token_id]
                    new_beams.append((score, new_seq, h_new, c_new))

            beams = sorted(
                new_beams,
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                reverse=True,
            )[:beam_size]

        best_seq = beams[0][1]

    words = [vocab.itos[i] for i in best_seq if i not in ignore_tokens]
    return " ".join(words) if words else "A scene captured by the model."


# ImageNet dog breeds: don't replace "dog" with a breed name (e.g. kuvasz) so we keep simpler "dog"
_IMAGENET_DOG_BREEDS = {
    "affenpinscher", "kuvasz", "golden retriever", "labrador retriever", "corgi",
    "beagle", "poodle", "bulldog", "chihuahua", "german shepherd", "dalmatian",
    "husky", "malamute", "terrier", "boxer", "pug", "rottweiler", "doberman",
    "mastiff", "schnauzer", "shih tzu", "samoyed", "collie", "shepherd", "retriever",
}


def correct_caption_with_imagenet(caption: str, imagenet_label: str) -> str:
    """
    When the caption mentions a different *type* of object than ImageNet (e.g. dog vs goldfish),
    replace the wrong subject with the ImageNet label. Skip when both are same category
    (e.g. "dog" vs "kuvasz") so we keep the simpler word.
    """
    if not caption or not imagenet_label or len(imagenet_label) < 2:
        return caption
    label_lower = imagenet_label.lower()
    caption_lower = caption.lower()
    wrong_subjects = [
        "dog", "dogs", "cat", "cats", "bird", "birds", "horse", "horses",
        "cow", "cows", "sheep", "car", "cars", "person", "people", "man", "woman",
        "child", "boy", "girl",
    ]
    for w in wrong_subjects:
        if w not in caption_lower or label_lower in w or w in label_lower:
            continue
        # Don't replace "dog" with a dog breed (e.g. kuvasz) — keep "dog"
        if w in ("dog", "dogs") and label_lower in _IMAGENET_DOG_BREEDS:
            continue
        # Same for cat breeds if we had them
        if w in ("cat", "cats") and any(x in label_lower for x in ("tabby", "persian", "siamese", "egyptian", "tiger cat")):
            continue
        pattern = r"\b" + re.escape(w) + r"\b"
        match = re.search(pattern, caption, re.IGNORECASE)
        if not match:
            continue
        repl = imagenet_label.capitalize() if match.group(0)[0].isupper() else imagenet_label
        caption = caption[: match.start()] + repl + caption[match.end() :]
        break
    return caption


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Neural Storyteller", page_icon="📖", layout="wide")

st.title("Neural Storyteller")
st.markdown("### AI-Powered Image Captioning with Seq2Seq & ResNet50")
st.markdown("---")

# Resources
resources = load_all_resources()
if resources:
    model, vocab, device, resnet, transform, imagenet_fc, imagenet_categories = resources
else:
    st.error("Failed to load resources. Check logs.")
    st.stop()

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖼️ Upload Image")
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📝 Generated Caption")
    
    if uploaded_file:
        max_length = st.sidebar.slider("Max Sentence Length", 10, 100, 50)
        beam_size = st.sidebar.slider(
            "Caption quality (beam size)",
            min_value=3,
            max_value=7,
            value=5,
            help="Higher = better captions, slower.",
        )
        if st.button("Generate Narrative", use_container_width=True):
            with st.spinner("Analyzing image features..."):
                try:
                    # 1. Image to Tensor
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # 2. Extract Features using ResNet50
                    with torch.no_grad():
                        features = resnet(img_tensor).view(1, -1)
                        # ImageNet top-1 for subject correction
                        if len(imagenet_categories) > 0:
                            logits = imagenet_fc(features)
                            top1_idx = logits.argmax(dim=1).item()
                            imagenet_label = imagenet_categories[min(top1_idx, len(imagenet_categories) - 1)]
                        else:
                            imagenet_label = ""
                    
                    # 3. Generate Caption (beam search)
                    start_time = time.time()
                    caption = generate_caption_beam(
                        model,
                        features,
                        vocab,
                        max_len=max_length,
                        device=device,
                        beam_size=beam_size,
                        length_penalty=0.6,
                        repetition_penalty=1.2,
                    )
                    # 4. Correct subject when caption disagrees with ImageNet (e.g. dog vs goldfish)
                    if imagenet_label:
                        caption = correct_caption_with_imagenet(caption, imagenet_label)
                    duration = time.time() - start_time
                    
                    st.success(f"Processing complete in {duration:.2f}s")
                    st.markdown(f"> **AI Description:** {caption}")
                    
                except Exception as e:
                    st.error(f"Inference Error: {e}")
    else:
        st.info("Please upload an image to begin.")

# Architecture Visualization


# Footer Info
st.markdown("---")
with st.expander("System Details"):
    st.write(f"**Running on:** {device}")
    st.write(f"**Vocab Size:** {len(vocab)} tokens")
    st.write("**Backbone:** Pre-trained ResNet50 (ImageNet)")
