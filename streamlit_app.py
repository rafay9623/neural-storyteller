import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
import time
from PIL import Image
import numpy as np
from torchvision import models, transforms


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

        # 4. Load ResNet50 for Feature Extraction
        resnet = models.resnet50(weights='DEFAULT')
        # Strip the last layer (classification layer) to get 2048 features
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.to(device)
        resnet.eval()

        # 5. Define Image Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return model, vocab, device, resnet, transform

    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def generate_caption(model, features, vocab, max_len=50, device='cuda'):
    """Generate caption from image features"""
    model.eval()
    with torch.no_grad():
        # Prepare initial state
        features = features.unsqueeze(0).to(device)
        h = model.encoder(features).unsqueeze(0)
        c = torch.zeros(1, 1, 512).to(device)
        
        # Start token
        start_id = vocab.stoi["<START>"]
        token = torch.LongTensor([[start_id]]).to(device)
        caption = [start_id]
        
        for _ in range(max_len - 1):
            embedded = model.decoder.embedding(token)
            embedded = model.decoder.dropout(embedded)
            lstm_out, (h, c) = model.decoder.lstm(embedded, (h, c))
            logits = model.decoder.fc(lstm_out[:, 0, :])
            next_id = logits.argmax(dim=1).item()
            caption.append(next_id)
            
            if next_id == vocab.stoi["<END>"]:
                break
            
            token = torch.LongTensor([[next_id]]).to(device)
    
    # Filter out special tokens
    ignore_tokens = {vocab.stoi["<START>"], vocab.stoi["<PAD>"], vocab.stoi["<END>"]}
    words = [vocab.itos[i] for i in caption if i not in ignore_tokens]
    
    return ' '.join(words) if words else "A scene captured by the model."

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
    model, vocab, device, resnet, transform = resources
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
        
        if st.button("Generate Narrative", use_container_width=True):
            with st.spinner("Analyzing image features..."):
                try:
                    # 1. Image to Tensor
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # 2. Extract Features using ResNet50
                    with torch.no_grad():
                        # Output shape from resnet is (1, 2048, 1, 1) -> flatten to (1, 2048)
                        features = resnet(img_tensor).view(1, -1)
                    
                    # 3. Generate Caption
                    start_time = time.time()
                    caption = generate_caption(model, features, vocab, max_len=max_length, device=device)
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
