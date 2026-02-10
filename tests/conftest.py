"""Patch Streamlit so we can import streamlit_app and test core logic."""
import sys
from unittest.mock import MagicMock

# Mock Streamlit before any import of streamlit_app
st = MagicMock()
st.cache_resource = lambda f: f  # no caching in tests
st.error = lambda msg: None
st.set_page_config = lambda **kwargs: None
st.title = lambda x: None
st.markdown = lambda x: None
st.sidebar = MagicMock()
st.sidebar.slider = MagicMock(side_effect=[50, 5])  # max_length, beam_size
st.columns = lambda spec: [MagicMock() for _ in range(len(spec) if hasattr(spec, "__len__") else 2)]
st.subheader = lambda x: None
st.file_uploader = lambda *a, **k: None
st.image = lambda *a, **k: None
st.button = lambda *a, **k: False
st.spinner = lambda x: MagicMock()
st.success = lambda x: None
st.info = lambda x: None
st.expander = lambda x: MagicMock()
st.write = lambda x: None
st.stop = lambda: None

sys.modules["streamlit"] = st
