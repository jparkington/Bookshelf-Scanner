import streamlit as st  
import random
from PIL import Image, ImageOps
import numpy as np
from bookshelf_scanner import BookSegmenter

st.set_page_config(layout="wide", page_title="Bookshelf Scanner")
st.title("Bookshelf Scanner")
file = st.file_uploader(type=["jpg", "png"], label="Upload an image")
        
if file is None:
    print("No file uploaded")
else:
    image = Image.open(file)
    image = ImageOps.exif_transpose(image)

    st.image(image, use_container_width=True)
    