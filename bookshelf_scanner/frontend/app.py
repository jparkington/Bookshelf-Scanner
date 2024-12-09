import streamlit as st  
import random
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from bookshelf_scanner import BookSegmenter

st.set_page_config(page_title="Bookshelf Scanner")

def draw_bounding_boxes(image, bboxes, confidences):
    draw = ImageDraw.Draw(image, "RGBA")
    for bbox, confidence in zip(bboxes, confidences):
        if confidence > 0.75:
            color = (0, 255, 0, 128)  # Green with 50% transparency
        elif confidence > 0.5:
            color = (255, 165, 0, 128)  # Orange with 50% transparency
        else:
            color = (255, 0, 0, 128)  # Red with 50% transparency
        draw.rectangle(bbox, outline="red", fill=color, width=3)
    return image

def main():
    segmenter = BookSegmenter()
    st.title("Bookshelf Scanner")
    file = st.sidebar.file_uploader(type=["jpg", "png"], label="Upload an image")

    if file is None:
        print("No file uploaded")
    else:
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        books, bboxes, confidences = segmenter(np.array(image))
        image_with_boxes = draw_bounding_boxes(image, bboxes, confidences)
        st.image(image_with_boxes, use_container_width=True)

if __name__ == "__main__":
    main()