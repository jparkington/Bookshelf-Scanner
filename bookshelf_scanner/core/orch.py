import cv2
from pathlib import Path
from bookshelf_scanner.core.text_extractor.extractor import TextExtractor
from bookshelf_scanner.core.book_segmenter.base import BookSegmenter
from rapidfuzz import process, fuzz
import duckdb

# -------------------- Workflow Orchestrator --------------------

def main():
    # Initialize classes for segmentation and text extraction
    segmenter = BookSegmenter()
    text_extractor = TextExtractor(headless=True)

    # Define the directories for input and master book list
    image_dir = Path("bookshelf_scanner/images/books")
    master_db_path = Path("bookshelf_scanner/data/books.duckdb")

    # Check if the image directory exists
    if not image_dir.exists():
        print(f"Image directory does not exist: {image_dir}")
        return

    # Check if the database file exists
    if not master_db_path.exists():
        print(f"Master database file does not exist: {master_db_path}")
        return

    # Loop through all the images in the image directory
    for image_path in image_dir.glob("*"):
        if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            continue  # Skip files that are not images

        print(f"Processing image: {image_path}")

        # Step 1: Segment the image into individual book regions
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image {image_path}, skipping.")
            continue

        # Get the segments from the image
        segments = segmenter.segment(image)

        # Check if any segments were generated
        if not segments:
            print(f"No segments found for image {image_path}, skipping.")
            continue

        # Step 2: Extract text from each segment using OCR
        ocr_results = []
        for i, segment in enumerate(segments):
            # Validate the segment before proceeding
            if isinstance(segment, dict) and segment.get('image') is not None and segment['image'].size != 0:
                print(f"Processing segment {i + 1}/{len(segments)}")
                text = text_extractor.extract_text_headless([segment['image']])
                if text and list(text.values())[0]:  # Check if there is any extracted text
                    extracted_text = list(text.values())[0][0][0]
                    ocr_results.append(extracted_text)
            else:
                print(f"Skipping segment {i + 1} - invalid or empty segment.")

        # Step 3: Fuzzy match the extracted text with the master book list
        if ocr_results:
            # Load the list of books from your master database
            conn = duckdb.connect(str(master_db_path))
            book_titles = conn.execute("SELECT title FROM books").fetchall()
            book_titles = [title[0] for title in book_titles]

            # Perform fuzzy matching for each extracted text
            for ocr_text in ocr_results:
                matches = process.extract(ocr_text, book_titles, scorer=fuzz.ratio, limit=3)
                print(f"Extracted text: {ocr_text}")
                print("Top matches:")
                for match in matches:
                    print(f"  - {match[0]} (Score: {match[1]})")

# -------------------- Main Entry Point --------------------

if __name__ == "__main__":
    main()
