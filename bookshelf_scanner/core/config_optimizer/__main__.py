"""
Entry point for running the parameter optimizer as a module.
"""

from bookshelf_scanner import ConfigOptimizer, TextExtractor

def main():

    extractor   = TextExtractor()
    optimizer   = ConfigOptimizer(extractor = extractor, output_images = True)
    image_files = TextExtractor.find_image_files(subdirectory = 'Books')

    optimizer.optimize(image_files)

if __name__ == "__main__":
    main()
