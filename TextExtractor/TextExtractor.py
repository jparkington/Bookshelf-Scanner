import cv2
import easyocr
import json
import logging
import numpy as np

from dataclasses import dataclass, field
from pathlib     import Path
from PIL         import Image, ImageDraw, ImageFont
from ruamel.yaml import YAML
from typing      import Any, Optional

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = 'TextExtractor.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    image_idx       : int                   = 0     # Index of the current image
    window_height   : int                   = 800   # Window height in pixels
    last_params     : Optional[dict]        = None  # Previous processing parameters
    annotations     : dict[str, np.ndarray] = field(default_factory = dict)  # Cached annotations
    ocr_results     : dict[str, list]       = field(default_factory = dict)  # OCR results for each image
    original_image  : Optional[np.ndarray]  = None  # The original image
    processed_image : Optional[np.ndarray]  = None  # The processed image
    image_name      : str                   = ''    # Name of the current image

    def next_image(self, total_images: int):
        """
        Cycle to the next image and reset image-related state.
        """
        self.image_idx = (self.image_idx + 1) % total_images
        self.original_image = None

    def reset_image_state(self):
        """
        Resets processing-related state variables.
        """
        self.processed_image = None
        self.last_params     = None
        self.annotations     = {}

@dataclass
class Parameter:
    name          : str         # Internal name of the parameter
    display_name  : str         # Name to display in the UI
    value         : Any         # Current value of the parameter
    increase_key  : str         # Key to increase the parameter
    min           : Any = None  # Minimum value of the parameter
    max           : Any = None  # Maximum value of the parameter
    step          : Any = None  # Step size for incrementing/decrementing the parameter

    @property
    def decrease_key(self) -> str:
        """
        Returns the key to decrease the parameter.
        """
        return self.increase_key.lower()
    
    @property
    def display_value(self) -> str:
        """
        Returns the value formatted as a string for display purposes.
        """
        return f"{self.value:.2f}" if isinstance(self.value, float) else str(self.value)

    def adjust_value(self, increase: bool):
        """
        Adjusts the parameter value based on whether it should be increased or decreased.

        Args:
            increase: True to increase the value, False to decrease.
        """
        old_value = self.value
        delta     = self.step if increase else -self.step
        new_value = self.value + delta

        if isinstance(self.value, float):
            self.value = round(max(self.min, min(new_value, self.max)), 2)
        else:
            self.value = max(self.min, min(new_value, self.max))
        
        return old_value  # Return old value for logging purposes

@dataclass
class ProcessingStep:
    name          : str              # Internal name of the processing step
    display_name  : str              # Name to display in the UI
    toggle_key    : str              # Key to toggle this processing step
    parameters    : list[Parameter]  # List of parameter instances
    is_enabled    : bool = False     # Whether the step is enabled (default: False)

    def adjust_param(self, key_char: str) -> Optional[str]:
        """
        Adjusts the parameter value based on the provided key character and returns an action message.

        Args:
            key_char: The character representing the key pressed.

        Returns:
            A string describing the action taken, or None if no action was taken.
        """
        for param in self.parameters:
            if key_char in (param.increase_key, param.decrease_key):
                increase    = key_char == param.increase_key
                old_value   = param.adjust_value(increase)
                action_type = 'Increased' if increase else 'Decreased'
                return f"{action_type} '{param.display_name}' from {old_value} to {param.value}"
        return None
    
    def toggle(self) -> str:
        """
        Toggles the 'is_enabled' state of the processing step and returns an action message.

        Returns:
            A string describing the action taken.
        """
        self.is_enabled = not self.is_enabled
        return f"Toggled '{self.display_name}' to {'On' if self.is_enabled else 'Off'}"

# -------------------- Utility Functions --------------------

def find_image_files(
    target_subdirectory : str = 'images/books', 
    start_directory     : Optional[Path] = None
    ) -> list[Path]:
    """
    Retrieves a sorted list of image files from the specified target_subdirectory.

    Args:
        target_subdirectory : The subdirectory to search for images.
        start_directory     : The starting directory for the search.

    Returns:
        A list of image file paths.

    Raises:
        FileNotFoundError: If no image files are found.
    """
    start_directory    = start_directory or Path(__file__).resolve().parent
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    image_files = next(
        (
            sorted(
                file for file in (directory / target_subdirectory).rglob('*')
                if file.is_file() and file.suffix.lower() in allowed_extensions
            )
            for directory in [start_directory, *start_directory.parents]
            if (directory / target_subdirectory).is_dir()
        ),
        None
    )

    if image_files:
        return image_files

    raise FileNotFoundError(f"No image files found in '{target_subdirectory}' directory.")

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified file path.

    Args:
        image_path: The path to the image file.

    Returns:
        The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be found or loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def center_image_in_square(image: np.ndarray) -> np.ndarray:
    """
    Centers the image in a square canvas with sides equal to the longest side of the image.

    The background is set to black.

    Args:
        image: The image to center.

    Returns:
        The image centered in a square canvas.
    """
    height, width = image.shape[:2]
    max_side      = max(height, width)
    canvas        = np.zeros((max_side, max_side, 3), dtype = np.uint8)
    y_offset      = (max_side - height) // 2
    x_offset      = (max_side - width)  // 2

    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image
    return canvas

# -------------------- Initialization Function --------------------

def initialize_steps(params_override: dict = None) -> list[ProcessingStep]:
    """
    Initializes processing steps with default parameters or overrides.

    Args:
        params_override: Optional; a dictionary of parameters to override default settings.

    Returns:
        A list of initialized ProcessingStep instances.

    Raises:
        FileNotFoundError : If the configuration file is not found.
        Exception         : If there is an error parsing the configuration file.
    """
    params_file = Path(__file__).resolve().parent / 'params.yml'
    yaml        = YAML(typ = 'safe')

    try:
        with params_file.open('r') as f:
            step_definitions = yaml.load(f)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {params_file}")
        raise

    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

    steps = [
        ProcessingStep(
            name         = step_def['name'],
            display_name = step_def['display_name'],
            toggle_key   = str(index + 1),
            parameters   = [Parameter(**param_def) for param_def in step_def.get('parameters', [])]
        )
        for index, step_def in enumerate(step_definitions)
    ]

    # Apply parameter overrides if provided
    if params_override:
        step_map  = {f"use_{step.name}": step for step in steps}
        param_map = {param.name: param for step in steps for param in step.parameters}

        for key, value in params_override.items():
            if key in step_map:
                step_map[key].is_enabled = value
            elif key in param_map:
                param_map[key].value = value

    return steps

# -------------------- Image Processing Functions --------------------

def process_image(
    image : np.ndarray,
    steps : list[ProcessingStep]
) -> np.ndarray:
    """
    Processes the image according to the enabled processing steps.

    Args:
        image : The original image to process.
        steps : List of ProcessingStep instances.

    Returns:
        The processed image.
    """
    processed_image = image.copy()

    for step in steps:
        if step.is_enabled:
            # Extract parameters for the current step
            params = {param.name: param.value for param in step.parameters}

            if step.name == 'image_rotation':
                angle = params['rotation_angle']
                if angle % 360 != 0:
                    k = int(angle / 90) % 4
                    processed_image = np.rot90(processed_image, k)

            elif step.name == 'brightness_adjustment':
                value = params['brightness_value']
                hsv_image          = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
                hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], value)
                processed_image    = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            elif step.name == 'contrast_adjustment':
                alpha = params['contrast_value']
                processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=0)

            elif step.name == 'shadow_removal':
                kernel_size = int(params['shadow_kernel_size']) | 1  # Ensure odd
                median_blur = int(params['shadow_median_blur']) | 1  # Ensure odd
                kernel      = np.ones((kernel_size, kernel_size), np.uint8)
                channels    = list(cv2.split(processed_image))  # Convert tuple to list for modification

                # Process each channel separately
                for i in range(len(channels)):
                    dilated    = cv2.dilate(channels[i], kernel)
                    bg_image   = cv2.medianBlur(dilated, median_blur)
                    diff_image = 255 - cv2.absdiff(channels[i], bg_image)
                    channels[i] = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)

                processed_image = cv2.merge(channels)

            elif step.name == 'gaussian_blur':
                ksize = int(params['gaussian_kernel_size']) | 1  # Ensure odd
                sigma = params['gaussian_sigma']
                processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), sigma)

            elif step.name == 'color_clahe':
                clip_limit = params['clahe_clip_limit']
                lab_image  = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
                clahe      = cv2.createCLAHE(clipLimit=clip_limit)

                # Apply CLAHE to the L-channel
                lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
                processed_image    = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return processed_image

# -------------------- Image Annotation Functions --------------------

def extract_text_from_image(
    image          : np.ndarray,
    reader         : easyocr.Reader,
    min_confidence : float
) -> list:
    """
    Extracts text from a given image using EasyOCR.

    Args:
        image          : The image to perform OCR on.
        reader         : An instance of easyocr.Reader.
        min_confidence : Minimum confidence threshold for OCR results.

    Returns:
        List of tuples containing OCR results.
    """
    try:
        ocr_results = reader.readtext(
            image[..., ::-1],  # Convert BGR to RGB
            decoder       = 'greedy',
            rotation_info = [90, 180, 270]
        )

        # Filter results by confidence
        return [result for result in ocr_results if result[2] >= min_confidence]

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return []

def draw_rotated_text(
    source_image : np.ndarray,
    text         : str, 
    position     : tuple[int, int],
    angle        : float,
    scale        : float = 1.0,
    opacity      : float = 0.75
) -> np.ndarray:
    """
    Draws bold white text with semi-transparent background, keeping boxes within image bounds.
    
    Args:
        source_image : Image to draw on
        text         : Text to draw
        position     : (x,y) center position for text box
        angle        : Rotation angle in degrees
        scale        : Text size multiplier (1.0 = default size)
        opacity      : Background opacity (0.0 to 1.0)
    """
    # Setup drawing layers
    pil_image  = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw       = ImageDraw.Draw(text_layer)
    
    # Add thin spaces between characters
    THIN_SPACE = ' '
    spaced_text = THIN_SPACE.join(text)
    
    # Get text dimensions
    font        = ImageFont.load_default()
    bbox        = draw.textbbox((0, 0), spaced_text, font=font)
    text_width  = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate box padding based on scale
    box_padding  = max(4, int(20 * scale * 0.2))
    total_width  = text_width  + box_padding * 2
    total_height = text_height + box_padding * 2
    
    # Constrain position to keep text box within image bounds
    center_x, center_y = position
    bounded_x = max(total_width  // 2, min(source_image.shape[1] - total_width  // 2, center_x))
    bounded_y = max(total_height // 2, min(source_image.shape[0] - total_height // 2, center_y))
    
    # Calculate text position relative to center
    text_x = bounded_x - text_width  // 2
    text_y = bounded_y - text_height // 2
    
    # Draw background
    background_bbox = (
        text_x - box_padding,
        text_y - box_padding,
        text_x + text_width + box_padding,
        text_y + text_height + box_padding
    )
    draw.rectangle(background_bbox, fill = (0, 0, 0, int(255 * opacity)))
    
    # Draw text multiple times for boldness
    for offset_x, offset_y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        draw.text(
            (text_x + offset_x, text_y + offset_y),
            spaced_text,
            font  = font,
            fill  = (255, 255, 255, 255),
            angle = angle
        )
    
    # Composite text layer onto source image
    annotated_image = Image.alpha_composite(pil_image.convert('RGBA'), text_layer)
    return cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGBA2BGR)

def annotate_image_with_text(
    image   : np.ndarray,
    reader  : easyocr.Reader,
    params  : dict,
    state   : DisplayState
) -> np.ndarray:
    """
    Annotates the image with recognized text.

    Args:
        image  : Image to annotate.
        reader : EasyOCR reader instance.
        params : Dictionary of processing parameters.
        state  : DisplayState instance to store OCR results.

    Returns:
        Annotated image.
    """
    cache_key = f"{state.image_name}_{hash(frozenset(params.items()))}"

    if cache_key in state.annotations:
        return state.annotations[cache_key]

    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    ocr_results     = extract_text_from_image(
        image          = annotated_image,
        reader         = reader,
        min_confidence = params.get('ocr_confidence_threshold', 0.3)
    )

    image_text_results = []

    for bounding_box, text, confidence in ocr_results:
        coordinates = np.array(bounding_box).astype(int)

        # Draw bounding box around detected text
        cv2.polylines(annotated_image, [coordinates], True, (0, 255, 0), 2)
        logger.info(f"OCR Text: '{text}' with confidence {confidence:.2f}")

        x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
        top_edge_length    = np.hypot(x_coords[1] - x_coords[0], y_coords[1] - y_coords[0])
        right_edge_length  = np.hypot(x_coords[2] - x_coords[1], y_coords[2] - y_coords[1])

        # Determine text position and angle based on orientation
        if right_edge_length > top_edge_length:
            angle         = 270
            text_position = (int(np.max(x_coords) + 10), int(np.mean(y_coords)))
        else:
            angle         = 0
            text_position = (int(np.mean(x_coords)), int(np.min(y_coords) - 10))

        # Draw the rotated text annotation
        annotated_image = draw_rotated_text(
            source_image = annotated_image,
            text         = f"{text} ({confidence * 100:.1f}%)",
            position     = text_position,
            angle        = angle
        )

        # Collect OCR results for output
        image_text_results.append([text, confidence])

    # Store OCR results in state
    state.ocr_results[state.image_name] = image_text_results
    state.annotations[cache_key]        = annotated_image

    return annotated_image

# -------------------- UI and Visualization Functions --------------------

def generate_sidebar_content(
    steps      : list[ProcessingStep],
    image_name : str
) -> list[tuple[str, tuple[int, int, int], float]]:
    """
    Generates a list of sidebar lines with text content, colors, and scaling factors.

    Args:
        steps     : List of processing steps to display.
        image_name: Name of the current image file.

    Returns:
        List of tuples: (text content, RGB color, scale factor)
    """
    TEAL  = (255, 255, 0)
    WHITE = (255, 255, 255)
    GRAY  = (200, 200, 200)
    
    lines = [
        (f"[/] Current Image: {image_name}",  TEAL, 0.85),
        ("", WHITE, 1.0)  # Spacer
    ]
    
    for step in steps:
        status = 'On' if step.is_enabled else 'Off'
        lines.append((f"[{step.toggle_key}] {step.display_name}: {status}", WHITE, 1.0))

        for param in step.parameters:
            lines.append((
                f"   [{param.decrease_key} | {param.increase_key}] {param.display_name}: {param.display_value}",
                GRAY,
                0.85
            ))
        lines.append(("", WHITE, 1.0))  # Spacer
    
    lines.append(("[q] Quit", WHITE, 1.0))
    return lines

def render_sidebar(
    steps         : list[ProcessingStep],
    image_name    : str,
    window_height : int
) -> np.ndarray:
    """
    Renders the sidebar image with controls and settings.

    Args:
        steps         : List of processing steps to display.
        image_name    : Name of the current image file.
        window_height : Height of the window in pixels.

    Returns:
        Rendered sidebar image as a NumPy array.
    """
    lines             = generate_sidebar_content(steps, image_name)
    num_lines         = len(lines)
    margin            = int(0.05 * window_height)
    horizontal_margin = 20
    line_height       = max(20, min(int((window_height - 2 * margin) / num_lines), 40))
    font_scale        = line_height / 40
    font_thickness    = max(1, int(font_scale * 1.5))
    font              = cv2.FONT_HERSHEY_DUPLEX

    # Determine maximum text width
    max_text_width = max(
        cv2.getTextSize(text, font, font_scale * rel_scale, font_thickness)[0][0]
        for text, _, rel_scale in lines if text
    )

    sidebar_width = max_text_width + 2 * horizontal_margin
    sidebar       = np.zeros((window_height, sidebar_width, 3), dtype = np.uint8)
    y_position    = margin + line_height

    # Draw text lines onto the sidebar
    for text, color, rel_scale in lines:
        if text:
            cv2.putText(
                img       = sidebar,
                text      = text,
                org       = (horizontal_margin, y_position),
                fontFace  = font,
                fontScale = font_scale * rel_scale,
                color     = color,
                thickness = font_thickness,
                lineType  = cv2.LINE_AA
            )
        y_position += line_height

    return sidebar

# -------------------- Main Interactive Function --------------------

def interactive_experiment(
    image_files     : list[Path],
    params_override : dict = None,
    output_json     : bool = True,
    interactive_ui  : bool = True
):
    """
    Runs the interactive experiment allowing parameter adjustment and image processing.
    
    Args:
        image_files     : List of image file paths to process.
        params_override : Optional parameter overrides.
        output_json     : Whether to output OCR results to JSON file on exit.
        interactive_ui  : Whether to run the interactive UI.
    """
    if not image_files:
        raise ValueError("No image files provided")

    # Initialize components
    state  = DisplayState()
    steps  = initialize_steps(params_override)
    reader = easyocr.Reader(['en'], gpu = False)

    # Initialize UI if needed
    window_name = None
    if interactive_ui:
        window_name = 'Bookshelf Scanner'
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    try:
        while state.image_idx < len(image_files):
            # Load new image if needed
            if state.original_image is None:
                image_path           = image_files[state.image_idx]
                state.original_image = load_image(str(image_path))
                state.window_height  = max(state.original_image.shape[0], 800)
                state.image_name     = image_path.name
                state.reset_image_state()

            # Extract current parameters
            current_params = {
                param.name: param.value
                for step in steps if step.is_enabled
                for param in step.parameters
            }
            current_params.update({
                f"use_{step.name}": step.is_enabled for step in steps
            })

            # Process image if parameters have changed
            if state.last_params != current_params:
                state.processed_image = process_image(state.original_image, steps)
                state.last_params     = current_params.copy()
                state.annotations     = {}

            # Handle OCR processing
            ocr_enabled = any(step.is_enabled and step.name == 'ocr' for step in steps)
            display_image = state.processed_image

            if ocr_enabled:
                display_image = annotate_image_with_text(
                    image   = display_image,
                    reader  = reader,
                    params  = current_params,
                    state   = state
                )
            elif state.image_name not in state.ocr_results:
                # Extract text for JSON output even if not displaying
                ocr_results = extract_text_from_image(
                    image          = state.processed_image,
                    reader         = reader,
                    min_confidence = current_params.get('ocr_confidence_threshold', 0.3)
                )
                state.ocr_results[state.image_name] = [[text, confidence] for _, text, confidence in ocr_results]

            if not interactive_ui:
                # Process next image in headless mode
                state.image_idx += 1
                state.original_image = None
                continue

            # Prepare image for display
            display_image = center_image_in_square(display_image)
            if display_image.ndim == 2:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

            # Scale image to window height
            display_scale = state.window_height / display_image.shape[0]
            display_image = cv2.resize(
                display_image,
                (int(display_image.shape[1] * display_scale), state.window_height)
            )

            # Add sidebar and display
            sidebar_image   = render_sidebar(steps, state.image_name, state.window_height)
            combined_image = np.hstack([display_image, sidebar_image])
            cv2.imshow(window_name, combined_image)
            cv2.resizeWindow(window_name, combined_image.shape[1], combined_image.shape[0])

            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            char = chr(key)
            if char == 'q':
                break
            elif char == '/':
                state.next_image(len(image_files))
            else:
                # Handle parameter adjustments
                for step in steps:
                    if char == step.toggle_key:
                        logger.info(step.toggle())
                        state.last_params = None
                        break

                    action = step.adjust_param(char)
                    if action:
                        logger.info(action)
                        state.last_params = None
                        break

    finally:
        if interactive_ui:
            cv2.destroyAllWindows()

        if output_json and state.ocr_results:
            output_path = Path('ocr_results.json')

            with output_path.open('w', encoding = 'utf-8') as f:
                json.dump(state.ocr_results, f, ensure_ascii = False, indent = 4)
            logger.info(f"OCR results saved to {output_path}")

# -------------------- Entry Point --------------------

if __name__ == "__main__":
    image_files     = find_image_files('images/books')
    params_override = {
        'use_ocr'            : True,
        'use_shadow_removal' : True,
        'use_color_clahe'    : True,
        'use_image_rotation' : True,
        'rotation_angle'     : 90
    }

    interactive_experiment(
        image_files     = image_files,
        params_override = params_override,
        interactive_ui  = False
    )
