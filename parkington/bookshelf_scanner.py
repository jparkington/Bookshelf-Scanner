import cv2
import logging
import numpy as np
import pytesseract

from dataclasses import dataclass
from pathlib     import Path
from typing      import Any

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = 'bookshelf_scanner.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Class --------------------

@dataclass
class ProcessingStep:
    """
    Represents a processing step in the image processing pipeline.

    Attributes:
        name       (str)                  : Name of the processing step.
        toggle_key (str)                  : Key to toggle this processing step.
        parameters (list[dict[str, Any]]) : List of parameter dictionaries.
        is_enabled (bool)                 : Whether the step is enabled (default: False).
    """
    name       : str
    toggle_key : str
    parameters : list[dict[str, Any]]
    is_enabled : bool = False

    def toggle(self) -> str:
        """
        Toggles the 'is_enabled' state of the processing step.

        Returns:
            str: 'reprocess' to indicate that the image needs to be reprocessed.
        """
        previous_state  = self.is_enabled
        self.is_enabled = not self.is_enabled
        logger.info(
            f"Toggled '{self.name}' from "
            f"{'On' if previous_state else 'Off'} to "
            f"{'On' if self.is_enabled else 'Off'}"
        )
        return 'reprocess'

    def adjust_param(self, key: int) -> str:
        """
        Adjusts the parameter value based on the key pressed.

        Args:
            key (int) : ASCII value of the key pressed.

        Returns:
            str: 'reprocess' if parameter was adjusted and reprocessing is needed, None otherwise.
        """
        for param in self.parameters:

            if key == ord(param['increase_key']):
                old_value      = param['value']
                param['value'] = min(param['value'] + param['step'], param['max'])
                logger.info(
                    f"Increased '{param['display_name']}' from {old_value} to {param['value']}"
                )
                return 'reprocess'
            
            elif key == ord(param['decrease_key']):
                old_value      = param['value']
                param['value'] = max(param['value'] - param['step'], param['min'])
                logger.info(
                    f"Decreased '{param['display_name']}' from {old_value} to {param['value']}"
                )
                return 'reprocess'
            
        return None

# -------------------- Utility Functions --------------------

def find_images_directory() -> Path:
    """
    Searches for the 'images' directory in the parent directories.
    """
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        images_dir = current_dir / 'images'
        if images_dir.is_dir():
            return images_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Could not find 'images' directory in parent directories")

def get_image_files() -> list[Path]:
    """
    Retrieves a sorted list of image files from the 'images' directory.
    """
    images_dir       = find_images_directory()
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'
    }

    image_files = sorted(
        [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix in image_extensions
        ]
    )

    if not image_files:
        raise FileNotFoundError("No image files found in images directory")

    return image_files

def ensure_odd(value: int) -> int:
    """
    Ensures that the given integer value is odd.
    """
    return value | 1

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given file path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def extract_params(steps: list[ProcessingStep]) -> dict[str, Any]:
    """
    Extracts parameters from processing steps into a dictionary.
    """
    params = {
        param['name']: param['value']
        for step in steps
        for param in step.parameters
    }

    params.update({
        f"use_{step.name.lower().replace(' ', '_')}": step.is_enabled
        for step in steps
    })

    return params

# -------------------- Initialization Function --------------------

def initialize_steps(params_override: dict[str, Any] = None) -> list[ProcessingStep]:
    """
    Initializes processing steps with default parameters or overrides.

    Args:
        params_override (dict[str, Any], optional) : Parameters to override defaults.

    Returns:
        list[ProcessingStep]: List of initialized processing steps.
    """
    steps = []
    step_definitions = [
        {
            'name': 'Shadow Removal',
            'parameters': [
                {
                    'name'         : 'shadow_kernel_size',
                    'display_name' : 'Shadow Kernel Size',
                    'value'        : 15,
                    'min'          : 3,
                    'max'          : 21,
                    'step'         : 2,
                    'increase_key' : 'K',
                },
                {
                    'name'         : 'shadow_median_blur',
                    'display_name' : 'Shadow Median Blur',
                    'value'        : 21,
                    'min'          : 3,
                    'max'          : 99,
                    'step'         : 2,
                    'increase_key' : 'B',
                },
            ],
        },
        {
            'name': 'Gaussian Blur',
            'parameters': [
                {
                    'name'         : 'gaussian_kernel_size',
                    'display_name' : 'Gaussian Kernel Size',
                    'value'        : 5,
                    'min'          : 3,
                    'max'          : 21,
                    'step'         : 2,
                    'increase_key' : 'G',
                },
            ],
        },
        {
            'name': 'Color CLAHE',
            'parameters': [
                {
                    'name'         : 'clahe_clip_limit',
                    'display_name' : 'CLAHE Clip Limit',
                    'value'        : 2.0,
                    'min'          : 0.5,
                    'max'          : 10.0,
                    'step'         : 0.5,
                    'increase_key' : 'H',
                },
            ],
        },
        {
            'name': 'Edge Detection',
            'parameters': [
                {
                    'name'         : 'canny_threshold1',
                    'display_name' : 'Canny Threshold1',
                    'value'        : 50,
                    'min'          : 0,
                    'max'          : 255,
                    'step'         : 5,
                    'increase_key' : 'T',
                },
                {
                    'name'         : 'canny_threshold2',
                    'display_name' : 'Canny Threshold2',
                    'value'        : 150,
                    'min'          : 0,
                    'max'          : 255,
                    'step'         : 5,
                    'increase_key' : 'Y',
                },
            ],
        },
        {
            'name': 'Adaptive Thresholding',
            'parameters': [
                {
                    'name'         : 'adaptive_block_size',
                    'display_name' : 'Block Size',
                    'value'        : 11,
                    'min'          : 3,
                    'max'          : 99,
                    'step'         : 2,
                    'increase_key' : 'A',
                },
                {
                    'name'         : 'adaptive_c',
                    'display_name' : 'C Value',
                    'value'        : 2,
                    'min'          : -10,
                    'max'          : 10,
                    'step'         : 1,
                    'increase_key' : 'C',
                },
            ],
        },
        {
            'name': 'Morphology',
            'parameters': [
                {
                    'name'         : 'morph_kernel_size',
                    'display_name' : 'Morphology Kernel Size',
                    'value'        : 3,
                    'min'          : 1,
                    'max'          : 21,
                    'step'         : 2,
                    'increase_key' : 'M',
                },
                {
                    'name'         : 'erosion_iterations',
                    'display_name' : 'Erosion Iterations',
                    'value'        : 1,
                    'min'          : 0,
                    'max'          : 10,
                    'step'         : 1,
                    'increase_key' : 'E',
                },
                {
                    'name'         : 'dilation_iterations',
                    'display_name' : 'Dilation Iterations',
                    'value'        : 1,
                    'min'          : 0,
                    'max'          : 10,
                    'step'         : 1,
                    'increase_key' : 'D',
                },
            ],
        },
        {
            'name': 'Contour Adjustments',
            'parameters': [
                {
                    'name'         : 'min_contour_area',
                    'display_name' : 'Min Contour Area',
                    'value'        : 1000,
                    'min'          : 0,
                    'max'          : 10000,
                    'step'         : 250,
                    'increase_key' : 'N',
                },
                {
                    'name'         : 'max_contours',
                    'display_name' : 'Max Contours',
                    'value'        : 50,
                    'min'          : 0,
                    'max'          : 100,
                    'step'         : 5,
                    'increase_key' : 'X',
                },
            ],
        },
        {
            'name': 'Contour Approximation',
            'parameters': [],
        },
        {
            'name': 'OCR Settings',
            'parameters': [
                {
                    'name'         : 'oem',
                    'display_name' : 'OCR Engine Mode',
                    'value'        : 3,
                    'min'          : 0,
                    'max'          : 3,
                    'step'         : 1,
                    'increase_key' : 'O',
                },
                {
                    'name'         : 'psm',
                    'display_name' : 'Page Segmentation Mode',
                    'value'        : 6,
                    'min'          : 0,
                    'max'          : 13,
                    'step'         : 1,
                    'increase_key' : 'R',
                },
                {
                    'name'         : 'ocr_confidence_threshold',
                    'display_name' : 'OCR Confidence Threshold',
                    'value'        : 50,
                    'min'          : 0,
                    'max'          : 100,
                    'step'         : 5,
                    'increase_key' : 'F',
                },
            ],
        },
    ]

    # Initialize steps with automated toggle keys and decrease keys
    for index, step_def in enumerate(step_definitions):
        toggle_key = str(index + 1)
        for param in step_def['parameters']:
            param['decrease_key'] = param['increase_key'].lower()
        steps.append(ProcessingStep(
            name       = step_def['name'],
            toggle_key = toggle_key,
            parameters = step_def['parameters']
        ))

    # Apply any parameter overrides
    if params_override:
        for step in steps:
            toggle_key = f"use_{step.name.lower().replace(' ', '_')}"
            if toggle_key in params_override:
                step.is_enabled = params_override[toggle_key]
            for param in step.parameters:
                if param['name'] in params_override:
                    param['value'] = params_override[param['name']]

    return steps

# -------------------- Image Processing Functions --------------------

def process_image(
    image : np.ndarray,
    **params
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Processes the image according to the parameters provided.

    Args:
        image (np.ndarray) : Original image to process.
        **params           : Arbitrary keyword arguments containing processing parameters.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
            Processed color image, grayscale image, binary image, and list of contours.
    """
    processed = image.copy()

    # Shadow Removal
    if params.get('use_shadow_removal'):
        k_size    = ensure_odd(int(params['shadow_kernel_size']))
        blur_size = ensure_odd(int(params['shadow_median_blur']))
        kernel    = np.ones((k_size, k_size), np.uint8)
        channels  = list(cv2.split(processed))  # Split image into channels for processing

        for i, channel in enumerate(channels):
            dilated     = cv2.dilate(channel, kernel)
            background  = cv2.medianBlur(dilated, blur_size)
            difference  = 255 - cv2.absdiff(channel, background)
            channels[i] = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

        processed = cv2.merge(channels)  # Merge channels back into a color image

    # Gaussian Blur
    if params.get('use_gaussian_blur'):
        size      = ensure_odd(int(params['gaussian_kernel_size']))
        processed = cv2.GaussianBlur(processed, (size, size), 0)

    # Color CLAHE
    if params.get('use_color_clahe'):
        lab          = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        clahe        = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'])
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        processed    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to Grayscale
    grayscale = 255 - cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Edge Detection
    if params.get('use_edge_detection'):
        edges = cv2.Canny(
            image      = grayscale,
            threshold1 = params['canny_threshold1'],
            threshold2 = params['canny_threshold2']
        )
        grayscale = cv2.bitwise_or(grayscale, edges)

    # Adaptive Thresholding
    if params.get('use_adaptive_thresholding'):
        b_size = ensure_odd(int(params['adaptive_block_size']))
        binary = cv2.adaptiveThreshold(
            src            = grayscale,
            maxValue       = 255,
            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType  = cv2.THRESH_BINARY_INV,
            blockSize      = b_size,
            C              = params['adaptive_c']
        )

    else:
        _, binary = cv2.threshold(
            src    = grayscale,
            thresh = 0,
            maxval = 255,
            type   = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Morphological Transformations
    if params.get('use_morphology'):
        k_size = ensure_odd(int(params['morph_kernel_size']))
        kernel = np.ones((k_size, k_size), np.uint8)
        binary = cv2.erode(
            src        = binary,
            kernel     = kernel,
            iterations = int(params['erosion_iterations'])
        )
        binary = cv2.dilate(
            src        = binary,
            kernel     = kernel,
            iterations = int(params['dilation_iterations'])
        )

    # Find Contours
    contours, _ = cv2.findContours(
        image   = binary,
        mode    = cv2.RETR_EXTERNAL,
        method  = cv2.CHAIN_APPROX_SIMPLE
    )

    # Contour Adjustments
    if params.get('use_contour_adjustments'):
        min_area   = params['min_contour_area']
        image_area = image.shape[0] * image.shape[1]
        contours   = [
            contour for contour in contours
            if min_area <= cv2.contourArea(contour) <= 0.9 * image_area
        ]

        max_contours = int(params['max_contours'])
        contours     = contours[:max_contours]

    # Contour Approximation using minAreaRect
    if params.get('use_contour_approximation'):
        contours = [
            cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
            for contour in contours
        ]

    return processed, grayscale, binary, contours

def ocr_spine(spine_image: np.ndarray, **params) -> str:
    """
    Performs OCR on a given spine image using Tesseract.

    Args:
        spine_image (np.ndarray) : The image of the book spine to perform OCR on.
        **params                 : Arbitrary keyword arguments containing OCR parameters.

    Returns:
        str: Extracted text from the spine image.
    """
    config = f"--oem {int(params['oem'])} --psm {int(params['psm'])}"

    try:
        max_ocr_image_size = params.get('max_ocr_image_size', 1000)
        height, width      = spine_image.shape[:2]
        scaling_factor     = min(1.0, max_ocr_image_size / max(height, width))
        if scaling_factor < 1.0:
            spine_image = cv2.resize(
                spine_image,
                (int(width * scaling_factor), int(height * scaling_factor)),
                interpolation=cv2.INTER_AREA
            )

        data = pytesseract.image_to_data(
            spine_image, 
            lang        = 'eng', 
            config      = config, 
            output_type = pytesseract.Output.DICT
        )
        text = ''
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf >= params.get('ocr_confidence_threshold', 50):
                text += data['text'][i] + ' '
        return text.strip()
    
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ''

def draw_contours_and_text(
    image       : np.ndarray,
    ocr_image   : np.ndarray,
    contours    : list[np.ndarray],
    params      : dict[str, Any]
) -> np.ndarray:
    """
    Draws contours and recognized text on the image.

    Args:
        image       (np.ndarray)       : Original image (used for annotations).
        ocr_image   (np.ndarray)       : Image to use for OCR.
        contours    (list[np.ndarray]) : List of contours to draw.
        params      (dict[str, Any])   : Parameters including whether to perform OCR.

    Returns:
        np.ndarray: Annotated image with contours and text.
    """
    annotated     = image.copy()
    contour_color = (180, 0, 180)

    for contour in contours:
        cv2.drawContours(
            image      = annotated,
            contours   = [contour],
            contourIdx = -1,
            color      = contour_color,
            thickness  = 4
        )

        if params.get('use_ocr_settings'):
            x, y, w, h = cv2.boundingRect(contour)
            ocr_region = ocr_image[y:y+h, x:x+w]
            ocr_text   = ocr_spine(ocr_region, **params)

            if ocr_text:
                (text_width, text_height), _ = cv2.getTextSize(
                    text      = ocr_text,
                    fontFace  = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 0.6,
                    thickness = 2
                )
                text_x = x + (w - text_width)  // 2
                text_y = y + (h + text_height) // 2

                # Draw background rectangle for text
                cv2.rectangle(
                    img       = annotated,
                    pt1       = (text_x - 5, text_y - text_height - 5),
                    pt2       = (text_x + text_width + 5, text_y + 5),
                    color     = (0, 0, 0),
                    thickness = -1
                )

                # Put text over rectangle
                cv2.putText(
                    img       = annotated,
                    text      = ocr_text,
                    org       = (text_x, text_y),
                    fontFace  = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 0.6,
                    color     = (255, 255, 255),
                    thickness = 2
                )

    return annotated

# -------------------- UI and Visualization Functions --------------------

def create_sidebar(
    steps           : list[ProcessingStep],
    sidebar_width   : int,
    current_display : str,
    image_name      : str,
    window_height   : int
) -> np.ndarray:
    """
    Creates a sidebar image displaying the controls and current settings.

    Args:
        steps           (list[ProcessingStep]) : List of processing steps.
        sidebar_width   (int)                  : Width of the sidebar.
        current_display (str)                  : Name of the current display option.
        image_name      (str)                  : Name of the current image file.
        window_height   (int)                  : Height of the window.

    Returns:
        np.ndarray: Image of the sidebar.
    """
    sidebar        = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)
    scale_factor   = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    font_scale     = 0.8 * scale_factor
    line_height    = int(32 * scale_factor)
    y_position     = int(30 * scale_factor)
    font_thickness = max(1, int(scale_factor * 1.5))

    def put_text(
        text  : str,
        x     : int,
        y     : int,
        color : tuple[int, int, int] = (255, 255, 255),
        scale : float = 1.0
    ):
        cv2.putText(
            img       = sidebar,
            text      = text,
            org       = (x, y),
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = font_scale * scale,
            color     = color,
            thickness = font_thickness,
            lineType  = cv2.LINE_AA
        )

    # Display current view option and image name
    put_text(
        text  = f"[/] View Options for {current_display}",
        x     = 10,
        y     = y_position,
        color = (255, 255, 0),
        scale = 1.1
    )
    y_position += line_height
    put_text(
        text  = f"   [?] Current Image: {image_name}",
        x     = 10,
        y     = y_position,
        color = (255, 255, 0),
        scale = 0.9
    )
    y_position += int(line_height * 1.5)

    # Display controls for each processing step
    for step in steps:
        put_text(
            text  = f"[{step.toggle_key}] {step.name}: {'On' if step.is_enabled else 'Off'}",
            x     = 10,
            y     = y_position,
            scale = 1.1
        )
        y_position += line_height

        for param in step.parameters:
            value_str = (
                f"{param['value']:.3f}" if isinstance(param['value'], float)
                else str(param['value'])
            )
            key_text = f"[{param['decrease_key']} | {param['increase_key']}]"
            put_text(
                text  = f"   {key_text} {param['display_name']}: {value_str}",
                x     = 20,
                y     = y_position,
                color = (200, 200, 200),
                scale = 0.9
            )
            y_position += line_height

        y_position += line_height

    # Display quit option
    put_text(
        text = "[q] Quit",
        x    = 10,
        y    = window_height - int(line_height * 1.5)
    )

    return sidebar

# -------------------- Main Interactive Function --------------------

def interactive_experiment(
    image_files     : list[Path],
    params_override : dict[str, Any] = None
):
    """
    Runs the interactive experiment allowing the user to adjust image processing parameters.

    Args:
        image_files     (list[Path])               : List of image file paths to process.
        params_override (dict[str, Any], optional) : Parameters to override default settings.

    Raises:
        ValueError: If no image files are provided.
    """
    if not image_files:
        raise ValueError("No image files provided")

    # Initialize variables and UI elements
    window_name        = 'Bookshelf Scanner'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    current_image_idx  = 0
    display_options    = ['Processed Image', 'Binary Image', 'Annotated Image']
    current_display    = 0
    steps              = initialize_steps(params_override)
    original_image     = load_image(str(image_files[current_image_idx]))
    window_height      = max(original_image.shape[0], 800)
    scale_factor       = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    sample_text        = "   [X | Y] Very Long Parameter Name: 100000.000"
    (width, _), _      = cv2.getTextSize(
        text      = sample_text,
        fontFace  = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.8 * scale_factor,
        thickness = 1
    )
    sidebar_width      = max(400, int(width * 1.2))
    last_params        = None
    cached_results     = None

    # Resize window to accommodate sidebar
    cv2.resizeWindow(
        window_name,
        original_image.shape[1] + sidebar_width,
        window_height
    )

    # Define key actions
    key_actions = {
        ord('q') : 'quit',
        ord('/') : 'toggle_display',
        ord('?') : 'next_image',
    }

    def create_adjust_param_function(step, key_char):
        key_code = ord(key_char)
        return lambda: step.adjust_param(key_code)

    def create_toggle_function(step):
        return lambda: step.toggle()

    for step in steps:
        key_actions[ord(step.toggle_key)] = create_toggle_function(step)
        for param in step.parameters:
            key_actions[ord(param['increase_key'])] = create_adjust_param_function(step, param['increase_key'])
            key_actions[ord(param['decrease_key'])] = create_adjust_param_function(step, param['decrease_key'])

    # Main loop
    while True:
        current_image_path = image_files[current_image_idx]
        if last_params is None:
            # Load new image and adjust window height
            original_image = load_image(str(current_image_path))
            window_height  = max(original_image.shape[0], 800)
            cv2.resizeWindow(
                window_name,
                original_image.shape[1] + sidebar_width,
                window_height
            )

        current_params = extract_params(steps)
        if last_params != current_params:
            # Reprocess the image if parameters have changed
            processed_image, _, binary_image, contours = process_image(
                image = original_image,
                **current_params
            )
            annotated_image = draw_contours_and_text(
                image     = original_image,
                ocr_image = processed_image,
                contours  = contours,
                params    = current_params
            )
            cached_results = (processed_image, binary_image, annotated_image)
            last_params    = current_params.copy()

        display_image = cached_results[current_display]
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        display_image = cv2.resize(
            src   = display_image,
            dsize = (
                int(display_image.shape[1] * (window_height / display_image.shape[0])),
                window_height
            )
        )

        # Create sidebar and display images
        sidebar_image = create_sidebar(
            steps           = steps,
            sidebar_width   = sidebar_width,
            current_display = display_options[current_display],
            image_name      = current_image_path.name,
            window_height   = window_height
        )

        main_display = np.hstack([display_image, sidebar_image])
        cv2.imshow(window_name, main_display)

        key = cv2.waitKey(1) & 0xFF

        action = key_actions.get(key)
        if action:
            if action == 'quit':
                break

            elif action == 'toggle_display':
                current_display = (current_display + 1) % len(display_options)
                logger.info(f"Switched to view: {display_options[current_display]}")

            elif action == 'next_image':
                current_image_idx = (current_image_idx + 1) % len(image_files)
                last_params       = None
                logger.info(f"Switched to image: {image_files[current_image_idx].name}")

            else:
                result = action()
                if result == 'reprocess':
                    last_params = None

        for handler in logger.handlers:
            handler.flush()

    cv2.destroyAllWindows()

# -------------------- Entry Point --------------------

if __name__ == "__main__":
    params_override = {
        'use_shadow_removal'      : True,
        'shadow_kernel_size'      : 11,
        'use_contour_adjustments' : True,
        'min_contour_area'        : 1000
    }

    image_files = get_image_files()
    interactive_experiment(image_files, params_override)
