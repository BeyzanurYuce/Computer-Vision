from PIL import Image
import numpy as np
import cv2


def gamma_correction(image, gamma=1.0):
    # Convert image to a NumPy array
    img_array = np.array(image)
    # Normalize pixel values
    normalized_image = img_array / 255.0
    result_img = np.power(normalized_image, 1 / gamma)
    # Denormalize the image
    result_img = (result_img * 255).astype(np.uint8)

    return Image.fromarray(result_img)


def prewitt_edge_detection(image):
    grayscale = image.convert('L')
    img_array = np.array(grayscale, dtype=np.float32)

    # Detect edges in horizontal direction
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    # Detect edges in vertical direction
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    edge_x = cv2.filter2D(img_array, -1, prewitt_x)
    edge_y = cv2.filter2D(img_array, -1, prewitt_y)

    result_img = Image.fromarray(np.uint8(np.sqrt(edge_x ** 2 + edge_y ** 2)))

    return result_img


def dct_transform(image):
    grayscale = image.convert('L')
    img_array = np.array(grayscale, dtype=float)

    result_img = cv2.dct(img_array)

    return Image.fromarray(np.uint8(result_img))

# Get the input image
input_image = Image.open('input.jpg')

# Apply the methods
gamma_corrected_image = gamma_correction(input_image, gamma=1.5)
edge_image = prewitt_edge_detection(input_image)
dct_image = dct_transform(input_image)

gamma_corrected_image.save('gamma_corrected.jpg')
edge_image.save('edge_detected.jpg')
dct_image.save('dct_transformed.jpg')
