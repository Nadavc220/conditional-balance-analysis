import numpy as np
import cv2
from PIL import Image


CANNY_VALUES = 10, 100

def prepare_canny_image(image_pil, canny_values=CANNY_VALUES):
    image_pil = image_pil.convert("RGB")
    image = np.array(image_pil)

    image = cv2.Canny(image, *canny_values)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image