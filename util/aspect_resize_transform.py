import numpy as np
import PIL
from torchvision.transforms.transforms import _setup_size
import torch
from PIL import Image


class AspectResize(torch.nn.Module):
    """
   Resize image while keeping the aspect ratio.
   Extra parts will be covered with 255(white) color value
   """
    
    def __init__(self, size, background=255):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.background = background
    
    @staticmethod
    def fit_image_to_canvas(image: Image, canvas_width, canvas_height, background=255) -> Image:
        # Get the dimensions of the image
        image_width, image_height = image.size
        
        # Calculate the aspect ratio of the image
        image_aspect_ratio = image_width / float(image_height)
        
        # Calculate the aspect ratio of the canvas
        canvas_aspect_ratio = canvas_width / float(canvas_height)
        
        # Calculate the new dimensions of the image to fit the canvas
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = canvas_height * image_aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / image_aspect_ratio
        
        # Resize the image to the new dimensions
        image = image.resize((int(new_width), int(new_height)), PIL.Image.BICUBIC)
        
        # Create a blank canvas of the specified size
        canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
        canvas[:, :, :] = background
        
        # Calculate the position to paste the resized image on the canvas
        x = int((canvas_width - new_width) / 2)
        y = int((canvas_height - new_height) / 2)
        
        # Paste the resized image onto the canvas
        canvas[y:y + int(new_height), x:x + int(new_width)] = np.array(image)
        
        return PIL.Image.fromarray(canvas)
    
    def forward(self, image: Image) -> Image:
        image = self.fit_image_to_canvas(image, self.size[0], self.size[1], self.background)
        return image
