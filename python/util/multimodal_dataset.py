import os
import pathlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import PIL
from torchvision.transforms.transforms import _setup_size


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


class MultimodalDataset(Dataset):
    """
    A multimodal dataset wrapper. The directory must contain the following sub-directories and files:
    data_root
        | - train
        |      | - JPEGImages
        |      |       | -  FILE01.jpeg
        |      |       | -  FILE02.jpeg
        |      |       | -  FILE03.jpeg
        |      | - Captions
        |              | -  FILE01.txt
        |              | -  FILE02.txt
        |              | -  FILE03.txt
        | - classes.txt
    """
    
    def __init__(self, dataset_path, classnames, doc_length, img_dim,
                 image_transform=None, caption_transform=None, target_transform=None,
                 split="train",
                 images_dir="JPEGImages", caption_dir="Captions", image_extension=".jpeg"):
        assert split in ("train", "text")
        self.dataset_path = pathlib.Path(dataset_path)
        self.images_dir = dataset_path / split / images_dir
        self.caption_dir = dataset_path / split / caption_dir
        # read class names
        self.classnames = classnames
        assert len(self.classnames) > 0, "A list of classnames is required"
        self.doc_length = doc_length
        # read images
        self.images = [os.path.join(self.images_dir, file) for file in os.listdir(self.images_dir) if
                       file.endswith(image_extension)]
        # read captions
        self.captions = [os.path.join(self.caption_dir, file) for file in os.listdir(self.caption_dir) if
                         file.endswith(".txt")]
        assert len(self.images) == len(self.captions), \
            "The count of images and captions must be same, got {}!={}".format(len(self.images), len(self.captions))
        if image_transform is None:
            image_transform = transforms.Compose([
                transforms.Resize(img_dim),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        if caption_transform is None:
            caption_transform = transforms.Compose([
                transforms.Lambda(self._basic_char_encoder),
                transforms.ToTensor(),
            ])
        
        if target_transform is None:
            target_transform = transforms.Compose([
                transforms.Lambda(self._class_encoder),
                transforms.ToTensor(),
            ])
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.target_transform = target_transform
        self.vocabulary = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.vocab_length = len(self.vocabulary)
    
    def _class_encoder(self, label):
        for cls_idx, lbl in enumerate(self.classnames):
            if label == lbl:
                return cls_idx
        return -1
    
    def _basic_char_encoder(self, caption):
        # lowercase
        caption = caption.lower()
        # truncate if the sentence is large
        if len(caption.split()) > self.doc_length:
            caption = "".join(caption.split()[:self.doc_length])
        char_encode = torch.zeros(self.doc_length, self.vocab_length)
        char_idx = [self.vocabulary.index(char) for char in caption.split()]
        for idx, char_idx in enumerate(char_idx):
            char_encode[idx, char_idx] = 1
        return char_encode
    
    def get_img(self, img_path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img = self.image_transform(img)
        return img
    
    def get_caption(self, caption_path) -> (str, int):
        with open(caption_path) as fp:
            captions = fp.readlines()
            captions_ix = random.randint(0, len(captions) - 1)
            caption = captions[captions_ix]
            class_id = None
            for label in self.classnames:
                if label in caption:
                    class_id = label
                    break
            caption = self.caption_transform(caption)
            class_id = self.target_transform(class_id)
            return caption, class_id
    
    def __getitem__(self, idx):
        image_file = self.images[idx]
        caption_file = self.captions[idx]
        image = self.get_img(image_file)
        caption, label = self.get_caption(caption_file)
        return caption, image, label
    
    def __len__(self):
        return len(self.images)
