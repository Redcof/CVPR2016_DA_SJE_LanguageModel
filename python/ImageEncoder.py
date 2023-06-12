import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, input_size, hid_size, noop):
        super(ImageEncoder, self).__init__()
        self.noop = noop

        if self.noop == 1:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(input_size, hid_size)

    def forward(self, x):
        return self.encoder(x)
