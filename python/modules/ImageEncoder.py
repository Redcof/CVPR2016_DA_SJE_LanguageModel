import torch
import torch.nn as nn

from python.modules.common_model_interface import CommonModel


class ImageEncoder(CommonModel):
    def __init__(self, input_size, hid_size, noop=False):
        super(ImageEncoder, self).__init__()
        self.noop = noop
        
        if self.noop:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(input_size, hid_size)
        
        self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
        self.cnn.eval()
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.encoder(x)
        return x


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    print(model)
