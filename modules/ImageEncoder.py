import pathlib

import torch
import torch.nn as nn

from modules.common_model_interface import CommonModel


class ImageEncoder(CommonModel):
    def __init__(self, output_size, noop=False):
        super(ImageEncoder, self).__init__()
        self.noop = noop
        # googlelenet frozen model
        # self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', source='github', weights=GoogLeNet_Weights.DEFAULT)
        self.cnn = torch.hub.load(str(pathlib.Path.home() / '.cache/torch/hub/pytorch_vision_v0.10.0'), 'googlenet',
                                  source='local',
                                  weights='GoogLeNet_Weights.IMAGENET1K_V1')
        self.cnn.eval()
        dim_googlelenet_output_dim = 1000  # as we know, googlelenet ends with 1000 neurons
        if self.noop:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(in_features=dim_googlelenet_output_dim, out_features=output_size)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.encoder(x)
        return x


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    print(model)
