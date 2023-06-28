from torch import nn


class CommonModel(nn.Module):

    def get_learnable_params(self):
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
        return params
