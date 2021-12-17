from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """implement as section 2.5 in our document"""

    def __init__(self):
        super(Model, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return F.log_softmax(logits, dim=1)

    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.state_dict().items()} 
        

    def set_weights(self, weights: OrderedDict[str, torch.Tensor]):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
