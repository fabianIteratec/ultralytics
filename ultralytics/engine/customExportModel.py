from ultralytics.utils import ops
from torch import nn

class ExportModel(nn.Module): 
  def __init__(self, model): 
    super().__init__()
    self.model = model

    self.stride = model.stride
    self.yaml = model.yaml
    self.pt_path = model.pt_path
    self.task = model.task

  def forward(self, im): 
    preds = self.model(im)
    preds = ops.non_max_suppression(
            preds
        )
    print(preds)
    return preds[0]

  def fuse(self): 
    self.model = self.model.fuse()
    return self