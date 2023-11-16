from torch import nn
from typing import Literal
from typing import NamedTuple, Final
import torch
from backbone import make_layers, darknet


class Yolov1(nn.Module):
    def __init__(self,classes:int,
                 backbone:nn.Module = make_layers(darknet,3),
                 split_size=7,num_boxes=2,
                 p:float=0.5,
                 *args,
                 **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.classes:Final = classes
        self.split_size:Final = split_size
        self.backbone:Final = backbone
        self.num_boxes:Final = num_boxes
        
        self.avgpool:Final = nn.AdaptiveAvgPool1d(self.split_size**2*1024)
        self.l1:Final = nn.Linear(self.split_size**2*1024,4096)
        self.dropout:Final = nn.Dropout1d(p)
        self.l2:Final = nn.Linear(4096,self.split_size*self.split_size*(self.classes +5*self.num_boxes))
        

 
    
    def forward(self,x:torch.Tensor):
        x = self.backbone(x)
        x = torch.flatten(x,start_dim=1)
        # torch.Size([1, 50176])
        # torch.Size([4, 50176])
        x = self.avgpool(x) 
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = torch.flatten(x,start_dim=1)
        x = x.reshape(-1,self.split_size,self.split_size,self.classes +5*self.num_boxes)
        return x 
    

if __name__ == "__main__":
    model = Yolov1(backbone=darknet, in_channels=3,split_size=10,classes=20)
    img =  torch.randn((4,3,448,448))
    print(model(img).shape)

