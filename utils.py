import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
import torch
from numpy.typing import NDArray
from typing import Literal
from functools import partial
from torchvision.ops import box_iou,nms as nms2

class Corners(NamedTuple):
    xs:int 
    xe:int 
    ys:int
    ye:int 

class MidPoint(NamedTuple):
    midX:int 
    midY:int 
    width:int
    height:int 


def IOU(predictions:torch.Tensor, labels:torch.Tensor, mode:Corners|MidPoint = Corners(0,1,2,3)):

    if isinstance(mode,Corners):
        xs1 = torch.unsqueeze(predictions[...,mode.xs],-1)
        xe1 = torch.unsqueeze(predictions[...,mode.xe],-1)
        ys1 = torch.unsqueeze(predictions[...,mode.ys],-1)
        ye1 = torch.unsqueeze(predictions[...,mode.ye],-1)

        xs2 = torch.unsqueeze(labels[...,mode.xs],-1)
        xe2 = torch.unsqueeze(labels[...,mode.xe],-1)
        ys2 = torch.unsqueeze(labels[...,mode.ys],-1)
        ye2 = torch.unsqueeze(labels[...,mode.ye],-1)

    elif isinstance(mode,MidPoint):

        xs1 = torch.unsqueeze(predictions[...,mode.midX] - predictions[...,mode.width] / 2 ,-1)  
        xe1 = torch.unsqueeze(predictions[...,mode.midX] + predictions[...,mode.width] / 2 ,-1)  
        ys1 = torch.unsqueeze(predictions[...,mode.midY] - predictions[...,mode.height] / 2 ,-1) 
        ye1 = torch.unsqueeze(predictions[...,mode.midY] + predictions[...,mode.height] / 2 ,-1) 

        xs2 = torch.unsqueeze(labels[...,mode.midX] - labels[...,mode.width] / 2, -1)
        xe2 = torch.unsqueeze(labels[...,mode.midX] + labels[...,mode.width] / 2, -1)
        ys2 = torch.unsqueeze(labels[...,mode.midY] - labels[...,mode.height] / 2,-1)
        ye2 = torch.unsqueeze(labels[...,mode.midY] + labels[...,mode.height] / 2,-1) 
        

    intersection = (torch.min(xe1,xe2) - torch.max(xs1,xs2)).clamp(0)*(torch.min(ye1,ye2)-torch.max(ys1,ys2)).clamp(0)
    predictions_area =torch.abs((xe1 - xs1)*(ye1 -ys1))
    labels_area = torch.abs((xe2 - xs2)*(ye2 -ys2))
    
    return intersection/(predictions_area + labels_area - intersection +1e-6)


def FIOU(mode:Corners|MidPoint=MidPoint(midX=0,midY=1,width=2,height=3)):
    return partial(IOU, mode=mode)

def nms(boxes:torch.Tensor,scores:torch.Tensor,iou_threshold:float = 0.9)-> torch.Tensor:
    
    keep = torch.ones(size=scores.size()).to(dtype=torch.bool)
    for index in torch.arange(boxes.shape[-2]):
        bb = torch.select(boxes,dim=-2,index=index)

        rest = torch.arange(0,boxes.shape[-2]) == index
        rest = (~rest).to(dtype=torch.bool) 
        #rest = keep & rest
        rest = torch.squeeze(torch.nonzero(rest))
        other_bbs = torch.index_select(boxes, dim=-2,index=rest)
        bb = torch.tile(bb,dims=(other_bbs.shape[-2],1))

        iou_above = iou_threshold < IOU(bb,other_bbs)
        iou_above = torch.squeeze(iou_above)
        curent_obj_not_best = (scores[...,index] < scores[...,rest]) 
        is_bad_bb = torch.any(iou_above & curent_obj_not_best)
        keep[index] = ~is_bad_bb

    good_bb_indices = torch.squeeze(torch.nonzero(keep))
    sorted_indices = torch.argsort(scores[good_bb_indices], descending=True)
    return good_bb_indices[sorted_indices]


if __name__ == "__main__":

    x = torch.ones(size=(10,1))
    y = torch.ones(size=(10,1))
    x2 = x + torch.abs(torch.randn(size=(10,1)))
    y2 = y + torch.abs(torch.randn(size=(10,1)))
    M = torch.concat([x,x2,y,y2],dim=1)
    scores = torch.abs(torch.randn(10))

    resX = nms(
        M,
        scores,
        iou_threshold = 0.2
    )
    resY = nms2(torch.concat([x,y,x2,y2],dim=1),scores,0.2)
   
    print(resY,scores[resY])
    print(resX,scores[resX])
    #m1 = MidPoint(midX=0,midY=1,width=2,height=3)
    #m2 = Corners(xs=0,xe=1,ys=2,ye=3)
    #
    #preds1 = torch.tensor([[2.5,3,1.5,2]])
    #labels1 = torch.tensor([[2.5,2.5,0.5,0.5]])
    #
    #preds2 = torch.tensor([[1,4,1,5]])
    #labels2 = torch.tensor([[2,3,2,3]])
    #
    #c  = FIOU(m2)
    #
    #x = c(preds2,labels2)
    #
    #print(x)

