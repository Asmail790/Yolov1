import torch 
import torch.nn as nn
from utils import FIOU,MidPoint,Corners

IOU = FIOU(MidPoint(midX=0,midY=1,width=2,height=3))
from torchvision.ops import box_iou,box_convert


class YoloLoss(nn.Module):
    def __init__(self,
                classes:int,
                split_size:int = 7,
                num_boxes:int = 2,
                noobj_scale:float = 0.05,
                coord_scale:float = 0.5,
                class_scale:float = 0.1,
                object_scale:float = 0.1,
                *args,
                box_iou_by_pytorch:bool = False,
                **kwargs
                ) -> None:
        super().__init__(*args, **kwargs)


        # orignaly sum
        self._mse = nn.MSELoss(reduction="mean")
        self._classes = classes
        self._split_size = split_size
        self._num_boxes = num_boxes
        self._noobj_scale  = noobj_scale
        self._coord_scale = coord_scale
        self._box_iou_by_pytorch = box_iou_by_pytorch
        self._object_scale  = object_scale
        self._class_scale = class_scale

       
    def forward(self, predictions:torch.Tensor,target:torch.Tensor):
        predictions = predictions.flatten(end_dim=-2)
        target = target.flatten(end_dim=-2)

       
        ious = torch.Tensor([]).reshape(predictions.shape[0],0).to(device=predictions.device)
        first_bb_indices = [21,22,23,24]

        if self._box_iou_by_pytorch is False:
            for i in range(self._num_boxes):
                bb_indices = self._classes + i*5 + 1 + torch.arange(4)
                ious_i = IOU(predictions[:,bb_indices],target[:,first_bb_indices]).reshape(-1,1)
                ious = torch.concat([ious,ious_i],dim=1)
        
        else:
            
            target_bb_dims = box_convert(target[:,first_bb_indices], in_fmt="cxcywh",out_fmt="xywh")

            for i in range(self._num_boxes):
                bb_indices = self._classes + i*5 + 1 +  torch.arange(4)
                prediction_bb_dims = box_convert(predictions[:,bb_indices], in_fmt="cxcywh",out_fmt="xywh")

                ious_i = box_iou(prediction_bb_dims,target_bb_dims)
                ious_i = torch.diagonal(ious_i).reshape(-1,1)
                ious = torch.concat([ious,ious_i],dim=1)
                 
        first_bb_width_idx,first_bb_height_idx = 23,24
        target_bb_dims = torch.sign(target[:,[first_bb_width_idx,first_bb_height_idx]]) * torch.sqrt(
            target[:,[first_bb_width_idx,first_bb_height_idx]]
        )


        iou_max_indices = torch.argmax(ious,dim=1) 
        width_indices = self._classes+iou_max_indices*5+3
        height_indices = self._classes+iou_max_indices*5+4

        predictions_bb_dims = torch.concat([
                predictions[torch.arange(predictions.shape[0]),width_indices].reshape(-1,1),
                predictions[torch.arange(predictions.shape[0]),height_indices].reshape(-1,1)
            ],
            dim = 1)

        predictions_bb_dims = torch.sign(predictions_bb_dims) * torch.sqrt( torch.abs(predictions_bb_dims + 1e-6))
        
        first_obj_score_index = self._classes
        exists_box = target[:,first_obj_score_index].reshape(-1,1)

        x_indices = self._classes+iou_max_indices*5+1
        y_indices = self._classes+iou_max_indices*5+2

        predictions_bb_position = torch.concat([
                predictions[torch.arange(predictions.shape[0]),x_indices].reshape(-1,1),
                predictions[torch.arange(predictions.shape[0]),y_indices].reshape(-1,1)
            ],
            dim = 1)
        
        first_bb_xpos_idx,first_bb_ypos_idx = 21,22
        target_bb_position =  target[:,[first_bb_xpos_idx,first_bb_ypos_idx]]
        
        
        box_loss = self._mse(exists_box * torch.concat([predictions_bb_position,predictions_bb_dims],dim=1),
                              exists_box * torch.concat([target_bb_position,target_bb_dims],dim=1)
        )

        obj_score_indices = self._classes+iou_max_indices*5
        object_loss = self._mse(
            exists_box*predictions[torch.arange(predictions.shape[0]),obj_score_indices].reshape(-1,1),
            exists_box*target[:,first_obj_score_index].reshape(-1,1)
        )



        no_object_loss = 0
        for i in range(self._num_boxes):
            obj_score_index = self._classes + i * 5 
            no_object_loss += self._mse( 
                (1 - exists_box) * predictions[:,obj_score_index].reshape(-1,1),
                (1 - exists_box) * target[:,first_obj_score_index].reshape(-1,1)
            )



        class_loss = self._mse(
            exists_box * predictions[:,:20],
            exists_box * target[:,:20]
        )

        total_loss =  self._coord_scale * box_loss + self._object_scale * object_loss + self._noobj_scale * no_object_loss + self._class_scale * class_loss

        
        return total_loss

