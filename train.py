from torch import nn 
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from typing import cast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import date

from loss import YoloLoss
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import cast
from model import Yolov1

from typing import Final
from utils import FIOU,MidPoint,Corners
IOU = FIOU(MidPoint(midX=0,midY=1,width=2,height=3))

def train(
        model:nn.Module,
        train_dataloader:DataLoader,
        validation_dataloader:DataLoader,
        writer:SummaryWriter,
        save = False, 
        cuda=False, 
        epochs = 1
    ):
    loss_fn = YoloLoss(classes=20)
    optimizer = Adam(model.parameters(), lr=1e-6)
    timestamp = date.today()


    smallest_average_validation_loss = float("inf")
    for epoch in range(epochs):
        train_loss = []
    
        model.train()
        mAP_train = MeanAveragePrecision(box_format="cxcywh")
        

        with tqdm(train_dataloader, total=len(train_dataloader), unit="batch") as bar:
            bar.set_description(f"Epoch {epoch}")
            for  imgs, labels in bar:
                imgs = cast(torch.Tensor,imgs)
                labels = cast(torch.Tensor,labels)

                if cuda:
                    imgs = imgs.cuda()
                    labels= labels.cuda()

                optimizer.zero_grad()

                predicted_labels = model(imgs)
                predicted_labels = cast(torch.Tensor,predicted_labels)

                loss = loss_fn(predicted_labels,labels)
                loss = cast(torch.Tensor,loss)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                
                bar.set_postfix(
                    loss=float(loss),
                )

                ############################### Todo make not hardcoded
                bb1 = predicted_labels[...,[21,22,23,24]]
                bb2 = predicted_labels[...,[26,27,28,29]]
                bb_target = predicted_labels[...,[21,22,23,24]]

                obj_score1 = predicted_labels[...,[20]]
                obj_score2 = predicted_labels[...,[25]]
                
                ious1 = IOU(bb1,bb_target)
                ious2 = IOU(bb2,bb_target)
                    
                ious = torch.argmax(torch.concat([ious1,ious2],dim=-1),dim=-1).flatten()
                    
                bbs_pred = torch.stack([bb1,bb2],dim=-1).flatten(end_dim=-3)
                obj_score_pred = torch.stack([obj_score1,obj_score2],dim=-1).flatten(end_dim=-2)
                    
                bbs_pred = bbs_pred[torch.arange(392),:,ious].reshape(8,7,7,-1)
                obj_score_pred = obj_score_pred[torch.arange(392),ious].reshape(8,7,7,-1)
                labels_pred = torch.unsqueeze(torch.argmax(predicted_labels[...,:20],dim=-1),dim=-1)

                bbs_pred = bbs_pred.reshape(8,-1,4)
                obj_score_pred = obj_score_pred.reshape(8,-1,1)
                labels_pred = labels_pred.reshape(8,-1,1)


                preds = [{
                        "boxes":bbs_pred[i],
                        "scores":torch.squeeze(obj_score_pred[i]),
                        "labels":torch.squeeze(labels_pred[i])
                        } for i in range(8)
                ]
                    
                bbs_target = labels[...,[21,22,23,24]].reshape(8,-1,4)
                obj_score_target =  labels[...,[20]].reshape(8,-1,1)
                labels_target = torch.argmax(labels[...,:20],dim=-1).reshape(8,-1,1)
                
                target = [
                    {
                    "boxes":bbs_target[i],
                    "scores":torch.squeeze(obj_score_target[i]),
                    "labels":torch.squeeze(labels_target[i])
                    } for i in range(8)
                ]

                mAP_train.update(preds=preds,target=target)
                #####################################################

        average_train_loss_in_epoch = sum(train_loss)/len(train_dataloader)
        print("tranning",mAP_train.compute())       
        writer.add_scalar("avg_train_loss",average_train_loss_in_epoch,epoch)
        print(f"average train loss in epoch {epoch} {average_train_loss_in_epoch}")

     
        valid_loss = []

        model.eval()
        mAP_test = MeanAveragePrecision(box_format="cxcywh")
        
        with tqdm(validation_dataloader, total=len(validation_dataloader),desc='validation',unit="batch") as bar:
            for imgs,labels in bar:
                imgs = cast(torch.Tensor,imgs)
                labels = cast(torch.Tensor,labels)
                
                if cuda:
                    imgs = imgs.cuda()
                    labels= labels.cuda()

                with torch.no_grad():
                    predicted_labels = model(imgs)
                    predicted_labels = cast(torch.Tensor,predicted_labels)

                    loss = loss_fn(predicted_labels,labels)
                        
                    loss = cast(torch.Tensor,loss)
                    
                    valid_loss.append(loss)


                    bar.set_postfix(
                        loss=float(loss),
                    )

                    ############################### Todo make not hardcoded
                    bb1 = predicted_labels[...,[21,22,23,24]]
                    bb2 = predicted_labels[...,[26,27,28,29]]
                    bb_target = predicted_labels[...,[21,22,23,24]]

                    obj_score1 = predicted_labels[...,[20]]
                    obj_score2 = predicted_labels[...,[25]]
                
                    ious1 = IOU(bb1,bb_target)
                    ious2 = IOU(bb2,bb_target)
                    
                    ious = torch.argmax(torch.concat([ious1,ious2],dim=-1),dim=-1).flatten()
                    
                    bbs_pred = torch.stack([bb1,bb2],dim=-1).flatten(end_dim=-3)
                    obj_score_pred = torch.stack([obj_score1,obj_score2],dim=-1).flatten(end_dim=-2)
                    
                    bbs_pred = bbs_pred[torch.arange(392),:,ious].reshape(8,7,7,-1)
                    obj_score_pred = obj_score_pred[torch.arange(392),ious].reshape(8,7,7,-1)
                    labels_pred = torch.unsqueeze(torch.argmax(predicted_labels[...,:20],dim=-1),dim=-1)

                    bbs_pred = bbs_pred.reshape(8,-1,4)
                    obj_score_pred = obj_score_pred.reshape(8,-1,1)
                    labels_pred = labels_pred.reshape(8,-1,1)



                    preds = [{
                        "boxes":bbs_pred[i],
                        "scores":torch.squeeze(obj_score_pred[i]),
                        "labels":torch.squeeze(labels_pred[i])
                        } for i in range(8)
                    ]
                    
                    bbs_target = labels[...,[21,22,23,24]].reshape(8,-1,4)
                    obj_score_target =  labels[...,[20]].reshape(8,-1,1)
                    labels_target = torch.argmax(labels[...,:20],dim=-1).reshape(8,-1,1)
                
                    target = [
                        {
                        "boxes":bbs_target[i],
                        "scores":torch.squeeze(obj_score_target[i]),
                        "labels":torch.squeeze(labels_target[i])
                        } for i in range(8)
                    ]

                    mAP_test.update(preds=preds,target=target)
                    #####################################################
        average_validation_loss = sum(valid_loss)/len(validation_dataloader)
        if average_validation_loss<smallest_average_validation_loss:
            smallest_average_validation_loss = average_validation_loss
            if save:
                torch.save(model, f"models/model_{timestamp}_{epoch}.pt")
            
        writer.add_scalar("avg_vd_loss",average_validation_loss,epoch)
        print("validation",mAP_test.compute())
        print(f"average validation loss in epoch {epoch} {average_validation_loss}")
            



    #return train_acc,train_loss

