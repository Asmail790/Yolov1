import torch
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt
from matplotlib import patches
from transformer import Transformer
from torchvision.datasets import VOCDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import Yolov1

classes =  [
"person",
"bird",
"cat",
"cow",
"dog",
"horse",
"sheep",
"aeroplane",
"bicycle",
"boat",
"bus",
"car",
"motorbike",
"train",
"bottle",
"chair",
"dining table",
"potted plant",
"sofa",
"tv/monitor"
]

def convert_to_xywh(label_row:torch.Tensor,sq_row:int,sq_col:int, split_pixel_size:float):
    in_sq_x,in_sq_y,relative_w,relative_h = label_row 
    
    x = sq_col * split_pixel_size +  in_sq_x * split_pixel_size
    y = sq_row * split_pixel_size +  in_sq_y * split_pixel_size
    w = relative_w * split_pixel_size 
    h = relative_h * split_pixel_size 

    return [x,y,w,h]    
    

dataset = VOCDetection(".", transforms=Transformer(classes,split_size=7,new_img_size=448))
dataset = Subset(dataset, list(range(0,100)))

loader = DataLoader(dataset, batch_size=8,drop_last=True)


model=torch.load("models/model_2023-11-16_0.pt",map_location=torch.device('cpu'))

model.eval()

class MetricCollector:

    def __init__(self) -> None:
        
        self.bbox_per_img = {
            "preds":[],
            "targets":[]
        }

        self.__pred_bbox_in_img = {
            "boxes":list(),
            "scores":list(),
            "labels":list()
        }
        
        self.__target_bbox_in_img = {
            "boxes":list(),
            "scores":list(),
            "labels":list()
        }


    def add_bbox(self, 
        target_bb:torch.FloatTensor,
        target_obj_score:torch.FloatTensor,
        target_class:torch.IntTensor,


        predicted_bb:torch.FloatTensor,
        predicted_obj_score:torch.FloatTensor,
        predicted_class:torch.IntTensor
    ):
      
       
        self.__pred_bbox_in_img["boxes"].append(predicted_bb)
        self.__pred_bbox_in_img["scores"].append(predicted_obj_score)
        self.__pred_bbox_in_img["labels"].append(predicted_class)
        
     
        self.__target_bbox_in_img["boxes"].append(target_bb)
        self.__target_bbox_in_img["scores"].append(target_obj_score)
        self.__target_bbox_in_img["labels"].append(target_class)
        
    def group(self):
        pred_boxes = torch.stack(self.__pred_bbox_in_img["boxes"], dim=0)
        pred_scores = torch.concat(self.__pred_bbox_in_img["scores"], dim=0)
        pred_labels = torch.concat(self.__pred_bbox_in_img["labels"], dim=0)

        target_boxes = torch.stack(self.__target_bbox_in_img["boxes"], dim=0)
        target_scores = torch.concat(self.__target_bbox_in_img["scores"], dim=0)
        target_labels = torch.concat(self.__target_bbox_in_img["labels"], dim=0)

        self.bbox_per_img["preds"].append({
            "boxes":pred_boxes,
            "scores":pred_scores,
            "labels":pred_labels
        })

        self.bbox_per_img["targets"].append({
            "boxes":target_boxes,
            "scores":target_scores,
            "labels":target_labels
        })

        self.__target_bbox_in_img["boxes"].clear()
        self.__target_bbox_in_img["scores"].clear()
        self.__target_bbox_in_img["labels"].clear()

        self.__pred_bbox_in_img["boxes"].clear()
        self.__pred_bbox_in_img["scores"].clear()
        self.__pred_bbox_in_img["labels"].clear()



    


collectors = MetricCollector()
with torch.no_grad():
    for imgs,target_matrixs in loader:

        predicted_matrixs = model(imgs) #model(imgs))

        for i in range(8):

            preds = []
            targets = []
            target_matrix = target_matrixs[i]
            predicted_matrix = predicted_matrixs[i] 

            img = imgs[i]
            img = torch.movedim(img,0,-1)

     

            for row in range(7):
                for col in range(7): 
                    target_sq = target_matrix[row][col]
                    predicted_sq = predicted_matrix[row][col]

                    if target_sq[20] == 1:
                        print("predicted",predicted_sq)
                        print("targets",target_sq)
                        if  predicted_sq[25] < predicted_sq[20] :
                            predicted_sq= torch.concat([predicted_sq[:20], predicted_sq[20:25]], dim=0)
                        else:
                            predicted_sq= torch.concat([predicted_sq[:20], predicted_sq[25:30]], dim=0)
                        
                        
                        predicted_score = predicted_sq[20]
                        predicted_xmin, predicted_ymin,predicted_w,predicted_h=convert_to_xywh(predicted_sq[21:],row,col, 448/7) 
                        predicted_class =  torch.argmax(predicted_sq[...,:20])

                        
                        target_score = target_sq[20]
                        target_xmin, target_ymin,target_w,target_h=convert_to_xywh(target_sq[21:],row,col, 448/7) 
                        target_class = torch.argmax(target_sq[...,:20])
                       
                        rect_predicted = patches.Rectangle((predicted_xmin, predicted_ymin),predicted_w,predicted_h, linewidth=1, edgecolor='r', facecolor='none')
                        rect_target = patches.Rectangle((target_xmin,target_ymin),target_w,target_h, linewidth=1, edgecolor='b', facecolor='none')


                        axImage = plt.imshow(img)
                        axImage.axes.add_patch(rect_predicted)
                        axImage.axes.add_patch(rect_target)
                        
                        plt.title(f"target:{target_class} pred:{predicted_class}")
                        plt.show()
                        
                        
                        
                        collectors.add_bbox(
                            torch.FloatTensor([target_xmin, target_ymin,target_w,target_h]),
                            torch.FloatTensor([1]),
                            torch.IntTensor([target_class]),
                            torch.FloatTensor([ predicted_xmin, predicted_ymin,predicted_w,predicted_h]),
                            torch.FloatTensor([predicted_score]),
                            torch.IntTensor([predicted_class])
                        )

            collectors.group()



mAP = MeanAveragePrecision(box_format="xywh")
mAP.update(preds=collectors.bbox_per_img["preds"],target=collectors.bbox_per_img["targets"])                      
print(mAP.compute())