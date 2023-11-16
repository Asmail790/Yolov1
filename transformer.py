from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from pprint import pprint
from typing import NamedTuple,cast
from torchvision.transforms.functional import to_tensor,resize,normalize
import torch
import matplotlib.pyplot as plt
from matplotlib import patches



class Transformer():
    def __init__(self,classes:list[str],new_img_size:float,split_size:int,normalize_img=False) -> None:
        self.classes = classes
        self.new_img_size = new_img_size
        self.split_size = split_size
        self.normalize_img = normalize_img

    def to_number(self,str:str):
        # TODO   split all "x/y" and check x or y or x/y
        if str == "tvmonitor" or str == "tv" or "monitor":
            str = "tv/monitor"
    
        return self.classes.index(str)

    def to_str(self,id:int):

        return self.classes[id]


    def make_bbs_relative(self,bbs:torch.Tensor,size:float):
        bbs =  bbs/size
        return bbs

    def resize_sample(self,img:torch.Tensor,bbs:torch.Tensor,new_size:float):
        _,height,width = img.shape

        ratioX = (new_size/width)
        ratioY = (new_size/height)
        img = resize(img,(new_size,new_size),antialias=True)
        bbs = bbs * torch.Tensor([ratioX,ratioX,ratioY,ratioY])
        return img,bbs


    def create_label_matrix(self,boxes:torch.Tensor,nbr_of_classes:int,split_size:int):
        label_matrix = torch.zeros((split_size,split_size,nbr_of_classes + 5))
        
        
        for (class_label,xmax,xmin,ymax,ymin) in boxes: 
            x = xmin 
            y = ymin 
            width =     xmax-xmin 
            height =    ymax-ymin

            x = torch.clamp(x,max= 1-1e-6) # make less than 1
            y = torch.clamp(y,max= 1-1e-6) # make less than 1
            
            class_label = int(class_label)
            cell_x,cell_y = int(split_size*x), int(split_size*y) # 0 < ceil_x < split_size
            in_cell_x,in_cell_y = split_size*x -cell_x, split_size*y -cell_y  # 0 < x < 1 
            width_cell ,height_cell = (width * split_size, height* split_size) # 0 < width_ceil < self._split_size

            object_score_index = nbr_of_classes
            x_index = nbr_of_classes + 1 
            y_index = nbr_of_classes + 2 
            width_index = nbr_of_classes + 3
            height_index = nbr_of_classes + 4


            #if label_matrix[cell_y,cell_x,object_score_index] == 1:
            #    raise ValueError("label_matrix should be zero at this point")
            
            if label_matrix[cell_y,cell_x,object_score_index] == 0:

                label_matrix[cell_y,cell_x][class_label] = 1
                label_matrix[cell_y,cell_x][object_score_index] = 1
                label_matrix[cell_y,cell_x][x_index] = in_cell_x
                label_matrix[cell_y,cell_x][y_index] = in_cell_y
                label_matrix[cell_y,cell_x][width_index] = width_cell
                label_matrix[cell_y,cell_x][height_index] = height_cell
            
        return label_matrix 


    def __call__(self,image:torch.Tensor,data:dict):

        img = to_tensor(image)
        if self.normalize_img:
            mean = torch.mean(img)
            std = torch.std(img)
            img = normalize(img,mean,std)
        
        bbs_info = [ [     
            self.to_number(object["name"]),
            int(object["bndbox"]['xmax']),
            int(object["bndbox"]['xmin']),
            int(object["bndbox"]['ymax']),
            int(object["bndbox"]['ymin']),
            ] for object in cast(dict,data["annotation"]["object"]) ]
    
        bbs_info =  torch.Tensor(bbs_info)
        bbs_dims =  bbs_info[:,1:]
        bbs_class =    bbs_info[:,[0]]
        img,bbs_dims = self.resize_sample(img,bbs_dims,self.new_img_size)
        bbs_dims = self.make_bbs_relative(bbs_dims,size=self.new_img_size)
        bbs_info = torch.concat([bbs_class,bbs_dims],dim=1) 

        label_matrix = self.create_label_matrix(bbs_info, split_size=self.split_size, nbr_of_classes=len(self.classes))

        return img,label_matrix 


if __name__ == "__main__":
    classes = [
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
    dataset = VOCDetection(".", transforms=Transformer(classes,split_size=7,new_img_size=448))
    loader = DataLoader(dataset, batch_size=8)


    def recreate(label_row:torch.Tensor,sq_row:int,sq_col:int, split_pixel_size:float):
        in_sq_x,in_sq_y,relative_w,relative_h = label_row 
        
        x = sq_col * split_pixel_size +  in_sq_x * split_pixel_size
        y = sq_row * split_pixel_size +  in_sq_y * split_pixel_size
        w = relative_w * split_pixel_size 
        h = relative_h * split_pixel_size 

        return [x,y,w,h]
        




    for imgs,matrixs in loader:
        for i in range(8):
            matrix = matrixs[i]
            img = imgs[i]
            img = torch.movedim(img,0,-1)

            for row in range(7):
                for col in range(7): 

                    sq = matrix[row][col]
                    if sq[20] == 1:
                        xmin, ymin,w,h=recreate(sq[21:],row,col, 448/7) 

                        
                        rect = patches.Rectangle((xmin, ymin),w,h, linewidth=1, edgecolor='r', facecolor='none')
                        circle = patches.Circle((xmin, ymin),radius=3)

                        axImage = plt.imshow(img)
                        axImage.axes.add_patch(rect)
                        axImage.axes.add_patch(circle)
                        #plt.title(to_str(int(id)))
                        plt.show()