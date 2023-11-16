from torchvision.datasets import VOCDetection 
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter
from model import Yolov1
from train import train
import torch
from transformer import Transformer

from torchvision.datasets import VOCDetection


device = "GPU"
useGPU = device == "GPU"
if useGPU and torch.cuda.is_available() == False:
    raise Exception("Nvidia says no can do sir")


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
train_dataset =  VOCDetection(".",image_set="train",transforms=Transformer(classes,split_size=7,new_img_size=448))
test_dataset  =  VOCDetection(".",image_set="val",transforms=Transformer(classes,split_size=7,new_img_size=448))

if useGPU:
    threads = torch.get_num_threads()
    train_dataloder = DataLoader(train_dataset, batch_size=8,shuffle=True,drop_last=True,pin_memory=True, num_workers=threads) 
    test_dataloder = DataLoader(test_dataset, batch_size=8,shuffle=True,drop_last=True,pin_memory=True,num_workers=threads)

else: 
    train_dataloder = DataLoader(train_dataset, batch_size=32,shuffle=True,drop_last=True) 
    test_dataloder = DataLoader(test_dataset, batch_size=32,shuffle=True,drop_last=True)

if useGPU:
    model=Yolov1(classes=20).cuda()
else:
    model=Yolov1(classes=20)


writer = SummaryWriter()
#dataiter = iter(train_dataloder)
#imgs,labels = dataiter._next_data()
#model(imgs)
#writer.add_graph(model,imgs

train(model, train_dataloader=train_dataloder, save=True,validation_dataloader=train_dataloder, writer=writer,cuda=useGPU, epochs=100)
writer.close()