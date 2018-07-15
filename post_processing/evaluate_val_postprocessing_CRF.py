import sys,os,numpy as np
sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myLoss import  CrossEntropyLoss2d
import torch
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from myutils.myUtils import pred2segmentation, iou_loss,showImages
from myutils.myVisualize import Dashboard
import platform
import torch.backends.cudnn as cudnn
from PIL import Image, ImageOps
import torch.nn.functional as F
from myutils.myCrf import dense_crf
import matplotlib.pyplot as plt



cuda_device = "0"

root = "datasets/ISIC2018"

class_number = 2
use_cuda = True
number_workers=1
batch_size=1

net = Enet(class_number)
net = net.cuda() if (torch.cuda.is_available() and use_cuda) else net
map_location=lambda storage, loc: storage
checkpoint_name='ENet_0.841_equal_True.pth'

Equalize = True if checkpoint_name.find('equal_True')>=0 else False

if (use_cuda and torch.cuda.is_available()):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load('../checkpoint/'+checkpoint_name,map_location=map_location))

valdata = ISICdata(root=root,model='train',transform=True,dataAugment=False,equalize=Equalize)
val_loader = DataLoader(valdata,batch_size=batch_size,shuffle=False,num_workers=number_workers,pin_memory=True)

iou_meter_val = AverageValueMeter()
iou_crf_meter_val = AverageValueMeter()
iou_meter_val.reset()
iou_crf_meter_val.reset()

net.eval()
plt.ion()
with torch.no_grad():
    for i, (img, mask, (img_path,mask_path)) in tqdm(enumerate(val_loader)):
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
        orginal_img = Image.open(os.path.join(valdata.root,'ISIC2018_Task1-2_Training_Input',img_path[0])).resize((384,384))
        pred_val = F.softmax(net(img),dim=1)

        full_prediction = dense_crf(np.array(orginal_img).astype(np.uint8), pred_val[0,1].cpu().data.numpy().astype(np.float32))
        plt.imshow(full_prediction)
        plt.show()

        iou_val = iou_loss(pred2segmentation(pred_val), mask.squeeze(1).float(), class_number)[1]
        iou_crf_val = iou_loss(full_prediction,mask[0,0].cpu().data.numpy(),class_number)[1]
        iou_meter_val.add(iou_val)
        iou_crf_meter_val.add(iou_crf_val)

print(iou_meter_val.value()[0])
print(iou_crf_meter_val.value()[0])

