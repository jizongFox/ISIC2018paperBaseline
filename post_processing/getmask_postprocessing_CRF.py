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
from torchvision import transforms
from myutils.myUtils import image_transformation,fill_in_holes



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
net.eval()

## dataset
root = '../datasets/ISIC2018'
val_in = 'ISIC2018_Task1-2_Validation_Input'
outdir = 'ISIC2018_Task1-2_Validation_Output'
imgs = [x for x in os.listdir(os.path.join(root,val_in)) if x.find('jpg')>0]

# imgs =[
# 'ISIC_0012346',
# 'ISIC_0012576',
# 'ISIC_0012623',
# 'ISIC_0012627',
# 'ISIC_0016804',
# 'ISIC_0017702',
# 'ISIC_0019794',
# 'ISIC_0020999',
# 'ISIC_0021037',
# 'ISIC_0021504',
# 'ISIC_0021914',
# 'ISIC_0022221',
# 'ISIC_0023508',
# 'ISIC_0023755',
# 'ISIC_0023936',
# 'ISIC_0036098',
# 'ISIC_0036247',
# ]
#
# imgs = [x+".jpg" for x in imgs]


for img in tqdm(imgs):
    input_name = os.path.basename(img)
    with torch.no_grad():
        img, (l, h), img_orignal = image_transformation(os.path.join(root,val_in,img), Equalize=Equalize)

        prediction = F.softmax(net(img.unsqueeze(0).to("cuda")),dim=1)[0,1].cpu().data.numpy()
        prediction_resized = np.array(Image.fromarray(np.uint8(prediction*255)).resize((l,h)))/255.0
        full_prediction = dense_crf(img_orignal.astype(np.uint8), prediction_resized.astype(np.float32))
        full_prediction_1 = fill_in_holes(full_prediction)

        fig1=plt.figure(1)
        plt.clf()
        ax0 = fig1.add_subplot(221)
        ax0.imshow(img_orignal)
        ax0.set_axis_off()
        ax0.set_title('original image')
        ax1= fig1.add_subplot(222)
        ax1.imshow(prediction_resized)
        ax1.set_title('output prediction')
        ax1.set_axis_off()
        ax3= fig1.add_subplot(223)
        ax3.imshow(full_prediction)
        ax3.set_title('DCF output')
        ax3.set_axis_off()
        ax2 = fig1.add_subplot(224)
        ax2.imshow(full_prediction_1)
        ax2.set_title('Gaussian filter and fill holes')
        ax2.set_axis_off()
        plt.show(block=False)
        plt.pause(0.001)
        mask = Image.fromarray(np.uint8(full_prediction_1 * 255))
        mask.save(os.path.join(os.path.join(root,outdir), input_name.replace('jpg','png')), 'PNG', optimize=True, progressive=True)



