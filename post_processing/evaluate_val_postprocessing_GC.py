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

from medpy.graphcut import graph_from_voxels
from medpy.graphcut.energy_voxel import boundary_difference_linear, boundary_difference_exponential


cuda_device = "0"

root = "datasets/ISIC2018"

class_number = 2
use_cuda = True
number_workers=1
batch_size=1

net = Enet(class_number)
net = net.cuda() if (torch.cuda.is_available() and use_cuda) else net
map_location=lambda storage, loc: storage
checkpoint_name='ENet_0.815_equal_True.pth'
net.load_state_dict(torch.load('checkpoint/'+checkpoint_name,map_location=map_location))
Equalize = True if checkpoint_name.find('equal_True')>=0 else False

if (use_cuda and torch.cuda.is_available()):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

valdata = ISICdata(root=root,model='train',transform=True,dataAugment=False,equalize=Equalize)
val_loader = DataLoader(valdata,batch_size=batch_size,shuffle=False,num_workers=number_workers,pin_memory=True)

iou_meter_val = AverageValueMeter()
iou_crf_meter_val = AverageValueMeter()
iou_meter_val.reset()
iou_crf_meter_val.reset()

def graphcut_as_postprocessing(heatmap,image):

    fgmarkers = (heatmap>0.99).astype(np.float)
    fgmarkers_ = np.zeros(shape=(*fgmarkers.shape,3))
    for i in range (3):
        fgmarkers_[:,:,i] = fgmarkers

    bgmarkers = (heatmap<0.01).astype(np.float)
    bgmarkers_ = np.zeros(shape=(*bgmarkers.shape,3))
    for i in range(3):
        bgmarkers_[:,:,i]= bgmarkers

    image = image.cpu().data.numpy()
    image = np.moveaxis(image,0,2)
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    # image = image[:,:,0]

    #
    # image = np.asarray([[0, 0, 0, 0, 0],
    #                        [0, 0, 2, 0, 0],
    #                        [0, 0, 2, 0, 0],
    #                        [0, 0, 2, 0, 0],
    #                        [0, 0, 2, 0, 0]], dtype=np.float)
    # fgmarkers = np.asarray([[0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 1, 0, 0]], dtype=np.bool)
    # bgmarkers = np.asarray([[1, 0, 0, 0, 1],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0, 0]], dtype=np.bool)
    # expected = image.astype(np.bool)
    graph = graph_from_voxels(fgmarkers,
                              bgmarkers,
                              boundary_term=boundary_difference_exponential,
                              boundary_term_args=(image, 1000, (10,10))
                              )

    try:
        graph.maxflow()
    except Exception as e:
        print(e)

    # reshape results to form a valid mask
    result = np.zeros(image.size, dtype=np.bool)
    for idx in range(len(result)):
        result[idx] = 0 if graph.termtype.SINK == graph.what_segment(idx) else 1
    return (result*1).reshape(image.shape)

net.eval()
plt.ion()
with torch.no_grad():
    for i, (img, mask, (img_path,mask_path)) in tqdm(enumerate(val_loader)):
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
        orginal_img = Image.open(os.path.join(valdata.root,'ISIC2018_Task1-2_Training_Input',img_path[0])).resize((384,384))
        orginal_img = np.array(orginal_img).astype(np.uint8)
        pred_val = F.softmax(net(img),dim=1)



        a= graphcut_as_postprocessing(pred_val[0,1].cpu().data.numpy(),img[0])

        plt.figure(1)
        plt.subplot(2,2,1)
        plt.imshow(orginal_img)
        plt.subplot(2,2,2)
        plt.imshow(pred_val[0,1])
        plt.subplot(2,2,4)
        plt.imshow(a)
        plt.subplot(2,2,3)
        plt.imshow(mask.cpu().data.numpy().squeeze())
        plt.show()
        #
        # iou_val = iou_loss(pred2segmentation(pred_val), mask.squeeze(1).float(), class_number)[1]
        # iou_crf_val = iou_loss(full_prediction,mask[0,0].cpu().data.np(),class_number)[1]
        # iou_meter_val.add(iou_val)
        # iou_crf_meter_val.add(iou_crf_val)

print(iou_meter_val.value()[0])
print(iou_crf_meter_val.value()[0])

