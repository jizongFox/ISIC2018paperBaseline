import numpy as np
import os
import sys
import pandas as pd
sys.path.extend([os.path.dirname(os.getcwd())])
from myutils.myENet import Enet
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.nn.functional as F
from myutils.myCrf import dense_crf
import matplotlib.pyplot as plt
from scipy import ndimage
from torchnet.meter import AverageValueMeter
from myutils.myUtils import image_transformation, fill_in_holes,calculate_iou

cuda_device = "0"
# method ='baseline'
# method = 'fillholes'
# method = 'gaussian'
method = 'crf'

root = "datasets/ISIC2018"

class_number = 2
use_cuda = True
number_workers = 1
batch_size = 1

net = Enet(class_number)
net = net.cuda() if (torch.cuda.is_available() and use_cuda) else net
map_location = lambda storage, loc: storage
checkpoint_name = 'ENet_0.841_equal_True.pth'

Equalize = True if checkpoint_name.find('equal_True') >= 0 else False

if (use_cuda and torch.cuda.is_available()):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load('../checkpoint/' + checkpoint_name, map_location=map_location))
net.eval()

## dataset
root = '../datasets/ISIC2018'
val_in = 'ISIC2018_Task1-2_Training_Input'
outdir = 'Development_set_Output_Mask'
csv_root = os.path.join(root,'ISIC_Segmenation_dataset_split')
if not os.path.isdir(os.path.join(root, outdir)):
    os.mkdir(os.path.join(root, outdir))
img_gts_list = pd.read_csv(os.path.join(csv_root, 'val.csv'))
imgs = img_gts_list['img'].values
gts = img_gts_list['label'].values

mean_iou_meter = AverageValueMeter()
fore_iou_meter = AverageValueMeter()
mean_iou_meter.reset()
fore_iou_meter.reset()
for (img,gt) in tqdm(zip(imgs,gts)):
    input_name = os.path.basename(img)
    with torch.no_grad():
        img, (l, h), img_orignal = image_transformation(os.path.join(root,  img), Equalize=Equalize)
        groundtruth_mask = np.array(Image.open(os.path.join(root, gt)).convert('L'))/255.0

        prediction = F.softmax(net(img.unsqueeze(0).to("cuda")), dim=1)[0, 1].cpu().data.numpy()
        prediction_resized = np.array(Image.fromarray(np.uint8(prediction * 255)).resize((l, h))) / 255.0

        if method=='baseline':
            final_prediction = (prediction_resized > 0.5).astype(np.int_)
            assert final_prediction.shape == groundtruth_mask.shape
            # mean iou: 0.8709129795448528
            # foreground iou: 0.8050291207321197

        if method=="fillholes":
            prediction_resized = (prediction_resized > 0.5).astype(np.int_)
            final_prediction = fill_in_holes(prediction_resized)
            # meaniou: 0.8709865610158177
            # foreground iou: 0.8051885879455437
        if method == 'gaussian':
            prediction_resized = (prediction_resized > 0.5).astype(np.int_)
            final_prediction = ndimage.gaussian_filter(prediction_resized, sigma=5).astype(int)
            # sigma ==1:
            # mean iou: 0.8673420208414067
            # foreground iou: 0.7982967857510189
            # sigma ==3
            # mean iou: 0.8536097849945127
            # foreground iou: 0.7749834090238301
            # sigma ==5
            # mean iou: 0.8330956802575635
            # foreground iou: 0.7405888607817745
        if method =="crf":
            final_prediction = dense_crf(img_orignal.astype(np.uint8), prediction_resized.astype(np.float32))
            # mean iou: 0.8617552757360335
            # foreground iou: 0.7904958657627422

        if method =='crf+fillholes':
            pass
        if method =='crf+gaussian':
            pass

             # full_prediction_1 = fill_in_holes(prediction_resized)


        iou = calculate_iou(final_prediction,groundtruth_mask)
        mean_iou_meter.add(np.array(iou).mean())
        fore_iou_meter.add(iou[1])

        mask = Image.fromarray(np.uint8(final_prediction * 255))
        mask.save(os.path.join(os.path.join(root, outdir), input_name.replace('jpg', 'png')), 'PNG', optimize=True,
                  progressive=True)
print('mean iou:',mean_iou_meter.value()[0])
print('foreground iou:', fore_iou_meter.value()[0])