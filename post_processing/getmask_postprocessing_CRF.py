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
from myutils.myUtils import image_transformation, fill_in_holes,calculate_iou,dilation
import shutil

cuda_device = "0"
# method ='baseline'
# method = 'fillholes'
# method = 'gaussian'
# method = 'crf'
# method = 'dilation'
method ='crf+dilation'

root = '../datasets/ISIC2018'
val_in = 'ISIC2018_Task1-2_Validation_Input'
outdir = 'crf+dilation'

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

csv_root = os.path.join(root,'ISIC_Segmenation_dataset_split')
if not os.path.isdir(os.path.join(root, outdir)):
    os.mkdir(os.path.join(root, outdir))
# img_gts_list = pd.read_csv(os.path.join(csv_root, 'val.csv'))
# imgs = img_gts_list['img'].values
# gts = img_gts_list['label'].values

imgs =[x for x in  os.listdir(os.path.join(root,val_in)) if x.find('jpg')>0]

mean_iou_meter = AverageValueMeter()
fore_iou_meter = AverageValueMeter()
mean_iou_meter.reset()
fore_iou_meter.reset()

def plot_graphs(image, groundtruth, raw_prediction, final_prediction):
    plt.close('all')
    fig,ax4 = plt.subplots(nrows=1, ncols=1)
    # plt.suptitle('\n')
    # ax1.imshow(image)
    # ax1.title.set_text('original image')
    # ax1.set_axis_off()
    #
    # ax2.imshow(image)
    # ax2.contour(groundtruth, level=[0], colors="red", alpha=1, linewidth=0.001)
    # ax2.title.set_text('ground truth')
    # ax2.set_axis_off()
    #
    # ax3.imshow(image)
    # ax3.contour((raw_prediction>0.5)*1, level=[0], colors="green", alpha=1, linewidth=0.001)
    # ax3.title.set_text('raw prediction')
    # ax3.set_axis_off()

    ax4.imshow(image)
    ax4.contour(groundtruth, level=[0], colors="red", alpha=1, linewidth=0.001)
    ax4.contour((raw_prediction>0.5)*1, level=[0], colors="green", alpha=1, linewidth=0.001)
    ax4.contour(final_prediction, level=[0], colors="blue", alpha=1, linewidth=0.001)
    # ax4.set_title('final prediction and ground truth',loc='left')
    ax4.set_axis_off()




for img in tqdm(imgs):
    input_name = os.path.basename(img)
    with torch.no_grad():
        img, (l, h), img_orignal = image_transformation(os.path.join(root, val_in ,img), Equalize=Equalize)
        try:
            groundtruth_mask = np.array(Image.open(os.path.join(root, gt)).convert('L'))/255.0
        except:
            pass

        prediction = F.softmax(net(img.unsqueeze(0).to("cuda")), dim=1)[0, 1].cpu().data.numpy()
        prediction_resized = np.array(Image.fromarray(np.uint8(prediction * 255)).resize((l, h))) / 255.0

        if method=='baseline':
            final_prediction = (prediction_resized > 0.5).astype(np.int_)
            # assert final_prediction.shape == groundtruth_mask.shape
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

        if method =='dilation':
            prediction_resized = (prediction_resized > 0.5).astype(np.int_)
            for i in range (3):
                prediction_resized=dilation(prediction_resized,structure=2)

            final_prediction = prediction_resized
            # structure =1
            # meaniou: 0.8713472311313888
            # foreground iou: 0.8060365433221706

            # structure =2
            # mean_iou: 0.8713437592695996
            # foreground_iou: 0.8060562227233957

            # for iterations:
            # i = 5
            # mean_iou: 0.8714133740696663
            # foreground_iou: 0.8075761948327599
            # i=10
            #     mean_iou: 0.8687775320857908
            # foreground_iou: 0.8052052400702532
            #  i=3
            # mean_iou: 0.8716408208825901
            # foreground_iou: 0.8072118252244899
            #  i=4
            # mean_iou: 0.8715822911988028
            # foreground_iou: 0.8074848751554229
            # i=2
            # mean_iou: 0.8715621904935187
            # foreground_iou: 0.8067410512789404


        if method =='crf+dilation':
            prediction_resized = dense_crf(img_orignal.astype(np.uint8), prediction_resized.astype(np.float32))
            for i in range (3):
                prediction_resized=dilation(prediction_resized,structure=2)

            final_prediction = prediction_resized
            # mean_iou: 0.8688208351664619
            # foreground_iou: 0.8024578721796595

             # full_prediction_1 = fill_in_holes(prediction_resized)

        try:
            final_iou = calculate_iou(final_prediction,groundtruth_mask)
            mean_iou_meter.add(np.array(final_iou).mean())
            fore_iou_meter.add(final_iou[1])
            original_iou = calculate_iou((prediction_resized > 0.5).astype(np.int_),groundtruth_mask)
        except:
            pass

        # plot_graphs(img_orignal.astype(np.uint8),groundtruth_mask,prediction_resized,final_prediction)
        # plt.suptitle('original vs final prediction iou:%.3f, %.3f, with improvement of %.3f'%(original_iou[1],final_iou[1],final_iou[1]-original_iou[1] ) )
        # plt.tight_layout()
        # plt.show()

        mask = Image.fromarray(np.uint8(final_prediction * 255))
        mask.save(os.path.join(os.path.join(root, outdir), input_name.replace('jpg', 'png')), 'PNG', optimize=True,
                  progressive=True)

shutil.make_archive(os.path.join(root, outdir), 'zip', os.path.join(root, outdir))
print('mean_iou:',mean_iou_meter.value()[0])
print('foreground_iou:', fore_iou_meter.value()[0])

from upload.upload_script import automatic_upload

uploader = automatic_upload(os.path.abspath(os.path.join(root, outdir))+'.zip')
uploader.fill_login()
uploader.get_excel()
