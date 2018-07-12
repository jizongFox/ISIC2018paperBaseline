import sys,os
sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.tiramisu import FCDenseNet67
from myutils.myLoss import  CrossEntropyLoss2d
import torch
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from myutils.myUtils import pred2segmentation, iou_loss,showImages
from myutils.myVisualize import Dashboard
import platform
import torch.backends.cudnn as cudnn


cuda_device = "1"
os.environ['CUDA_VISIBLE_DEVICES']=cuda_device
root = "datasets/ISIC2018"

class_number = 2
lr = 1e-4
weigth_decay=1e-5
use_cuda=True
number_workers=8
batch_size=8
max_epoch=100
train_print_frequncy =50
val_print_frequncy = 3
## visualization
board_train_image = Dashboard(server='http://localhost',env="image_train")
board_val_image = Dashboard(server='http://localhost',env="image_val")
board_loss = Dashboard(server='http://localhost',env="loss")

Equalize = False
## data
traindata = ISICdata(root=root,model='train',transform=True,dataAugment=True,equalize=Equalize)
valdata = ISICdata(root=root,model='dev',transform=True,dataAugment=False,equalize=Equalize)

train_loader = DataLoader(traindata,batch_size=batch_size,shuffle=True,num_workers=number_workers,pin_memory=True)
val_loader = DataLoader(valdata,batch_size=batch_size,shuffle=True,num_workers=number_workers,pin_memory=True)

## network and optimiser
net =FCDenseNet67(class_number)
net = net.cuda() if (torch.cuda.is_available() and use_cuda) else net

if (use_cuda and torch.cuda.is_available()):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
map_location = lambda storage, loc: storage
# net.load_state_dict(torch.load('checkpoint/ENet_0.798_equal_False.pth',map_location=map_location))

optimiser = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weigth_decay)
## loss
class_weigth=[1*1.8, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).cuda() if (torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(class_weigth)
highest_iou = -1

def train():
    net.train()
    iou_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for epoch in range(max_epoch):
        iou_meter.reset()
        loss_meter.reset()
        if epoch % 5 == 0:
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr * (0.9 ** (epoch // 5))
                print('learning rate:', param_group['lr'])

        for i,(img,mask,_) in tqdm(enumerate(train_loader)):
            (img,mask) = (img.cuda(),mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
            optimiser.zero_grad()
            pred = net(img)
            loss = criterion(pred,mask.squeeze(1))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-3)
            optimiser.step()
            loss_meter.add(loss.item())
            iou = iou_loss(pred2segmentation(pred),mask.squeeze(1).float(),class_number)[1]
            loss_meter.add(loss.item())
            iou_meter.add(iou)

            if i %train_print_frequncy==0:
                showImages(board_train_image, img, mask, pred2segmentation(pred))

        board_loss.plot('train_iou_per_epoch',iou_meter.value()[0])
        board_loss.plot('train_loss_per_epoch',loss_meter.value()[0])

        val(net,val_loader)


def val(net, dataloader_):
    global highest_iou
    net.eval()
    iou_meter_val = AverageValueMeter()
    loss_meter_val = AverageValueMeter()
    iou_meter_val.reset()
    for i, (img, mask, _) in tqdm(enumerate(dataloader_)):
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
        pred_val = net(img)
        loss_val = criterion(pred_val,mask.squeeze(1))
        loss_meter_val.add(loss_val.item())
        iou_val = iou_loss(pred2segmentation(pred_val), mask.squeeze(1).float(), class_number)[1]
        iou_meter_val.add(iou_val)
        if i%val_print_frequncy ==0:
            showImages(board_val_image,img,mask,pred2segmentation(pred_val))

    board_loss.plot('val_iou_per_epoch',iou_meter_val.value()[0])
    board_loss.plot('val_loss_per_epoch',loss_meter_val.value()[0])
    net.train()
    if highest_iou < iou_meter_val.value()[0]:
        highest_iou = iou_meter_val.value()[0]
        torch.save(net.state_dict(), 'checkpoint/ENet_%.3f_%s.pth' %( iou_meter_val.value()[0],'equal_'+str(Equalize)))
        print('The highest IOU is:%.3f'%iou_meter_val.value()[0],'Model saved.')

if __name__=="__main__":
    train()






