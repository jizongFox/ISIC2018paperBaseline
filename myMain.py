import sys, os

sys.path.extend([os.path.dirname(os.getcwd())])
from torch.utils.data import DataLoader
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.tiramisu import FCDenseNet67
from myutils.myLoss import CrossEntropyLoss2d
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from myutils.myUtils import pred2segmentation, iou_loss, showImages, int2onehot
from myutils.myVisualize import Dashboard
import platform
import torch.backends.cudnn as cudnn
from myutils.myDiscriminator import segDiscriminator

cuda_device = "0"
root = "datasets/ISIC2018"

class_number = 2
lr = 1e-4
weigth_decay = 1e-5
use_cuda = False
number_workers = 2
batch_size = 2
max_epoch = 100
train_print_frequncy = 50
val_print_frequncy = 10
## visualization
board_train_image = Dashboard(server='http://localhost', env="image_train")
board_val_image = Dashboard(server='http://localhost', env="image_val")
board_loss = Dashboard(server='http://localhost', env="loss")

Equalize = True
## data
traindata = ISICdata(root=root, model='train', transform=True, dataAugment=True, equalize=Equalize)
valdata = ISICdata(root=root, model='dev', transform=True, dataAugment=False, equalize=Equalize)

train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=number_workers, pin_memory=False)
val_loader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=number_workers, pin_memory=False)

## network and optimiser
net = Enet(class_number)
net = net.cuda() if (torch.cuda.is_available() and use_cuda) else net

discriminator = segDiscriminator()
discriminator = discriminator.cuda() if (torch.cuda.is_available() and use_cuda) else discriminator

if (use_cuda and torch.cuda.is_available()):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
map_location = lambda storage, loc: storage
# net.load_state_dict(torch.load('checkpoint/ENet_0.798_equal_False.pth',map_location=map_location))

s_optimiser = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weigth_decay)
d_optimiser = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weigth_decay)
## loss
class_weigth = [1 * 1.8, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion_ce = CrossEntropyLoss2d(class_weigth).cuda() if (
            torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(class_weigth)
criterion_bce = nn.BCELoss()
highest_iou = -1

zeros = torch.zeros((batch_size), requires_grad=False)
ones = torch.ones((batch_size), requires_grad=False)

def train():
    net.train()
    iou_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for epoch in range(max_epoch):
        iou_meter.reset()
        loss_meter.reset()
        if epoch % 5 == 0:
            for param_group in s_optimiser.param_groups:
                param_group['lr'] = lr * (0.9 ** (epoch // 5))
            for param_group in d_optimiser.param_groups:
                param_group['lr'] = lr * (0.9 ** (epoch // 5))

        for i, (img, mask, _) in tqdm(enumerate(train_loader)):
            (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
            d_optimiser.zero_grad()
            s_optimiser.zero_grad()
            pred = net(img)

            fake_score = discriminator(img, F.softmax(pred, dim=1).detach())
            fake_loss = criterion_bce(fake_score, zeros)

            real_score = discriminator(img, int2onehot(mask))
            real_loss = criterion_bce(real_score,ones)
            (real_loss + fake_loss).backward()
            d_optimiser.step()
            # create a graph

            d_optimiser.zero_grad()
            s_optimiser.zero_grad()
            pred = net(img)
            fake_score = discriminator(img, F.softmax(pred, dim=1))
            loss_s = criterion_ce(pred, mask.squeeze(1)) + 0.5 * criterion_bce(fake_score, ones)
            loss_s.backward()
            s_optimiser.step()

            iou = iou_loss(pred2segmentation(pred), mask.squeeze(1).float(), class_number)[1]
            loss_meter.add(loss_s.item())
            iou_meter.add(iou)

            if i % train_print_frequncy == 0:
                showImages(board_train_image, img, mask, pred2segmentation(pred))

        board_loss.plot('train_iou_per_epoch', iou_meter.value()[0])
        board_loss.plot('train_loss_per_epoch', loss_meter.value()[0])

        val(net, val_loader)


def val(net, dataloader_):
    global highest_iou
    net.eval()
    iou_meter_val = AverageValueMeter()
    loss_meter_val = AverageValueMeter()
    iou_meter_val.reset()
    for i, (img, mask, _) in tqdm(enumerate(dataloader_)):
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)
        pred_val = net(img)
        loss_val = criterion_ce(pred_val, mask.squeeze(1))
        loss_meter_val.add(loss_val.item())
        iou_val = iou_loss(pred2segmentation(pred_val), mask.squeeze(1).float(), class_number)[1]
        iou_meter_val.add(iou_val)
        if i % val_print_frequncy == 0:
            showImages(board_val_image, img, mask, pred2segmentation(pred_val))

    board_loss.plot('val_iou_per_epoch', iou_meter_val.value()[0])
    board_loss.plot('val_loss_per_epoch', loss_meter_val.value()[0])
    net.train()
    if highest_iou < iou_meter_val.value()[0]:
        highest_iou = iou_meter_val.value()[0]
        torch.save(net.state_dict(),
                   'checkpoint/ENet_%.3f_%s.pth' % (iou_meter_val.value()[0], 'equal_' + str(Equalize)))
        print('The highest IOU is:%.3f' % iou_meter_val.value()[0], 'Model saved.')


if __name__ == "__main__":
    train()
