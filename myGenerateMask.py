import os
from myutils.myNetworks import UNet
from myutils.myENet import Enet
import torch
from myEvaluation import evaluate
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm

image_input_path = 'datasets/ISIC2018/ISIC2018_Task1-2_Validation_Input'

def main(args):
    use_cuda = True
    args.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    imgs = [x for x in os.listdir(image_input_path) if x.find('.jpg')>=0]
    imgs.sort()

    if args.model == "enet":
        net = Enet(2).to(args.device)
        if (use_cuda and torch.cuda.is_available()):
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        map_location = lambda storage, loc: storage
        net.load_state_dict(torch.load(os.path.join('checkpoint',args.checkpoint),map_location=map_location))
    else:
        raise NotImplemented

    for img in tqdm(imgs):
        args.input_name=img
        evaluate(args,net)

class configs:
    model = "unet"
    img_dir = image_input_path
    checkpoint = None
    input_name = None
    out_dir ="datasets/ISIC2018/ISIC2018_Task1-2_Validation_Input_Mask"
    Equalize=True

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Mask Generation')
    parser.add_argument('--model', default='enet', type=str, help='model name "enet" or "unet".')
    parser.add_argument('--checkpoint', required=False, default="ENet_0.841_equal_True.pth", type=str, help='checkpoint_path')
    ar = parser.parse_args()
    args = configs()
    args.model = ar.model
    args.checkpoint = ar.checkpoint
    main(args)






