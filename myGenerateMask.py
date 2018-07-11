import os
from myutils.myNetworks import UNet
from myutils.myENet import Enet
import torch
from myEvaluation import evaluate
import argparse

image_input_path = 'datasets/ISIC2018/ISIC2018_Task3_Training_Input'

def main(args):
    use_cuda = True
    args.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

    imgs = [x for x in os.listdir(image_input_path) if x.find('.jpg')>=0]
    imgs.sort()

    if args.model == "enet":
        net = Enet(2).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint))
    elif args.model == "unet":
        net = UNet(2).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint))

    for img in imgs:
        args.input_name=img
        evaluate(args,net)

class configs:
    model = "unet"
    img_dir = image_input_path
    checkpoint = "UNet_0.726.pth"
    input_name = None
    out_dir ="datasets/ISIC2018/ISIC2018_Task3_Training_Input_Mask"
    kernel_size= 25

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Mask Generation')
    parser.add_argument('--model', default='enet', type=str, help='model name "enet" or "unet".')
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint_path')
    parser.add_argument('--kernel_size', type=int, default=25, help='dilation size')
    ar = parser.parse_args()
    args = configs()
    args.model = ar.model
    args.checkpoint = ar.checkpoint
    args.kernel_size= ar.kernel_size
    main(args)






