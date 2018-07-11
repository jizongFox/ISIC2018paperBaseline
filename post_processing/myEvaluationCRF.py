#coding=utf8
import torch
from PIL import Image, ImageOps
# import matplotlib
# matplotlib.use('agg')
import numpy as np
from myutils.myNetworks import UNet
from myutils.myENet import Enet
import os
from torchvision import transforms
import argparse
import torch.nn.functional as F
from myutils.myCrf import dense_crf


# means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
# stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])

img_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])


def image_transformation(img_path):
    img = Image.open(img_path)
    (l,w)= img.size
    img = ImageOps.equalize(img)
    img = img_transform(img)
    return img , (l,w)


def evaluate(args ,net ):

    with torch.no_grad():

        img,(l,h) = image_transformation(os.path.join(args.input_name))
        orginal_img = Image.open(args.input_name)
        prediction = F.softmax(net(img.unsqueeze(0).to(args.device)),dim=1)
        full_prediction = Image.fromarray(np.uint8(prediction[0,1]*255)).resize((l,h))
        full_prediction = np.asarray(full_prediction)/255.0
        full_prediction = dense_crf(np.array(orginal_img).astype(np.uint8),full_prediction.astype(np.float32))
        full_prediction = Image.fromarray(np.uint8(full_prediction * 255))
        output_name = os.path.basename(args.input_name)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        full_prediction.save(os.path.join(args.out_dir,output_name),'JPEG',optimize=True, progressive=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Mask Generation')
    parser.add_argument('--model', default='enet', type=str, help='model name "enet" or "unet".')
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint_path')
    parser.add_argument('--input_name', required=True, help='input image path')
    parser.add_argument('--out_dir',default='output', help='output image path')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if args.model == "enet":
        net = Enet(2).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint))
    elif args.model == "unet":
        net = UNet(2).to(args.device)
        net.load_state_dict(torch.load(args.checkpoint))
    evaluate(args,net)






