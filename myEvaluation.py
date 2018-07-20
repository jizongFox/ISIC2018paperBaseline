#coding=utf8
import torch
from PIL import Image, ImageOps
# import matplotlib
# matplotlib.use('agg')
import cv2
import numpy as np
import os
from torchvision import transforms
import argparse
from myutils.myUtils import pred2segmentation

means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])

img_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=means,
    #                      std=stds)
])

def image_transformation(img_path):
    img = Image.open(img_path)
    (l,w)= img.size
    img = ImageOps.equalize(img)
    img = img_transform(img)
    return img , (l,w)

def dilate_segmentation(seg,size):
    # seg = Image.fromarray(seg.astype(np.int32))
    kernel = np.ones((size, size), np.uint8)
    seg_dilation = cv2.dilate(seg.astype(np.float32),kernel,iterations = 1)
    return seg_dilation


def evaluate(args ,net ):

    with torch.no_grad():

        img,(l,h) = image_transformation(os.path.join(args.img_dir,args.input_name))
        prediction = net(img.unsqueeze(0).to(args.device))
        segmentation = pred2segmentation(prediction).cpu().data.numpy().squeeze(0)
        seg_dil =dilate_segmentation(segmentation,args.kernel_size)
        seg_dil = Image.fromarray(np.uint8(seg_dil*255)).resize((l,h))
        output_name = os.path.basename(args.input_name)
        if not os.path.exists(os.path.join(args.out_dir+'__kernel__'+str(args.kernel_size))):
            os.makedirs(os.path.join(args.out_dir+'__kernel__'+str(args.kernel_size)))

        seg_dil.save(os.path.join(args.out_dir+'__kernel__'+str(args.kernel_size),output_name),'JPEG',optimize=True, progressive=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Mask Generation')
    parser.add_argument('--model', default='enet', type=str, help='model name "enet" or "unet".')
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint_path')
    parser.add_argument('--input_name', required=True, help='input image path')
    parser.add_argument('--kernel_size', type=int, default=25, help='dilation size')
    parser.add_argument('--out_dir',required=True, help='output image path')
    args = parser.parse_args()
    evaluate(args)






