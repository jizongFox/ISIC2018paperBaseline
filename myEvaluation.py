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

img_transform_std = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means,
                         std=stds)
])

img_transform_equal = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])

def image_transformation(img_path,Equalize):
    img = Image.open(img_path)
    (l,w)= img.size
    img = ImageOps.equalize(img)
    img = img_transform_equal(img) if Equalize else img_transform_std(img)
    return img , (l,w)

def dilate_segmentation(seg,size):
    # seg = Image.fromarray(seg.astype(np.int32))
    kernel = np.ones((size, size), np.uint8)
    seg_dilation = cv2.dilate(seg.astype(np.float32),kernel,iterations = 1)
    return seg_dilation


def evaluate(args, net, return_score=False ):

    with torch.no_grad():

        img,(l,h) = image_transformation(os.path.join(args.img_dir,args.input_name),args.Equalize)
        prediction = net(img.unsqueeze(0).to(args.device))
        if return_score:
            return prediction
        segmentation = pred2segmentation(prediction).cpu().data.numpy().squeeze(0)
        seg_dil = Image.fromarray(np.uint8(segmentation*255)).resize((l,h))
        output_name = os.path.basename(args.input_name).replace('.jpg','.png')
        if not os.path.exists(os.path.join(args.out_dir)):
            os.makedirs(os.path.join(args.out_dir))

        seg_dil.save(os.path.join(args.out_dir,output_name),'PNG',optimize=True, progressive=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Mask Generation')
    parser.add_argument('--model', default='enet', type=str, help='model name "enet" or "unet".')
    parser.add_argument('--checkpoint', default='checkpoint/ENet_0.841_equal_True.pth', type=str, help='checkpoint_path')
    parser.add_argument('--input_name', required=True, help='input image path')
    parser.add_argument('--kernel_size', type=int, default=25, help='dilation size')
    parser.add_argument('--out_dir',required=True, help='output image path')
    parser.add_argument('--equalize',default=False,action='store_true')
    args = parser.parse_args()
    evaluate(args)






