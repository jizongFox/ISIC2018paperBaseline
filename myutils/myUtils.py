import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from torchvision import transforms


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0), -1)
    tflat = target.view(input.size(0), -1)
    intersection = (iflat * tflat).sum(1)

    return ((2. * intersection + smooth).float() / (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean()


def iou_loss(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            try:
                ious.append(float(intersection) / max(union, 1).cpu().data.numpy())
            except:
                ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
        # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:
                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print('error in colorize.')
        return color_image


def showImages(board, image_batch, mask_batch, segment_batch):
    color_transform = Colorize()
    means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
    stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])

    if image_batch.min() < 0:
        for i in range(3):
            image_batch[:, i, :, :] = (image_batch[:, i, :, :]) * stds[i] + means[i]

    board.image(image_batch[0], 'original image')
    board.image(color_transform(mask_batch[0]), 'ground truth image')
    board.image(color_transform(segment_batch[0]), 'prediction given by the net')


def int2onehot(mask):
    batch_size = mask.shape[0]
    nb_digits = 2
    y_onehot = torch.zeros((batch_size, nb_digits, *mask.shape[2:]))
    for i in range(nb_digits):
        y_onehot[:, i] = (mask == i).squeeze()
    return y_onehot


means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])

img_transform_std = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means,
                         std=stds)
])

img_transform_equal = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])


def image_transformation(img_path, Equalize):
    img = Image.open(img_path)
    img_original = np.array(img)
    (l, w) = img.size
    img = ImageOps.equalize(img)
    img = img_transform_equal(img) if Equalize else img_transform_std(img)
    return img, (l, w), img_original


def fill_in_holes(predictions):
    img_fill_holes = predictions
    for i in range(5, 3, -1):
        img_fill_holes = ndimage.binary_fill_holes(predictions, structure=np.ones((i, i))).astype(int)
    img_fill_holes = ndimage.binary_fill_holes(img_fill_holes).astype(int)


    return img_fill_holes

def calculate_iou(predicted_mask, mask):
    '''
    :param predicted_mask: (h,w) with 0 and 1 from numpy
    :param image: (h,w) with 0 and 1 from numpy
    :return: (background_iou, foreground_iou)
    '''
    assert mask.ravel().max()<=1
    iou = []
    for cls in {0,1}:
        pred_inds = predicted_mask == cls
        target_inds = mask == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        iou.append(intersection/union)
    return iou

def dilation(predicted_mask,structure=1):
    struct1 = ndimage.generate_binary_structure(2, 1)
    struct2 = ndimage.generate_binary_structure(2, 2)
    if structure==1:
        return ndimage.binary_dilation(predicted_mask, structure=struct1).astype(predicted_mask.dtype)
    else:
        return ndimage.binary_dilation(predicted_mask, structure=struct2).astype(predicted_mask.dtype)