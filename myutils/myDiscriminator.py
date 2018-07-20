# this discriminator is based on the proposal in https://arxiv.org/pdf/1611.08408.pdf
# this is a conditional classifier to tell whether the input image is the ground truth or the predicted ones conditioned on the input image.

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, matplotlib.pyplot as plt

class segDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_feature_extract = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1)
        )
        self.seg_feature_extract = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5,stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        self.feature_classifier = nn.Sequential(
            nn.Conv2d(128,128,3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=1),
            nn.Conv2d(128, 256, 3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,1),
            nn.Conv2d(256, 512, 3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,2,3,stride=1, padding=1),
        )
    def forward(self, rgb_image, gt_or_segment):
        image_feature = self.image_feature_extract(rgb_image)
        gt_or_seg_feature = self.seg_feature_extract(gt_or_segment)
        combined_features = torch.cat((image_feature,gt_or_seg_feature),dim=1)

        classifier_1 = self.feature_classifier(combined_features)
        score = F.avg_pool2d(classifier_1,(classifier_1.shape[2],classifier_1.shape[3]))
        n, _, _, _ = rgb_image.size()
        p = F.softmax(score, dim=1).view(n,-1).transpose(0,1)[0]
        return p

if __name__=="__main__":
    image = torch.FloatTensor(2,3,512,512).random_(0, 1)
    gth = torch.LongTensor(2,2,512,512).random_(0,2)
    seg = F.softmax(torch.FloatTensor(1,2,512,512),dim=1)
    net = segDiscriminator()
    output = net(image,gth.float())
    print(output)






