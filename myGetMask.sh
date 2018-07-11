#!/usr/bin/env bash

python myGenerateMask.py --model unet --checkpoint UNet_0.726.pth --kernel_size 0

python myGenerateMask.py --model unet --checkpoint UNet_0.726.pth --kernel_size 25

python myGenerateMask.py --model unet --checkpoint UNet_0.726.pth --kernel_size 50

python myGenerateMask.py --model unet --checkpoint UNet_0.726.pth --kernel_size 75