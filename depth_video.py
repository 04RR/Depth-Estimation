import os
import numpy as np
from PIL import Image
import cv2
import models
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp


device = 'cuda'
model_path = 'D:\\DL Projects\\Depth Estimation\\pretrained\\'

encoder = models.ResnetEncoder(18, False)
encoder_path = r'D:\DL Projects\Depth Estimation\pretrained\encoder.pth'
loaded_dict_enc = torch.load(encoder_path, map_location=device)

img_height = loaded_dict_enc['height']
img_width = loaded_dict_enc['width']

filtered_dict_enc = {k: v for k,
                     v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

depth_decoder = models.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))
depth_decoder_path = model_path + 'depth.pth'
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()


def estimate_depth(img_path):
    input_image = Image.open(img_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((img_width, img_height), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = F.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    scaled_disp = disp_to_depth(
        disp_resized, 0.1, 100).squeeze().cpu().detach().numpy()

    return scaled_disp


depth_images = []

for img_path in os.listdir(r'D:\DL Projects\Depth Estimation\image_00\data')[:240]:
    img_path = 'D:\DL Projects\Depth Estimation\image_00\data' + '\\' + img_path
    depth_images.append(Image.fromarray(estimate_depth(img_path)*255.0))

depth_images[0].save('depth_small.gif', format='GIF',
                     append_images=depth_images[1:], save_all=True, duration=50, loop=1)
