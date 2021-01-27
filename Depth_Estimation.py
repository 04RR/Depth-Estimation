import os
import numpy as np
from PIL import Image
import models
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp


image_path = r'D:\Desktop\Depth Estimation\test.jpg'
device = 'cuda'
model_path = 'D:\\Desktop\\Depth Estimation\\pretrained\\'

encoder = models.ResnetEncoder(18, False)
encoder_path = model_path + 'encoder.pth'
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

input_image = Image.open(image_path).convert('RGB')
original_width, original_height = input_image.size
input_image = input_image.resize((img_width, img_height), Image.LANCZOS)
input_image = transforms.ToTensor()(input_image).unsqueeze(0)

input_image = input_image.to(device)
features = encoder(input_image)
outputs = depth_decoder(features)

disp = outputs[("disp", 0)]
disp_resized = F.interpolate(
    disp, (original_height, original_width), mode="bilinear", align_corners=False)

output_name = f'{image_path[:-4]}_disp.npy'
scaled_disp = disp_to_depth(
    disp_resized, 0.1, 100).squeeze().cpu().detach().numpy()

np.save(output_name, scaled_disp)
