from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = create_model('./models/cldm_v15.yaml').cpu()
print('model create success')
model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



input_image = cv2.imread("test_imgs/bag_scribble.png")
#batch_size
num_samples = 1
#ddim_num_steps
ddim_steps=20

img = resize_image(HWC3(input_image), 512)
H, W, C = img.shape
print("shape ",img.shape)
detected_map = np.zeros_like(input_image, dtype=np.uint8)
detected_map[np.min(input_image, axis=2) < 127] = 255
control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
control = torch.stack([control for _ in range(num_samples)], dim=0)
control = einops.rearrange(control, 'b h w c -> b c h w').clone()

cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(['colorful and true'] * num_samples)]}

# if seed == -1:
seed = random.randint(0, 2147483647)
seed_everything(seed)

shape = (4, H // 8, W // 8)


samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False)


x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)] + [255 - detected_map]

cv2.imshow("output",results[0])
cv2.waitKey(0)

# return [255 - detected_map] + results


