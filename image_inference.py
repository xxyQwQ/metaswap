import sys
import hydra

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from facial.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from facial.mtcnn import *
from model.generator import InjectiveGenerator
import cv2
import PIL.Image as Image
import numpy as np


@hydra.main(version_base=None, config_path='./config', config_name='standard')
def main(config):
    device = torch.device('cuda') if config.image_inference.device == 'gpu' else torch.device('cpu')

    source_image_path = config.image_inference.source_image_path
    target_image_path = config.image_inference.target_image_path
    save_path = config.image_inference.result_image_save_path

    detector = MTCNN()
    
    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/weight.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator()
    generator_model.eval()
    generator_model.load_state_dict(torch.load(config.image_inference.generator_path, map_location=torch.device('cpu')))
    generator_model = generator_model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    source_image_raw = cv2.imread(source_image_path)
    try:
        source_image = detector.align(Image.fromarray(source_image_raw[:, :, ::-1]), crop_size=(224, 224))
    except Exception as e:
        print('the source image is wrong, please change the image')
    source_image = transform(source_image)
    source_image = source_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        embeds = identity_model(F.interpolate(source_image, (112, 112), mode='bilinear', align_corners=True))

    target_image_raw = cv2.imread(target_image_path)
    try:
        target_image, inversion = detector.align(Image.fromarray(target_image_raw[:, :, ::-1]), crop_size=(224, 224), return_trans_inv=True)
    except Exception as e:
        print('the target image is wrong, please change the image')
    target_image = transform(target_image)
    target_image = target_image.unsqueeze(0).to(device)
    target_image_raw = target_image_raw.astype(np.float64) / 255.0

    mask = np.zeros([224, 224], dtype=np.float64)
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i-112)**2 + (j-112)**2) / 112
            dist = np.minimum(dist, 1)
            mask[i, j] = 1-dist
    mask = cv2.dilate(mask, None, iterations=20)

    with torch.no_grad():
        output_image, _ = generator_model(target_image, embeds)
        output_image = output_image.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        output_image = output_image[:, :, ::-1]
        
        output_image = cv2.warpAffine(output_image, inversion, (np.size(target_image_raw, 1), np.size(target_image_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask, inversion, (np.size(target_image_raw, 1), np.size(target_image_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        
        result = mask_ * output_image + (1-mask_) * target_image_raw
        
        cv2.imwrite(save_path, result * 255)
        print("the result image has been saved")
        cv2.waitKey(0)

if __name__ == '__main__':
    main()