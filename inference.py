import os
import sys

import cv2
import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from facial.mtcnn import MTCNN
from facial.model import Backbone
from utils.logger import Logger
from model.generator import InjectiveGenerator


@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    source_image_path = str(config.parameter.source_image_path)
    target_image_path = str(config.parameter.target_image_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'inference.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    detector_model = MTCNN()

    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/weight.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator().to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image
    source_image = cv2.imread(source_image_path)
    source_image_aligned = detector_model.align(Image.fromarray(source_image[:, :, ::-1]), crop_size=(224, 224))
    source_image_aligned = transform(source_image_aligned).unsqueeze(0).to(device)

    target_image = cv2.imread(target_image_path)
    target_image_aligned, target_inversion = detector_model.align(Image.fromarray(target_image[:, :, ::-1]), crop_size=(224, 224), return_trans_inv=True)
    target_image_aligned = transform(target_image_aligned).unsqueeze(0).to(device)

    print('source shape: {}'.format(source_image.transpose([2, 0, 1]).shape))
    print('target shape: {}\n'.format(target_image.transpose([2, 0, 1]).shape))

    # start inference
    with torch.no_grad():
        source_identity = identity_model(F.interpolate(source_image_aligned, (112, 112), mode='bilinear', align_corners=True))
        result_image_aligned, _ = generator_model(target_image_aligned, source_identity)
        result_image_aligned = (0.5 * result_image_aligned + 0.5).squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])[:, :, ::-1]

    # create mask
    mask_aligned = np.zeros((224, 224), dtype=np.float32)
    for i in range(224):
        for j in range(224):
            mask_aligned[i, j] = 1 - np.minimum(1, np.sqrt(np.square(i - 112) + np.square(j - 112)) / 112)
    mask_aligned = cv2.dilate(mask_aligned, None, iterations=20)

    # generate result
    mask = cv2.warpAffine(mask_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))[:, :, np.newaxis]
    result_image = cv2.warpAffine(result_image_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))
    result_image = (1 - mask) * target_image + mask * (255 * result_image)
    
    # save result
    cv2.imwrite(os.path.join(checkpoint_path, 'source.jpg'), source_image)
    cv2.imwrite(os.path.join(checkpoint_path, 'target.jpg'), target_image)
    cv2.imwrite(os.path.join(checkpoint_path, 'result.jpg'), result_image)
    print('save inference result in: {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
