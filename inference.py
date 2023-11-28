import os
import sys

import cv2
import hydra
import time
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import glob


import torch
import torch.nn.functional as F
from torchvision import transforms

from facial.mtcnn import MTCNN
from facial.arcface import Backbone
from utils.logger import Logger
from model.generator import InjectiveGenerator



def images_inference(source_identity, target_images_aligned, generator_model, device):
    # start inference
    source_identity = torch.repeat_interleave(source_identity, target_images_aligned.shape[0], dim=0)
    with torch.no_grad():
        result_images_aligned, _ = generator_model(target_images_aligned, source_identity)
        result_images_aligned = (0.5 * result_images_aligned + 0.5).detach().cpu().numpy().transpose([0, 2, 3, 1])[:, :, :, ::-1]
    return result_images_aligned

def image_recover(result_image_aligned, target_image, target_inversion):
    mask = cv2.warpAffine(mask_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))[:, :, np.newaxis]
    result_image = cv2.warpAffine(result_image_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))
    result_image = (1 - mask) * target_image + mask * (255 * result_image)
    return result_image
    
@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    time1 = time.time()
    # load configuration
    inference_type = str(config.parameter.inference_type)
    model_path = str(config.parameter.model_path)
    source_image_path = str(config.parameter.source_image_path)
    target_image_path = str(config.parameter.target_image_path)
    target_video_path = str(config.parameter.target_video_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    frames_path = os.path.join(checkpoint_path, 'frames')
    os.makedirs(frames_path)
    result_video_path = os.path.join(checkpoint_path, 'result_video.mp4')
    video_fps = int(config.parameter.video_fps)
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
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator().to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # create mask
    global mask_aligned
    mask_aligned = np.zeros((224, 224), dtype=np.float32)
    for i in range(224):
        for j in range(224):
            mask_aligned[i, j] = 1 - np.minimum(1, np.sqrt(np.square(i - 112) + np.square(j - 112)) / 112)
    mask_aligned = cv2.dilate(mask_aligned, None, iterations=20)
    
    # create transform
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #source image
    source_image = cv2.imread(source_image_path)
    source_image_aligned = detector_model.align(Image.fromarray(source_image[:, :, ::-1]), crop_size=(224, 224))
    source_image_aligned = transform(source_image_aligned).unsqueeze(0).to(device)
    print('source shape: {}'.format(source_image.transpose([2, 0, 1]).shape))
    with torch.no_grad():
        source_identity = identity_model(F.interpolate(source_image_aligned, 112, mode='bilinear', align_corners=True))
    
    if inference_type == 'image':
        target_image = cv2.imread(target_image_path)
        target_image_aligned, target_inversion = detector_model.align(Image.fromarray(target_image[:, :, ::-1]), crop_size=(224, 224), return_trans_inv=True)
        target_image_aligned = transform(target_image_aligned).unsqueeze(0).to(device)

        result_image_aligned = images_inference(source_identity, target_image_aligned, generator_model, device).squeeze(0)
                
        result_image = image_recover(result_image_aligned, target_image, target_inversion)
       
        #save result
        cv2.imwrite(os.path.join(checkpoint_path, 'source.jpg'), source_image)
        cv2.imwrite(os.path.join(checkpoint_path, 'target.jpg'), target_image)
        cv2.imwrite(os.path.join(checkpoint_path, 'result.jpg'), result_image)
        print('save inference result in: {}\n'.format(checkpoint_path))
        
    elif(inference_type == 'video'):
        
        #Video Capture
        capture = cv2.VideoCapture(target_video_path)
        success = capture.isOpened()
        frame_count = 0
        if success == False:
            print("error opening video stream or file!")
        try:
            while success:
                frame_count += 1
                success, frame = capture.read()
                frame_path = os.path.join(frames_path, '%08d.jpg'%frame_count)
                cv2.imwrite(frame_path, frame)
                if (frame_count % 50 == 0):
                    print("%dth frame has been processed" % frame_count)
        except Exception as e:
            print("video has been processed to frames")
        capture.release()
        
        #load frames
        frames = glob.glob(os.path.join(frames_path,'*.*g'))
        frames.sort()
        index = 0
        batch_size = 16
        target_index = []
        target_images = []
        target_inversions = []
        target_images_aligned = torch.zeros(1)
        size = ()
        
        for frame in frames:
            target_image = cv2.imread(frame)
            size = (target_image.shape[1], target_image.shape[0])
            index += 1
            if(index % 50 == 0):
                print("%dth frame has been processed"%index)
            try:
                target_image_aligned, target_inversion = detector_model.align(Image.fromarray(target_image[:, :, ::-1]), crop_size=(224, 224), return_trans_inv=True)
            except Exception as e:
                print('skip one frame')
                continue
            if target_image_aligned == None:
                continue
            target_image_aligned = transform(target_image_aligned).unsqueeze(0).to(device)
            if len(target_index) == 0:
                target_index = [index]
                target_images = [target_image]
                target_inversions = [target_inversion]
                target_images_aligned = target_image_aligned
            else:
                target_index.append(index)
                target_images.append(target_image)
                target_inversions.append(target_inversion)
                target_images_aligned = torch.cat((target_images_aligned, target_image_aligned), dim=0)
            if len(target_index) == batch_size:
                result_images_aligned = images_inference(source_identity, target_images_aligned, generator_model, device)
                for _ in range(batch_size):
                    result_image = image_recover(result_images_aligned[_], target_images[_], target_inversions[_])
                    cv2.imwrite(os.path.join(frames_path, '%08d.jpg' % target_index[_]), result_image)
                target_index = []
        #the left frames
        if len(target_index) > 0:
            batch_size = len(target_index)
            result_images_aligned = images_inference(source_identity, target_images_aligned, generator_model, device)
            for _ in range(batch_size):
                result_image = image_recover(result_images_aligned[_], target_images[_], target_inversions[_])
                cv2.imwrite(os.path.join(frames_path, '%08d.jpg' % target_index[_]), result_image)
                
        print("start generate video")
        videowriter = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, size)

        frames = glob.glob(os.path.join(frames_path, '*.*g'))
        frames.sort()
        frame_count = 0
        for frame in frames:
            image = cv2.imread(frame)
            videowriter.write(image)
            if(frame_count % 50 == 0):
                print("%dth frames has been convert to video" % frame_count)
            frame_count +=1
    else:
        print("Inference type error!")
    time2 = time.time()
    print("total processing time is %f"%(time2-time1))



if __name__ == '__main__':
    main()
