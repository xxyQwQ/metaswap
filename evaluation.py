import os
import tqdm
import torchvision
import torch
import hydra
from tl2.proj.fvcore.checkpoint import Checkpointer
import numpy as np
from PIL import Image
from tl2 import tl2_utils
from model.generator import InjectiveGenerator
from facial.mtcnn import *
import PIL.Image as Image
import torch.nn.functional as F
import torchvision.transforms.functional as tv_f
from facial.model import Backbone
from pathlib import Path
import facial.hopenet


def inference(source,target):

    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/weight.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator()
    generator_model.eval()
    generator_model.load_state_dict(torch.load('./facial/generator.pth', map_location=torch.device('cpu')))
    generator_model = generator_model.to(device)
    
    source_image = Image.open(source)
    source_image = tv_f.to_tensor(source_image)
    source_image = (source_image*2-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        source_identity = identity_model(F.interpolate(source_image, 112, mode='bilinear', align_corners=True))

    target_image = Image.open(target)
    target_image = tv_f.to_tensor(target_image)
    target_image = (target_image*2-1).unsqueeze(0).to(device)

    result_image, _ = generator_model(target_image, source_identity)
    result_image = torch.squeeze(result_image, dim=0)

    result_image = result_image.detach().cpu().numpy()
    result_image = result_image * 0.5 +0.5
    result_image = result_image * 255

    return result_image


def generate_swapped_img(oringin_evaluation_dataset_path,chekpoint_path):

    new_folder_name = "origin_as_source"
    source_root = os.path.join(chekpoint_path, new_folder_name)
    if not os.path.exists(source_root):
        os.makedirs(source_root)

    new_folder_name = "origin_as_target"
    target_root = os.path.join(chekpoint_path, new_folder_name)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"] 
    oringin_evaluation_dataset_path = Path(oringin_evaluation_dataset_path)
    image_files = [file for file in oringin_evaluation_dataset_path.iterdir() if file.suffix.lower() in image_extensions]
    cnt = 0
    for source_image in tqdm.tqdm(image_files):
        
        origin_as_source_name = source_image.stem
        origin_as_source_name_folder_path = os.path.join(source_root, origin_as_source_name)

        if not os.path.exists(origin_as_source_name_folder_path):
            os.makedirs(origin_as_source_name_folder_path)

        for target_image in image_files:
            
            origin_as_target_name = target_image.stem
            origin_as_target_name_folder_path = os.path.join(target_root, origin_as_target_name)

            if not os.path.exists(origin_as_target_name_folder_path):
                os.makedirs(origin_as_target_name_folder_path)

            swapped_img = inference(str(source_image), str(target_image))
            cnt+=1
            swapped_img = swapped_img.transpose([1, 2, 0])
            Image.fromarray((swapped_img).astype(np.uint8)).save(os.path.join(origin_as_source_name_folder_path, f"{cnt}.jpg"))
            Image.fromarray((swapped_img).astype(np.uint8)).save(os.path.join(origin_as_target_name_folder_path, f"{cnt}.jpg"))

    return source_root,target_root


def resize_func(x,size=(112, 112)):
        
    if tuple(x.shape[-2:]) != size:
        out = F.interpolate(x, size=size, mode='bilinear', antialias=True)
    else:
        out = x
    
    return out


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def poseture(origin_evaluation_dataset_path,target_root):
    model = facial.hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load("./facial/hopenet_robust_alpha1.pkl")
    model.load_state_dict(saved_state_dict)
    model = model.to(device)
    model.eval()
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    poseture = 0
    cnt = 0
    origin_path_list = tl2_utils.get_filelist_recursive(origin_evaluation_dataset_path, )
    for _ , origin_path in enumerate(tqdm.tqdm(origin_path_list)):
        origin_img = Image.open(origin_path)
        origin_img = tv_f.to_tensor(origin_img)
        origin_img = (origin_img * 2 - 1).unsqueeze(0).to(device)

        yaw, pitch, roll = model(origin_img)
        yaw_predicted = softmax_temperature(yaw.data, 1)
        pitch_predicted = softmax_temperature(pitch.data, 1)
        roll_predicted = softmax_temperature(roll.data, 1)

        yaw_predicted_ori = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted_ori = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted_ori = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
        
        target_dir = f"{target_root}/{origin_path.stem}"
        target_path_list = tl2_utils.get_filelist_recursive(target_dir)

        for target_path in target_path_list:
            target_img = Image.open(target_path)
            target_img = tv_f.to_tensor(target_img)
            target_img = (target_img * 2 - 1).unsqueeze(0).to(device)

            x,y,z = model(target_img)
            yaw_pre = softmax_temperature(x.data, 1)
            pitch_pre = softmax_temperature(y.data, 1)
            roll_pre = softmax_temperature(z.data, 1)

            yaw_pre_tar = torch.sum(yaw_pre * idx_tensor, 1).cpu() * 3 - 99
            pitch_pre_tar = torch.sum(pitch_pre * idx_tensor, 1).cpu() * 3 - 99
            roll_pre_tar = torch.sum(roll_pre * idx_tensor, 1).cpu() * 3 - 99

            dx = np.square(yaw_pre_tar - yaw_predicted_ori)
            dy = np.square(pitch_pre_tar - pitch_predicted_ori)
            dz = np.square(roll_pre_tar - roll_predicted_ori)
            dist = np.sqrt(dx + dy + dz)
            poseture += dist
            cnt += 1
            
    poseture /= cnt

    print("Posture Loss:{:.2f}".format(poseture.item()))


def id_retrieval(swapped_root,oringin_evaluation_dataset_path):

    facenet = Backbone(50, 0.6, 'ir_se').eval().requires_grad_(False).to(device)
    Checkpointer(facenet).load_state_dict_from_file('./facial/weight.pth')

    source_path_list = tl2_utils.get_filelist_recursive(oringin_evaluation_dataset_path, )

    emb_content_list = []
    emb_transfer_list = []
    label_id_list = []

    for label_id, source_path in enumerate(tqdm.tqdm(source_path_list)):

        img_c_pil = Image.open(source_path)
        img_c_tensor = tv_f.to_tensor(img_c_pil)
        img_c_tensor = (img_c_tensor * 2 - 1).unsqueeze(0).to(device)
        img_c_tensor = resize_func(img_c_tensor)
        emb_c = facenet(img_c_tensor)
        emb_content_list.append(emb_c)

        swapped_dir = f"{swapped_root}/{source_path.stem}"
        swapped_path_list = tl2_utils.get_filelist_recursive(swapped_dir)

        for transfer_path in swapped_path_list:

            img_t_pil = Image.open(transfer_path)
            img_t_tensor = tv_f.to_tensor(img_t_pil)
            img_t_tensor = (img_t_tensor * 2 - 1).unsqueeze(0).to(device)
            img_t_tensor = resize_func(img_t_tensor)
            emb_t = facenet(img_t_tensor)
            emb_transfer_list.append(emb_t)
            label_id_list.append(label_id)

    emb_content = torch.cat(emb_content_list, dim=0)
    emb_transfer = torch.cat(emb_transfer_list, dim=0)
    label_id = torch.tensor(label_id_list, device=device)

    diff = emb_content.unsqueeze(-1) - emb_transfer.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    _ , min_idx = torch.min(dist, dim=0)

    id_retrieval = (min_idx == label_id).float().sum() / len(label_id)
    print("ID_retrieval:{:.2%}".format(id_retrieval))
    

@hydra.main(version_base=None, config_path='./config', config_name='evaluation')
def main(config):
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    dataset_origin_path = str(config.parameter.oringin_evaluation_dataset_path)
    Checkpoint_path = str(config.parameter.checkpoint_path)
    source_root,target_root = generate_swapped_img(dataset_origin_path,chekpoint_path=Checkpoint_path)
    id_retrieval(source_root, dataset_origin_path)
    poseture(dataset_origin_path,target_root=target_root)

    
if __name__ == '__main__':

    main()