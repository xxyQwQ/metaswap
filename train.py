import os
import sys
import time

import hydra
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


from facial.model import Backbone
from utils.logger import Logger
from utils.dataset import FaceDataset
from utils.function import hinge_loss, plot_sample
from model.generator import InjectiveGenerator
from model.discriminator import MultiscaleDiscriminator


@hydra.main(version_base=None, config_path='./config', config_name='light')
def main(config):
    # load configuration
    device = torch.device('cuda') if config.training.device == 'gpu' else torch.device('cpu')
    dataset_path = str(config.training.dataset_path)
    checkpoint_path = str(config.training.checkpoint_path)
    batch_size = int(config.training.batch_size)
    num_workers = int(config.training.num_workers)
    learning_rate = float(config.training.learning_rate)
    num_iterations = int(config.training.num_iterations)
    report_interval = int(config.training.report_interval)
    save_interval = int(config.training.save_interval)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))

    # print configuration
    print('device: {}'.format(device))
    print('dataset_path: {}'.format(dataset_path))
    print('checkpoint_path: {}'.format(checkpoint_path))
    print('batch_size: {}'.format(batch_size))
    print('num_workers: {}'.format(num_workers))
    print('learning_rate: {}'.format(learning_rate))
    print('num_iterations: {}'.format(num_iterations))
    print('report_interval: {}'.format(report_interval))
    print('save_interval: {}\n'.format(save_interval))

    # create model
    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/weight.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator(identity_channels=512).to(device)
    generator_model.train()

    discriminator_model = MultiscaleDiscriminator(num_scales=3).to(device)
    discriminator_model.train()

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)

    # load dataset
    dataset = FaceDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    print('num_images: {}'.format(len(dataset)))

    # start training
    current_iteration = 0
    current_time = time.time()

    while current_iteration < num_iterations:
        for source_image, target_image, same_identity in dataloader:
            current_iteration += 1

            source_image, target_image, same_identity = source_image.to(device), target_image.to(device), same_identity.to(device)

            with torch.no_grad():
                source_identity = identity_model(F.interpolate(source_image, 112, mode='bilinear', align_corners=True))

            # generator
            generator_optimizer.zero_grad()

            result_image, target_attribute = generator_model(target_image, source_identity)

            prediction_result = discriminator_model(result_image)
            loss_adversarial = 0
            for prediction in prediction_result:
                loss_adversarial += hinge_loss(prediction, True)

            result_identity = identity_model(F.interpolate(result_image, 112, mode='bilinear', align_corners=True))
            loss_identity = (1 - torch.cosine_similarity(source_identity, result_identity, dim=1)).mean()

            result_attribute = generator_model.attribute(result_image)
            loss_attribute = 0
            for i in range(len(target_attribute)):
                loss_attribute += 0.5 * torch.mean(torch.square(target_attribute[i] - result_attribute[i]))

            loss_reconstruction = torch.sum(same_identity * 0.5 * torch.mean(torch.square(result_image - target_image).reshape(batch_size, -1), dim=1)) / (torch.sum(same_identity) + 1e-6)

            loss_generator = 1 * loss_adversarial + 5 * loss_attribute + 20 * loss_identity + 5 * loss_reconstruction
            loss_generator.backward()
            generator_optimizer.step()

            # discriminator
            discriminator_optimizer.zero_grad()
            prediction_fake = discriminator_model(result_image.detach())
            loss_fake = 0
            for prediction in prediction_fake:
                loss_fake += hinge_loss(prediction, False)

            prediction_real = discriminator_model(source_image)
            loss_true = 0
            for prediction in prediction_real:
                loss_true += hinge_loss(prediction, True)

            loss_discriminator = 0.5 * (loss_true + loss_fake)
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # report
            if current_iteration % report_interval == 0:
                last_time = current_time
                current_time = time.time()
                iteration_time = (current_time - last_time) / report_interval

                print('iteration {} / {}:'.format(current_iteration, num_iterations))
                print('time: {:.6f} seconds per iteration'.format(iteration_time))
                print('discriminator loss: {:.6f}, generator loss: {:.6f}'.format(loss_discriminator.item(), loss_generator.item()))
                print('adversarial loss: {:.6f}, identity loss: {:.6f}, attribute loss: {:.6f}, reconstruction loss: {:.6f}\n'.format(loss_adversarial.item(), loss_identity.item(), loss_attribute.item(), loss_reconstruction.item()))
            
            # save
            if current_iteration % save_interval == 0:
                save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))
                os.makedirs(save_path, exist_ok=True)

                image_path = os.path.join(save_path, 'image.jpg')
                generator_path = os.path.join(save_path, 'generator.pth')
                discriminator_path = os.path.join(save_path, 'discriminator.pth')

                image = plot_sample(source_image, target_image, result_image).transpose([1, 2, 0])
                Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
                torch.save(generator_model.state_dict(), generator_path)
                torch.save(discriminator_model.state_dict(), discriminator_path)

                print('save sample image in: {}'.format(image_path))
                print('save generator model in: {}'.format(generator_path))
                print('save discriminator model in: {}\n'.format(discriminator_path))
            
            if current_iteration >= num_iterations:
                break


if __name__ == '__main__':
    main()
