import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, to_scalar
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

import pickle
from collections import defaultdict
from torch.utils import data
import os

from my_classes import Dataset

import multiprocessing
multiprocessing.set_start_method('spawn', True)

def train(data_loader, model, optimizer, args, writer):
    #for images, _ in data_loader:
        #images = images.to(args.device)
    for local_batch, local_labels in data_loader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(args.device), local_labels.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(local_batch)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, local_labels)
        #loss_recons = F.mse_loss(x_tilde[:,0:3,:,:], local_labels[:,0:3,:,:])
        #loss_reward = F.mse_loss(torch.mean(x_tilde[:,3,:,:]), torch.mean(local_labels[:,3,:,:]))
        #loss_reward = F.mse_loss(x_tilde[:,3,:,:], local_labels[:,3,:,:])
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit# + 0.1 * loss_reward
        loss = torch.clamp(loss, loss.item(), 10)
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        args.steps += 1

def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        #for images, _ in data_loader:
            #images = images.to(args.device)
        for local_batch, local_labels in data_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(args.device), local_labels.to(args.device)
            x_tilde, z_e_x, z_q_x = model(local_batch)
            loss_recons += F.mse_loss(x_tilde, local_labels)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)

    if args.dataset == 'atari':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dictDir = args.data_folder+'data_dict/'
        dataDir = args.data_folder+'data_traj/'
        #dictDir = args.data_folder+'test_dict/'
        #dataDir = args.data_folder+'test_traj/'
        all_partition = defaultdict(list)
        all_labels = defaultdict(list)
        # Datasets
        for dictionary in os.listdir(dictDir):
            ########
            if args.out_game not in dictionary:
                #######
                dfile = open(dictDir+dictionary, 'rb')
                d = pickle.load(dfile)
                dfile.close()
                if("partition" in dictionary):
                    for key in d:
                        all_partition[key] += d[key]
                elif("labels" in dictionary):
                    for key in d:
                        all_labels[key] = d[key]
                else:
                    print("Error: Unexpected data dictionary")
        #partition = # IDs
        #labels = # Labels

        # Generators
        training_set = Dataset(all_partition['train'], all_labels, dataDir)
        train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        validation_set = Dataset(all_partition['validation'], all_labels, dataDir)
        valid_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
            num_workers=args.num_workers, pin_memory=True)
        test_loader = data.DataLoader(validation_set, batch_size=16, shuffle=True)
        input_channels = 13
        output_channels = 4

    # Define the data loaders
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #     batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #     batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, fixed_y = next(iter(test_loader))
    fixed_y = fixed_y[:,0:3,:,:]
    fixed_grid = make_grid(fixed_y, nrow=8, range=(0, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(input_channels, output_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    reconstruction_image = reconstruction[:,0:3,:,:]
    grid = make_grid(reconstruction_image.cpu(), nrow=8, range=(0, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    print("Starting to train...")
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)
        print("Finished Epoch: " + str(epoch) + "   Validation Loss: " + str(loss))
        reconstruction = generate_samples(fixed_images, model, args)
        reconstruction_image = reconstruction[:,0:3,:,:]
        grid = make_grid(reconstruction_image.cpu(), nrow=8, range=(0, 1), normalize=True)
        writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,default='dataset/',
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--out-game', type=str,default='IceHockey',
        help='name of the 11th game')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
