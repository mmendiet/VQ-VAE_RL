#parse_data
import os
import numpy as np
from PIL import Image
import torch
from collections import defaultdict

dataDir = "/mnt/AI_HDD/Atari/"
saveDir = "/home/mmendiet/fall2019/rl/pytorch-vqvae/data/"
innerFiles = ['curr_state', 'action', 'reward', 'next_state']

partition = defaultdict(list)
labels = defaultdict(list)
train_list = []
validation_list = []
count = 0
outfile = "IceHockey"
for episode in os.listdir(dataDir):
    print(episode)
    #check that 11th game not in episode name
    if outfile not in episode:
        ep =  np.load(dataDir+episode)
        end = len(ep['curr_state'])

        for traj in range(0, end):
            ID = episode.split('.')[0] + '_' + str(traj)
            curr_state = ep[innerFiles[0]][traj]
            curr_state_rs = np.concatenate((curr_state[:,:,:,0],curr_state[:,:,:,1],curr_state[:,:,:,2],curr_state[:,:,:,3]), axis=2)

            #np.all(curr_state_rs[:,:,0:3]==curr_state[:,:,:,0])s
            action = ep[innerFiles[1]][traj]
            action_image = np.full((84,84,1), action)
            curr_state_rsa = np.concatenate((curr_state_rs, action_image), axis=2)

            reward = ep[innerFiles[2]][traj]
            reward_image = np.full((84,84,1), reward)
            next_state = ep[innerFiles[3]][traj]
            next_frame = next_state[:,:,:,3]
            next_frame_r = np.concatenate((next_frame, reward_image), axis=2)

            input_tensor = torch.from_numpy(curr_state_rsa).float()
            output_tensor = torch.from_numpy(next_frame_r).float()

            torch.save(input_tensor, saveDir+str(ID)+".pt")
            if(count < 80000000):#80000000  8000000
                train_list.append(ID)
            else:
                validation_list.append(ID)
            labels[ID] = output_tensor
            count += 1
            #img = Image.fromarray(data, 'RGB')
            #break
    if(count>10):
        break
partition['train'] = train_list
partition['validation'] = validation_list