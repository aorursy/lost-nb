#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('bash', '', '# Install deps from \n# https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux\napt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \\\nnasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \\\nlibopenal-dev timidity libwildmidi-dev unzip\n\n# Boost libraries\napt-get install libboost-all-dev\n\n# Lua binding dependencies\napt-get install liblua5.1-dev')




get_ipython().system('pip install vizdoom')




get_ipython().system('git clone https://github.com/simoninithomas/Deep_reinforcement_learning_Course.git')




from __future__ import division
from __future__ import print_function
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser




get_ipython().system('cat "Deep_reinforcement_learning_Course/Deep Q Learning/Doom/basic.cfg"')




#DEFAULT_MODEL_SAVEFILE = "drive/My Drive/Colab Notebooks/tmp/model"
DEFAULT_CONFIG = "Deep_reinforcement_learning_Course/Deep Q Learning/Doom/basic.cfg"
DEFAULT_SCN = "Deep_reinforcement_learning_Course/Deep Q Learning/Doom/basic.wad"




# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path , scenario_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_doom_scenario_path(scenario_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game




doom=initialize_vizdoom(DEFAULT_CONFIG , DEFAULT_SCN)




import random
import time
import matplotlib.pyplot as plt
def test_environment(game):
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 1
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()




import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
from collections import namedtuple
import torch
import torch.optim as optim
import numpy as np
stacked_size = 4

def preprocess(frame):
    """
    screen frames are in grayscale format defalut
    1.normalize the images
    2. apply some transformations
    """
    frame = frame / 255.0
    frame = transform.resize(frame ,[96,96])
    
    return frame
    
    
def stack_frames(stacked_frames , frame , new_episode=False):
    """
    stack multiple frames with each other to idenitify the temporal movemnts of the objects
    1. preprocess the new frame
    """
    processed_frame = preprocess(frame)
    if(new_episode):
        stacked_frames = deque([ np.zeros([96,96] ,dtype=int) for i_dx in range(stacked_size) ] , maxlen=stacked_size)
        #for initial step after new episode add same frame to all the stacked frames
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
        
        stack_states = np.stack(stacked_frames , axis=0)
        
        return stack_states , stacked_frames
    
    else:
        stacked_frames.append(processed_frame)
        
        stack_states = np.stack(stacked_frames , axis=0)
        
        return stack_states , stacked_frames




device='cuda:0' if torch.cuda.is_available() else 'cpu'
class ReplayBuffer :
    def __init__(self , batch_size , buffer_size , seed):
        self.batch_size = batch_size 
        self.buffer_size = buffer_size 
        self.seed = seed
        random.seed(self.seed)
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience' , field_names=['state','action','reward','next_state','done'])
        
    def add(self , state , action , reward , next_state , done):
        experience = self.experience(state , action , reward , next_state , done)
        self.memory.append(experience)
        
    def sample(self):
        
        experinece_batch = random.sample(self.memory , k=self.batch_size)
        
        states = torch.from_numpy( np.stack([ e.state.reshape(4,96,96) for e in experinece_batch if e is not None ],axis=0) ).float().to(device)
        action = torch.from_numpy( np.stack([ (e.action).reshape(1,) for e in experinece_batch if e is not None] , axis=0) ).long().to(device)
        reward = torch.from_numpy( np.stack([ np.array((e.reward)).reshape(1,) for e in experinece_batch if e is not None] , axis=0) ).float().to(device)
        next_state = torch.from_numpy( np.stack([ e.next_state.reshape(4,96,96) for e in experinece_batch if e is not None ] , axis=0) ).float().to(device)
        done = torch.from_numpy( np.stack([ np.array((e.done)).reshape(1,) for e in experinece_batch if e is not None], axis=0).astype(np.uint8) ).float().to(device)
        
        return ( states , action , reward , next_state , done )
    
    def __len__(self):
        return len(self.memory)




BATCH_SIZE = 16
UPDATE_EVERY = 4
SEED= 1244
BUFFER_SIZE=int(1e3)
GAMMA= 0.99
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 

class Agent :
    def __init__(self,state_size , action_size , seed):
        
        self.state_size  = state_size 
        self.action_size = action_size
        self.seed  = random.seed(seed)
        self.train_loss = deque(maxlen=500)
        """
        step --> add the data to experience tuple and if in the update instance train the model
        act --> output the action for states
        learn --> train the model using experince tuple
        soft_update --> update the target network 
        """
        """
        1 --> define local and target  Q networks
        2 --> initialize the optimizer
        3 --> initialize the raply buffer
        
        """
        self.q_local = DeepQNet(self.state_size , self.action_size , seed).to(device)
        self.q_target = DeepQNet(self.state_size , self.action_size , seed).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters() , lr=LR)
        
        self.replay_buffer = ReplayBuffer(BATCH_SIZE , BUFFER_SIZE , SEED)
        self.t_step = 0    # for determine the update instances
        
    def step(self,state , action , reward , next_state , done):
        
        self.replay_buffer.add(state , action , reward , next_state , done)
        self.t_step = (self.t_step + 1)%UPDATE_EVERY
        if(self.t_step == 0):
            if(len(self.replay_buffer) > BATCH_SIZE):
                train_sample = self.replay_buffer.sample()
                self.learn(train_sample , GAMMA)
                
    def learn(self , train_sample , GAMMA) :
        
        state , action , reward , next_state , done = train_sample
        """
        1 --> infernce the local model with state and take the Q values for actions associated
        2 --> indernce the target model and get the maximum Q value
        3 --> obtain the q target with ( reward + GAMMA * max(Q_target_model(next_state))*(1-dones))
        4 --> MSELoss(Q_local , Q_target)
        5 --> optimize the model
        """
        Q_expected = self.q_local(state).gather(1,action)
        Q_target = self.q_target(next_state).detach().max(1)[0].unsqueeze(1)
        Q_target = reward + GAMMA*(Q_target*(1-done))
        
        q_loss = F.mse_loss(Q_expected , Q_target)
        self.train_loss.append(q_loss.to('cpu').detach().item())
        # reset the optimizer
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        #update the weights of target model weights
        self.soft_update(self.q_target , self.q_local ,TAU)
        
    def act(self , state , eps=0.):
        """
        expand the dim 0 of the state tensor
        base on eps either select max action or a random action
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_local.eval()
        with torch.no_grad():
            action_tensor = self.q_local(state_tensor)
        self.q_local.train()
        
        # if the random.random > eps --> select a random action else select greedy action
        if(random.random() > eps ):
            return np.argmax(action_tensor.to('cpu').detach().numpy())
        
        else :
            return random.choice(np.arange(self.action_size))
        
    def soft_update(self , target_model , local_model , tau) :
        
        for target_param , local_param in zip(target_model.parameters() , local_model.parameters()):
            
            target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)
        




import torch.nn as nn
import torch.nn.functional as F




def conv_block(in_channels , out_channles , kernel_size =3 , stride=1 , padding=1 , batch_norm=True , maxpool=False ):
    layers =[]
    conv_layer = nn.Conv2d(in_channels=in_channels , out_channels=out_channles , kernel_size=kernel_size ,
                          stride=stride , padding=padding , padding_mode='reflect', bias=False)
    layers.append(conv_layer)
    
    if(batch_norm):
        bn = nn.BatchNorm2d(out_channles)
        layers.append(bn)
    layers.append(nn.ReLU())
    
    if(maxpool):
        max_layer = nn.MaxPool2d(kernel_size=4, stride=2 , padding=1)
        layers.append(max_layer)
        
    return nn.Sequential(*layers)




class DeepQNet(nn.Module):
    def __init__(self,stack_size , action_size , seed):
        super(DeepQNet , self).__init__()
        """
        define a simple model with some conv layers and later with fully connected layers
        """
        self.in_size = stack_size
        self.out_size = action_size
        self.seed = torch.manual_seed(seed)
        # 224 * 224 * 4 --> 224 * 224 * 32
        self.conv_block1 = conv_block(self.in_size , 32 )
        # 224 * 224 * 32 --> 112 * 112 * 64
        self.conv_block2 = conv_block(32 , 64 , maxpool=True)
        #112 * 112 * 64 --> 112 * 112 * 128
        self.conv_block3 = conv_block(64 , 128 )
        # 112 * 112 * 128 --> 56 * 56 * 256
        self.conv_block4 = conv_block(128,256,maxpool=True)
        # 56 * 56 * 256 --> 56 * 56 * 512
        self.conv_block5 = conv_block(256 , 512)
        # 56 * 56 * 512 --> 28 * 28 * 1024 
        self.conv_block6 = conv_block(512 , 1024 , maxpool=True)
        self.flatten_size = 12*12*1024
        self.fc1 = nn.Linear(self.flatten_size , 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512 , self.out_size)
        
    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = x.view(-1,self.flatten_size)
        x = F.dropout(F.relu(self.fc_bn(self.fc1(x))) ,p=0.4 )
        x = self.fc2(x) 
        
        return x




agent = Agent(state_size=4, action_size=3, seed=1243)




agent.q_local.load_state_dict(torch.load('../input/doom-rl/local_model.pth'))
agent.q_target.load_state_dict(torch.load('../input/doom-rl/target_model.pth'))




# Init the game
shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]
total_episodes=2500
game = initialize_vizdoom(DEFAULT_CONFIG , DEFAULT_SCN)
explore_probability = 0.4746
explore_probability_end=0.01
eps_decay = 0.9995
max_steps = 500
for episode in range(total_episodes):
    # Set step to 0
    step = 0
            
    # Initialize the rewards of the episode
    episode_rewards = []
            
    # Make a new episode and observe the first state
    game.new_episode()
    state = game.get_state().screen_buffer
            
    # Remember that stack frame function also call our preprocess function.
    state, stacked_frames = stack_frames(None, state, True)

    while step < max_steps:
        step += 1 
        # Predict the action to take and take it
        action  = agent.act(state , eps=explore_probability)

        # Do the action
        reward = game.make_action(actions[action])

        # Look if the episode is finished
        done = game.is_episode_finished()
                
        # Add the reward to total reward
        episode_rewards.append(reward)

        # If the game is finished
        if done:
            # the episode ends so no next state
            next_state = np.zeros((96,96), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Set step = max_steps to end the episode
            step = max_steps

            # Get the total reward of the episode
            total_reward = np.sum(episode_rewards)
            if(episode%10==0):
                print('Episode: {}'.format(episode),
                        'Total reward: {}'.format(total_reward),
                        'Training loss: {:.4f}'.format(np.mean(agent.train_loss)),
                        'Explore P: {:.4f}'.format(explore_probability))

            agent.step(state, action, reward, next_state, done)

        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
                    
            # Stack the frame of the next_state
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    
            # Add experience to memory
            agent.step(state, action, reward, next_state, done)
                    
            # st+1 is now our current state
            state = next_state
    
    explore_probability = max(explore_probability_end, explore_probability*eps_decay)




torch.save(agent.q_local.state_dict(),'local_model.pth')
torch.save(agent.q_target.state_dict(),'target_model.pth')




img_frames = []
max_steps=500
game = initialize_vizdoom(DEFAULT_CONFIG , DEFAULT_SCN)
# Make a new episode and observe the first state
game.new_episode()

state = game.get_state().screen_buffer
img_frames.append(state)          
# Remember that stack frame function also call our preprocess function.
state, stacked_frames = stack_frames(None, state, True)
agent.q_local.eval()
step=0
while step < max_steps:
    step += 1 
    # Predict the action to take and take it
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action_tensor = agent.q_local(state_tensor)
    action = np.argmax(action_tensor.to('cpu').detach().numpy()) 
    # Do the action
    reward = game.make_action(actions[action])

    # Look if the episode is finished
    done = game.is_episode_finished()
                
    # Add the reward to total reward
    episode_rewards.append(reward)
    # If the game is finished
    if done:
        # the episode ends so no next state
        next_state = np.zeros((96,96), dtype=np.int)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Set step = max_steps to end the episode
        step = max_steps

        # Get the total reward of the episode
        total_reward = np.sum(episode_rewards)
        print('Total reward: {}'.format(total_reward))

    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        img_frames.append(next_state)        
        # Stack the frame of the next_state
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    
        # st+1 is now our current state
        state = next_state




data_path='data'
import PIL
for idx , image  in enumerate(img_frames):
 PIL.Image.fromarray(image).save('data/{}.png'.format(str(idx)))




import os
os.system('ffmpeg -r 10 -i data/%1d.png -vcodec libx264 -b 10M -y FlowVideo.mp4  ')




from IPython.display import HTML
from base64 import b64encode
mp4 = open('FlowVideo.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)




for image in img_frames:
    plt.imshow(image)
    plt.show()






