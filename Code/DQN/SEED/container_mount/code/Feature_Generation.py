#Code for Feature generation: save activations of the last layer neurons
#python3 Feature_Extraction_SeedRL.py -r /workspace/ROMs/enduro.bin -cp 0/ckpt-98 -g enduro
#This code contrains portions originating from Lasse Espeholt, Raphaël Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski
#and is licensed under the Apache 2.0 License. See: https://github.com/google-research/seed_rl
#Code, Functions and structures were adopted from the original repository. Original code is marked with [*]. 


import argparse
import os
import numpy as np
from ale_python_interface import ALEInterface
import pygame

from seed_rl.atari import networks
from seed_rl.common import utils
import math
import tensorflow as tf
import cv2
import gym

class RunSeed:
    
    def create_agent(self): #[*]
        return networks.DuelingLSTMDQNNet(18, (84, 84, 1), 4)
    
    def create_optimizer_fn(self, unused_final_iteration): #[*]
        learning_rate_fn = lambda iteration: 0.00048
        optimizer = tf.keras.optimizers.Adam(0.00048, epsilon=1e-3)
        return optimizer, learning_rate_fn
    
    def _pool_and_resize(self, screen_buffer):
        transformed_image = cv2.resize(screen_buffer, (84, 84), interpolation=cv2.INTER_LINEAR)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return int_image
    
    def __init__(self, game_name, checkpoint_path):
        #code from here: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        
        self.agent = self.create_agent() #[*]
        target_agent = self.create_agent() #[*]

        
        optimizer, learning_rate_fn = self.create_optimizer_fn(None) #[*]
        
        ckpt = tf.train.Checkpoint(agent=self.agent, target_agent=target_agent, optimizer=optimizer) #[*]
        
        if game_name == 'enduro.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/checkpoints/Enduro', 
                                      checkpoint_path))
            env = gym.make('EnduroNoFrameskip-v4', full_action_space=True) 
        elif game_name == 'breakout.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/checkpoints/Breakout', 
                                      checkpoint_path))
            env = gym.make('BreakoutNoFrameskip-v4', full_action_space=True) 
        elif game_name == 'space_invaders.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/checkpoints/SpaceInvaders', 
                                      checkpoint_path))
            env = gym.make('SpaceInvadersNoFrameskip-v4', full_action_space=True) 
        else:
            raise FileExistsError(game_name + ' not found!')
        
        env.seed(0)  
        env.reset()
        env.step(0) 

        obs_dims = env.observation_space 
        screen_buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)

        env.observation_space.shape = (84, 84, 1) 

        env_output_specs = utils.EnvOutput(
            tf.TensorSpec([], tf.float32, 'reward'),
            tf.TensorSpec([], tf.bool, 'done'),
            tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                          'observation'),
            tf.TensorSpec([], tf.bool, 'abandoned'),
            tf.TensorSpec([], tf.int32, 'episode_step')) #[*]
        
        action_specs = tf.TensorSpec([], tf.int32, 'action')  #[*]
        agent_input_specs = (action_specs, env_output_specs) #[*]

        self.initial_agent_state = self.agent.initial_state(1) #[*]
        agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), 
                                                  self.initial_agent_state) #[*]
        self.input_ = tf.nest.map_structure(lambda s: tf.zeros([1] + list(s.shape), s.dtype), 
                                            agent_input_specs) #[*]

        self.current_agent_state = self.initial_agent_state
        
        self.last_15hz_screen_1_84_84_1 = np.empty([1, 84, 84, 1], dtype=np.uint8)

        self.episode_step_60hz = 0

    
    def __call__(self, screen_Gray_max, observed_reward, action_played, episode_end_flag):

        last_15hz_screen_max_84x84 = self._pool_and_resize(screen_Gray_max)
        self.last_15hz_screen_1_84_84_1[0, :, :, 0] = last_15hz_screen_max_84x84

        observed_reward_np = np.zeros((1,), dtype=np.float32)
        observed_reward_np[0] = observed_reward
        episode_end_flag_np = np.zeros((1,), dtype=np.bool)
        episode_end_flag_np[0] = episode_end_flag
        self.episode_step_60hz += 4
        episode_step_60hz_np = np.zeros((1,), dtype=np.int32)
        episode_step_60hz_np[0] = self.episode_step_60hz
        
        input_action, input_env = self.input_ 
        input_env = input_env._replace(observation=tf.convert_to_tensor(self.last_15hz_screen_1_84_84_1,
                                                                 dtype=np.uint8))
        input_env = input_env._replace(reward=tf.convert_to_tensor(observed_reward_np, 
                                                                 dtype=np.float32))
        input_env = input_env._replace(done=tf.convert_to_tensor(episode_end_flag_np, 
                                                                 dtype=np.bool))
        input_env = input_env._replace(episode_step=tf.convert_to_tensor(episode_step_60hz_np, 
                                                                 dtype=np.int32))
        #input_action = tf.convert_to_tensor(action_played, dtype=np.int32)
        self.input_ = (input_action, input_env)
        
        if episode_end_flag:
            self.current_agent_state = self.initial_agent_state
            self.episode_step_60hz = 0
            print(' ')
            print('Episode ended, set LSTM core to initial state')
                
        agent_out = self.agent(self.input_, self.current_agent_state)
        AgentOutput, AgentState = agent_out
        self.current_agent_state = AgentState
        q_values = AgentOutput.q_values 
        
        return q_values 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ROM_path', help='ROM file path', type=str, required=True)
    parser.add_argument('-cp', '--checkpoint_path', help='Path for checkpoint files', type=str,
    required=True)
    parser.add_argument('-g', '--game', help='Game name', type=str,
                    required=True)
    args = parser.parse_args()

    game_name = os.path.basename(args.ROM_path)
    n_frames_60hz = 20000 
    n_frames_15hz = n_frames_60hz//4

    n_sessions = 7

    prob_file = open('/workspace/container_mount/code/Code_Subjects','r')
    prob_code = prob_file.read().splitlines()

    for participant in prob_code:

        pseudo_code = participant

        for session_no in range(1,n_sessions+1):
            print(session_no)
            run_seed = RunSeed(game_name, args.checkpoint_path)
            ActValuesLayerOutput = np.zeros((n_frames_15hz,1,18))

            episode_end_flag = False 
            episode_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game + '/'
                             + pseudo_code +  '_' + args.game + '_E_' + str(session_no) + '_episode_vec.csv'
            episode_vec = np.loadtxt(episode_path, dtype=np.int)
            episode_vec = np.append(episode_vec,0) 
            current_episode = 0
            screen_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game + '/'
                          + pseudo_code + '_' + args.game  + '_E_' + str(session_no) + '_screen_15hz_Gray.npy'
            Video15Hz = np.load(screen_path)
            responses_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game + '/'
                             + pseudo_code + '_' + args.game  + '_E_' + str(session_no) + '_responses_vec.csv'
            Responses_human = np.loadtxt(responses_path, dtype=np.int)

            reward_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game + '/'
                          + pseudo_code + '_' + args.game  + '_E_' + str(session_no) + '_reward_vec.csv' 
            reward = np.loadtxt(reward_path, dtype=np.float32)
            reward  = np.append(reward,0)

            for loop_count_15hz in range(n_frames_15hz):
                print(loop_count_15hz)

                Screen15HzCurrent = Video15Hz[:,:,:,loop_count_15hz]	
                screen1 = Screen15HzCurrent[:,:,0]
                screen2 = Screen15HzCurrent[:,:,1]
                screen_Gray_max = np.amax(np.dstack((screen1, screen2)), axis=2)

                observed_reward = 0.0
                action_played = 0

                for i in range(0,4):

                    if (episode_vec[loop_count_15hz*4+i] - episode_vec[loop_count_15hz*4+i-1]) == 0:  
                        observed_reward += (reward[loop_count_15hz*4+i] - reward[loop_count_15hz*4+i-1])
                    else:
                        observed_reward += reward[loop_count_15hz*4+i]        

                    if Responses_human[loop_count_15hz*4+i]>0: 
                        action_played = Responses_human[loop_count_15hz*4+i]

                #action_played = Responses_human[loop_count_15hz*4+2]

                if episode_vec[loop_count_15hz*4+2] == current_episode:
                    episode_end_flag = False
                else:
                    episode_end_flag = True 
                    current_episode += 1

                LastLayer_15hz = run_seed(screen_Gray_max, observed_reward, action_played, episode_end_flag)

                ActValuesLayerOutput[loop_count_15hz] = LastLayer_15hz.numpy()

            model_name = args.checkpoint_path.replace('/', '-')

            del run_seed

if __name__ == '__main__':
    main()
