import gymnasium as gym
import math
import numpy as np
from itertools import count
import torch
import torch.nn as nn
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 50
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v1'
PRINT_INTERVAL = 50

env = gym.make(ENV_NAME)
state_shape = len(env.reset()[0])
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    # TODO implement an epsilon-greedy strategy
    if np.random.random() < EPS_EXPLORATION:
        return torch.from_numpy(np.array(env.action_space.sample()).reshape((1,1))).to(device)
    else:
        return model.select_action(torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0))

def optimize_model(state, action, next_state, reward, done):
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    not_done = torch.logical_not(done).to(torch.float32)
    
    targetQ = reward + GAMMA * not_done * torch.max(target(next_state), dim=1)[0]

    forward_pass = model(state)
    m, _ = forward_pass.shape
    loss = nn.functional.mse_loss(targetQ, forward_pass[torch.arange(m), action.squeeze()])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")
    
    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state, _ = env.reset()
        for t in count():
            action = choose_action(state) # Should be size torch(1,1)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0][0])
            memory.push(state, action.cpu().numpy()[0][0], next_state, reward, terminated)
            steps_done += 1
            episode_total_reward += reward

            if len(memory) >= BATCH_SIZE:
                optimize_model(*memory.sample(BATCH_SIZE))
            else:
                optimize_model(*memory.sample(len(memory)))

            state = next_state

            if render:
                env.render()

            if terminated or truncated:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
