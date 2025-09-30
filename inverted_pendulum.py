
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DQN(nn.Module):
    def __init__(self,n_obs,n_acts,h_dim=128):
        super(DQN,self).__init__()
        self.layer_a=nn.Linear(n_obs,h_dim)
        self.layer_b=nn.Linear(h_dim,h_dim)
        self.layer_c=nn.Linear(h_dim,n_acts)

    def forward(self,x):
        x=F.relu(self.layer_a(x))
        x=F.relu(self.layer_b(x))
        return self.layer_c(x)

class ReplayBuffer:
    def __init__(self,max_mem,batch_size):
        self.memory=deque(maxlen=max_mem)
        self.batch_size=batch_size

    def add(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def sample(self):
        experiences=random.sample(self.memory,k=self.batch_size)
        states=torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions=torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards=torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states=torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones=torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return states,actions,rewards,next_states,dones

    def __len__(self):
        return len(self.memory)

class DQNAgent: 
    def __init__(self, n_obs, n_acts, lr=0.001):
        self.n_obs=n_obs
        self.n_acts=n_acts
        self.lr=lr

        # Hyperparameters
        self.max_mem=10000
        self.batch_size=64
        self.gamma=0.99
        self.tau=0.001
        self.eps_value=1.0
        self.eps_value_min=0.01
        self.eps_value_decay=0.900
        self.learn_frequency=4

        self.qnetwork_local=DQN(n_obs,n_acts)
        self.target_net=DQN(n_obs,n_acts)
        self.optimizer=optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory=ReplayBuffer(self.max_mem, self.batch_size)
        self.t_step=0

    def step(self,state,action,reward,next_state,done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step=(self.t_step+1)%self.learn_frequency
        if self.t_step==0 and len(self.memory)>self.batch_size:
            experiences=self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=None):
        if eps is None:
            eps=self.eps_value
        state=torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values=self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_acts))

    def learn(self, experiences):
        states,actions,rewards,next_states,dones=experiences
        Q_targets_next=self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets=rewards+(self.gamma * Q_targets_next * (1 - dones))
        Q_expected=self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.target_net, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train_dqn(n_episodes=2000, max_t=500, solve_score=195.0):
    env=gym.make('CartPole-v1')
    agent=DQNAgent(n_obs=4, n_acts=2)
    scores=[]
    scores_window=deque(maxlen=100)
    start_time=time.time()

    for i_episode in range(1,n_episodes+1):
        state=env.reset()
        if isinstance(state, tuple):
            state=state[0]
        score=0
        for t in range(max_t):
            action = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, done, _, _ = result
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        agent.eps_value = max(agent.eps_value_min, agent.eps_value_decay * agent.eps_value)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= solve_score:
            torch.save(agent.qnetwork_local.state_dict(), 'dqn_final.pth')
            break

    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = total_time % 60
    print(f"\nTotal Training Time: {int(minutes)} min {int(seconds)} sec")
    env.close()
    return scores, agent

def plot_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN training progress')
    plt.show()

if __name__ == "__main__":
    print("Training DQN Agent on cartpole")
    scores, trained_agent = train_dqn()
    print("\nPLOTTING")
    plot_scores(scores)
    print("\n ALL DONE")
