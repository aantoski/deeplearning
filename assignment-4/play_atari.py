import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from PIL import Image

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNPolicyNet(nn.Module):
    def __init__(self, num_actions):
        super(CNNPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=2, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1) 
        self.fc = nn.Linear(128 * 12 * 12, 128)

        # Policy head
        self.policy_head = nn.Linear(128, num_actions)  # Output: num_actions

        # Value head
        self.value_head = nn.Linear(128, 1)  # Output: single value

    def forward(self, x):
        x = torch.moveaxis(x.float().to(device), 3, 1) / 255. # Normalize and move channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        
        policy = self.policy_head(x)  # Policy output
        value = self.value_head(x)    # Value output
        return policy, value

    def predict(self, x):  # for compatibility with OpenAI Gym
        # Convert obs to a torch tensor if it's a NumPy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        policy_logits, _ = self.forward(x)
        action_probs = F.softmax(policy_logits, dim=-1)
        return action_probs

# Play the game using the specified control policy
def render_env(env, policy, max_steps=500):
    obs = env.reset()
    for i in range(max_steps):
        with torch.no_grad():
            actionProbs = policy.predict(obs)
        # action = torch.multinomial(actionProbs, 1).item()  # randomly pick an action according to policy
        action = torch.argmax(actionProbs).item()  # pick the action with the highest probability
        obs, reward, done, info = env.step([action])
        if done:
            break  # game over

    env.close()

def train_model(model, data_loader, batch_size=100, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_value = nn.MSELoss()  # For value head
    criterion_policy = nn.CrossEntropyLoss()  # For policy head
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (observations, targets_policy, targets_value) in enumerate(data_loader):
            observations, targets_policy, targets_value = (
            observations.to(device),
            targets_policy.to(device),
            targets_value.to(device)
            )
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_output, value_output = model(observations)
            target_values = targets_value.view(-1, 1)
            
            # Calculate losses
            policy_loss = criterion_policy(policy_output, targets_policy)
            value_loss = criterion_value(value_output, target_values)
            
            # Combine the losses
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], "
                  f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    print("Training complete!\n Saving model...")        
    torch.save(model.state_dict(), "o128-k2-s2.cpt")
    print("Model saved!")

# Change for training or rendering
train = False

observations = torch.load('pong_observations.pt')
actions = torch.load('pong_actions.pt')
batch_size = 256

# Create the Pong environment
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=2, env_kwargs={'render_mode':"human"})
env = VecFrameStack(env, n_stack=4)

if train:
    # Create a DataLoader
    dataset = TensorDataset(observations, actions, torch.rand((15261, 1)))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    model = CNNPolicyNet(env.action_space.n).to(device)
    train_model(model, train_loader, batch_size=batch_size, num_epochs=10, learning_rate=0.001)
else:
    # Load the model and render the environment
    model = CNNPolicyNet(env.action_space.n).to(device)
    model.load_state_dict(torch.load("o128-k2-s2.cpt", map_location=device))
    render_env(env, model, 5000)
