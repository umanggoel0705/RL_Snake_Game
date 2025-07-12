import sys
import traceback
import torch
import random
import numpy as np
import pygame
from collections import deque
from game_script import SnakeGame, Directions, Point
from model import Linear_Qnet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class Agent:

    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        
        pt_l = Point(head.x-20, head.y)
        pt_r = Point(head.x+20, head.y)
        pt_u = Point(head.x, head.y-20)
        pt_d = Point(head.x, head.y+20)

        dir_l = game.direction == Directions.LEFT
        dir_r = game.direction == Directions.RIGHT
        dir_d = game.direction == Directions.DOWN
        dir_u = game.direction == Directions.UP

        state = [
            # Danger straight
            (dir_l == game.is_collision(pt_l)) or
            (dir_r == game.is_collision(pt_r)) or
            (dir_u == game.is_collision(pt_u)) or
            (dir_d == game.is_collision(pt_d)),

            # Danger right
            (dir_l == game.is_collision(pt_u)) or
            (dir_r == game.is_collision(pt_d)) or
            (dir_u == game.is_collision(pt_r)) or
            (dir_d == game.is_collision(pt_l)),

            # Danger left
            (dir_l == game.is_collision(pt_u)) or
            (dir_r == game.is_collision(pt_d)) or
            (dir_u == game.is_collision(pt_l)) or
            (dir_d == game.is_collision(pt_r)),

            # Moving directions
            dir_l,
            dir_r, 
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # Food left
            game.food.x > game.head.x, # Food right
            game.food.y < game.head.y, # Food up
            game.food.y > game.head.y  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # print(mini_sample)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves : tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move
        result = game.play_game(final_move)

        if result is None:
            # User closed the window
            pygame.quit()
            import sys
            sys.exit()

        reward, done, score = result

        # get new state
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game:", agent.n_game, "Score:",score, "Reward:",reward)
            # break
            plot_scores.append(score)
            total_scores += score
            mean_scores = total_scores / agent.n_game
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)