import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Directions(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

block_size = 20
speed = 15

class SnakeGame():

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.iter = 0
        self.reset()

    def reset(self):
        self.direction = Directions.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-block_size, self.head.y),
                      Point(self.head.x-(2*block_size), self.head.y)]
        # self.snake.insert(0, self.head)

        self.score = 0
        self.food = None
        self.iter = 0
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-block_size)//block_size)*block_size
        y = random.randint(0, (self.h-block_size)//block_size)*block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_game(self, action):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True, self.score

        # move
        self._move(action)
        self.snake.insert(0, self.head)

        self.iter += 1
        # print(self.iter)

        # check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            # print("Iter: ",self.iter)
            reward = -10
            game_over = True
            return reward, game_over, self.score
        
        if self.iter > 1000:
            reward = -10
            game_over = True
            return reward, game_over, self.score
        
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self.update_ui()
        self.clock.tick(speed)

        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x > self.w-block_size or pt.x < 0 or pt.y > self.h-block_size or pt.y < 0:
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        text = font.render("Score: "+str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def _move(self, action):

        clock_wise = [Directions.RIGHT, Directions.DOWN, Directions.LEFT, Directions.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1,0,0]): # Same direction
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]): # Take right turn
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        elif np.array_equal(action, [0,0,1]): # Take left turn
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Directions.RIGHT:
            x += block_size
        elif self.direction == Directions.LEFT:
            x -= block_size
        elif self.direction == Directions.DOWN:
            y += block_size
        elif self.direction == Directions.UP:
            y -= block_size

        self.head = Point(x,y)


# if __name__ == '__main__':
#     game = SnakeGame()

#     action = [1, 0, 0]  # Start moving in the current direction

#     while True:
#         action = [1, 0, 0]
#         # Handle key presses for direction
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_RIGHT:
#                     action = [0, 1, 0]  # turn right
#                 elif event.key == pygame.K_LEFT:
#                     action = [0, 0, 1]  # turn left

#         reward, game_over, score = game.play_game(action)
#         if game_over:
#             print(f"Final Score: {score}")
#             pygame.time.wait(2000)
#             game.reset()
