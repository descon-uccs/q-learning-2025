# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:53:30 2025

@author: phili
"""

from typing import Tuple, Optional
import random
import numpy as np


class InertialGridworldEnv:
    def __init__(self, width: int, height: int, start: Tuple[int, int], goal: Tuple[int, int]):
        self.width = width
        self.height = height
        self.start_pos = start
        self.goal_pos = goal

        self.motions = ['none', 'up', 'down', 'left', 'right']
        self.actions = ['thrust_up', 'thrust_down', 'thrust_left', 'thrust_right']

        self.motion_map = {
            'thrust_up': 'up',
            'thrust_down': 'down',
            'thrust_left': 'left',
            'thrust_right': 'right',
        }
        self.opposite = {
            'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'
        }

        self.state = None  # (x, y, motion)

    def reset(self) -> Tuple[int, int, str]:
        self.state = (*self.start_pos, 'none')
        return self.state

    def step(self, action: str) -> Tuple[Tuple[int, int, str], int, bool, dict]:
        x, y, motion = self.state

        # --- Update motion ---
        thrust_dir = self.motion_map[action]
        if motion == 'none':
            new_motion = thrust_dir
        elif self.opposite[motion] == thrust_dir:
            new_motion = 'none'
        else:
            new_motion = motion

        # --- Apply motion ---
        dx, dy = self._motion_delta(new_motion)
        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        self.state = (new_x, new_y, new_motion)

        # --- Compute reward and done ---
        done = (new_x, new_y) == self.goal_pos

        delay_penalty = -1 if not done else 0
        thrust_penalty = -1 if motion != new_motion else 0
        reward = 10 if done else delay_penalty + thrust_penalty

        return self.state, reward, done, {}

    def render(self):
        AsciiVisualizer.render(self)

    def _motion_delta(self, motion: str) -> Tuple[int, int]:
        return {
            'none': (0, 0),
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }[motion]

    def encode_state(self, state: Tuple[int, int, str]) -> int:
        x, y, motion = state
        motion_idx = self.motions.index(motion)
        return (y * self.width + x) * len(self.motions) + motion_idx

    def decode_state(self, index: int) -> Tuple[int, int, str]:
        motion_idx = index % len(self.motions)
        flat_pos = index // len(self.motions)
        x = flat_pos % self.width
        y = flat_pos // self.width
        return (x, y, self.motions[motion_idx])


class AsciiVisualizer:
    @staticmethod
    def render(env: InertialGridworldEnv):
        grid = [[' . ' for _ in range(env.width)] for _ in range(env.height)]
        gx, gy = env.goal_pos
        grid[gy][gx] = ' G '

        x, y, motion = env.state
        agent_char = {
            'none': ' A ',
            'up': ' ↑ ',
            'down': ' ↓ ',
            'left': ' ← ',
            'right': ' → '
        }[motion]
        grid[y][x] = agent_char

        print('\n'.join(''.join(row) for row in grid))
        print()


class QTableAgent:
    def __init__(self, env: InertialGridworldEnv, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.num_states = env.width * env.height * len(env.motions)
        self.num_actions = len(env.actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def select_action(self, state: Tuple[int, int, str]) -> str:
        state_idx = self.env.encode_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            action_idx = int(np.argmax(self.q_table[state_idx]))
            return self.env.actions[action_idx]

    def update(self, state, action, reward, next_state):
        s = self.env.encode_state(state)
        a = self.env.actions.index(action)
        s_prime = self.env.encode_state(next_state)

        td_target = reward + self.gamma * np.max(self.q_table[s_prime])
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def train(self, episodes=500, max_steps=100):
        for ep in range(episodes):
            state = self.env.reset()
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                if done:
                    break




if __name__ == "__main__":
    # Create a 5x5 grid, start at top-left, goal at bottom-right
    env = InertialGridworldEnv(width=5, height=5, start=(0, 0), goal=(4, 4))
    state = env.reset()
    print("Initial state:")
    env.render()

    # Hardcoded action sequence to test inertial movement
    actions = [
        'thrust_right',  # sets motion → right
        'thrust_down',   # ignored, motion stays right
        'thrust_left',   # cancels motion → none
        'thrust_down',   # sets motion → down
        'thrust_down',   # motion continues down
        'thrust_up',     # cancels motion → none
    ]

    for i, action in enumerate(actions):
        print(f"Step {i+1}: Action = {action}")
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        if done:
            break

    Q = QTableAgent(env)
    Q.train()
    
