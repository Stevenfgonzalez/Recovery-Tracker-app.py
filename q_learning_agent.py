import json
import numpy as np
import os
import random

class QLearningAgent:
    def __init__(self, actions, state_space=11, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_file="q_table.json"):
        self.actions = actions
        self.state_space = state_space  # pain levels 0-10
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "r") as f:
                return json.load(f)
        else:
            # Initialize Q-table with zeros
            return {str(s): [0.0 for _ in self.actions] for s in range(self.state_space)}

    def save_q_table(self):
        with open(self.q_table_file, "w") as f:
            json.dump(self.q_table, f)

    def choose_action(self, state):
        state = str(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        a_idx = self.actions.index(action)
        q_predict = self.q_table[state][a_idx]
        q_target = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][a_idx] += self.alpha * (q_target - q_predict)
        self.save_q_table() 