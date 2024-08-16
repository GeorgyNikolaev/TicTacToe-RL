import random

import numpy as np
from keras.api.models import Sequential
from keras.api import layers


class TicTacToe:
    def __init__(self,
                 polygone_size: int = 3,
                 award: int = 1):
        self.playing_field = None
        self.polygone_size = polygone_size
        self.award = award

        self.reset()

    def step(self,
             action: tuple[int, int]):
        done = False
        reward = 0

        # Ход.
        pos_value = self.playing_field[action[0], action[1]]

        if pos_value != 0:
            reward -= 100 * self.award
        else:
            self.playing_field[action[0], action[1]] = 1

        cnt = self.get_reward(1)
        if cnt == -1:
            done = True
            reward += 5 * self.award
        else:
            reward += cnt * self.award

        # Рандомный ход.
        move_index_probability = []
        for i in range(self.polygone_size):
            for j in range(self.polygone_size):
                if self.playing_field[i, j] == 0:
                    move_index_probability.append([i, j])

        cnt_move = len(move_index_probability)
        if cnt_move < 1:
            done = True
        elif cnt == 1:
            move = move_index_probability[0]
        else:
            move = move_index_probability[random.randint(0, len(move_index_probability) - 1)]

        if not done:
            self.playing_field[move[0], move[1]] = 2

        enemy_cnt = self.get_reward(2)
        if enemy_cnt == -1:
            reward -= 5 * self.award
            done = True
        else:
            reward -= cnt * self.award * 0.5

        if np.count_nonzero(np.reshape(self.playing_field, (self.polygone_size ** 2)) == 0) == 0:
            done = True

        return self.playing_field, reward, done

    def get_reward(self,
                   side: int) -> int:
        reward = 0
        for i in range(self.polygone_size):
            # По вертикали.
            cnt = np.count_nonzero(self.playing_field[i, :] == side)
            if cnt == self.polygone_size:
                return -1
            reward += (cnt - 1) if (cnt > 0) else 0

            # По горизонтали.
            cnt = np.count_nonzero(self.playing_field[:, i] == side)
            if cnt == self.polygone_size:
                return -1
            reward += (cnt - 1) if (cnt > 0) else 0

        cnt1 = cnt2 = 0
        for i in range(self.polygone_size):
            cnt1 += 1 if (self.playing_field[i, i] == side) else 0
            cnt2 += 1 if (self.playing_field[i, -i - 1] == side) else 0
        if cnt1 == self.polygone_size or cnt2 == self.polygone_size:
            return -1
        reward += (cnt1 - 1) if (cnt1 > 0) else 0
        reward += (cnt2 - 1) if (cnt2 > 0) else 0

        return reward

    def reset(self):
        self.playing_field = np.zeros(shape=(self.polygone_size, self.polygone_size))
        return self.playing_field
