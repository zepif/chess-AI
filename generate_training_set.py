import numpy as np
import chess
import chess.engine
import random 
import os
from state import Board

state = Board()

score = None

while score == None:
    board = state.random_board()
    score = state.stockfish(board, 15)
    state.board = board
    state.split_dims()

arr1 = np.empty(0)
arr2 = np.empty(0)
arr1 = np.append(arr1, state.board3d)
arr2  = np.append(arr2, score)

np.savez('dataset.npz', b=arr1, v=arr2)

data = np.load('dataset.npz', allow_pickle=True, encoding='latin1')

print(data['b'].shape)

sz = 100 # 10^6

new_data = {}
new_data['b'] = data['b']
print(new_data['b'])
new_data['v'] = data['v']

i = 1
while i < sz:
    board = state.random_board()
    score = state.stockfish(board, 15)
    if score == None:
        i -= 1
        continue
    state.board = board
    state.split_dims()

    new_data['b'] = np.append(new_data['b'], state.board3d)
    new_data['v'] = np.append(new_data['v'], score)
    i += 1

print(new_data['b'])
print(new_data['b'].dtype)
print(new_data['b'].shape)
print(new_data['v'])
print(new_data['v'].size)
np.savez('dataset.npz', b=new_data['b'], v=new_data['v'])
