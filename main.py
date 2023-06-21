import numpy as np
import chess
import chess.engine
import random 
import os
from state import Board
from tensorflow import keras

model = keras.models.load_model("model.h5")

state = Board()

def minimax_eval(state):
    state.split_dims()
    state.board3d = np.expand_dims(state.board3d, 0)
    return model.predict(state.board3d)[0][0]

def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.board.is_game_over():
        return minimax_eval(state.board)
    
    if maximizing_player:
        max_eval = -np.inf
        for move in state.board.legal_moves:
            state.board.push(move)
            eval = minimax(state, depth - 1, alpha, beta, False)
            state.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
            return max_eval
    else:
        min_eval = np.inf
        for move in state.board.legal_moves:
            state.board.push(move)
            eval = minimax(state, depth - 1, alpha, beta, True)
            state.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
            return min_eval

def get_ai_move(state, depth):
    max_move = None
    max_eval = -np.inf

    for move in state.board.legal_moves:
        state.board.push(move)
        eval = minimax(state, depth - 1, -np.inf, np.inf, False)
        state.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
    
    return max_move

with chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-2022-x86-64-avx2') as engine:
    while True:
        move = get_ai_move(state, 5)
        state.board.push(move)
        print(f'\n{state.board}')
        if state.board.is_game_over():
            break

        move = engine.analyse(board, chess.engine.Limit(time=1), info=chess.engine.INFO)
        state.board.push(move)
        if state.board.is_game_over():
            break
