import numpy as np
import chess
import chess.engine
import random 
import os

class Board(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
    
    def random_board(self, max_depth=200):
        board = chess.Board()
        depth = random.randrange(0, max_depth)

        for i in range(0, depth):
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break
        return board
    
    def stockfish(self, board, depth):
        with chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-2022-x86-64-avx2') as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth))
            score = result['score'].white().score()
            return score

    # h3 -> 17
    @staticmethod
    def square_to_index(square): 
        squares_index = {
            'a' : 0,
            'b' : 1,
            'c' : 2,
            'd' : 3,
            'e' : 4,
            'f' : 5,
            'g' : 6,
            'h' : 7
        }
        letter = chess.square_name(square)
        return 8 - int(letter[1]), squares_index[letter[0]]

    def split_dims(self):
        board3d = np.zeros((14, 8, 8), dtype=np.int8)

        for piece in chess.PIECE_TYPES:
            for square in self.board.pieces(piece, chess.WHITE):
                idx = np.unravel_index(square, (8, 8))
                board3d[piece - 1][7 - idx[0]][idx[1]] = 1
            
            for square in self.board.pieces(piece, chess.BLACK):
                idx = np.unravel_index(square, (8, 8))
                board3d[piece + 5][7 - idx[0]][idx[1]] = 1
        
        aux = self.board.turn
        self.board.turn = chess.WHITE
        for move in self.board.legal_moves:
            i, j = Board.square_to_index(move.to_square)
            board3d[12][i][j] = 1
        
        self.board.turn = chess.BLACK
        for move in self.board.legal_moves:
            i, j = Board.square_to_index(move.to_square)
            board3d[13][i][j] = 1

        self.board.turn = aux

        self.board3d = board3d
    
