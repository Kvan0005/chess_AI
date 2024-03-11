from typing import NoReturn
import numpy as np
import torch as th
import chess
from .abstact_ai import AbstractAI


class DeepQNetwork(AbstractAI):
    """_summary_ Deep Q Network AI"""

    def __init__(
        self,
        color: chess.Color,
        model: th.nn.Module,
        gamma: float = 0.99,
        lr: float = 0.01,
    ):
        super().__init__(color)
        DeepQNetwork.check_model(model)
        self.network: th.nn.Module = model
        self.target_network: th.nn.Module = model
        self.loss = th.nn.MSELoss()
        self.optimizer = th.optim.SGD(self.network.parameters(), lr=lr)
        self.gamma: float = gamma

    def get_move(self, board: chess.Board):
        """_summary_ get move function

        Args:
            board (_type_): _description_ chess board
        """
        assert board.turn == self.color, "It's not my turn"
        nn_info_array = to_info_array(board)
        
        pass

    def available_moves(self, board: chess.Board):
        """_summary_ available moves function

        Args:
            board (_type_): _description_ chess board
        """
        return list(board.legal_moves)

    def q_values(self, board_information: np.ndarray):
        """_summary_ q values function

        Args:
            board (_type_): _description_ chess board
        """
        return self.network.foward(th.from_numpy(board_information))

    def v_values(self, board_information: np.ndarray):
        return self.q_values(board_information).max()
    
    
    def update(
        self,
        board: chess.Board,
        move: chess.Move,
        reward: float,
        next_board: chess.Board,
    ):
        """_summary_ update function

        Args:
            board (_type_): _description_ chess board
            move (_type_): _description_ move
            reward (_type_): _description_ reward
            next_board (_type_): _description_ next board
        """
        pass

    @staticmethod
    def check_model(model: th.nn.Module) -> None | NoReturn:
        """_summary_ check model function

        Args:
            model (_type_): _description_ model
        """
        assert isinstance(model, th.nn.Module), "model must be a torch.nn.Module"
        assert isinstance(
            model, th.nn.Sequential
        ), "model must be a torch.nn.Sequential"
        # model needs to have only one output in the last layer
        assert (
            model[-1].out_features == 1
        ), "model must have only one output in the last layer"
        # TBD: input size

    @staticmethod
    def load_model(path: str) -> th.nn.Module:
        """_summary_ load model function

        Args:
            path (_type_): _description_ path
        """
        return th.load(path)

    def save_model(self, path: str) -> None:
        """_summary_ save model function

        Args:
            path (_type_): _description_ path
        """
        th.save(self.network, path)


def to_info_array(board: chess.Board) -> np.ndarray:
    """_summary_ convert board to info array
    the info array is a 8x8x13 numpy array with the following structure:
    layers:
        - 0-5: white pieces
        - 6-11: black pieces
        - 12:
            - (0,0): 1 if white can castle kingside, 0 otherwise
            - (0,1): 1 if white can castle queenside, 0 otherwise
            - (0,2): 1 if black can castle kingside, 0 otherwise
            - (0,3): 1 if black can castle queenside, 0 otherwise
            - (0,4): 1 if check, 0 otherwise
            - (0,5): 1 if checkmate, 0 otherwise
            - (0,6): 1 if stalemate, 0 otherwise
            - (0,7): 1 if white turn, 0 otherwise


    Args:
        board (chess.Board): _description_

    Returns:
        np.ndarray: _description_
    """
    numpy_arrray = np.zeros((64, 13), dtype=np.uint8)
    for color in chess.COLORS:
        for piece_t in chess.PIECE_TYPES:
            bit_board = board.pieces_mask(piece_t, color)
            # convert a python int to a 64 bits numpy array
            numpy_arrray[:, 6 * color + piece_t] = np.unpackbits(
                np.array([bit_board], dtype=np.uint64).view(np.uint8)
            )

    # castling
    numpy_arrray[0, 12] = board.has_kingside_castling_rights(chess.WHITE)
    numpy_arrray[1, 12] = board.has_queenside_castling_rights(chess.WHITE)
    numpy_arrray[2, 12] = board.has_kingside_castling_rights(chess.BLACK)
    numpy_arrray[3, 12] = board.has_queenside_castling_rights(chess.BLACK)

    # check, checkmate, stalemate
    numpy_arrray[4, 12] = board.is_check()
    numpy_arrray[5, 12] = board.is_checkmate()
    numpy_arrray[6, 12] = board.is_stalemate()
    numpy_arrray[7, 12] = board.turn
    return numpy_arrray
