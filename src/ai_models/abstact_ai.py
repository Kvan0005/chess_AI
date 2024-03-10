from abc import ABC, abstractmethod
import chess


class AbstractAI(ABC):
    """_summary_ mother class for AI

    Args:
        ABC (_type_): _description_ abstract class
    """

    def __init__(self, color: chess.Color):
        self.color = color

    def foward(self, board: chess.Board):
        """_summary_ foward function

        Args:
            board (_type_): _description_ chess board
        """
        return self.get_move(board)

    @abstractmethod
    def get_move(self, board: chess.Board):
        """_summary_ get move function

        Args:
            board (_type_): _description_ chess board
        """
