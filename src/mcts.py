import math
# import chess
import torch
# from chess_model import ChessNet
from fen_to_tensor import fen_to_tensor
from move_encoder import move_to_policy_index

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior  # P from the NN
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0

class MCTS:
    def __init__(self, model, sims=800, c_puct=1.4):
        self.model = model
        self.sims = sims
        self.c_puct = c_puct

    def run(self, board):
        self.root = MCTSNode(board)

        # Expand root using neural network
        self.expand(self.root)

        for _ in range(self.sims):
            node = self.select(self.root)
            value = self.expand(node)
            self.backpropagate(node, value)

        # Pick best move by visits
        best_child = max(self.root.children.items(), key=lambda c: c[1].N)
        return best_child[0]  # the move

    def select(self, node):
        while node.children:
            node = max(
                node.children.values(),
                key=lambda c: c.Q + self.c_puct * c.prior *
                               math.sqrt(node.N) / (1 + c.N)
            )
        return node

    def expand(self, node):
        board = node.board

        # Terminal
        if board.is_game_over():
            outcome = board.outcome().result()
            if outcome == "1-0": return 1
            if outcome == "0-1": return -1
            return 0

        # Prepare input
        x = torch.from_numpy(fen_to_tensor(board.fen())).float().unsqueeze(0)


        with torch.no_grad():
            policy_logits, value = self.model(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze()

        # Create children
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            idx = move_to_policy_index(move, board)   # YOU already have this from your model
            prior = float(policy[idx])
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, node, prior)

        return float(value)

    def backpropagate(self, node, value):
        while node:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent
