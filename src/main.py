# cd my-chesshacks-bot/devtools
# print('hello')
# from utils import chess_manager, GameContext
from chess import Move
import torch
import math
import torch.nn.functional as F
from src.utils import chess_manager, GameContext
from src.chess_model import ChessNet
from src.fen_to_tensor import fen_to_tensor
from src.move_encoder import move_to_policy_index
# from src.mcts import MCTS
# from chess_model import ChessNet
# from fen_to_tensor import fen_to_tensor
# from move_encoder import move_to_policy_index


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
# download weights from huggingface, etc.
model = ChessNet(num_filters=128, num_residual_blocks=5)
model.load_state_dict(torch.load("chess_model_big.pth", map_location='cpu'))
model.eval()

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

mcts = MCTS(model, sims=200)

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position
 
    board = ctx.board
    board_tensor = fen_to_tensor(board.fen())
    board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0)

    with torch.no_grad():
        policy, value = model(board_tensor)

    policy = policy.squeeze(0)

    # Get legal moves and their scores
    legal_moves = list(board.generate_legal_moves())
    print(f"\nLegal moves: {len(legal_moves)}")

    maxv = -1e9
    bestMove = ''
    for move in legal_moves:
        row, col, plane = move_to_policy_index(move, board)
        score = policy[row, col, plane].item()
        if score > maxv:
            maxv = score
            bestMove = move
    print("asdadasdad", bestMove)
    return bestMove

    # # Run MCTS
    # best_move = mcts.run(board)

    # # Optional: Log visit counts or probabilities
    # # move_probs = {
    # #     move.uci(): child.N
    # #     for move, child in mcts.root.children.items()
    # # }
    # # ctx.logProbabilities(move_probs)

    # return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
