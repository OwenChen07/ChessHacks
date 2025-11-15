# Create a simple test file: test_bot.py

import chess
from chess_model import ChessNet
from fen_to_tensor import fen_to_tensor
from move_encoder import move_to_policy_index
import torch
import torch.nn.functional as F

# Load model
model = ChessNet(num_filters=128, num_residual_blocks=5)
model.load_state_dict(torch.load("chess_model.pth", map_location='cpu'))
model.eval()

# Test on starting position
board = chess.Board()
print(board)

# Get prediction
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
print("Best move is", bestMove)

print(f"\nPosition evaluation: {value.item():.3f}")