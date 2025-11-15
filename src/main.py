# cd my-chesshacks-bot/devtools
# print('hello')
# from utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn.functional as F
from src.utils import chess_manager, GameContext
from src.chess_model import ChessNet
from src.fen_to_tensor import fen_to_tensor
from src.move_encoder import move_to_policy_index
# from chess_model import ChessNet
# from fen_to_tensor import fen_to_tensor
# from move_encoder import move_to_policy_index


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
# download weights from huggingface, etc.
model = ChessNet(num_filters=128, num_residual_blocks=5)
model.load_state_dict(torch.load("chess_model.pth", map_location='cpu'))
model.eval()

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    # print("Cooking move...")
    # print(ctx.board.move_stack)
    # time.sleep(0.1)

    # legal_moves = list(ctx.board.generate_legal_moves())
    # if not legal_moves:
    #     ctx.logProbabilities({})
    #     raise ValueError("No legal moves available (i probably lost didn't i)")

    # move_weights = [random.random() for _ in legal_moves]
    # total_weight = sum(move_weights)
    # # Normalize so probabilities sum to 1
    # move_probs = {
    #     move: weight / total_weight
    #     for move, weight in zip(legal_moves, move_weights)
    # }
    # ctx.logProbabilities(move_probs)

    # return random.choices(legal_moves, weights=move_weights, k=1)[0]
    # Get prediction
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

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
