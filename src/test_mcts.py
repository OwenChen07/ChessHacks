import torch
import chess
from chess_model import ChessNet
from fen_to_tensor import fen_to_tensor
from move_encoder import move_to_policy_index
from mcts import MCTS   # Your MCTS implementation


# ----------------------------------------------------------
# Load trained NN model
# ----------------------------------------------------------
def load_trained_model(model_path="chess_model_big.pth"):
    print(f"Loading model from {model_path}...")

    model = ChessNet(num_filters=128, num_residual_blocks=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("✓ Model loaded successfully!")
    return model


# ----------------------------------------------------------
# Run MCTS on a FEN
# ----------------------------------------------------------
def get_mcts_move(model, fen, sims=200):
    board = chess.Board(fen)

    mcts = MCTS(model=model, sims=sims)
    best_move = mcts.run(board)

    return best_move.uci()


# ----------------------------------------------------------
# Compare MCTS move to Stockfish’s move
# ----------------------------------------------------------
def compare_to_stockfish_mcts(model, fen, stockfish_move, sims=200):
    board = chess.Board(fen)

    print("\n" + "="*60)
    print("MCTS + NEURAL NET VS STOCKFISH")
    print("="*60)
    print(board)
    print(f"\nFEN: {fen}")

    predicted = get_mcts_move(model, fen, sims=sims)

    print(f"\nStockfish recommends: {stockfish_move}")
    print(f"MCTS predicts:       {predicted}")
    print(f"Match: {'✓ YES' if predicted == stockfish_move else '✗ NO'}")

    return predicted


# ----------------------------------------------------------
# Demo tests
# ----------------------------------------------------------
if __name__ == "__main__":

    print("="*60)
    print("MCTS + NN CHESS TESTING")
    print("="*60)

    model = load_trained_model("chess_model.pth")

    # Test 1: Starting position
    print("\n" + "="*60)
    print("TEST 1: STARTING POSITION")
    print("="*60)

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stockfish_move = "e2e4"  # Usually best at depth 20+

    compare_to_stockfish_mcts(model, start_fen, stockfish_move)

    # Test 2: Mid-game
    print("\n" + "="*60)
    print("TEST 2: MID-GAME POSITION")
    print("="*60)

    mid_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    stockfish_move = "d2d4"  # Best move in this position

    compare_to_stockfish_mcts(model, mid_fen, stockfish_move)

    print("\n✓ MCTS Testing complete!")
