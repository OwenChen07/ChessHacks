import torch
import chess
from chess_model import ChessNet
from fen_to_tensor import fen_to_tensor
from chess_dataset import ChessDataset


def load_trained_model(model_path="chess_model_v3.pth", num_filters=128, num_residual_blocks=5):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model weights
        num_filters: Numsber of filters (must match training)
        num_residual_blocks: Number of blocks (must match training)
        
    Returns:
        model: Loaded ChessNet model
    """
    print(f"Loading model from {model_path}...")
    
    # Create model with same architecture as training
    model = ChessNet(num_filters=num_filters, num_residual_blocks=num_residual_blocks)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set to evaluation mode
    
    print("✓ Model loaded successfully!")
    return model


def test_on_position(model, fen, verbose=True):
    """
    Test model on a single chess position.
    
    Args:
        model: Trained ChessNet model
        fen: FEN string of the position
        verbose: Print details
        
    Returns:
        move_uci: Best move in UCI format
        confidence: Confidence of the move
        value: Position evaluation
    """
    board = chess.Board(fen)
    
    if verbose:
        print("\n" + "="*60)
        print("POSITION:")
        print(board)
        print(f"\nFEN: {fen}")
        print(f"Turn: {'White' if board.turn else 'Black'}")
    
    # Get board tensor
    board_tensor = fen_to_tensor(fen)
    board_tensor = torch.from_numpy(board_tensor).float()
    
    # Get prediction
    move_uci, confidence, value = model.get_best_move_uci(board_tensor, board)
    
    if verbose:
        print(f"\nModel's Best Move: {move_uci}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Position Evaluation: {value:.3f}")
        print(f"  ({value:.3f} means ", end="")
        if value > 0.2:
            print("White is winning)")
        elif value < -0.2:
            print("Black is winning)")
        else:
            print("Position is equal)")
    
    return move_uci, confidence, value


def test_on_dataset(model, csv_path, num_examples=10):
    """
    Test model on examples from the dataset.
    Shows how often the model predicts the correct move.
    
    Args:
        model: Trained ChessNet model
        csv_path: Path to CSV with test data
        num_examples: Number of examples to test
    """
    print("\n" + "="*60)
    print(f"TESTING ON {num_examples} POSITIONS FROM DATASET")
    print("="*60)
    
    dataset = ChessDataset(csv_path, max_rows=num_examples)
    
    correct = 0
    total = 0
    
    for i in range(min(num_examples, len(dataset))):
        board_tensor, target_move_tensor, result = dataset[i]
        
        # Get FEN and create board
        fen = dataset.df.iloc[i]['fen']
        target_move_uci = dataset.df.iloc[i]['move']
        board = chess.Board(fen)
        
        # Get model prediction
        predicted_move, confidence, value = model.get_best_move_uci(board_tensor, board)
        
        # Check if correct
        is_correct = (predicted_move == target_move_uci)
        if is_correct:
            correct += 1
        total += 1
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{i+1}. {status} Predicted: {predicted_move} | Target: {target_move_uci} | Conf: {confidence:.2%}")
    
    accuracy = (correct / total) * 100
    print("\n" + "="*60)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("="*60)
    
    return accuracy


def interactive_test(model):
    """
    Interactive testing - enter FEN positions and see predictions.
    """
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("Enter FEN positions to see model predictions.")
    print("Type 'quit' to exit, 'start' for starting position.")
    print()
    
    while True:
        fen_input = input("Enter FEN (or 'quit'/'start'): ").strip()
        
        if fen_input.lower() == 'quit':
            break
        elif fen_input.lower() == 'start':
            fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        try:
            test_on_position(model, fen_input, verbose=True)
        except Exception as e:
            print(f"Error: {e}")
            print("Please enter a valid FEN string.")

def test_value_head(model):
    print("\n" + "="*60)
    print("TESTING VALUE HEAD (0–1 OUTPUT RANGE)")
    print("="*60)

    test_positions = [
        ("White up a queen",
         "8/8/8/8/8/8/8/3Qk3 w - - 0 1",
         "White"),

        ("Black up a rook",
         "8/8/8/8/8/8/3r4/4K3 b - - 0 1",
         "Black"),

        ("Equal position",
         "8/8/8/8/8/8/3k4/4K3 w - - 0 1",
         "Equal"),

        ("White has mate in 1",
         "6rk/5Q1p/8/8/8/8/8/7K w - - 0 1",
         "White"),

        ("Black has mate in 1",
         "7k/8/8/8/8/8/5q2/7K b - - 0 1",
         "Black")
    ]

    for desc, fen, expected in test_positions:
        board_tensor = torch.from_numpy(fen_to_tensor(fen)).float()
        board = chess.Board(fen)

        with torch.no_grad():
            move_uci, conf, value = model.get_best_move_uci(board_tensor, board)

        print(f"\n--- {desc} ---")
        print(board)
        print(f"Expected side winning: {expected}")
        print(f"Model raw value: {value:.3f}")

        # Interpretation
        if value > 0.70:
            verdict = "White winning"
        elif value < 0.30:
            verdict = "Black winning"
        else:
            verdict = "Equal / unclear"

        print(f"Model interpretation: {verdict}")

    print("\n" + "="*60)
    print("VALUE HEAD TEST COMPLETE")
    print("="*60)


def compare_to_stockfish(model, fen, stockfish_move):
    """
    Compare model's move to Stockfish's recommended move.
    
    Args:
        model: Trained model
        fen: Position FEN
        stockfish_move: Move recommended by Stockfish (e.g., "e2e4")
    """
    board = chess.Board(fen)
    board_tensor = fen_to_tensor(fen)
    board_tensor = torch.from_numpy(board_tensor).float()
    
    predicted_move, confidence, value = model.get_best_move_uci(board_tensor, board)
    
    print("\n" + "="*60)
    print("COMPARISON WITH STOCKFISH")
    print("="*60)
    print(board)
    print(f"\nStockfish recommends: {stockfish_move}")
    print(f"Model predicts: {predicted_move}")
    print(f"Match: {'✓ YES' if predicted_move == stockfish_move else '✗ NO'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Position evaluation: {value:.3f}")


if __name__ == "__main__":
    print("="*60)
    print("CHESS MODEL TESTING")
    print("="*60)
    
    # Load trained model
    model = load_trained_model("chess_model_v3.pth")
    
    # Test 1: Starting position
    print("\n" + "="*60)
    print("TEST 1: STARTING POSITION")
    print("="*60)
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    test_on_position(model, starting_fen)
    
    # Test 2: Mid-game position
    print("\n" + "="*60)
    print("TEST 2: MID-GAME POSITION")
    print("="*60)
    midgame_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    test_on_position(model, midgame_fen)
    
    # Test 3: Check accuracy on dataset
    print("\n" + "="*60)
    print("TEST 3: ACCURACY ON DATASET")
    print("="*60)
    test_on_dataset(model, "data/data.csv", num_examples=50)
    
    # Test 4: Interactive (optional)
    print("\n" + "="*60)
    print("TEST 4: INTERACTIVE TESTING (OPTIONAL)")
    print("="*60)
    response = input("Would you like to test interactively? (y/n): ").lower()
    if response == 'y':
        interactive_test(model)
    
    test_value_head(model)

    print("\n✓ Testing complete!")