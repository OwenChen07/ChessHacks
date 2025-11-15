import pandas as pd
import chess
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Import your encoding functions
from fen_to_tensor import fen_to_tensor
from move_encoder import encode_move_to_tensor

class ChessDataset(Dataset):
    """
    Dataset for loading chess positions from CSV.
    
    CSV format: fen, move, result
    """
    
    def __init__(self, csv_path, max_rows=None):
        """
        Load the CSV file.
        
        Args:
            csv_path: Path to your CSV file (e.g., "data/data.csv")
            max_rows: Optional - only load first N rows (useful for testing)
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path, nrows=max_rows)
        print(f"Loaded {len(self.df)} positions")
    
    def __len__(self):
        """Return total number of examples in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single example by index.
        
        Args:
            idx: Which example to get (0 to len-1)
            
        Returns:
            board_tensor: (8, 8, 18) - the board position
            move_tensor: (8, 8, 73) - the move to play
            result: float - game result (0.0, 0.5, or 1.0)
        """
        # Get the row from CSV
        row = self.df.iloc[idx]
        fen = row['fen']
        move_uci = row['move']
        result = float(row['result'])
        
        # Create chess board
        board = chess.Board(fen)
        
        # Convert FEN to tensor using your function
        board_tensor = fen_to_tensor(fen)  # Returns (8, 8, 18)
        
        # Convert move to tensor using your function
        move = chess.Move.from_uci(move_uci)
        move_tensor = encode_move_to_tensor(move, board)  # Returns (8, 8, 73)
        
        # Convert numpy arrays to PyTorch tensors
        board_tensor = torch.from_numpy(board_tensor).float()
        move_tensor = torch.from_numpy(move_tensor).float()
        result = torch.tensor(result).float()
        
        return board_tensor, move_tensor, result


def create_train_val_loaders(csv_path, batch_size=64, train_split=0.9, max_rows=None):
    """
    Create training and validation data loaders.
    
    Args:
        csv_path: Path to your CSV file
        batch_size: How many positions to process at once (default: 64)
        train_split: Fraction for training (default: 0.9 = 90%)
        max_rows: Optional - only use first N rows
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    print("\n=== Creating Train/Val Split ===")
    
    # Load the full dataset
    dataset = ChessDataset(csv_path, max_rows=max_rows)
    total_size = len(dataset)
    
    # Calculate split sizes
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    print(f"Total examples: {total_size}")
    print(f"Training: {train_size} ({train_split*100:.0f}%)")
    print(f"Validation: {val_size} ({(1-train_split)*100:.0f}%)")
    
    # Split the dataset randomly
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Same split every time
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=0  # Use 0 for debugging, increase to 4 later
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=0
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Test code
if __name__ == "__main__":
    print("=== Testing ChessDataset ===")
    
    # Create dataset with first 100 positions
    dataset = ChessDataset("data/data.csv", max_rows=100)
    
    print(f"\nDataset has {len(dataset)} examples")
    
    # Get first example
    print("\nGetting example #0:")
    board, move, result = dataset[0]
    
    print(f"  Board shape: {board.shape}")
    print(f"  Move shape: {move.shape}")
    print(f"  Result: {result.item()}")
    
    # Test train/val split
    print("\n" + "="*50)
    train_loader, val_loader = create_train_val_loaders(
        "data/data.csv",
        batch_size=32,
        train_split=0.8,  # 80% train, 20% val
        max_rows=100
    )
    
    # Get one batch from training
    print("\n=== Testing Training Batch ===")
    for boards, moves, results in train_loader:
        print(f"Board batch shape: {boards.shape}")
        print(f"Move batch shape: {moves.shape}")
        print(f"Result batch shape: {results.shape}")
        break  # Just test first batch
    
    # Get one batch from validation
    print("\n=== Testing Validation Batch ===")
    for boards, moves, results in val_loader:
        print(f"Board batch shape: {boards.shape}")
        print(f"Move batch shape: {moves.shape}")
        print(f"Result batch shape: {results.shape}")
        break
    
    print("\n✓ All tests passed!")