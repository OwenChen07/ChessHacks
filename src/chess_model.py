import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
}

def material_eval(board: chess.Board):
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            score += PIECE_VALUES[piece.symbol()]
    return score


class ResidualBlock(nn.Module):
    """
    A residual block for the chess neural network.
    Helps the network learn deeper patterns without training problems.
    """
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        
        # Two convolutional layers
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual (this is the key to residual networks!)
        out += residual
        out = F.relu(out)
        
        return out


class ChessNet(nn.Module):
    """
    Neural network for chess move prediction and position evaluation.
    
    Architecture inspired by AlphaZero:
    - Input: Board position (8x8x18)
    - Output 1 (Policy): Move probabilities (8x8x73)
    - Output 2 (Value): Position evaluation (-1 to +1)
    """
    
    def __init__(self, num_filters=128, num_residual_blocks=5):
        """
        Args:
            num_filters: Number of convolutional filters (default: 128)
            num_residual_blocks: Number of residual blocks (default: 5)
        """
        super(ChessNet, self).__init__()
        
        print(f"Creating ChessNet with {num_filters} filters and {num_residual_blocks} residual blocks")
        
        # Initial convolutional layer
        # Takes board (8x8x18) and converts to (8x8x num_filters)
        self.conv_input = nn.Conv2d(18, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head (move prediction)
        self.policy_conv = nn.Conv2d(num_filters, 73, kernel_size=1)
        
        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Board tensor of shape (batch, 8, 8, 18)
            
        Returns:
            policy: Move probabilities (batch, 8, 8, 73)
            value: Position evaluation (batch, 1)
        """
        # Convert from (batch, 8, 8, 18) to (batch, 18, 8, 8)
        # PyTorch expects channels first
        x = x.permute(0, 3, 1, 2)
        
        # Initial convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)  # (batch, 73, 8, 8)
        policy = policy.permute(0, 2, 3, 1)  # Convert back to (batch, 8, 8, 73)
        
        # Value head
        value = self.value_conv(x)  # (batch, 1, 8, 8)
        value = value.view(-1, 8 * 8)  # Flatten to (batch, 64)
        value = F.relu(self.value_fc1(value))  # (batch, 256)
        value = torch.tanh(self.value_fc2(value))  # (batch, 1), range [-1, 1]
        
        return policy, value
    
    def predict_move(self, board_tensor):
        """
        Predict the best move for a single position.
        
        Args:
            board_tensor: Single board (8, 8, 18) or batch (batch, 8, 8, 18)
            
        Returns:
            policy: Move probabilities
            value: Position evaluation
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Add batch dimension if single board
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            
            policy, value = self.forward(board_tensor)
            return policy, value
    
    def extract_best_move(self, policy_tensor, board):
        """
        Extract the best move from the policy tensor.
        
        Args:
            policy_tensor: Policy output from model, shape (8, 8, 73) or (batch, 8, 8, 73)
            board: chess.Board object for the current position
            
        Returns:
            best_move: chess.Move object (or list of moves if batch)
            probability: Probability of the best move (or list if batch)
        """
        from move_encoder import decode_tensor_to_move
        
        # Handle batch vs single position
        is_batch = policy_tensor.dim() == 4
        if not is_batch:
            policy_tensor = policy_tensor.unsqueeze(0)  # Add batch dim
        
        batch_size = policy_tensor.size(0)
        best_moves = []
        probabilities = []
        
        for i in range(batch_size):
            # Get policy for this position
            policy = policy_tensor[i]  # (8, 8, 73)
            
            # Apply softmax to get probabilities
            policy_flat = policy.view(-1)  # Flatten to (4672,)
            probs = F.softmax(policy_flat, dim=0)
            
            # Find the move with highest probability
            best_idx = torch.argmax(probs).item()
            best_prob = probs[best_idx].item()
            
            # Convert flat index back to (row, col, plane)
            # Formula: idx = row * (8 * 73) + col * 73 + plane
            row = best_idx // (8 * 73)
            remainder = best_idx % (8 * 73)
            col = remainder // 73
            plane = remainder % 73
            
            # Create a one-hot tensor for decoding
            move_tensor = torch.zeros(8, 8, 73)
            move_tensor[row, col, plane] = 1.0
            
            # Decode to chess move
            try:
                move = decode_tensor_to_move(move_tensor.numpy(), board)
                best_moves.append(move)
                probabilities.append(best_prob)
            except Exception as e:
                print(f"Warning: Could not decode move at ({row}, {col}, {plane}): {e}")
                best_moves.append(None)
                probabilities.append(best_prob)
        
        # Return single move if not batch
        if not is_batch:
            return best_moves[0], probabilities[0]
        else:
            return best_moves, probabilities
    
    def get_best_move_uci(self, board_tensor, board):
        """
        Complete pipeline: predict and return best move in UCI format.
        
        Args:
            board_tensor: Board position (8, 8, 18)
            board: chess.Board object
            
        Returns:
            move_uci: Best move in UCI format (e.g., "e2e4")
            probability: Confidence of the move
            value: Position evaluation
        """
        # Get predictions
        policy, value = self.predict_move(board_tensor)
        
        # Extract best move
        best_move, prob = self.extract_best_move(policy.squeeze(0), board)
        
        if best_move is None:
            return None, prob, value.item()
        
        return best_move.uci(), prob, value.item()


# # Test the model
# if __name__ == "__main__":
#     print("=== Testing ChessNet ===\n")
    
#     # Create a model
#     model = ChessNet(num_filters=128, num_residual_blocks=5)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\nTotal parameters: {total_params:,}")
    
#     # Create a dummy input (batch of 2 boards)
#     dummy_input = torch.randn(2, 8, 8, 18)
#     print(f"\nInput shape: {dummy_input.shape}")
    
#     # Forward pass
#     policy_out, value_out = model(dummy_input)
    
#     print(f"\nOutput shapes:")
#     print(f"  Policy: {policy_out.shape}")  # Should be (2, 8, 8, 73)
#     print(f"  Value: {value_out.shape}")    # Should be (2, 1)
    
#     print(f"\nValue range: [{value_out.min().item():.3f}, {value_out.max().item():.3f}]")
    
#     # Test single board prediction
#     print("\n=== Testing single board prediction ===")
#     single_board = torch.randn(8, 8, 18)
#     policy, value = model.predict_move(single_board)
#     print(f"Policy shape: {policy.shape}")
#     print(f"Value: {value.item():.3f}")
    
#     print("\n" + "="*50)
#     print("=== Testing with Real Chess Data ===")
    
#     try:
#         from chess_dataset import ChessDataset
#         from fen_to_tensor import fen_to_tensor
        
#         # Load a few real positions
#         dataset = ChessDataset("data/data.csv", max_rows=5)
        
#         print(f"\nLoaded {len(dataset)} real chess positions")
        
#         # Get first position
#         board_tensor, target_move, result = dataset[0]
        
#         # Get the FEN to create a board object
#         fen = dataset.df.iloc[0]['fen']
#         board = chess.Board(fen)
        
#         print(f"\nFirst position:")
#         print(f"  FEN: {fen}")
#         print(f"  Board shape: {board_tensor.shape}")
#         print(f"  Actual result: {result.item()}")
        
#         # Make prediction with model
#         policy_pred, value_pred = model.predict_move(board_tensor)
        
#         print(f"\nModel prediction:")
#         print(f"  Policy shape: {policy_pred.shape}")
#         print(f"  Predicted value: {value_pred.item():.3f}")
#         print(f"  Target value: {result.item()}")
        
#         # Extract best move
#         print("\n" + "="*50)
#         print("=== Extracting Best Move ===")
        
#         move_uci, probability, value = model.get_best_move_uci(board_tensor, board)
        
#         print(f"\nModel's best move: {move_uci}")
#         print(f"Confidence: {probability:.4f}")
#         print(f"Position evaluation: {value:.3f}")
        
#         # Show target move for comparison
#         target_uci = dataset.df.iloc[0]['move']
#         print(f"\nTarget move from game: {target_uci}")
#         print(f"Match: {move_uci == target_uci}")
        
#         # Test with a batch
#         print("\n" + "="*50)
#         print("=== Testing with a Batch ===")
        
#         # Get 3 positions
#         boards = torch.stack([dataset[i][0] for i in range(3)])
#         target_moves = torch.stack([dataset[i][1] for i in range(3)])
#         results = torch.stack([dataset[i][2] for i in range(3)])
        
#         print(f"\nBatch shapes:")
#         print(f"  Boards: {boards.shape}")
#         print(f"  Target moves: {target_moves.shape}")
#         print(f"  Results: {results.shape}")
        
#         # Forward pass
#         policy_batch, value_batch = model(boards)
        
#         print(f"\nModel batch predictions:")
#         print(f"  Policy: {policy_batch.shape}")
#         print(f"  Values: {value_batch.shape}")
#         print(f"  Predicted values: {[f'{v:.3f}' for v in value_batch.squeeze().tolist()]}")
#         print(f"  Target values: {results.tolist()}")
        
#         print("\n✓ All tests passed!")
        
#     except ImportError:
#         print("\nSkipping real data test (chess_dataset.py not found)")
#     except Exception as e:
#         print(f"\nError testing with real data: {e}")
#         print("Make sure chess_dataset.py, fen_to_tensor.py, and move_encoder.py are in the same directory")