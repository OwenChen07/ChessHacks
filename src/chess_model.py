import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

def evaluate_material(board):
    """
    Calculate material balance for the current position.
    Positive = white is winning, Negative = black is winning
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    return score

def is_piece_hanging(board, move):
    """
    Check if making this move hangs a piece (leaves it undefended and attackable).
    """
    temp_board = board.copy()
    temp_board.push(move)
    
    # Get the square the piece moved to
    to_square = move.to_square
    piece = temp_board.piece_at(to_square)
    
    if piece is None:
        return False
    
    # Check if the piece is attacked by opponent
    opponent_attacks = temp_board.attackers(not piece.color, to_square)
    if not opponent_attacks:
        return False
    
    # Check if the piece is defended
    friendly_defenders = temp_board.attackers(piece.color, to_square)
    
    # If attacked and not defended, it's hanging
    return len(opponent_attacks) > 0 and len(friendly_defenders) == 0

def get_captures(board):
    """Get all capturing moves."""
    return [move for move in board.legal_moves if board.is_capture(move)]

def get_checks(board):
    """Get all checking moves."""
    checks = []
    for move in board.legal_moves:
        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_check():
            checks.append(move)
    return checks


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
    
    def extract_best_move(self, policy_tensor, board, use_tactical_search=True):
        """
        Extract the best move from the policy tensor with tactical awareness.
        
        Args:
            policy_tensor: Policy output from model, shape (8, 8, 73) or (batch, 8, 8, 73)
            board: chess.Board object for the current position
            use_tactical_search: Whether to use tactical filtering (default: True)
            
        Returns:
            best_move: chess.Move object (or list of moves if batch)
            probability: Probability of the best move (or list if batch)
        """
        from move_encoder import decode_tensor_to_move, move_to_policy_index
        
        # Handle batch vs single position
        is_batch = policy_tensor.dim() == 4
        if not is_batch:
            policy_tensor = policy_tensor.unsqueeze(0)
        
        batch_size = policy_tensor.size(0)
        best_moves = []
        probabilities = []
        
        for i in range(batch_size):
            policy = policy_tensor[i]  # (8, 8, 73)
            legal_moves = list(board.legal_moves)
            
            # Priority 1: Check for checkmate in 1
            for move in legal_moves:
                temp_board = board.copy()
                temp_board.push(move)
                if temp_board.is_checkmate():
                    best_moves.append(move)
                    probabilities.append(1.0)
                    break
            
            if len(best_moves) == i + 1:
                continue
            
            # Priority 2: If tactical search is enabled, filter out terrible moves
            if use_tactical_search:
                initial_material = evaluate_material(board)
                
                # Get all legal moves with their model scores
                move_scores = []
                for move in legal_moves:
                    try:
                        row, col, plane = move_to_policy_index(move, board)
                        score = policy[row, col, plane].item()
                        move_scores.append((move, score))
                    except:
                        continue
                
                # Sort by model probability (descending)
                move_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Tactical filters
                captures = get_captures(board)
                checks = get_checks(board)
                
                # Check if we can capture a free piece
                free_captures = []
                for move in captures:
                    temp_board = board.copy()
                    temp_board.push(move)
                    captured_piece = board.piece_at(move.to_square)
                    
                    # Check if the piece we're capturing is undefended
                    if captured_piece:
                        attackers = board.attackers(not board.turn, move.to_square)
                        if len(attackers) == 0:  # Undefended piece
                            free_captures.append(move)
                
                # Priority 2a: Take free pieces (especially high-value ones)
                if free_captures:
                    # Sort by captured piece value
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9
                    }
                    
                    free_captures_valued = []
                    for move in free_captures:
                        captured = board.piece_at(move.to_square)
                        value = piece_values.get(captured.piece_type, 0)
                        # Get model score for this move
                        try:
                            row, col, plane = move_to_policy_index(move, board)
                            model_score = policy[row, col, plane].item()
                        except:
                            model_score = 0.0
                        free_captures_valued.append((move, value, model_score))
                    
                    # Prefer higher value captures, but also consider model score
                    free_captures_valued.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    best_move = free_captures_valued[0][0]
                    
                    try:
                        row, col, plane = move_to_policy_index(best_move, board)
                        prob = policy[row, col, plane].item()
                    except:
                        prob = 0.5
                    
                    best_moves.append(best_move)
                    probabilities.append(prob)
                    continue
                
                # Filter out moves that hang pieces (lose material for free)
                safe_moves = []
                for move, score in move_scores:
                    temp_board = board.copy()
                    temp_board.push(move)
                    new_material = evaluate_material(temp_board)
                    material_loss = initial_material - new_material
                    
                    # If we're losing more than a pawn without compensation, skip it
                    if material_loss > 1.5:
                        # Check if it's a check or forces something
                        if temp_board.is_check():
                            safe_moves.append((move, score))  # Checks can be worth it
                        # Otherwise skip moves that hang material
                        continue
                    else:
                        safe_moves.append((move, score))
                
                # If we filtered out all moves, fall back to original list
                if not safe_moves:
                    safe_moves = move_scores
                
                # Priority 2b: Consider checks if they're in top moves
                if checks:
                    for move, score in safe_moves[:5]:  # Top 5 moves
                        if move in checks:
                            best_moves.append(move)
                            probabilities.append(score)
                            break
                    
                    if len(best_moves) == i + 1:
                        continue
                
                # Take the best safe move according to the model
                if safe_moves:
                    best_move, best_score = safe_moves[0]
                    best_moves.append(best_move)
                    probabilities.append(best_score)
                    continue
            
            # Fallback: Use pure model prediction
            # Apply softmax to get probabilities
            policy_flat = policy.view(-1)
            probs = F.softmax(policy_flat, dim=0)
            
            # Get top-k moves and pick the first legal one
            top_k = 50
            top_indices = torch.topk(probs, k=min(top_k, len(probs))).indices
            
            found_move = False
            for idx in top_indices:
                idx = idx.item()
                row = idx // (8 * 73)
                remainder = idx % (8 * 73)
                col = remainder // 73
                plane = remainder % 73
                
                move_tensor = torch.zeros(8, 8, 73)
                move_tensor[row, col, plane] = 1.0
                
                try:
                    move = decode_tensor_to_move(move_tensor.numpy(), board)
                    if move in legal_moves:
                        best_moves.append(move)
                        probabilities.append(probs[idx].item())
                        found_move = True
                        break
                except:
                    continue
            
            if not found_move:
                # Last resort: return a random legal move
                import random
                move = random.choice(legal_moves)
                best_moves.append(move)
                probabilities.append(0.0)
        
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