import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from chess_model import ChessNet
from chess_dataset import create_train_val_loaders


def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_total_loss: Average total loss
    """
    model.train()  # Set to training mode
    
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for boards, moves, results in pbar:
        # Move data to device (GPU or CPU)
        boards = boards.to(device)
        moves = moves.to(device)
        results = results.to(device)
        
        # Forward pass
        policy_out, value_out = model(boards)
        
        # Prepare targets for loss calculation
        batch_size = boards.size(0)
        
        # Policy loss - convert one-hot moves to class indices
        policy_out_flat = policy_out.reshape(batch_size, -1)  # (batch, 8*8*73)
        moves_flat = moves.reshape(batch_size, -1)  # (batch, 8*8*73)
        move_indices = torch.argmax(moves_flat, dim=1)  # (batch,)
        
        policy_loss = nn.CrossEntropyLoss()(policy_out_flat, move_indices)
        
        # Value loss - MSE between predicted and actual result
        value_loss = nn.MSELoss()(value_out.squeeze(), results)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'policy': f'{policy_loss.item():.4f}',
            'value': f'{value_loss.item():.4f}',
            'total': f'{loss.item():.4f}'
        })
    
    return total_policy_loss / num_batches, total_value_loss / num_batches, total_loss / num_batches


def validate(model, val_loader, device):
    """
    Validate the model on validation set.
    
    Returns:
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_total_loss: Average total loss
    """
    model.eval()  # Set to evaluation mode
    
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # Don't calculate gradients
        for boards, moves, results in val_loader:
            boards = boards.to(device)
            moves = moves.to(device)
            results = results.to(device)
            
            # Forward pass
            policy_out, value_out = model(boards)
            
            # Calculate losses (same as training)
            batch_size = boards.size(0)
            
            policy_out_flat = policy_out.reshape(batch_size, -1)
            moves_flat = moves.reshape(batch_size, -1)
            move_indices = torch.argmax(moves_flat, dim=1)
            
            policy_loss = nn.CrossEntropyLoss()(policy_out_flat, move_indices)
            value_loss = nn.MSELoss()(value_out.squeeze(), results)
            loss = policy_loss + value_loss
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
    
    return total_policy_loss / num_batches, total_value_loss / num_batches, total_loss / num_batches


def train_model(
    csv_path,
    num_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    num_filters=128,
    num_residual_blocks=5,
    max_rows=None,
    save_path="chess_model.pth"
):
    """
    Main training function.
    
    Args:
        csv_path: Path to CSV with chess data
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_filters: Number of filters in model
        num_residual_blocks: Number of residual blocks in model
        max_rows: Optional limit on data rows
        save_path: Where to save the trained model
    """
    print("="*60)
    print("CHESS MODEL TRAINING")
    print("="*60)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_loader, val_loader = create_train_val_loaders(
        csv_path,
        batch_size=batch_size,
        train_split=0.9,
        max_rows=max_rows
    )
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = ChessNet(num_filters=num_filters, num_residual_blocks=num_residual_blocks)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Train
        train_policy_loss, train_value_loss, train_total_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        print("Validating...")
        val_policy_loss, val_value_loss, val_total_loss = validate(
            model, val_loader, device
        )
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train - Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f}, Total: {train_total_loss:.4f}")
        print(f"  Val   - Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}, Total: {val_total_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved! (val_loss: {val_total_loss:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    # Example: Train on 10,000 positions
    model = train_model(
        csv_path="data/data.csv",  # Changed from ../data/data.csv
        num_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        num_filters=128,
        num_residual_blocks=5,
        max_rows=40000,  # Use first 10k positions for testing
        save_path="chess_model.pth"
    )
    
    print("\n✓ Training finished!")