import numpy as np
import chess

# This file converts the FEN string into a Tensor
#   A Tensor in the context of Chess ML is 3D
#   (Height, Width, Channels)
#   8 × 8 × 18
#   Channel 0: Where the white pawns are
#   Channel 1: Where the white knights are
#   Channel 2: Where the white bishops are
#   Channel 3: Where the white rooks are
#   Channel 4: Where the white queens are
#   Channel 5: Where the white king is
#   Channels 6–11: Same for black pieces
#   Channel 12: Whose turn it is
#   Channels 13–16: Castling rights (K, Q, k, q)
#   Channel 17: En passant square

pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 18), dtype=np.float32)

    # 1. Piece planes (channels 0-11)
    piece_map = board.piece_map()
    channel = 0
    
    for piece_type in pieces:
        # White piece plane
        for sq, piece in piece_map.items():
            if piece.piece_type == piece_type and piece.color == chess.WHITE:
                tensor[sq // 8, sq % 8, channel] = 1
        channel += 1

        # Black piece plane
        for sq, piece in piece_map.items():
            if piece.piece_type == piece_type and piece.color == chess.BLACK:
                tensor[sq // 8, sq % 8, channel] = 1
        channel += 1

    # 2. Side-to-move plane (channel 12)
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1

    # 3. Castling planes (channels 13-16)
    fen_parts = board.fen().split()
    castling_rights = fen_parts[2]
    
    if "K" in castling_rights:
        tensor[:, :, 13] = 1
    if "Q" in castling_rights:
        tensor[:, :, 14] = 1
    if "k" in castling_rights:
        tensor[:, :, 15] = 1
    if "q" in castling_rights:
        tensor[:, :, 16] = 1

    # 4. En passant plane (channel 17)
    if board.ep_square is not None:
        sq = board.ep_square
        tensor[sq // 8, sq % 8, 17] = 1

    return tensor  # shape (8, 8, 18)


# # Example usage
# if __name__ == "__main__":
#     # Starting position
#     fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#     tensor = fen_to_tensor(fen)
    
#     print(f"Tensor shape: {tensor.shape}")
#     print(f"White pawns (channel 0):\n{tensor[:, :, 0]}")
#     print(f"Side to move (channel 12):\n{tensor[:, :, 12][0, 0]}")  # Should be 1.0 for white
#     print(f"Total pieces: {np.sum(tensor[:, :, 0:12])}")  # Should be 32