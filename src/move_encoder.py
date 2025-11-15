import chess
import numpy as np

# Note: this implementation uses the convention:
#  - square index 0 == a1, 7 == h1, 8 == a2, ... 63 == h8
#  - row = square // 8 (row 0 = rank 1, row 7 = rank 8)
#  - planes: 0..55 sliding (8 directions * 7 distances),
#            56..63 knight moves (8),
#            64..72 underpromotions (3 pieces * 3 directions)
#  - Queen promotions are encoded as a normal sliding "forward" move
#    (this matches your original approach).

KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]

DIRECTION_VECTORS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1)   # NW
]

def move_to_policy_index(move: chess.Move, board: chess.Board):
    from_sq = move.from_square
    to_sq = move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)

    # compute deltas in board coords (rank-increasing = north)
    row_diff = to_row - from_row
    col_diff = to_col - from_col

    # Convert deltas to the MOVERS perspective: forward should be +1
    # If Black is to move, invert both deltas
    if not board.turn:  # False -> black to move
        row_diff = -row_diff
        col_diff = -col_diff

    # PROMOTIONS
    if move.promotion is not None:
        # Queen promotion: treat as normal forward queen move (falls through)
        if move.promotion != chess.QUEEN:
            # col_diff must be -1,0,1 for promotions
            if col_diff not in (-1, 0, 1):
                raise ValueError("Unexpected promotion column delta")
            # map -1 -> 0 (left), 0 -> 1 (straight), 1 -> 2 (right)
            dir_idx = col_diff + 1
            # piece offsets: Knight, Bishop, Rook  (we exclude queen here)
            if move.promotion == chess.KNIGHT:
                piece_off = 0
            elif move.promotion == chess.BISHOP:
                piece_off = 1
            elif move.promotion == chess.ROOK:
                piece_off = 2
            else:
                raise ValueError("Unexpected promotion piece")
            plane = 64 + dir_idx * 3 + piece_off
            return from_row, from_col, plane
        # else: queen promotion falls through to sliding handling below

    # KNIGHT
    for k_idx, (kr, kc) in enumerate(KNIGHT_MOVES):
        if row_diff == kr and col_diff == kc:
            return from_row, from_col, 56 + k_idx

    # SLIDING / STEPPING: 8 directions * distance 1..7
    dist = max(abs(row_diff), abs(col_diff))
    if dist >= 1 and dist <= 7:
        # normalize direction: divide components by distance when non-zero
        nr = row_diff // dist if row_diff != 0 else 0
        nc = col_diff // dist if col_diff != 0 else 0
        # find matching direction index
        try:
            dir_idx = DIRECTION_VECTORS.index((nr, nc))
        except ValueError:
            dir_idx = None
        if dir_idx is not None:
            plane = dir_idx * 7 + (dist - 1)
            return from_row, from_col, plane

    raise ValueError(f"Cannot encode move {move.uci()} (from {from_row},{from_col} dr={row_diff} dc={col_diff})")


def encode_move_to_tensor(move: chess.Move, board: chess.Board):
    tensor = np.zeros((8, 8, 73), dtype=np.float32)
    r, c, p = move_to_policy_index(move, board)
    tensor[r, c, p] = 1.0
    return tensor


def decode_tensor_to_move(tensor: np.ndarray, board: chess.Board):
    # find max element
    flat_idx = int(np.argmax(tensor))
    from_row, from_col, plane = np.unravel_index(flat_idx, tensor.shape)
    from_sq = from_row * 8 + from_col

    # UNDERPROMOTIONS (64..72)
    if plane >= 64:
        offset = plane - 64
        dir_idx = offset // 3  # 0=left,1=straight,2=right
        piece_off = offset % 3  # 0=knight,1=bishop,2=rook
        col_diff = dir_idx - 1  # -1,0,1
        # forward is +1 from mover perspective; invert for black
        dr = 1 if board.turn else -1
        to_row = from_row + (1 if board.turn else -1)
        to_col = from_col + col_diff
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise ValueError("Decoded promotion out of bounds")
        to_sq = to_row * 8 + to_col
        promotion_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion_piece = promotion_map[piece_off]
        return chess.Move(from_sq, to_sq, promotion=promotion_piece)

    # KNIGHT (56..63)
    if 56 <= plane <= 63:
        kr, kc = KNIGHT_MOVES[plane - 56]
        # remember: kr,kc are from mover's perspective; invert for black
        if not board.turn:
            kr, kc = -kr, -kc
        to_row = from_row + kr
        to_col = from_col + kc
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise ValueError("Decoded knight out of bounds")
        to_sq = to_row * 8 + to_col
        return chess.Move(from_sq, to_sq)

    # SLIDING (0..55)
    dir_idx = plane // 7
    dist = (plane % 7) + 1
    vr, vc = DIRECTION_VECTORS[dir_idx]
    # vr,vc are in mover's perspective; invert for black
    if not board.turn:
        vr, vc = -vr, -vc
    to_row = from_row + vr * dist
    to_col = from_col + vc * dist
    if not (0 <= to_row < 8 and 0 <= to_col < 8):
        raise ValueError("Decoded sliding move out of bounds")
    to_sq = to_row * 8 + to_col

    # handle queen promotions that were encoded as sliding forward to last rank
    # if the move lands on the promotion rank, set promotion=QUEEN
    if board.turn and from_row == 6 and to_row == 7:
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if (not board.turn) and from_row == 1 and to_row == 0:
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return chess.Move(from_sq, to_sq)
