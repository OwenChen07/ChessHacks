import chess.pgn, sys, csv
from tqdm import tqdm

# This file processes PGN files into a CSV format suitable for training
# Each row in the CSV will contain: 
#   FEN string (board state)
#   move in UCI format, 
#   game result (win: 1, tie: 0.5, loss: 0)

def prepare_data(pgn_path, out_csv, max_games=int(1e9)):
    with open(pgn_path) as f, open(out_csv, "w", newline='') as out:
        writer = csv.writer(out)
        writer.writerow(["fen","move","result"])
        for _ in tqdm(range(max_games)):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result","*")
            # convert to numeric result from White POV
            if result == "1-0": res = 1.0
            elif result == "1/2-1/2": res = 0.5
            elif result == "0-1": res = 0.0
            else: res = 0.5
            node = game
            board = game.board()
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                # save the position before the move
                writer.writerow([board.fen(), move.uci(), res])
                board.push(move)
                node = next_node

prepare_data("data/lichess_elite_2022-01.pgn", "data/data2.csv", 480109)