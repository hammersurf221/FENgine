import chess
import chess.svg
import cairosvg
from PIL import Image
import io
import os
import random

def generate_image_from_fen(fen, output_path):
    board = chess.Board(fen)
    svg = chess.svg.board(board, size=256)
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data))
    image.save(output_path)

def generate_random_fens(count):
    fens = set()
    while len(fens) < count:
        board = chess.Board()
        for _ in range(random.randint(10, 60)):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        fens.add(board.fen())
    return list(fens)

def generate_dataset(num_samples=1000, out_dir="data/train"):
    os.makedirs(out_dir, exist_ok=True)
    fens = generate_random_fens(num_samples)
    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        for i, fen in enumerate(fens):
            path = os.path.join(out_dir, f"{i:04d}.png")
            generate_image_from_fen(fen, path)
            f.write(f"{i:04d}.png {fen}\n")

if __name__ == "__main__":
    generate_dataset(5000)
