
import os
import random
import colorsys
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import cv2
import chess

OUT_DIR = "data/train_png"
IMAGE_SIZE = 256
TILE_SIZE = 80

ASSET_DIR = "assets"
PIECE_DIR = os.path.join(ASSET_DIR, "pieces")
EMPTY_LIGHT_PATH = os.path.join(ASSET_DIR, "empty_light.png")
EMPTY_DARK_PATH = os.path.join(ASSET_DIR, "empty_dark.png")

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

def apply_augmentations(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if random.random() < 0.7:
        top = random.randint(1, 4)
        bottom = random.randint(1, 4)
        left = random.randint(1, 4)
        right = random.randint(1, 4)
        h, w = img.shape[:2]
        img = img[top:h - bottom, left:w - right]

    if random.random() < 0.6:
        tx = random.randint(-3, 3)
        ty = random.randint(-3, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.95, 1.05))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.95, 1.05))

    return pil

def random_board_colors():
    h = random.random()
    s = 0.4 + random.random() * 0.4
    v = 0.85 + random.random() * 0.1

    light_rgb = colorsys.hsv_to_rgb(h, s * 0.4, 1.0)
    dark_rgb = colorsys.hsv_to_rgb(h, s, v * 0.65)

    light = tuple(int(c * 255) for c in light_rgb)
    dark = tuple(int(c * 255) for c in dark_rgb)

    return light, dark

def tint_image(image, color):
    img = image.convert("RGBA")
    r, g, b = color
    tint = Image.new("RGBA", img.size, (r, g, b, 0))
    return Image.blend(tint, img, alpha=0.5)

def render_board(fen):
    board = chess.Board(fen)
    base = Image.new("RGBA", (TILE_SIZE * 8, TILE_SIZE * 8), (255, 255, 255, 255))

    light, dark = random_board_colors()
    empty_light = tint_image(Image.open(EMPTY_LIGHT_PATH).resize((TILE_SIZE, TILE_SIZE)), light)
    empty_dark = tint_image(Image.open(EMPTY_DARK_PATH).resize((TILE_SIZE, TILE_SIZE)), dark)

    squares = list(chess.SQUARES)
    highlight_piece = random.choice([sq for sq in squares if board.piece_at(sq)])
    highlight_empty = random.choice([sq for sq in squares if not board.piece_at(sq)])

    for square in squares:
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        is_dark = (file + rank) % 2 == 1
        tile = empty_dark.copy() if is_dark else empty_light.copy()

        # Highlight if needed
        if square in [highlight_piece, highlight_empty]:
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, TILE_SIZE, TILE_SIZE], fill=(255, 0, 0, 80))

        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            color = "b" if symbol.islower() else "w"
            piece_type = symbol.upper()
            piece_path = os.path.join(PIECE_DIR, f"{color}{piece_type}.png")
            if os.path.exists(piece_path):
                piece_img = Image.open(piece_path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
                tile.alpha_composite(piece_img)

        base.paste(tile, (file * TILE_SIZE, rank * TILE_SIZE))

    return base.convert("RGB")

def generate_dataset(num_samples=1000):
    os.makedirs(OUT_DIR, exist_ok=True)
    fens = generate_random_fens(num_samples)

    with open(os.path.join(OUT_DIR, "labels.txt"), "w") as f:
        for i, fen in enumerate(fens):
            base_img = render_board(fen)
            final_img = apply_augmentations(base_img)
            filename = f"{i:04d}.png"
            final_img.save(os.path.join(OUT_DIR, filename))
            f.write(f"{filename} {fen}\n")

    print(f"✅ Generated {num_samples} PNG-based augmented boards.")

if __name__ == "__main__":
    generate_dataset(num_samples=5000)
