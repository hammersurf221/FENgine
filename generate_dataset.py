import chess
import chess.svg
import cairosvg
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import io
import os
import random
import cv2

OUT_DIR = "data/train"
IMAGE_SIZE = 256

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

def random_color_palette():
    palettes = [
        ("#f0d9b5", "#b58863"),
        ("#f0f0d0", "#769656"),
        ("#dcdcdc", "#4b4b4b"),
        ("#ffefc2", "#bc987e"),
        ("#eee4da", "#7a6f5a"),
        ("#ffe4e1", "#ff7f50")
    ]
    return random.choice(palettes)

def generate_random_highlight():
    hue = random.randint(0, 360)
    color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 1, 1))
    return color

def render_board(fen):
    import colorsys
    board = chess.Board(fen)

    # 🎨 Generate light/dark square colors from random hue
    def random_board_colors():
        h = random.random()
        s = 0.4 + random.random() * 0.4  # saturation
        v = 0.85 + random.random() * 0.1  # brightness

        light_rgb = colorsys.hsv_to_rgb(h, s * 0.4, 1.0)
        dark_rgb = colorsys.hsv_to_rgb(h, s, v * 0.65)

        light = tuple(int(c * 255) for c in light_rgb)
        dark = tuple(int(c * 255) for c in dark_rgb)

        return f"#{light[0]:02x}{light[1]:02x}{light[2]:02x}", f"#{dark[0]:02x}{dark[1]:02x}{dark[2]:02x}"

    light, dark = random_board_colors()

    # Identify piece and empty squares
    empty_squares = []
    piece_squares = []
    fen_rows = fen.split()[0].split('/')
    for rank in range(8):
        file = 0
        for char in fen_rows[rank]:
            if char.isdigit():
                for _ in range(int(char)):
                    empty_squares.append((file, rank))
                    file += 1
            else:
                piece_squares.append((file, rank))
                file += 1

    highlight_piece = random.choice(piece_squares) if piece_squares else (0, 0)
    highlight_empty = random.choice(empty_squares) if empty_squares else (7, 7)

    # Render SVG board with pieces
    svg = chess.svg.board(
        board,
        size=320,
        coordinates=False,
        colors={
            "square light": light,
            "square dark": dark,
            "margin": "#ffffff",
            "border": "#ffffff"
        }
    )
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data)).convert("RGBA")

    # Overlay highlights between board and pieces
    highlight_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight_layer)

    cell_size = image.size[0] // 8
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.8, 1)]
    highlight_color = (r, g, b, 120)

    for fx, ry in [highlight_piece, highlight_empty]:
        x0 = fx * cell_size
        y0 = ry * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        draw.rectangle([x0, y0, x1, y1], fill=highlight_color)

    image = Image.alpha_composite(highlight_layer, image)
    return image.convert("RGB")





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

    print(f"✅ Generated {num_samples} augmented boards.")

if __name__ == "__main__":
    generate_dataset(num_samples=5000)
