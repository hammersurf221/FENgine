import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataset import PIECE_TO_IDX
from ccn_model import CCN
from infer import load_image
import argparse

# Reverse mapping
IDX_TO_PIECE = {v: k for k, v in PIECE_TO_IDX.items()}

def fen_to_matrix(fen):
    rows = fen.split()[0].split('/')
    matrix = []
    for row in rows:
        expanded = []
        for c in row:
            if c.isdigit():
                expanded.extend(['.'] * int(c))
            else:
                expanded.append(c)
        matrix.append(expanded)
    return matrix

def draw_prediction(image_path, model, expected_fen=None, save_path="output_debug.png"):
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    draw = ImageDraw.Draw(img)

    input_tensor = load_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        preds = output.argmax(dim=-1).squeeze(0).cpu().numpy()

    expected = None
    if expected_fen:
        expected = fen_to_matrix(expected_fen)

    font = ImageFont.load_default()
    cell_size = 256 // 8

    for r in range(8):
        for c in range(8):
            pred_char = IDX_TO_PIECE[preds[r][c]]
            x, y = c * cell_size, r * cell_size
            box = [x, y, x + cell_size, y + cell_size]

            color = "white"
            if expected:
                true_char = expected[r][c]
                if true_char == pred_char:
                    color = "green" if true_char != "." else "gray"
                else:
                    color = "red"

            draw.text((x + 5, y + 2), pred_char, fill=color, font=font)

    img.save(save_path)
    print(f"✅ Saved visual debug to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="test_screenshot.png", help="Path to board screenshot")
    parser.add_argument("--fen", default=None, help="Optional: true FEN to compare predictions")
    args = parser.parse_args()

    model = CCN()
    model.load_state_dict(torch.load("ccn_model.pth", map_location=torch.device("cpu")))
    model.eval()

    draw_prediction(args.img, model, args.fen)
