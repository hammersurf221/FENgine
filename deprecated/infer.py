import torch
import numpy as np
from PIL import Image
from ccn_model import CCN
from dataset import PIECE_TO_IDX

IDX_TO_PIECE = {v: k for k, v in PIECE_TO_IDX.items()}

model = CCN()
model.load_state_dict(torch.load("ccn_model.pth", map_location=torch.device("cpu")))
model.eval()

def load_image(path):
    img = Image.open(path).convert("RGB").resize((256, 256))
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)

def predict_fen(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        preds = output.argmax(dim=-1).squeeze(0)

    fen_rows = []
    for row in preds:
        fen_row = ''
        empty = 0
        for idx in row:
            piece = IDX_TO_PIECE[int(idx)]
            if piece == '.':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    return '/'.join(fen_rows) + ' w - - 0 1'

if __name__ == "__main__":
    image_tensor = load_image("test_screenshot.png")
    fen = predict_fen(image_tensor)
    print("Predicted FEN:")
    print(fen)
