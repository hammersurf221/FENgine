import os
import torch
from infer import load_image, predict_fen
from ccn_model import CCN
from visual_debug import draw_prediction

# Load trained model
model = CCN()
model.load_state_dict(torch.load("ccn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Path to folder with screenshots
FOLDER = "test_screenshots"

# Loop through all PNGs


for filename in os.listdir(FOLDER):
    if filename.endswith(".png"):
        path = os.path.join(FOLDER, filename)
        save_path = f"output_debugs//output_debug_{filename}"
        draw_prediction(path, model, save_path=save_path)
        print(f"✅ {filename} → {save_path}")

