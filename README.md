# ♟️ FENgine – Chess Position Recognition from Images (FEN Predictor)

**FENgine** is an AI-powered tool that can recognize a chess board from an image and generate a valid FEN (Forsyth–Edwards Notation) string. It's built using a Convolutional Chess Network (CCN) trained on real chess screenshots and is ready for developers, chess enthusiasts, or tool builders who want to integrate computer vision into their chess projects.

---

## 📸 What It Does

FENgine takes a screenshot of a chess board and outputs its **FEN string**, a compact representation of the board state. This can then be used in chess engines (like Stockfish), web apps, GUIs, or databases.

---

## 📁 Folder Structure Overview

```
FENgine_Commercial/
├── src/fengine/                # Core model & logic
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   ├── fen_predictor.py        # Predict FEN from image - currently tailored to my own GUI, infer.py inside /deprecated is more general
│   ├── ccn_model.py            # The neural network
│   ├── dataset.py              # Piece mapping (IDX↔Piece)
│   └── utils/
│       ├── __init__.py
│       └── visual_debug.py     # Debugging/visualizing output
│
├── scripts/                    # Training & data generation
│   ├── train.py
│   ├── batch_test.py
│   ├── generate_dataset.py
│   ├── generate_dataset_custom.py
│   └── generate_lichess_dataset.py
│
├── test_screenshots/           # Example inputs
│   ├── test1.png
│   └── test_screenshot.fen
│
├── MODELS/                     # Pretrained weights (.pth)
│   ├── ccn_model_lichess.pth
│   └── ccn_model_chesscom_icysea.pth
│
├── output_debugs/              # Visual prediction heatmaps
├── assets/                     # Piece images & board assets
├── deprecated/                 # Old versions of scripts
├── requirements.txt
├── setup.py
└── README.md
```

---

## 💻 Installation Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourname/FENgine.git
cd FENgine
```

### 2. Install dependencies
Make sure you're using Python 3.8–3.11.

```bash
pip install -r requirements.txt
```

### 3. Install the package
This makes `fengine` available as a Python module and enables CLI support.
```bash
pip install .
```

---

## 🚀 Using the CLI

### 📸 Predict FEN from an image
```bash
python -m fengine path/to/image.png w
```

- `image.png`: path to your board screenshot
- `w` (or `b`): perspective — white's or black's POV

### ✅ Example:
```bash
python -m fengine test_screenshots/test1.png w
```

**Output:**
```
r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1
```

---

## 🧠 Using It as a Python Module

```python
from fengine.fen_predictor import load_model, load_image, predict_fen

# Load model
model = load_model("MODELS/ccn_model_lichess.pth")

# Load an image and predict
img_tensor = load_image("test_screenshots/test1.png", my_color="w")
fen = predict_fen(model, img_tensor, my_color="w")

print("Predicted FEN:", fen)
```

---

## 🧪 Testing Your Model

To batch-test predictions on multiple screenshots:
```bash
python scripts/batch_test.py
```

To visualize what the model is "seeing", use:
```bash
python -m fengine.utils.visual_debug
```

---

## 🏗️ Training a Model (Optional)

### 1. Generate Dataset
Use one of the dataset generators (e.g., from Lichess):
```bash
python scripts/generate_lichess_dataset.py
```

### 2. Train Your CCN
```bash
python scripts/train.py
```

Model will be saved as `.pth` in `MODELS/`.

---

## 📦 Model Format

Model weights are saved as `.pth` files using PyTorch. You can swap them out or fine-tune them using your own data.

---

## ❓ What Is FEN?

FEN stands for *Forsyth–Edwards Notation*. It’s a standard notation for describing the state of a chess game. Example:

```
r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1
```

Learn more: https://en.wikipedia.org/wiki/Forsyth–Edwards_Notation

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it even commercially.

---

## 🙌 Contributing

Pull requests, feedback, and new ideas are always welcome!

---

## 👤 Author

Made by William Samiri

- GitHub: [github.com/hammersurf221](https://github.com/hammersurf221)
- Contact: williamsamiri011@gmail.com