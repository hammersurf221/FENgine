
# ♟️ Convolutional Chess Network (CCN)

This project trains a convolutional neural network to **recognize chess pieces from a screenshot of a digital chess board** and generate the correct **FEN (Forsyth-Edwards Notation)**.

It supports:
- Training a custom model on synthetic board images
- Real-time FEN prediction from screenshots
- Visual debugging of predictions
- Integration into GUI or chess helper apps

## 🚀 Features

- 🔍 **Accurate piece recognition** across all 12 chess pieces + empty squares
- 🎯 **FEN output** for use with engines like Stockfish
- 🧠 Built with PyTorch + torchvision
- 🧪 Includes visual debugging and test scripts
- 📦 Fully modular: train, infer, and visualize

## 📁 Project Structure

```
.
├── ccn_model.py              # The CNN architecture
├── dataset.py                # Custom Dataset class + FEN encoder
├── generate_dataset.py       # Augmented dataset generator
├── train.py                  # Model training loop
├── infer.py                  # Simple FEN prediction script
├── visual_debug.py           # Generates visual overlays of predictions
├── screenshots/              # Folder to test screenshots
├── data/train/               # Auto-generated training data
├── confusion_matrix_*.png    # Saved evaluation outputs
└── ccn_model.pth             # Saved trained model
```

## 🛠️ Setup

1. **Install requirements**
```bash
pip install torch torchvision matplotlib seaborn cairosvg python-chess
```

2. **Generate training data**
```bash
python generate_dataset.py
```

3. **Train the model**
```bash
python train.py
```

4. **Test on an image**
```bash
python infer.py                # Uses test_screenshot.png
python visual_debug.py --img screenshots/test1.png --fen "optional_true_fen"
```

## 🧪 Batch Testing (Optional)

Use `batch_test.py` to test all screenshots in a folder and output their predicted FENs.

## 🔍 Output Example

**Input**: Screenshot of a digital chessboard  
**Output**:  
```
r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

## 🧱 Model Details

- 3 convolutional layers
- Global average pooling + 1×1 convolution
- Dropout for regularization
- Output shape: `[batch_size, 8, 8, 13]` (12 pieces + empty)

## 📌 Notes

- The model is trained entirely on **synthetic images**, using random boards rendered in SVG and converted to PNG.
- Augmentations include rotation, shift, crop, brightness/contrast jitter, and theme randomization.

## 🤝 License

MIT License — use and modify freely.
