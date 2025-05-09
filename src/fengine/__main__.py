from .fen_predictor import load_model, load_image, predict_fen
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m fengine <image_path> [w|b]")
        sys.exit(1)

    image_path = sys.argv[1]
    my_color = sys.argv[2] if len(sys.argv) > 2 else "w"

    model = load_model()
    image = load_image(image_path, my_color=my_color)
    fen = predict_fen(model, image, my_color=my_color)
    print(fen)