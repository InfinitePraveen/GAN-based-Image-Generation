from pathlib import Path
import io
import base64

import torch
import torch.nn as nn
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "generator.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_SIZE = 100
IMAGE_SIZE = 32


class Generator(nn.Module):
    """DCGAN-style generator for 32x32 RGB images."""

    def __init__(self, latent_size=100, channels=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)


generator = None


def load_generator():
    global generator

    if generator is not None:
        return generator

    if not MODEL_PATH.exists():
        return None

    model = Generator(latent_size=LATENT_SIZE).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    generator = model
    return generator


def image_grid_to_base64(images):
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    grid = make_grid(images, nrow=4, normalize=True, value_range=(-1, 1))
    image = to_pil_image(grid.cpu())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    error = None
    count = 16

    if request.method == "POST":
        try:
            count = int(request.form.get("count", 16))
            count = max(1, min(count, 32))

            model = load_generator()

            if model is None:
                error = (
                    "Trained Generator not found. Run "
                    "notebooks/01_GAN_Training.ipynb first and create "
                    "models/generator.pth."
                )
            else:
                noise = torch.randn(count, LATENT_SIZE, 1, 1, device=DEVICE)
                with torch.no_grad():
                    fake_images = model(noise)

                image_data = image_grid_to_base64(fake_images)

        except Exception as exc:
            error = f"Could not generate images: {exc}"

    return render_template(
        "index.html",
        image_data=image_data,
        error=error,
        count=count,
        device=str(DEVICE),
    )


if __name__ == "__main__":
    app.run(debug=True)
