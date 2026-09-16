# GAN-based Image Generation

A beginner-friendly Generative Adversarial Network project that trains a Deep Convolutional GAN (DCGAN) with PyTorch to create synthetic RGB images.

The project uses the open CIFAR-10 dataset and contains notebook-based training/evaluation plus a small Flask web app for generating images from random latent noise.

## Project Highlights

- GAN architecture with a **Generator** and **Discriminator**
- PyTorch implementation
- CIFAR-10 dataset through `torchvision`
- Notebook-first workflow for learning and demonstration
- Fixed-noise samples to visualize training progress
- Saved Generator checkpoint for the web application
- Flask interface for generating a grid of synthetic images
- LinkedIn and GitHub links included in the web app
- Simple repository structure with no `src/`, preprocessing modules, or extra Python modules

The architecture follows the core DCGAN ideas described in the PyTorch DCGAN tutorial: convolutional layers in the discriminator, transposed convolutions in the generator, batch normalization, LeakyReLU/ReLU activations, and Tanh output normalization.

## Dataset

This project uses **CIFAR-10**, which contains 60,000 32×32 RGB images across 10 classes. The dataset is downloaded automatically by `torchvision` when the first notebook is executed.

CIFAR-10 is used here as a compact, practical dataset for learning GAN training rather than as a claim that the model produces photorealistic high-resolution images.

## Repository Structure

```text
GAN-based-Image-Generation/
│
├── notebooks/
│   ├── 01_GAN_Training.ipynb
│   └── 02_GAN_Generation_Demo.ipynb
│
├── models/
│   └── README.md
│
├── generated/
│   └── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

> The trained model files are intentionally not included in the source repository archive. Run the training notebook first; it creates the Generator checkpoint required by the Flask app.

## How the GAN Works

The model contains two neural networks:

**Generator**

Takes a random latent vector and attempts to transform it into an image that looks like it came from the training dataset.

**Discriminator**

Receives either a real CIFAR-10 image or a generated image and learns to distinguish between real and generated samples.

Training alternates between improving the discriminator and improving the generator. As training progresses, the generator attempts to produce samples that the discriminator cannot easily distinguish from real images.

## Notebook Workflow

### 1. Install dependencies

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

### 2. Train the GAN

Open:

```text
notebooks/01_GAN_Training.ipynb
```

Run the notebook from top to bottom.

The notebook:

1. Imports PyTorch and torchvision.
2. Downloads CIFAR-10.
3. Normalizes images to `[-1, 1]`.
4. Defines the Generator.
5. Defines the Discriminator.
6. Initializes model weights.
7. Trains the GAN.
8. Records Generator and Discriminator losses.
9. Saves generated sample grids.
10. Saves the Generator checkpoint into `models/generator.pth`.
11. Saves the Discriminator checkpoint into `models/discriminator.pth`.

For a first run on a CPU, use the smaller epoch setting in the notebook. A CUDA-capable GPU is strongly recommended for longer training.

### 3. Generate images in a notebook

Open:

```text
notebooks/02_GAN_Generation_Demo.ipynb
```

This notebook loads the trained Generator and produces new images from randomly sampled latent vectors.

### 4. Run the Flask web app

From the repository root:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web page lets you choose how many images to generate and displays the resulting image grid.

The page also contains the project owner's GitHub and LinkedIn profiles:

- GitHub: https://github.com/InfinitePraveen
- LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Important Note About the Generated Images

CIFAR-10 images are only 32×32 pixels. Therefore, the generated samples are intended to demonstrate the GAN learning process and synthetic image generation rather than professional photorealistic image synthesis.

For interviews, explain:

- why a GAN needs two competing networks,
- how the latent vector is converted into an image,
- how the discriminator provides a learning signal,
- why GAN training can be unstable,
- what mode collapse means,
- why the losses should not be interpreted like ordinary classifier accuracy,
- and how the same architecture could be scaled to a larger dataset.

## Main Hyperparameters

The notebook uses a practical DCGAN-style setup:

- Latent vector size: 100
- Image size: 32×32
- Channels: 3
- Batch size: 128
- Learning rate: 0.0002
- Adam beta1: 0.5
- Loss: Binary Cross Entropy with logits
- Optimizer: Adam

These values are intentionally kept visible in the notebook so they can be explained during an interview.

## Interview Discussion Points

### Why GANs?

GANs learn a data distribution without requiring class labels for the generation objective. The generator learns to synthesize samples while the discriminator learns to identify generated samples.

### Why DCGAN?

DCGAN is a natural starting point for image generation because convolutional layers are well suited to spatial image structure.

### What happens if the discriminator becomes too strong?

The generator may receive weak or unhelpful gradients. GAN training therefore requires a balance between the two networks.

### What is mode collapse?

Mode collapse occurs when the generator produces limited varieties of samples instead of covering the diversity of the training distribution.

### Why use a fixed noise batch?

A fixed latent batch provides a consistent visual reference during training, making it easier to see how generated images change from epoch to epoch.

## License

This project is intended for educational and portfolio use. Check the dataset terms before redistributing dataset files.
