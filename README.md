# GAN-based Image Generation

A beginner-friendly Generative Adversarial Network project that trains a Deep Convolutional GAN (DCGAN) with PyTorch to generate synthetic RGB images.

The project uses the open-source CIFAR-10 dataset and provides notebook-based training and inference workflows along with a Flask web application for interactive image generation.

## Project Highlights

* GAN architecture with a Generator and Discriminator
* PyTorch implementation
* CIFAR-10 dataset through `torchvision`
* DCGAN-style convolutional architecture
* Notebook-based training workflow
* Notebook-based image generation demo
* Generator and Discriminator model checkpoints
* Generated image samples
* GAN training loss visualization
* Flask web application
* Interactive synthetic image generation
* GitHub and LinkedIn profile links in the web app
* Simple repository structure
* No `src/` directory
* No separate preprocessing modules
* No unnecessary utility or configuration modules

## Dataset

This project uses the **CIFAR-10** dataset.

CIFAR-10 contains 60,000 32×32 RGB images divided into 10 classes. The dataset is downloaded automatically through `torchvision` when the training notebook is executed.

CIFAR-10 is intentionally used as a compact dataset for demonstrating GAN training and image generation.

## How the GAN Works

The project contains two neural networks:

### Generator

The Generator receives a random latent vector and transforms it into a synthetic RGB image.

The Generator progressively increases the spatial resolution of the input using transposed convolution layers.

```text
Random Latent Vector
        ↓
Generator
        ↓
Synthetic 32×32 RGB Image
```

### Discriminator

The Discriminator receives both real CIFAR-10 images and generated images.

Its objective is to determine whether an image is real or generated.

```text
Real Image ──────────┐
                     ├──→ Discriminator → Real / Fake
Generated Image ─────┘
```

During training, the Generator attempts to produce increasingly realistic images while the Discriminator learns to distinguish generated images from real training examples.

## DCGAN Architecture

The project follows the main ideas of a Deep Convolutional GAN.

### Generator

```text
100 × 1 × 1
     ↓
512 × 4 × 4
     ↓
256 × 8 × 8
     ↓
128 × 16 × 16
     ↓
64 × 32 × 32
     ↓
3 × 32 × 32
```

The final Generator layer uses `Tanh`, while the training images are normalized to approximately `[-1, 1]`.

### Discriminator

The Discriminator progressively reduces the spatial dimensions of the input image and produces a single output representing its real/fake prediction.

## Repository Structure

```text
GAN-based-Image-Generation/
│
├── generated/
│   ├── final_generated_samples.png
│   ├── inference_samples.png
│   ├── .gitkeep
│   └── README.md
│
├── models/
│   ├── generator.pth
│   ├── discriminator.pth
│   ├── .gitkeep
│   └── README.md
│
├── notebooks/
│   ├── 01_GAN_Training.ipynb
│   └── 02_GAN_Generation_Demo.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

## File Description

| File / Directory                         | Purpose                                            |
| ---------------------------------------- | -------------------------------------------------- |
| `notebooks/01_GAN_Training.ipynb`        | Complete GAN training workflow                     |
| `notebooks/02_GAN_Generation_Demo.ipynb` | Loads the trained Generator and creates new images |
| `models/generator.pth`                   | Trained Generator checkpoint                       |
| `models/discriminator.pth`               | Trained Discriminator checkpoint                   |
| `generated/final_generated_samples.png`  | Final generated samples from GAN training          |
| `generated/inference_samples.png`        | Samples generated during inference                 |
| `app.py`                                 | Flask web application                              |
| `templates/index.html`                   | Web application interface                          |
| `static/style.css`                       | Web application styling                            |
| `models/README.md`                       | Model checkpoint information                       |
| `generated/README.md`                    | Generated image information                        |
| `CONTRIBUTING.md`                        | Contribution guidelines                            |
| `CHANGELOG.md`                           | Project change history                             |
| `requirements.txt`                       | Python dependencies                                |
| `.gitignore`                             | Git ignore configuration                           |

## Requirements

Use Python **3.11 or 3.12**.

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the GAN

Open:

```text
notebooks/01_GAN_Training.ipynb
```

Run the notebook from beginning to end.

The notebook will:

1. Import the required libraries.
2. Download the CIFAR-10 dataset.
3. Normalize the images.
4. Create the Generator.
5. Create the Discriminator.
6. Initialize the model weights.
7. Configure the loss function and optimizers.
8. Train the GAN.
9. Track Generator and Discriminator losses.
10. Generate fixed-noise samples during training.
11. Save the trained Generator.
12. Save the trained Discriminator.
13. Save final generated image samples.

After training, the following model files are available:

```text
models/generator.pth
models/discriminator.pth
```

## Generate New Images

Open:

```text
notebooks/02_GAN_Generation_Demo.ipynb
```

This notebook loads:

```text
models/generator.pth
```

and generates new synthetic images using randomly sampled latent vectors.

The generated inference output is saved as:

```text
generated/inference_samples.png
```

## Run the Flask Web App

After training the GAN, run the application from the repository root:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The web application allows you to select the number of images to generate and displays them as an image grid.

## Web Application

The Flask application provides a simple interface for demonstrating the trained GAN.

It includes:

* Number of images selection
* Random latent noise generation
* Generator inference
* Generated image grid
* CPU/CUDA device information
* Error handling when the model checkpoint is missing
* Project information
* GitHub profile
* LinkedIn profile

### Profiles

**GitHub**

https://github.com/InfinitePraveen

**LinkedIn**

https://www.linkedin.com/in/infinitepraveen/

## Main Hyperparameters

| Parameter          |             Value |
| ------------------ | ----------------: |
| Latent Vector Size |               100 |
| Image Size         |             32×32 |
| Image Channels     |                 3 |
| Batch Size         |               128 |
| Learning Rate      |            0.0002 |
| Adam Beta 1        |               0.5 |
| Loss Function      | BCEWithLogitsLoss |
| Optimizer          |              Adam |

These parameters are intentionally kept simple and visible in the training notebook so the complete training process can be explained during an interview.

## Generated Samples

The repository contains generated examples from the trained model:

```text
generated/final_generated_samples.png
generated/inference_samples.png
```

These images demonstrate the output produced by the Generator after training.

Because CIFAR-10 images are only 32×32 pixels, the generated samples are intended primarily to demonstrate the GAN learning process and synthetic image generation rather than high-resolution photorealistic image synthesis.

## Interview Discussion Points

This project can be used to demonstrate understanding of:

* Generative Adversarial Networks
* Generator and Discriminator architecture
* DCGAN
* Latent vectors
* Transposed convolution
* Convolutional neural networks
* Batch normalization
* ReLU and LeakyReLU
* Tanh output normalization
* BCEWithLogitsLoss
* Adam optimization
* Adversarial training
* GAN training instability
* Mode collapse
* Generator vs Discriminator losses
* Fixed latent vectors
* Synthetic image generation
* Model checkpointing
* PyTorch inference
* Flask deployment

## What Is Mode Collapse?

Mode collapse occurs when a GAN Generator learns to produce only a limited variety of outputs instead of representing the diversity present in the training data.

This is one of the important challenges to discuss when explaining GANs during an interview.

## Why Use Fixed Noise?

The training notebook uses a fixed set of latent vectors to generate samples after each epoch.

This provides a consistent reference for visually observing how the Generator changes during training.

## Limitations

This project intentionally uses CIFAR-10 because it is relatively small and practical for experimentation.

The main limitations are:

* Generated images are only 32×32 pixels.
* GAN training can be unstable.
* Generated samples may contain artifacts.
* Mode collapse can occur.
* CPU training can be slow.
* Results depend on training duration and hardware.

## Future Improvements

Possible extensions include:

* Training for more epochs
* Higher-resolution image generation
* Larger datasets
* Conditional GANs
* Wasserstein GAN
* WGAN-GP
* Progressive image generation
* Better evaluation metrics
* Improved web interface
* GPU-based training
* Experiment tracking

## License

This project is intended for educational and portfolio purposes.

Please check the applicable dataset terms before redistributing dataset files.

## Author

**Praveen Kumar**

Data Science | Machine Learning | Deep Learning

GitHub:
https://github.com/InfinitePraveen

LinkedIn:
https://www.linkedin.com/in/infinitepraveen/
