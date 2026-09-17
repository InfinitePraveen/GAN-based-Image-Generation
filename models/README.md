# Models

The training notebook creates the trained model files in this directory:

```text
generator.pth
discriminator.pth
```

These checkpoint files are generated after training and are ignored by Git because model binaries can become large.

The Flask application expects:

```text
models/generator.pth
```

Run `notebooks/01_GAN_Training.ipynb` first to create it.
