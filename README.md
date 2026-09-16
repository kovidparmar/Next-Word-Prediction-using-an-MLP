# Next Word Prediction using an MLP

This repository contains a small machine learning project portfolio for Assignment 3. It combines multiple experiments in image classification, regularization, and text generation, with the main work stored in Jupyter notebooks and a Streamlit demo app.

## Contents

- `MNIST_and_CNN_Experiments.ipynb`  
  Explores training and evaluation on MNIST/Fashion-MNIST, compares MLP and CNN models, and visualizes learned representations.

- `Moons_Dataset_and_Regularization.ipynb`  
  Investigates classification on the two-moons dataset and compares regularization effects such as L1 and L2.

- `nn_nxt_word.ipynb`  
  Builds and tests a neural next-word prediction model using context windows and embeddings.

- `cocoapp.py`  
  A Streamlit app that loads a trained next-word generation model and generates text from a user prompt.

- `vocab.json` and `coding_vocab.json`  
  Vocabulary files used by the text generator for English and code-style text.

## Setup

Use a Python environment and install the required dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install torch torchvision torchaudio streamlit scikit-learn matplotlib numpy pandas jupyter
```

Core libraries used by the project include:

- PyTorch
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Jupyter

## Run the notebooks

To open and run the notebooks:

```bash
jupyter notebook
```

Then open one of the notebooks in the project folder, such as:

- `MNIST_and_CNN_Experiments.ipynb`
- `Moons_Dataset_and_Regularization.ipynb`
- `nn_nxt_word.ipynb`

## Run the text generator app

From the project root, start the app with:

```bash
streamlit run cocoapp.py
```

The app allows you to:

- select English or code vocabulary
- choose embedding size and context length
- adjust the generation temperature
- provide a prompt and generate continuation text

## Project focus

This assignment covers several core machine learning topics:

- representation learning and feature visualization
- regularization and overfitting control
- CNNs for image classification
- language modeling and autoregressive next-word prediction

## Notes

- The text generation app depends on trained model checkpoint files being available in the project directory.
- The notebooks are the main place to inspect the experiments, model setup, and result interpretation.
- This repository is designed as a teaching/assignment project and is best used for study and reproducible experimentation.
