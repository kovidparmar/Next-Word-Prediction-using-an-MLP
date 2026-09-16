# Machine Learning Assignment 3

This project combines three core machine learning tasks into a single assignment: image classification, regularization analysis, and next-word prediction. The main focus of this repository is the next-word prediction task using a multi-layer perceptron (MLP), while the other notebooks explore related concepts in neural networks and model generalization.

## Project goal

The overall goal of this work is to explore how different machine learning models behave under different data settings and learning objectives. The assignment demonstrates:

- how simple neural networks can learn from structured data
- how regularization affects model robustness and generalization
- how feed-forward networks can model sequential language data
- how visual and textual representations can be learned and evaluated

## Main focus: Next Word Prediction using an MLP

The next-word prediction component uses a feed-forward neural network to model language as a sequence prediction problem. Given a short context of previous words, the model predicts the most likely next word from a vocabulary.

### Model idea

The model uses:

- an embedding layer for word representations
- a fixed-size context window of previous tokens
- fully connected layers to combine the context information
- a final output layer that predicts probabilities across the vocabulary

This is a simple neural language model that captures local word relationships without using recurrent or transformer architectures.

### Why this is useful

This task demonstrates how machine learning can model language patterns by learning from context. It is a foundational example of natural language processing, and shows how neural models can convert text into vector representations and use them for prediction.

## Other tasks in the project

### 1. MNIST and CNN experiments

The notebook `MNIST_and_CNN_Experiments.ipynb` focuses on image classification using the MNIST and Fashion-MNIST datasets. It compares several approaches, including:

- logistic regression or simple baselines
- multilayer perceptrons
- feature visualization using embedding plots
- convolutional neural networks (CNNs)

This part of the assignment demonstrates how deeper models improve performance on image data and how learned features can be interpreted.

### 2. Moons dataset and regularization

The notebook `Moons_Dataset_and_Regularization.ipynb` studies how regularization affects model generalization on the synthetic two-moons dataset. It compares models trained with different regularization settings, such as:

- L1 regularization
- L2 regularization
- different levels of complexity and bias-variance tradeoff

This task highlights the importance of controlling overfitting and finding a suitable model complexity for the dataset.

## Project files

- `nn_nxt_word.ipynb`  
  Main notebook for training and evaluating the next-word prediction model.

- `cocoapp.py`  
  Streamlit web app for generating text from a prompt using a pretrained MLP model.

- `MNIST_and_CNN_Experiments.ipynb`  
  Notebook for image classification experiments and CNN analysis.

- `Moons_Dataset_and_Regularization.ipynb`  
  Notebook for the regularization study on the moons dataset.

- `vocab.json`  
  English vocabulary used by the text-generation app.

- `coding_vocab.json`  
  Code-oriented vocabulary used for code-style generation.

## Running the app

To run the language model app locally:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install torch streamlit numpy pandas matplotlib scikit-learn jupyter
streamlit run cocoapp.py
```

The app allows users to:

- choose English or code text generation
- select the embedding dimension and context length
- adjust temperature for more or less randomness
- generate text continuations from a custom prompt

## Technical summary

This assignment illustrates several major machine learning ideas:

- representation learning
- regularization and generalization
- image classification with CNNs
- sequence modeling and language generation
- neural network training and model evaluation

## Conclusion

Overall, this project shows how machine learning models can be applied to different kinds of data: images, synthetic classification problems, and text. The next-word prediction MLP is the central demonstration of how neural networks can learn contextual patterns in language, while the other tasks provide complementary insights into generalization, architecture design, and model behavior.
