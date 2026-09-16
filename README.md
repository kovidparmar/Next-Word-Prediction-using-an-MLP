# Next-Word Prediction Using an MLP

This repository is primarily a next-word prediction project. It trains a feed-forward neural language model on Project Gutenberg text and provides a Streamlit interface for generating continuations from a prompt. The model uses learned word embeddings, a fixed context window, and fully connected layers to predict the next token without an RNN, LSTM, or Transformer.

Two smaller notebooks provide supporting machine learning studies: image classification on MNIST/Fashion-MNIST and regularization on a synthetic two-moons dataset. They are included as comparative experiments; the language model is the main subject of this repository.

## What the project demonstrates

- Tokenization, vocabulary construction, and integer encoding for language modeling.
- Representation learning with an embedding layer.
- Supervised next-token prediction with cross-entropy loss.
- Temperature-controlled text sampling and prompt continuation.
- The effect of context size and embedding size on a feed-forward language model.
- Practical limitations of local-context text generation and checkpoint compatibility.

## Repository contents

| File | Description |
| --- | --- |
| `nn_nxt_word.ipynb` | Trains and evaluates the MLP language model on Project Gutenberg text. |
| `cocoapp.py` | Streamlit interface for loading checkpoints and generating text. |
| `MNIST_and_CNN_Experiments.ipynb` | MNIST/Fashion-MNIST baselines, CNN comparisons, visualizations, and timing. |
| `Moons_Dataset_and_Regularization.ipynb` | Two-moons classification and regularization experiments. |
| `vocab.json` | English vocabulary plus `stoi` and `itos` mappings. |
| `coding_vocab.json` | Code-oriented vocabulary plus `stoi` and `itos` mappings. |
| `README.md` | Project documentation and interpretation of the recorded results. |

## Main project: next-word prediction

### Problem formulation

The language model receives five consecutive tokens and learns to predict the sixth:

```text
context:  w1 w2 w3 w4 w5
target:                  w6
```

Every token is represented by an integer ID. The embedding layer maps each ID to a dense vector, and the five vectors are flattened and passed through fully connected layers. The final layer produces one logit for every vocabulary item. Applying softmax converts those logits into a probability distribution for the next token.

This is a fixed-window model. It can learn local relationships such as common phrases and syntactic patterns, but it has no explicit memory beyond the selected context length.

### Dataset and preprocessing

The recorded notebook run uses the Project Gutenberg text for *The Adventures of Sherlock Holmes* (`1661-0.txt`). The preprocessing pipeline:

1. Converts the corpus to lowercase.
2. Keeps lowercase letters, digits, spaces, and periods.
3. Splits the cleaned text on whitespace.
4. Builds a vocabulary and `stoi`/`itos` mappings.
5. Creates sliding-window examples with five input tokens and one target token.
6. Uses the first 90 percent of the sequence for training and the final 10 percent for validation.

Recorded dataset statistics:

| Statistic | Value |
| --- | ---: |
| Total tokens | 107,510 |
| Vocabulary size | 10,489 |
| Context length | 5 tokens |
| Training split | First 90 percent |
| Validation split | Final 10 percent |

The sequential split is simple and preserves the text order. It is useful for this demonstration, although a random split would make the validation set less representative of future, later text.

### MLP architecture

The main recorded model is:

```text
Input: five token IDs
Embedding(vocabulary_size, 64)
Flatten: 5 x 64 = 320 features
Linear(320, 1024) + ReLU
Linear(1024, 1024) + ReLU
Linear(1024, vocabulary_size)
Output: vocabulary-sized logits
```

The embedding layer learns a vector representation for each token. Flattening combines the five positions into one feature vector. The two hidden layers then learn nonlinear combinations of the context, and the output layer scores every possible next word.

This architecture is intentionally straightforward. It is easier to inspect than a recurrent or attention-based model, but it does not model long-range dependencies efficiently and has a large output layer because the vocabulary contains 10,489 tokens.

### Training configuration

The recorded run used:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Cross-entropy |
| Epochs | 1,000 |
| Nominal batch size | 64 |
| Activation | ReLU |
| Recorded device | CUDA |

Important implementation detail: the training loop samples only the first 64 shuffled examples in each epoch rather than iterating through every mini-batch in the training set. Consequently, an epoch in the notebook is not a complete pass over all training examples. This affects how the loss curve and the 1,000-epoch result should be interpreted.

### Recorded language-model results

At epoch 1,000, the notebook recorded:

| Metric | Loss |
| --- | ---: |
| Training loss | 5.623 |
| Validation loss | 6.894 |

The validation loss is higher than the training loss, which indicates a generalization gap and likely overfitting. The generated samples were sometimes syntactically plausible, but often incoherent. That behavior is expected from a small fixed-context model trained on a limited corpus with no mechanism for long-range context.

Cross-entropy is the main quantitative result here. It should not be compared directly with the image-classification accuracy values below because the tasks, labels, and metrics are different. For text generation, qualitative samples are useful, but a stronger evaluation would also report top-1 and top-k next-token accuracy, perplexity, and results on a held-out corpus.

### Text generation

`cocoapp.py` uses temperature-scaled multinomial sampling:

```text
probabilities = softmax(logits / temperature)
next_token = sample(probabilities)
```

Lower temperatures make the output more deterministic. Higher temperatures flatten the distribution and produce more varied, but potentially less coherent, text. Unknown prompt words are ignored, and short contexts are left-padded with token ID `0`.

The app exposes these model choices:

- English or code vocabulary.
- Embedding size 32 or 64.
- Context length 5 or 10.
- ReLU or sigmoid activation.
- Temperature from 0.3 to 1.5.
- Generation length from 5 to 100 words.

The checked-in repository does not include the eight checkpoint files referenced by the app (`model111.pth` through `model888.pth`). The app therefore requires those files to be added before generation can work. A checkpoint must match the selected vocabulary, embedding size, context length, hidden size, and activation function.

There is also an important reproducibility distinction: the notebook clearly documents an English Sherlock Holmes model, while the app offers code generation and multiple architecture combinations. The training process for the code checkpoints is not included in the accessible notebook files, so those models cannot be reproduced from this repository alone.

## Supporting experiments

The following notebooks are secondary studies included in the project. They are useful for comparing model architectures and generalization behavior, but they are not required to understand or run the next-word predictor.

### MNIST and CNN experiments

`MNIST_and_CNN_Experiments.ipynb` compares classical baselines with neural networks on handwritten-digit images. The recorded MNIST setup uses a stratified 10,000-image training subset and a 10,000-image test set. Images are normalized with mean `0.1307` and standard deviation `0.3081`.

Models compared:

- Random Forest baseline.
- Logistic Regression baseline.
- MLP with architecture `784 -> 30 -> 20 -> 10` and ReLU activations.
- Simple CNN with a 32-channel convolution, max pooling, and dense layers.
- MobileNetV2.
- ResNet18.

Recorded MNIST results:

| Model | Accuracy | F1 score |
| --- | ---: | ---: |
| Random Forest | 0.9515 | 0.9514 |
| Logistic Regression | 0.8843 | 0.8840 |
| MLP | 0.9289 | 0.9288 |
| Simple CNN | 0.9734 | 0.9733 |
| MobileNetV2 | 0.9487 | 0.9487 |
| ResNet18 | 0.9510 | 0.9509 |

The simple CNN achieved the strongest recorded score. This is consistent with the inductive bias of convolution: nearby pixels are treated as locally related, and the same learned filters can detect visual patterns in different positions. The MLP treats the flattened image as a vector and therefore does not encode spatial locality as naturally. The pretrained architectures did not outperform the small task-specific CNN in this setup, which may reflect input-size adaptation, transfer-learning choices, training configuration, and the relatively small evaluation setup.

The notebook also generates confusion matrices, t-SNE representations, parameter counts, and inference timings. These views complement accuracy by showing which classes are confused, how features cluster, how large each model is, and how much computation inference requires.

Fashion-MNIST transfer result:

The recorded Fashion-MNIST evaluation produced:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.0857 |
| F1 score | 0.0534 |

This very low result demonstrates that strong performance on MNIST digits does not automatically transfer to a different image domain. Fashion-MNIST contains clothing categories with different visual structure, textures, and class boundaries. A model trained for digits must be retrained or adapted for this task; it should not be expected to generalize without domain-appropriate training.

### Two-moons classification and regularization

`Moons_Dataset_and_Regularization.ipynb` uses a synthetic two-moons dataset with 500 samples. Three noise levels are considered: `0.1`, `0.2`, and `0.3`. The features are standardized using training-set statistics.

The experiment compares:

- A one-hidden-layer MLP with 64 hidden units.
- An MLP with early stopping.
- An MLP with L1 regularization.
- An MLP with L2 regularization.
- Degree-3 polynomial Logistic Regression.

### Regularization concepts

L1 regularization adds a penalty proportional to the absolute value of the weights:

$$L_{total} = L_{data} + \lambda \sum_i |w_i|$$

It encourages sparse parameters and can effectively remove less useful connections. L2 regularization adds a squared-weight penalty:

$$L_{total} = L_{data} + \lambda \sum_i w_i^2$$

It discourages very large weights and usually produces smoother decision boundaries. Early stopping regularizes by stopping training when validation performance stops improving, limiting the amount of fitting to noise.

### Recorded hyperparameters

| Method | Best recorded setting |
| --- | --- |
| L1 regularization | `1e-6` |
| L2 regularization / weight decay | `0.001` |
| Polynomial Logistic Regression | `C=100` |

The notebook uses decision-boundary visualizations to show how model complexity and noise interact. With more noise, overly complex boundaries may fit individual samples instead of the underlying two-moons structure. Regularization provides a way to trade a small amount of training fit for improved robustness.

The class-imbalance section is partly demonstrative: several retraining operations are omitted and replaced by two dummy model runs. Those results should therefore be treated as an illustration of the workflow rather than a complete, independently validated imbalance study.

## Setup for the language model

Python 3.9 or newer is recommended. Create a virtual environment and install the main dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate

pip install torch streamlit numpy pandas matplotlib scikit-learn jupyter
```

For the main next-word notebook, start Jupyter with:

```bash
jupyter notebook
```

The exact recorded metrics can vary with package versions, random seeds, CPU/GPU availability, pretrained-weight downloads, and dataset-loading behavior. There is currently no pinned `requirements.txt`, so reproducible reruns may require recording the installed package versions and random seeds.

## Running the text-generation app

After placing the required `.pth` checkpoint files next to `cocoapp.py`, run:

```bash
streamlit run cocoapp.py
```

Then select the vocabulary and architecture that match the checkpoint. The app loads the selected vocabulary, constructs the corresponding MLP, restores the checkpoint on CPU, and generates a continuation from the supplied prompt.

The application is an inference interface, not a training interface. It does not create missing checkpoints, retrain models, or verify that a checkpoint was trained with the selected activation function. A clear production version would validate checkpoint metadata before loading and report a helpful error when a file is missing or incompatible.

## Next-word model limitations

- The language model only sees a fixed local context and cannot reliably preserve long-range narrative information.
- The recorded language-model validation loss is substantially higher than training loss.
- The training loop does not process the full training set on every epoch.
- The corpus is small and domain-specific, so generated language is not a general-purpose language model.
- The vocabulary and checkpoint files must match exactly; an index mismatch changes the meaning of every output logit.
- The code-generation model training pipeline is not included.
- Checkpoints required by the Streamlit app are not currently included.
- Results are recorded notebook outputs, not guaranteed fresh reruns from a pinned environment.
- The supporting Fashion-MNIST and two-moons results are recorded notebook outputs rather than part of the language-model evaluation.

## Possible improvements

1. Add a training script that creates every checkpoint expected by `cocoapp.py`.
2. Add a `requirements.txt` or `pyproject.toml` with pinned versions.
3. Store random seeds, dataset versions, hardware information, and training logs.
4. Iterate over all mini-batches instead of using only the first 64 examples each epoch.
5. Add validation perplexity, top-k accuracy, and automated sample evaluation.
6. Add an explicit unknown-token and padding-token design instead of relying on token ID `0` implicitly.
7. Compare the MLP with an n-gram baseline, an RNN/LSTM, and a small Transformer.
8. Add checkpoint metadata so incompatible app selections are rejected before inference.
9. Complete the class-imbalance retraining experiments with real models and per-class metrics.
10. Add tests for vocabulary round-tripping, context padding, checkpoint loading, and temperature sampling.

## Conclusion

This project is a compact demonstration of several central machine learning ideas. The language-model notebook shows how embeddings and an MLP can learn local word relationships. The MNIST experiments show the value of architectural inductive bias, with the task-specific CNN outperforming the compared baselines in the recorded run. The two-moons experiments show how regularization and early stopping can control model complexity. Together, the notebooks provide a useful comparison of representation learning, optimization, generalization, and evaluation across text, images, and synthetic data.
