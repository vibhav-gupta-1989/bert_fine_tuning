# IMDb Sentiment Classification with BERT

Fine-tune **BERT (`bert-base-uncased`)** for binary sentiment classification on the IMDb movie-review dataset using **Hugging Face Transformers** and **PyTorch**.

The project demonstrates a typical NLP fine-tuning workflow: load a pretrained tokenizer, prepare a dataset, tokenize the text, configure training, and fine-tune a pretrained sequence-classification model.

## Project Overview

The notebook, `bert_fine_tune_imdb.ipynb`, implements these steps:

1. Load the uncased BERT tokenizer.
2. Load the IMDb dataset from Hugging Face Datasets.
3. Split the original training set into training and validation subsets (80/20, with a fixed seed).
4. Tokenize review text, truncating sequences to a maximum length of 200 tokens.
5. Load `bert-base-uncased` with a classification head for two labels.
6. Fine-tune the model with the Hugging Face `Trainer` API.
7. Calculate classification accuracy on the validation set and save training checkpoints according to the configured strategy.

## Dataset

The notebook loads the dataset using:

```python
from datasets import load_dataset

imdb_dataset = load_dataset("stanfordnlp/imdb")
```

The dataset contains movie reviews and binary sentiment labels. The notebook uses the dataset's `train` split to create training and validation subsets, while retaining the provided `test` split separately.

- **Training set:** 80% of the original training split
- **Validation set:** 20% of the original training split
- **Test set:** the dataset's predefined test split

The split uses `seed=42` for reproducibility.

## Model and Tokenization

- **Pretrained model:** `bert-base-uncased`
- **Task:** binary sequence classification
- **Number of labels:** 2
- **Tokenizer:** `AutoTokenizer` for `bert-base-uncased`
- **Maximum tokenized sequence length:** 200
- **Truncation:** enabled

BERT starts from pretrained language-model weights. A sequence-classification head is added and fine-tuned on the labeled IMDb reviews.

## Training Configuration

The notebook configures training with the following settings:

| Setting | Value |
|---|---|
| Epochs | 2 |
| Training batch size per device | 128 |
| Evaluation batch size per device | 128 |
| Learning rate | `2e-5` |
| Precision setting | `bf16=True` |
| Evaluation strategy | Every epoch |
| Logging strategy | Every epoch |
| Checkpoint saving strategy | Every epoch |
| Best-model selection metric | Accuracy |
| Load best model at end | Enabled |
| Random seed for PyTorch | 42 |
| Reporting integrations | Disabled (`report_to="none"`) |

The notebook places the model on CUDA (`.to("cuda")`). A compatible NVIDIA GPU and a PyTorch build with CUDA support are therefore expected. The `bf16=True` setting also requires hardware and software support for bfloat16 training.

## Results

The notebook's recorded training output shows the following validation results:

| Epoch | Training loss | Validation loss | Validation accuracy |
|---:|---:|---:|---:|
| 1 | 0.352519 | 0.273921 | 88.72% |
| 2 | 0.220761 | 0.246845 | **90.06%** |

The best recorded validation accuracy was **90.06% after epoch 2**, with a validation loss of `0.246845`. These are results on the validation split created from the original training data, not on the separate IMDb test set.

## Evaluation

The notebook defines an accuracy metric by taking the class with the highest predicted score and comparing it with the ground-truth label:

```python
def compute_accuracy(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}
```

The `Trainer` uses the validation set to report accuracy and select the best checkpoint according to that metric.

**Important:** The notebook tokenizes the test set, but the current final cell does not run evaluation on it. To measure final test accuracy after training, add:

```python
test_results = trainer.evaluate(tok_imdb_test_set)
print(test_results)
```

This evaluates the trained model on the test dataset. Keep the test set for final evaluation rather than using it to make training decisions.

## Requirements

Install the main dependencies in the Python environment used for the notebook:

```bash
pip install torch transformers datasets accelerate
```

For GPU training, install a PyTorch build compatible with your installed NVIDIA driver and CUDA environment. Check the official PyTorch installation instructions if you need a specific CUDA build.

## How to Run

1. Clone or download this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install the dependencies listed above.
4. Open `bert_fine_tune_imdb.ipynb` in Jupyter Notebook or VS Code.
5. Run the notebook cells in order.

The first run may download the tokenizer, pretrained BERT weights, and IMDb dataset, so an internet connection is required. Training time depends on your GPU, available memory, and software configuration.

## Outputs

- Fine-tuned model checkpoints are written under `my_imdb_model/` according to the configured save strategy.
- Training logs and validation metrics are produced by the Hugging Face `Trainer`.
- The notebook creates tokenized training, validation, and test datasets in memory.

The notebook does not currently include a final test-set evaluation cell or an explicit export step for a standalone final model directory; add those steps if you want to report test performance or reload the final model independently.

## Key Concepts Practiced

- Transfer learning with a pretrained Transformer
- BERT tokenization and sequence truncation
- Preparing training, validation, and test splits
- Fine-tuning for binary text classification
- Training with Hugging Face `Trainer`
- Evaluating a classifier using accuracy
- Saving checkpoints and selecting the best validation checkpoint

## Notes and Troubleshooting

- **CUDA errors:** The notebook explicitly uses `"cuda"`. If you do not have a working CUDA-enabled PyTorch installation and a compatible NVIDIA GPU, the model-loading cell will fail. Update the device configuration if you intend to run on CPU.
- **Bfloat16 errors:** If your GPU does not support bfloat16 training, set `bf16=False`. Use an appropriate precision option supported by your hardware.
- **GPU memory errors:** A batch size of 128 can require substantial GPU memory for BERT. If you encounter out-of-memory errors, reduce `per_device_train_batch_size` and `per_device_eval_batch_size`.
- **TrainingArguments compatibility:** If `TrainingArguments` raises an import or dependency error, check that `torch`, `transformers`, `accelerate`, and their versions are compatible in the active notebook environment.
- **Dataset download errors:** Confirm that the environment can access Hugging Face Hub and that the required packages are installed.

## Repository Structure

```text
.
└── bert_fine_tune_imdb.ipynb
```
