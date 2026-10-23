# Search Query Quality Rating with Neural Networks

## Overview

This project builds and evaluates a neural-network regression model that predicts the quality of a user's search query.

Each query receives a numerical **rating between 0 and 1**:

- A rating close to **0** represents a vague, unclear, incomplete, or meaningless query.
- A rating close to **1** represents a clear, specific, and meaningful query.

For example:

| Query | Example Interpretation |
|---|---|
| `asdfgh` | Little recognizable meaning |
| `help` | Understandable, but highly ambiguous |
| `python tutorial` | Clear topic, but limited context |
| `how to learn Python as a beginner step by step` | Clear topic, audience, and intent |

The project demonstrates an end-to-end text regression workflow that includes:

- Data loading and validation
- Text preprocessing
- Exploratory data analysis
- Training and validation splitting
- Baseline modeling
- Neural-network modeling
- Overfitting prevention
- Mean Absolute Error evaluation
- Model comparison
- Residual analysis
- Test-set prediction
- Submission generation
- Model persistence

---

## Problem Statement

A provider wants to rate user queries according to their relevance and meaningfulness.

The goal is to assign scores that can help prioritize responses so users receive relevant and useful information or services in a timely manner.

Given a dataset containing search queries, this project builds a neural network that predicts a **query-quality rating from 0 to 1**.

Because the predicted value is numerical and continuous, this is treated as a **regression problem**.

---

## Dataset

The project uses two primary datasets:

### Training Dataset

`train.csv` is used to train and evaluate the model.

It has exactly two columns:

| Field | Type | Description |
|---|---|---|
| `query` | String | User search query |
| `rating` | Float | Query-quality rating between 0 and 1 |

Example:

```csv
query,rating
how to learn Python as a beginner step by step,0.94
python tutorial,0.67
help,0.18
asdfgh,0.02
```

### Testing Dataset

`test.csv` contains the queries for which ratings must be predicted.

It has exactly one column:

| Field | Type | Description |
|---|---|---|
| `query` | String | User search query |

Example:

```csv
query
best laptop for computer science students
how to calculate compound interest
weather tomorrow
```

---

## Important Note About the Mock Data

The datasets included with this project are **synthetic mock datasets** created for development, testing, and demonstration.

The current mock datasets contain:

- **5,000 training observations**
- **1,000 testing observations**

The synthetic training ratings were generated using programmed mock-quality rules and controlled random variation.

They are **not human-validated measurements of real search-query quality**.

Therefore:

- The datasets are appropriate for testing the machine-learning pipeline.
- They are useful for demonstrating preprocessing, training, validation, and prediction.
- Validation MAE measures how well the model learns the synthetic rating patterns.
- The results should not be interpreted as proof that the model can accurately judge real-world user queries.

A production implementation would require carefully designed human-labeled data and additional evaluation.

---

## Repository Structure

The recommended repository structure is:

```text
query-rating-project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── outputs/
│   ├── submission.csv
│   ├── best_query_rating_model.keras
│   └── model_experiment_results.csv
│
├── query_rating_model.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The `outputs/` directory is created automatically by the notebook if it does not already exist.

---

# How the Project Works

## 1. Import Libraries and Configure the Project

The first notebook module imports all required libraries and configures:

- File locations
- Random seeds
- Validation size
- Batch size
- Maximum training epochs
- Vocabulary size
- Embedding dimensions
- Output locations

A fixed random seed is used to improve reproducibility:

```python
SEED = 42
```

The seed is applied to:

- Python's `random` module
- NumPy
- TensorFlow

---

## 2. Load and Validate the Data

The notebook loads the datasets using Pandas:

```python
data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
```

The training DataFrame is called:

```python
data
```

The testing DataFrame is called:

```python
test_data
```

The notebook verifies that `train.csv` contains exactly:

```text
query
rating
```

and `test.csv` contains exactly:

```text
query
```

If the expected files or columns are missing, the notebook stops with an informative error.

---

# Data Integrity

An important goal of the notebook is to keep the original training and testing DataFrames structurally unchanged.

The notebook does **not** add preprocessing or modeling columns directly to `data` or `test_data`.

For example, it avoids operations such as:

```python
data["query_clean"] = ...
data["word_count"] = ...
test_data["rating"] = ...
```

Instead, preprocessing information is stored in separate variables such as:

```python
all_train_queries
all_train_ratings
model_queries
model_ratings
test_queries
word_counts
character_counts
X_train
X_valid
y_train
y_valid
test_predictions
```

Near the end of the notebook, assertions verify that the original DataFrames still have their original shapes and columns.

---

# Query Preprocessing

The notebook uses intentionally conservative preprocessing.

A query-cleaning function:

1. Detects missing values.
2. Converts values to strings.
3. Replaces repeated whitespace with a single space.
4. Removes unnecessary whitespace from the beginning and end.

For example:

```text
"   learn    Python   "
```

becomes:

```text
"learn Python"
```

The cleaning process does not aggressively remove meaningful information from the query.

The cleaned query text is stored separately from the original DataFrames.

---

# Exploratory Data Analysis

Before training the model, the notebook examines the data.

The analysis includes:

- Rating distribution
- Query word counts
- Query character counts
- Relationship between query length and rating

Three visualizations are produced:

1. Rating histogram
2. Query-length histogram
3. Rating versus query-length scatterplot

This helps identify unusual patterns before model training begins.

---

# Neural-Network Sequence Length

Neural networks require sequences to have consistent dimensions.

Search queries naturally have different lengths, so the project calculates the **95th percentile of query word counts**.

That value is then constrained to practical minimum and maximum values.

Conceptually:

```python
SEQUENCE_LENGTH = int(
    np.clip(
        length_95,
        10,
        100
    )
)
```

This allows the model to retain most query content without allowing a few unusually long queries to create excessive padding.

---

# Training and Validation Split

The labeled training information is divided into:

- **80% training data**
- **20% validation data**

The training group is used to learn the relationship between search-query text and ratings.

The validation group is kept out of the model's weight updates and is used to estimate performance on unseen labeled examples.

The split uses:

```python
train_test_split(
    model_queries,
    model_ratings,
    test_size=0.20,
    random_state=42
)
```

The final `test.csv` queries are not used for model selection because their ratings are unknown.

---

# Evaluation Metric

The project uses **Mean Absolute Error**, abbreviated as **MAE**.

MAE measures the average absolute difference between:

- the true rating, and
- the predicted rating.

Conceptually:

```text
MAE = average(|actual - predicted|)
```

A lower MAE represents better predictions.

For example:

```text
Actual rating:    0.90
Predicted rating: 0.80

Absolute error:   0.10
```

An MAE of:

```text
0
```

would represent perfect predictions.

---

# Baseline Models

Before training neural networks, the notebook creates two reference models.

These baselines help answer an important question:

> Does the neural network perform better than a simpler approach?

---

## Mean Baseline

The simplest baseline predicts the average training rating for every validation query.

For example, if the average training rating were:

```text
0.65
```

the baseline would predict:

```text
0.65
```

for every query.

This baseline completely ignores the text.

A useful model should generally outperform this simple strategy.

---

## TF-IDF + Ridge Regression Baseline

The second baseline is more sophisticated.

It combines:

- Character-level TF-IDF
- Ridge regression

TF-IDF converts pieces of text into numerical features.

The notebook uses character n-grams, which can be especially useful for short search queries because they can capture:

- Parts of words
- Misspellings
- Word fragments
- Formatting patterns
- Similar query structures

Ridge regression then learns a regularized relationship between those numerical features and the query ratings.

This model provides a useful non-neural reference.

The final required model, however, is still a neural network.

---

# Text Vectorization

Neural networks cannot directly process ordinary written words.

The project therefore uses a Keras `TextVectorization` layer.

For example, a query such as:

```text
learn python programming
```

might be converted conceptually into:

```text
[52, 18, 91, 0, 0, 0, ...]
```

The exact token numbers depend on the learned vocabulary.

The vectorizer is initially adapted using only the training partition:

```python
vectorizer.adapt(
    tf.data.Dataset
    .from_tensor_slices(X_train)
    .batch(256)
)
```

This keeps validation information out of the preprocessing-learning process during model evaluation.

---

# Neural-Network Models

Two neural-network architectures are trained and compared.

## Model 1: AveragePoolingNN

The first architecture follows this general structure:

```text
Raw query
    ↓
Text vectorization
    ↓
Embedding layer
    ↓
Global average pooling
    ↓
Dense layer
    ↓
Dropout
    ↓
Dense layer
    ↓
Dropout
    ↓
Sigmoid output
    ↓
Predicted rating
```

### Embedding layer

The embedding layer learns a numerical vector representation for each token.

Instead of treating words simply as IDs, the neural network learns representations that are useful for predicting query ratings.

### Global average pooling

The token embeddings are averaged into a single representation of the query.

This creates a relatively simple and computationally efficient model.

---

## Model 2: BidirectionalGRUNN

The second architecture follows this general structure:

```text
Raw query
    ↓
Text vectorization
    ↓
Embedding layer
    ↓
Bidirectional GRU
    ↓
Dense layer
    ↓
Dropout
    ↓
Sigmoid output
    ↓
Predicted rating
```

The key difference is the **Bidirectional GRU**.

GRU stands for **Gated Recurrent Unit**.

Unlike average pooling, a GRU can consider the order of tokens in the query.

The bidirectional version processes the sequence in both directions.

This can help the network distinguish between queries that contain similar words arranged in different ways.

---

# Activation Functions

The hidden dense layers use:

```python
activation="relu"
```

ReLU introduces nonlinearity, allowing the neural network to learn more complex relationships.

The final output layer uses:

```python
activation="sigmoid"
```

The sigmoid function naturally restricts the output to approximately:

```text
0 ≤ rating ≤ 1
```

which matches the target-rating range.

---

# Optimizer

The models use the **Adam optimizer**.

Conceptually, the optimizer controls how the neural network changes its internal weights as it learns from prediction errors.

The learning rate begins at:

```python
0.001
```

The notebook can automatically reduce the learning rate when validation performance stops improving.

---

# Loss Function

The neural networks are compiled using **Mean Absolute Error** as the loss function.

```python
loss=keras.losses.MeanAbsoluteError()
```

This aligns the training objective with the project's required evaluation metric.

---

# Preventing Overfitting

Overfitting occurs when a model learns the training information too specifically and fails to generalize well to unseen queries.

The notebook uses multiple techniques to reduce this problem.

---

## Validation Monitoring

Training MAE and validation MAE are tracked separately.

If training error continues decreasing while validation error begins increasing, the model may be overfitting.

The notebook plots both values after training.

---

## Dropout

Dropout temporarily disables random neural-network units during training.

For example:

```python
layers.Dropout(0.30)
```

This helps prevent the network from depending too heavily on individual hidden units.

---

## L2 Regularization

Dense layers use L2 regularization:

```python
kernel_regularizer=keras.regularizers.l2(0.0001)
```

This discourages the model from creating unnecessarily large weights.

---

## Early Stopping

Early stopping monitors:

```text
validation loss
```

If validation performance stops improving for several epochs, training ends automatically.

The model then restores the weights from its best validation epoch.

Conceptually:

```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)
```

This helps prevent unnecessary training and overfitting.

---

## Learning-Rate Reduction

The notebook also uses:

```python
ReduceLROnPlateau
```

If validation improvement slows, the learning rate is reduced.

The optimizer can then make smaller changes to the model's weights.

---

# Model Selection

Both neural networks are evaluated using validation MAE.

The results are organized into a comparison table containing:

- Model name
- Validation MAE
- Best epoch
- Number of parameters

The neural network with the **lowest validation MAE** is selected.

The selected model is then compared with:

- Mean baseline
- TF-IDF + Ridge baseline

---

# Learning Curves

The notebook plots:

```text
Training MAE
Validation MAE
```

for each neural-network architecture.

These curves help answer questions such as:

- Is the model learning?
- Has the model stopped improving?
- Is the model beginning to overfit?
- Did early stopping activate at a reasonable point?

---

# Validation Diagnostics

After selecting the best neural network, the notebook performs additional diagnostics.

## Actual Versus Predicted Plot

This plot compares:

```text
actual rating
```

against:

```text
predicted rating
```

Perfect predictions would fall along a diagonal line.

---

## Residual Analysis

A residual is calculated as:

```text
residual = actual rating - predicted rating
```

A positive residual means the model predicted too low.

A negative residual means the model predicted too high.

The notebook creates:

- A residual histogram
- A residual-versus-prediction scatterplot

Ideally, residuals should generally be centered around zero without a strong visible pattern.

---

# Final Model Training

After selecting the best architecture, the model is rebuilt and trained again using all usable labeled training observations.

The text vectorizer is also rebuilt using all usable training queries.

The notebook reuses the best epoch count identified during validation rather than automatically training for the maximum number of epochs.

---

# Test Predictions

The final model generates one rating for every query in `test.csv`.

Conceptually:

```python
test_predictions = final_model.predict(test_queries)
```

Predictions are restricted to the required interval:

```python
test_predictions = np.clip(
    test_predictions,
    0.0,
    1.0
)
```

---

# Submission File

The notebook creates a **new DataFrame** for the submission.

It does not add a rating column to the original `test_data` DataFrame.

The output contains exactly:

```text
query
rating
```

Example:

```csv
query,rating
best laptop for computer science students,0.8421
how to calculate compound interest,0.9014
weather tomorrow,0.6172
```

The query values and row order are preserved from `test.csv`.

---

# Generated Files

After the notebook completes successfully, the `outputs/` directory contains:

```text
outputs/
├── submission.csv
├── best_query_rating_model.keras
└── model_experiment_results.csv
```

## `submission.csv`

Contains:

```text
query
rating
```

for every test observation.

## `best_query_rating_model.keras`

Contains the final trained neural network.

The text-vectorization layer is included within the saved model.

## `model_experiment_results.csv`

Contains information about the neural-network experiments, including:

- Validation MAE
- Best epoch
- Parameter count

---

# Data-Integrity Verification

Before saving the final outputs, the notebook verifies that the original DataFrames still have their original structure.

The notebook checks:

```python
assert tuple(data.columns) == original_train_columns
assert tuple(test_data.columns) == original_test_columns

assert data.shape == original_train_shape
assert test_data.shape == original_test_shape
```

This confirms that the notebook did not add or remove rows or columns from the loaded source DataFrames.

---

# Installation

A Python virtual environment is recommended.

## Create a Virtual Environment

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

# Install Dependencies

Install packages using:

```bash
pip install -r requirements.txt
```

A basic `requirements.txt` can contain:

```text
jupyter
matplotlib
numpy
pandas
scikit-learn
tensorflow
```

Exact versions may be pinned after the project is tested in its intended environment.

---

# Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
query_rating_model.ipynb
```

---

# Running the Notebook

The notebook should be executed from **top to bottom**.

The recommended workflow is:

1. Open `query_rating_model.ipynb`.
2. Restart the Jupyter kernel.
3. Run all cells from the beginning.
4. Allow both neural networks to finish training.
5. Review the validation MAE values.
6. Review the training curves.
7. Review the residual diagnostics.
8. Allow the selected model to refit.
9. Confirm that prediction files were created in `outputs/`.

---

# Understanding Jupyter Variables

Jupyter notebooks maintain variables in the active kernel's memory.

For example:

```python
data = pd.read_csv(TRAIN_PATH)
```

must execute before another cell can use:

```python
data
```

Similarly:

```python
import re
```

must execute before Python can use:

```python
re.sub(...)
```

If an error such as:

```text
NameError: name 'data' is not defined
```

or:

```text
NameError: name 're' is not defined
```

appears, restart the kernel and execute the notebook from the beginning.

This ensures that all imports and variables are created in the expected order.

---

# Loading the Saved Model

The final model can later be loaded without retraining:

```python
from tensorflow import keras

loaded_model = keras.models.load_model(
    "outputs/best_query_rating_model.keras"
)
```

Example new queries:

```python
import numpy as np

new_queries = np.array([
    "how to learn Python as a beginner",
    "help",
    "asdfgh"
])
```

Generate predictions:

```python
predictions = loaded_model.predict(
    new_queries
).reshape(-1)
```

Display them:

```python
for query, prediction in zip(
    new_queries,
    predictions
):
    print(
        query,
        round(float(prediction), 4)
    )
```

---

# Reproducibility

The project uses:

```python
SEED = 42
```

for:

- Python random operations
- NumPy random operations
- TensorFlow random operations

This improves reproducibility.

However, exact results may still vary somewhat across environments because of differences in:

- Python version
- TensorFlow version
- Operating system
- CPU architecture
- GPU hardware
- Numerical libraries
- Nondeterministic numerical operations

Therefore, small differences in final MAE values are possible.

---

# Suggested `.gitignore`

A suitable `.gitignore` for the project is:

```gitignore
# Python cache
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/
env/

# Jupyter temporary files
.ipynb_checkpoints/

# Environment variables and secrets
.env
.env.*

# Operating-system files
.DS_Store
Thumbs.db

# Generated model files
outputs/*.keras
outputs/*.h5

# Generated prediction and experiment files
outputs/submission.csv
outputs/model_experiment_results.csv
```

If generated outputs are required as part of a class submission, remove the corresponding entries from `.gitignore`.

---

# Limitations

## Synthetic Ratings

The current labels are synthetic.

They were not produced by human reviewers.

Therefore, the model primarily learns the patterns built into the mock-data generation process.

---

## Query Quality Is Context Dependent

Query quality is not always determined solely by the text.

For example:

```text
weather
```

may appear vague without additional information.

However, it may be completely sufficient if a system already knows:

- the user's location,
- the user's language,
- the current conversation, or
- the requested service.

The current model receives only query text.

---

## Missing User Context

The model does not use:

- User history
- Geographic location
- Session information
- Previous searches
- Language preferences
- Device information
- Conversation context
- User intent history

A production system might need some of these signals.

---

## Language Coverage

If most training observations use English, predictions may be less reliable for queries written in other languages.

Multilingual performance should be evaluated independently before using such a system broadly.

---

## Synthetic Data Distribution

The synthetic dataset may differ significantly from real search-query traffic.

Real-world query distributions can contain patterns not represented by mock data.

Therefore, strong validation results on the synthetic dataset do not guarantee equally strong performance on real-world information.

---

# Potential Improvements

Future work could explore:

- Human-reviewed query ratings
- Multiple human reviewers per query
- Reviewer-agreement analysis
- Larger and more diverse datasets
- Character-level neural networks
- Pretrained text embeddings
- Transformer-based language models
- Cross-validation
- Automated hyperparameter tuning
- More sophisticated query preprocessing
- Multilingual evaluation
- Query-category-specific evaluation
- Prediction uncertainty estimates
- Fairness testing
- Calibration analysis
- Robustness testing against spelling errors
- Robustness testing against extremely short queries
- Robustness testing against unusually long queries

---

# Ethical Considerations

A query-quality model should be used carefully.

A low predicted score does not necessarily mean the user's request is unimportant.

Short or grammatically unusual queries may come from:

- Users typing quickly
- Users on mobile devices
- Non-native speakers
- Users with limited literacy
- Users with disabilities
- Users relying on accessibility tools
- Users in urgent situations
- Users whose intent is already known from previous interactions

A production system should therefore avoid automatically denying service or access solely because a query receives a low score.

Additional safeguards may include:

- Human review
- Fallback handling
- Context-aware interpretation
- Multilingual testing
- Accessibility evaluation
- Fairness analysis
- User correction mechanisms

---

# Privacy Considerations

The datasets included with this project are synthetic and are intended for educational and testing purposes.

If real user queries are used in the future, additional safeguards may be required because search queries can contain sensitive personal information.

Examples can include:

- Names
- Addresses
- Financial information
- Health-related information
- Account information
- Personal concerns

Real-world datasets should be reviewed for appropriate privacy, security, retention, and access-control requirements before being used for machine learning.

---

# Dataset Distribution and GitHub

The mock datasets in this repository are synthetic.

This makes them useful for demonstrating the project without requiring access to a private dataset.

If these files are replaced with externally provided datasets, review the source's:

- License
- Privacy requirements
- Redistribution rules
- Usage restrictions

before committing those datasets to a public repository.

Do not commit passwords, API keys, access tokens, credentials, private user data, or other secrets to the repository.

---

# Interpretation of Results

A low validation MAE indicates that the model predicts the synthetic validation ratings relatively closely.

It does **not** necessarily demonstrate that the model understands query quality in the same way a human reviewer would.

Results should therefore be interpreted as evidence about the performance of the modeling pipeline against the synthetic dataset, not as proof of real-world query-quality understanding.

---

# Project Summary

This project demonstrates a complete neural-network text-regression workflow.

The main stages are:

1. Load the training and testing datasets.
2. Validate their schemas.
3. Preserve the original dataset structure.
4. Clean query text using separate arrays.
5. Explore rating and query-length distributions.
6. Create training and validation groups.
7. Establish simple prediction baselines.
8. Convert text into numerical token sequences.
9. Build multiple neural-network architectures.
10. Train the models using MAE.
11. Reduce overfitting with dropout, regularization, and early stopping.
12. Compare models using validation MAE.
13. Examine training curves and prediction residuals.
14. Select the best neural-network architecture.
15. Refit the selected architecture using all labeled observations.
16. Predict ratings for `test.csv`.
17. Create the required submission file.
18. Save the final model and experiment results.
19. Verify that the original source DataFrames remain structurally unchanged.

The project is designed to be understandable by both technical and nontechnical readers while demonstrating sound machine-learning practices.

---

## Author

Developed as a machine-learning project for predicting the quality and relevance of user search queries.

## Project Status

Educational and experimental.

The current version uses synthetic data and should not be treated as a production query-ranking system.