# Neural Network for Rating Search Queries

## Project goal

This project builds a machine-learning model that assigns a quality rating
to a user search query.

The rating is a decimal number between 0 and 1:

- A rating close to 0 represents a vague, unclear, or meaningless query.
- A rating close to 1 represents a clear, specific, and meaningful query.

For example:

| Query | Possible rating |
|---|---:|
| `asdfgh` | 0.02 |
| `help` | 0.18 |
| `python tutorial` | 0.65 |
| `how to learn Python as a beginner step by step` | 0.94 |

The project uses two CSV files:

- `train.csv` contains queries and known ratings.
- `test.csv` contains queries whose ratings must be predicted.

The notebook compares two neural-network architectures:

1. An embedding model with average pooling
2. An embedding model with a bidirectional gated recurrent unit (GRU).

The models are evaluated using Mean Absolute Error, abbreviated as MAE.
A smaller MAE represents more accurate predictions.

## Important data-handling rule

The notebook does not add columns to or remove columns from the original
training and testing DataFrames.

All cleaned text, query lengths, model inputs, and predictions are stored
in separate Python variables.