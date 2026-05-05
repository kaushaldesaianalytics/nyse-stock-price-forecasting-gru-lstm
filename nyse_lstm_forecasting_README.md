# NYSE Stock Price Forecasting
## Time Series Prediction with GRU and LSTM

Can a recurrent neural network learn the sequential patterns in historical stock prices well enough to forecast future closing values? This project trains a hybrid GRU and LSTM model on Adobe Systems (ADBE) closing prices from the NYSE dataset spanning 2010 to 2016, evaluating its ability to track the actual price trajectory on a held-out test period.

---

## Overview

Traditional regression models treat each observation as independent, ignoring the sequential structure of time series data. Recurrent neural networks maintain hidden state across time steps, allowing them to learn temporal dependencies and patterns like momentum, trend, and periodicity. The GRU and LSTM architectures used here are specifically designed for long sequences, with gating mechanisms that control how much past information to retain or discard at each step.

This project demonstrates the full deep learning time series workflow: data exploration, sequence construction via sliding windows, MinMax scaling for stable training, a hybrid recurrent architecture with Dropout regularization, training with adaptive callbacks, and inverse-transform evaluation in interpretable dollar values.

---

## Dataset

The [NYSE Prices dataset](https://www.kaggle.com/datasets/dgawlik/nyse) contains 851,264 daily price records across 501 S&P 500 companies from 2010 to 2016.

| File | Description |
|---|---|
| prices.csv | Daily open, close, low, high, and volume per ticker |
| securities.csv | Company name, sector, and ticker symbol metadata |

Adobe Systems (ADBE) is selected as the forecasting target due to its strong upward trend, consistent data coverage, and absence of major gaps across the full 2010 to 2016 period.

---

## Workflow

### Part 1: Dataset Exploration
Price record structure, ticker symbol coverage, null value audit, and descriptive statistics are examined across the full 851K-record dataset. Company metadata from securities.csv is loaded and joined for ticker identification.

### Part 2: Company Selection and Visualization
Six companies are selected across sectors (Yahoo, Xerox, Adobe, Microsoft, Facebook, Goldman Sachs) and their historical opening and closing prices are plotted. Adobe's consistent upward trend and clean data make it the target for forecasting.

### Part 3: Feature Scaling and Sequence Construction
Adobe's closing price series is extracted and scaled to [0, 1] using MinMaxScaler. A chronological 80/20 split preserves temporal order, preventing future data from leaking into the training window. The `process_data` function builds overlapping sliding windows of length `n_features=2`, producing sequential (X, y) pairs where X is two consecutive prices and y is the next price. Arrays are reshaped to the (samples, timesteps, features) format required by Keras recurrent layers.

### Part 4: Model Architecture
A hybrid sequential model is built:

- GRU (256 units, return_sequences=True): processes input sequences using update and reset gates
- Dropout (0.4): randomly zeros 40% of activations to reduce overfitting
- LSTM (256 units): refines temporal representations using cell state and hidden state
- Dropout (0.4): applied again after the LSTM layer
- Dense (64, ReLU): non-linear combination of recurrent outputs
- Dense (1): single-step price forecast output

### Part 5: Training
The model is compiled with Adam (learning_rate=0.0005) and MSE loss. Two callbacks manage the training process. ModelCheckpoint saves weights whenever validation loss improves, preserving the best model regardless of later epochs. ReduceLROnPlateau reduces the learning rate by 90% when validation loss stagnates, enabling finer convergence near the optimum. Training history (MSE and loss curves) is plotted for both train and validation sets.

### Part 6: Evaluation
MSE and RMSE are reported for both train and test sets. Predictions are inverse-transformed from scaled space back to USD using the fitted MinMaxScaler. The final plot overlays predicted and actual closing prices across the test period for direct visual evaluation of forecast quality.

---

## Results

| Metric | Value |
|---|---|
| Validation MSE | ~0.000328 |
| Train RMSE | ~0.01 |
| Test RMSE | ~0.018 |

The model closely tracks the directional trend and price magnitude of Adobe's closing prices on the held-out test period, with predicted and actual lines remaining within a small margin throughout.

---

## Key Concepts

**Chronological Train/Test Split:** Time series data must be split sequentially. A random split would allow future prices to appear in training data, producing artificially optimistic results that would not generalize to real forecasting.

**Sliding Window Sequences:** The model does not receive the full time series at once. It sees fixed-length windows of past values and learns to predict the next value. Longer windows capture more context but increase sequence length and training time.

**GRU vs. LSTM:** GRUs use two gates (update and reset) and are computationally lighter. LSTMs use three gates and maintain a separate cell state, giving them stronger capacity for long-range dependencies. Stacking both leverages the complementary strengths of each architecture.

**MinMax Scaling for Stable Training:** Raw price values in the range of tens to hundreds of dollars create large gradient magnitudes that can destabilize training. Scaling to [0, 1] normalizes the loss landscape. Inverse-transforming predictions at evaluation time restores interpretable dollar values.

**Dropout in Recurrent Networks:** Overfitting is a significant risk when training RNNs on a single stock's sequence. Dropout randomly disables neurons during training, forcing the network to learn redundant representations and improving generalization to the test period.

---

## Stack

- Python 3
- NumPy, Pandas
- Matplotlib
- scikit-learn (MinMaxScaler)
- Keras with TensorFlow backend (Sequential, GRU, LSTM, Dense, Dropout, Adam, ModelCheckpoint, ReduceLROnPlateau)

---

## File Structure

```
nyse-lstm-forecasting/
├── nyse_lstm_forecasting.ipynb   # Main project notebook
├── prices.csv                    # NYSE daily price records
├── securities.csv                # Company metadata
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn tensorflow`
3. Open `nyse_lstm_forecasting.ipynb` in Jupyter and run all cells

Note: Training 100 epochs on the full sequence is computationally intensive. A GPU runtime (Google Colab or local CUDA) is recommended for reasonable training time.
