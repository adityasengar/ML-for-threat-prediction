# ML for Threat Prediction

This project uses a Deep Neural Network (DNN) to predict a security score based on various API access and user session metrics. The original analysis was performed in a Jupyter Notebook and has been refactored into a structured, command-line-driven Python application.

## Project Overview

The tool performs the following steps:
1.  **Loads** tabular data representing user and API behavior.
2.  **Preprocesses** the data using StandardScaler and applies Principal Component Analysis (PCA) for dimensionality reduction.
3.  **Trains** a Keras-based DNN on the processed data to predict a security score.
4.  **Saves** the trained model for future use.
5.  **Predicts** scores on new data using the pre-trained model.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/ML-for-threat-prediction.git
    cd ML-for-threat-prediction
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This project requires TensorFlow. If you have a compatible GPU, `tensorflow` is recommended. Otherwise, you can use `tensorflow-cpu`.)*

---

## Usage

The application is controlled via `main.py` and has two primary modes: `train` and `predict`.

### Training the Model

To train the model on the dataset and save it, run the following command:

```bash
python main.py --mode train
```

This will:
- Load data from `data/dataset.csv`.
- Train the model for 10 epochs.
- Save the trained model to `threat_model.h5`.

### Making Predictions

Once the model is trained, you can use it to make predictions.

```bash
python main.py --mode predict
```

This will:
- Load the pre-trained model from `threat_model.h5`.
- Load the dataset and run inference on the test split.
- Print the evaluation metrics (MAE, R2 score) for the predictions.

### Command-Line Arguments

-   `--mode`: `train` or `predict`. (Required)
-   `--data_path`: Path to the input dataset. (Default: `data/dataset.csv`)
-   `--model_path`: Path to save (in train mode) or load (in predict mode) the model file. (Default: `threat_model.h5`)