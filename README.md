# IMDB Sentiment Analysis Project

This project involves sentiment analysis on movie reviews from the IMDB dataset using deep learning techniques with LSTM (Long Short-Term Memory) networks. The primary objective is to classify the reviews as positive or negative based on the textual content.

## Project Structure

- **Data Loading:** The dataset is loaded using pandas from a CSV file.
- **Data Preprocessing:** Text data is tokenized and converted into sequences that can be fed into an LSTM model.
- **Model Building:** A Sequential model is constructed with Embedding and LSTM layers using TensorFlow's Keras API.
- **Training and Evaluation:** The model is trained on the dataset and evaluated to determine its accuracy.

## Requirements

- TensorFlow
- Pandas
- Scikit-learn

Install the required libraries using:

```bash
pip install tensorflow pandas scikit-learn
```

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vedanth005/IMDB-Sentiment-Analysis.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd IMDB-Sentiment-Analysis
   ```

3. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

## Dataset

The project uses the IMDB movie reviews dataset. The dataset should be placed in the same directory as the notebook or the correct path should be specified in the code.

- **Dataset Format:** CSV file with columns for the review text and corresponding sentiment label.

## Model Architecture

- **Embedding Layer:** Converts text input into dense vectors of fixed size.
- **LSTM Layer:** Captures the sequential patterns in the text data.
- **Dense Layers:** Fully connected layers for final classification.

## How to Use

- Run the notebook cells sequentially to load data, preprocess, build the model, train, and evaluate.
- You can adjust hyperparameters such as the number of epochs, batch size, and LSTM units in the respective cells.

## Results

- The model's performance is evaluated using accuracy and loss metrics.
- Further improvements can be made by tuning the model or preprocessing steps.
