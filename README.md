# Handwritten Digit Recognizer (MNIST CNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/mnist_cnn.ipynb)

A convolutional neural network that classifies handwritten digits (0–9) from the MNIST dataset, built from raw idx-ubyte binary files (not the pre-packaged Keras loader) to practice real-world data ingestion, trained and evaluated end-to-end in TensorFlow/Keras.

> Replace `YOUR_USERNAME/YOUR_REPO_NAME` in the badge link above with your actual GitHub path once pushed, so the badge opens your exact notebook.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 97.76% |
| Test Loss | 0.0716 |
| Total misclassified (out of 10,000) | 224 |

Full per-class precision/recall/F1 breakdown is in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

**Notable finding:** the model's most common confusions follow well-known MNIST ambiguity patterns (e.g. 5s misread as 7s, 8s/9s having the lowest recall) — consistent with how visually similar these handwritten digit pairs can be, even to humans.

## Dataset

- **Source:** MNIST, downloaded via Kaggle in raw idx-ubyte format
- **Training set:** 60,000 grayscale images, 28×28 pixels
- **Test set:** 10,000 grayscale images, 28×28 pixels
- Parsed directly from the binary idx format using `idx2numpy`, rather than a pre-built dataset loader — done intentionally to practice reading raw binary dataset formats.

## Project Structure

```
mnist-digit-recognizer/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── mnist_cnn.ipynb
└── models/
    └── mnist_cnn.keras
```

## Setup & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MNIST idx-ubyte files from Kaggle and place them in a local `data/` folder (excluded from version control — see `.gitignore`).
4. Open `notebooks/mnist_cnn.ipynb` in Jupyter or Colab and run all cells top to bottom.

Or just click the **Open in Colab** badge above to run it directly in your browser, no local setup needed.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data handling:** NumPy, idx2numpy
- **Evaluation:** scikit-learn (classification report, confusion matrix)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab (T4 GPU)

## Author

Built by Adi — [GitHub](https://github.com/AadilaAnees) · [LinkedIn](https://linkedin.com/in/aadila-anees)
