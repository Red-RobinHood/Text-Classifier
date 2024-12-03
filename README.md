# Transformer-Based Text Classifier

## Introduction

This project implements a **Text Classifier** using an **Encoder-only Transformer** model. It is trained on the **AG_NEWS** dataset, which consists of news articles that can be classified into four categories:

- **World**
- **Sports**
- **Business**
- **Science/Technology**

This model leverages the power of Transformer-based architectures to accurately classify news articles into these categories.

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:

    ```bash
    git clone https://github.com/Red-RobinHood/Text-Classifier.git
    cd Text-Classifier
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you are using Python 3.8 or higher.

---

## Usage

### 1. Using the Pretrained Model

To use the pretrained model for text classification, follow these steps:

- Change the `custominput` flag in the `model.py` file (on line 368):
  - Set it to `True` to give input via the command line interface (CLI).
  - Set it to `False` to use the validation data from the `val.csv` file.
  
Once the flag is set, run the script:

```bash
python model.py
```

This will use the pretrained model to classify news articles.

### 2. Training on a Custom Dataset

To train the model on your own dataset:
1.	Delete the existing weights file from the weights folder or change the model_name parameter.

2.	Add your custom dataset to the appropriate subfolder within the Dataset folder.

3.	After this, run the training script to start training on your dataset.
---
## Acknowledgments

•	This project uses the AG_NEWS dataset for training and validation.

•	The approach is inspired by research on Transformer architectures, particularly for text classification tasks.

---

Feel free to contribute by opening issues or submitting pull requests!
