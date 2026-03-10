# DeepCSAT – Customer Satisfaction Prediction using Deep Learning

DeepCSAT is a Artificial Intelligence and Machine Learning and Natural Language Processing (NLP) project that predicts customer satisfaction (CSAT) from e-commerce support tickets.

E-commerce platforms receive thousands of customer complaints, reviews, and support requests daily. Manually analyzing this feedback to determine whether customers are satisfied or dissatisfied is inefficient and time-consuming.

This project builds an AI system that automatically analyzes customer remarks, product category, sentiment signals, and item price to classify whether a customer is **Happy** or **Unhappy**.

The model uses **TF-IDF based text representation**, **sentiment analysis**, and a **deep neural network classifier** to achieve high prediction accuracy.

---

# Project Objectives

• Automatically analyze customer support tickets
• Detect customer satisfaction levels from text feedback
• Use sentiment analysis to improve prediction signals
• Train a deep neural network for classification
• Provide visual insights using confusion matrices and probability graphs

---

# Key Features

### Natural Language Processing

Customer remarks are processed using NLP techniques to extract meaningful textual patterns.

### Sentiment Analysis

TextBlob is used to measure the emotional tone of customer comments.

### Keyword Boosting

Negative complaint words such as **broken, worst, refund, late, terrible** are amplified to help the model recognize dissatisfaction signals.

### Deep Neural Network

A multi-layer neural network learns complex relationships between textual and numerical features.

### Visual Analytics

The project generates:

• Confusion Matrix
• Prediction Probability Graphs

---

# Model Architecture

The prediction model uses a **Deep Neural Network (MLPClassifier)**.

Input Features:

• TF-IDF text vectors
• Sentiment score
• Item price

Neural Network Structure:

Input Layer

↓

Hidden Layer 1 – 1024 neurons (ReLU)

↓

Hidden Layer 2 – 512 neurons (ReLU)

↓

Hidden Layer 3 – 256 neurons (ReLU)

↓

Output Layer – Binary Classification (Happy / Unhappy)

---

# Feature Engineering

To improve prediction accuracy, the following techniques were applied.

### 1. Boosted Complaint Keywords

Critical complaint words are emphasized during training.

Example:

Original remark
"The screen was broken and delivery was late"

Boosted remark
"The screen was broken broken_critical and delivery was late late_critical"

This forces the model to recognize strong negative signals.

---

### 2. Sentiment Amplification

Sentiment scores extracted using TextBlob are multiplied to strengthen emotional signals.

Example:

Positive remark → sentiment +0.6 → boosted to +6

Negative remark → sentiment -0.8 → boosted to -8

---

# Dataset

Dataset used:

**E-Commerce Customer Support Dataset**

Key attributes include:

• Customer remarks
• Product category
• Item price
• Customer satisfaction score (CSAT)

The CSAT score is converted into a binary label:

CSAT ≥ 4 → Happy

CSAT < 4 → Unhappy

---

# Training Strategy

Customer satisfaction datasets are usually imbalanced.

To avoid biased predictions:

**Under-sampling** is applied so both classes contain equal samples.

Data split:

Training data – 85%

Testing data – 15%

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/DeepCSAT-Customer-Satisfaction-Prediction
cd DeepCSAT-Customer-Satisfaction-Prediction
```

Install required libraries:

```
pip install -r requirements.txt
```

---

# Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
textblob
joblib
scipy
```

---

# Running the Project

Run the training script:

```
python train_model.py
```

After training, the model files will be saved:

```
csat_model.pkl
tfidf.pkl
scaler.pkl
label_encoder.pkl
```

---

# Example Prediction

Example input:

Category: Electronics

Customer remark:
"The screen was completely shattered and delivery was late. I want a refund."

Item price: 899

AI Prediction:

```
Analyzing New Customer Ticket...

★★★★★★★★★★★★★★★★★★★★★★★★★★★★
FINAL AI VERDICT: UNHAPPY
★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

# Model Performance

Example output after training:

```
FINAL DEMO ACCURACY: 91.42%
```

Classification report example:

```
              precision    recall  f1-score   support

Happy           0.92       0.90       0.91       120
Unhappy         0.90       0.92       0.91       118

accuracy                              0.91       238
```

---

# Visualization Outputs

### Confusion Matrix

Shows how many predictions were correct vs incorrect.

Example:

Actual vs Predicted customer satisfaction

|         | Pred Happy | Pred Unhappy |
| ------- | ---------- | ------------ |
| Happy   | 108        | 12           |
| Unhappy | 9          | 109          |

---

### Prediction Confidence Graph

The model also displays probability confidence for predictions.

Example output:

Happy → 18%

Unhappy → 82%

This allows better interpretation of model certainty.

---

# Project Structure

```
DeepCSAT
│
├── data
│   └── eCommerce_Customer_support_data.csv
│
├── model
│   ├── csat_model.pkl
│   ├── tfidf.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── train_model.py
├── README.md
└── requirements.txt
```

---

# Future Improvements

Possible future upgrades include:

• Transformer models (BERT / RoBERTa)
• Real-time customer feedback dashboards
• API deployment using Flask or FastAPI
• Integration with customer support systems

---

# Technologies Used

Python

Machine Learning

Deep Neural Networks

Natural Language Processing (NLP)

Scikit-learn

TextBlob

Matplotlib

Seaborn

---

# Author

Madhan
