# 🛒 DeepCSAT-Eco — Customer Satisfaction Score Predictor

<p align="center">
  <img src="outputs/deepcsat_final_dashboard.png" alt="DeepCSAT Dashboard" width="900"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Accuracy-80.47%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.7775-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-ANN%20%7C%20MLP-purple?style=for-the-badge"/>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Goal](#-project-goal)
- [Dataset](#-dataset)
- [Data Cleaning & Audit](#-data-cleaning--audit)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Threshold Modes](#-threshold-modes)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Live Prediction](#-live-prediction)
- [Accuracy Improvement Journey](#-accuracy-improvement-journey)
- [Why Not 95%+](#-why-not-95)
- [Technologies Used](#-technologies-used)

---

## 📖 Overview

**DeepCSAT-Eco** is a deep learning project that predicts **Customer Satisfaction (CSAT) scores** from e-commerce customer support interactions using an Artificial Neural Network (ANN / MLP).

The model classifies each customer support ticket as either:

| Label | Meaning | CSAT Score |
|---|---|---|
| 😊 **HAPPY** | Satisfied customer | 4 or 5 |
| 😟 **UNHAPPY** | Dissatisfied customer | 1, 2, or 3 |

> Built for the **DeepCSAT-Eco** project — predicting satisfaction in real-time to help e-commerce businesses improve service quality and customer retention.

---

## 🎯 Project Goal

> Develop a deep learning model that accurately predicts CSAT scores based on customer interactions and feedback — providing e-commerce businesses with a powerful tool to monitor and enhance customer satisfaction in real-time.

**Key objectives:**
- Predict whether a customer will be Happy or Unhappy after a support interaction
- Identify the most important signals driving customer dissatisfaction
- Provide a live prediction tool for new incoming support tickets
- Deliver honest, explainable results — not inflated numbers

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | eCommerce Customer Support Data |
| Total Rows | 85,907 |
| Total Columns | 20 (raw) → 11 (after cleaning) |
| Target Column | `CSAT Score` (1–5) |
| Class Distribution | 82.5% Happy / 17.5% Unhappy |

### CSAT Score Distribution

```
Score 1 (Worst)  → 11,230 rows  (13.1%)  ← Unhappy
Score 2          →  1,283 rows  ( 1.5%)  ← Unhappy
Score 3          →  2,558 rows  ( 3.0%)  ← Unhappy
Score 4          → 11,219 rows  (13.1%)  ← Happy
Score 5 (Best)   → 59,617 rows  (69.4%)  ← Happy
```

---

## 🧹 Data Cleaning & Audit

A full data audit was performed **before** any training. This is the step most projects skip — and why they get unreliable models.

### Columns Dropped (9 removed)

| Column | Reason Dropped |
|---|---|
| `Unique id` | Random UUID — zero signal |
| `Order_id` | Random UUID — zero signal |
| `order_date_time` | **80% null** — can't use |
| `Customer_City` | **80% null** — can't use |
| `Product_category` | **80% null** — can't use |
| `Item_price` | **80% null** — can't impute reliably |
| `connected_handling_time` | **99.7% null** — only 242 rows had data |
| `Survey_response_Date` | Post-event data — would cause **data leakage** |
| `Manager` | Only 6 unique values — too low signal |

### Data Issues Fixed

| Issue Found | Count | Fix Applied |
|---|---|---|
| Negative response times (corrupted timestamps) | 3,128 | Replaced with median |
| Response times > 24 hours (outliers) | 2,877 | Capped at 1,440 minutes |
| Noise remarks ("Good", "Ok", "??", "5") | 7,315 | Treated as blank |
| Rare sub-categories (< 10 rows) | 4 | Grouped into `other_rare` |
| Duplicate rows | 0 | None found |
| Invalid CSAT scores (outside 1-5) | 0 | None found |

### Customer Remarks Quality

```
Total rows          : 85,907
Originally blank    : 57,165  (66.5%)
Noise removed       :  7,315  (e.g. "Good", "Ok", "??", numbers)
Genuinely useful    : 21,427  (25.0%)
```

---

## 🔧 Feature Engineering

All features were **fit on training data only** and then applied to test data — preventing data leakage.

### Feature Groups Used

| Feature | Type | How Created |
|---|---|---|
| `te_Agent_name` | Target Encoded | Agent's historical CSAT rate |
| `te_Supervisor` | Target Encoded | Supervisor's historical CSAT rate |
| `te_Sub-category` | Target Encoded | Sub-category's historical CSAT rate |
| `te_category` | Target Encoded | Category's historical CSAT rate |
| `te_channel_name` | Target Encoded | Channel's historical CSAT rate |
| `te_Agent Shift` | Target Encoded | Shift's historical CSAT rate |
| `te_Tenure Bucket` | Target Encoded | Tenure group's historical CSAT rate |
| `resp_min` | Numeric | Minutes between issue reported and responded |
| `reported_hour` | Numeric | Hour of day the issue was reported |
| `sentiment` | Numeric | Keyword-based score (positive − negative words) |
| `has_remarks` | Binary | 1 if customer left a remark, 0 if blank |
| `remark_len` | Numeric | Word count of the remark |
| TF-IDF (1,500 tokens) | Sparse Text | Bigram TF-IDF on category + sub-category + remarks |

**Total features: 1,512**

### Sentiment Keywords

```python
# Negative words (lower CSAT signal)
NEG = {"broken", "worst", "fake", "bad", "terrible", "refund", "late",
       "disappointed", "damaged", "wrong", "missing", "delayed", ...}

# Positive words (higher CSAT signal)
POS = {"great", "excellent", "satisfied", "perfect", "amazing",
       "resolved", "helpful", "fast", "appreciate", "delighted", ...}
```

---

## 🧠 Model Architecture

```
Input Layer    → 1,512 features
                 (1,500 TF-IDF + 7 Target-Encoded + 5 Numeric)

Hidden Layer 1 → 512 neurons  (ReLU activation)
Hidden Layer 2 → 256 neurons  (ReLU activation)
Hidden Layer 3 → 128 neurons  (ReLU activation)
Hidden Layer 4 →  64 neurons  (ReLU activation)

Output Layer   →   2 neurons  (Softmax → Happy / Unhappy probability)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Solver | Adam |
| Learning Rate | Adaptive |
| Initial LR | 0.001 |
| L2 Regularization (alpha) | 0.001 |
| Max Iterations | 150 |
| Early Stopping | ✅ Yes |
| Validation Fraction | 10% |
| Stop if no improvement for | 15 epochs |
| Class Weights | None (controlled by threshold instead) |
| Random Seed | 42 |

> **Key Design Decision:** No `sample_weight` is used during training. Class imbalance is handled via **threshold tuning** at prediction time — this gives higher accuracy while still allowing control over Unhappy recall.

---

## 📈 Results

```
=================================================================
   🎯 ACCURACY  : 80.47%
   📈 ROC-AUC   : 0.7775
   ⚙️  THRESHOLD : 0.75  (MODE = BALANCED)
=================================================================
              precision    recall  f1-score   support

     Unhappy       0.45      0.51      0.48      2261
       Happy       0.89      0.87      0.88     10626

    accuracy                           0.80     12887
   macro avg       0.67      0.69      0.68     12887
weighted avg       0.82      0.80      0.81     12887
```

### What These Numbers Mean

| Metric | Score | Plain English |
|---|---|---|
| **Accuracy** | 80.47% | Gets 8 out of 10 predictions right |
| **ROC-AUC** | 0.7775 | Correctly separates Happy/Unhappy 77.75% of the time |
| **Happy Precision** | 89% | When it says Happy — it's right 89% of the time |
| **Unhappy Recall** | 51% | Catches 51 out of every 100 genuinely unhappy customers |

---

## ⚙️ Threshold Modes

The model supports 3 modes — change one line at the top of the script:

```python
MODE = "balanced"   # ← change to "accuracy" or "recall"
```

| Mode | Threshold | Accuracy | Unhappy Caught | Best For |
|---|---|---|---|---|
| `"accuracy"` | 0.54 | **85.9%** | 31% | Project demo / presentation |
| `"balanced"` | 0.75 | **80.47%** | 51% | Real-world balanced use ✅ |
| `"recall"` | 0.85 | **74.2%** | 62% | Catching every upset customer |

---

## 📁 Project Structure

```
DeepCSAT/
│
├── data/
│   └── eCommerce_Customer_support_data.csv   ← Raw dataset
│
├── outputs/
│   ├── deepcsat_final_dashboard.png          ← 6-panel results dashboard
│   ├── deepcsat_prediction.png               ← Live prediction chart
│   ├── csat_nn.pkl                           ← Trained ANN model
│   ├── csat_tfidf.pkl                        ← TF-IDF vectorizer
│   ├── csat_scaler.pkl                       ← StandardScaler
│   └── csat_target_enc.pkl                   ← TargetEncoder
│
├── deepcsat_final_improved.py                ← ✅ Main script (this file)
├── requirements.txt                          ← Python dependencies
└── README.md                                 ← This file
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DeepCSAT-Eco.git
cd DeepCSAT-Eco
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
joblib>=1.3
```

---

## ▶️ How to Run

### Make sure the dataset is in the right place
```
DeepCSAT/
└── data/
    └── eCommerce_Customer_support_data.csv
```

### Run the main script
```bash
python deepcsat_final_improved.py
```

### What happens when you run it
```
1. Loads and audits the raw dataset
2. Drops 9 useless/null columns
3. Fixes corrupted response times
4. Removes noise from customer remarks
5. Splits data into train/test (stratified)
6. Builds features (target encoding + TF-IDF + numeric)
7. Trains the ANN model
8. Evaluates and prints results
9. Saves a 6-panel dashboard to outputs/
10. Runs a live prediction demo
```

---

## 🔮 Live Prediction

After training, you can predict any new customer ticket:

```python
predict(
    category    = "Returns",
    sub_category= "Reverse Pickup Enquiry",
    channel     = "Inbound",
    agent_name  = "Richard Buchanan",
    supervisor  = "Mason Gupta",
    shift       = "Morning",
    tenure      = ">90",
    remarks     = "The screen was completely shattered and delivery was late. I want a refund.",
    resp_min    = 45,
    hour        = 14
)
```

**Output:**
```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  VERDICT   : 😟 UNHAPPY
  Confidence: 71.9%
  Mode      : BALANCED  (threshold = 0.75)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 📉 Accuracy Improvement Journey

| Version | Key Problem | Accuracy |
|---|---|---|
| Original code | Under-sampling (lost 73% of data), only 2 features | 65.0% |
| First fix | Added features, no data cleaning | 69.65% |
| Clean pipeline | Proper cleaning + class weights | 68.57% |
| **Final version** | **No weights + threshold tuning** | **80.47% ✅** |

---

## ❓ Why Not 95%+

This is an honest project. Here's why 95%+ is mathematically impossible with this dataset:

**1. The naive baseline is already 82.5%**
A model that predicts "Happy" for every customer gets 82.5% free — because 82.5% of the data is Happy. Reaching 95% on top of that requires near-perfect identification of Unhappy customers, which the data simply doesn't support.

**2. 66% of remarks are blank**
Customer remarks are the strongest signal for satisfaction. But two-thirds of tickets have no remarks at all — leaving the model to rely only on agent name, category, and time of day.

**3. Irreducible error in the data**
Two identical tickets (same agent, same category, no remarks) can have CSAT scores of 1 and 5. No algorithm can predict both correctly from the same input.

> A fake 95% would require data leakage, overfitting, or removing the Unhappy class — all of which produce impressive screenshots but useless real-world models.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Model, encoders, vectorizer, metrics |
| MLPClassifier | Artificial Neural Network (ANN) |
| TargetEncoder | High-cardinality categorical encoding |
| TfidfVectorizer | Text feature extraction |
| StandardScaler | Numeric feature normalization |
| matplotlib | Dashboard and charts |
| seaborn | Confusion matrix heatmap |
| joblib | Model serialization |
| scipy.sparse | Efficient sparse matrix handling |

---

## 👤 Author

**Madhan**
- GitHub: [MadhanKairamkonda](https://github.com/MadhanKairamkonda)
- Project: DeepCSAT-Eco — Customer Satisfaction Score Prediction

---

## 📄 License

This project is for academic and educational purposes.

---

<p align="center">
  Made with ❤️ for the DeepCSAT-Eco Deep Learning Project
</p>
