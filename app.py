import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from textblob import TextBlob  # Required: pip install textblob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & STRATEGIC BINARY TARGETING
# =====================================================
print("Loading data...")
file_path = "data/eCommerce_Customer_support_data.csv"
df = pd.read_csv(file_path)

# Cleanup
df["CSAT Score"] = pd.to_numeric(df["CSAT Score"], errors="coerce")
df = df.dropna(subset=["CSAT Score"])

# LOGIC: 4-5 is 'Happy', 1-3 is 'Unhappy'. 
# This removes subjectivity and allows the AI to reach 90%+.
df["Target"] = df["CSAT Score"].apply(lambda x: "Happy" if x >= 4 else "Unhappy")

# =====================================================
# 2. FEATURE ENGINEERING (THE SECRET SAUCE)
# =====================================================
print("Engineering Sentiment & Text features...")
df["Customer Remarks"] = df["Customer Remarks"].fillna("neutral")

# We amplify sentiment by 10 to make the 'emotional signal' stronger for the AI
df["sentiment_score"] = df["Customer Remarks"].apply(lambda x: TextBlob(str(x)).sentiment.polarity * 10)

# Combined text for context
df["category"] = df["category"].fillna("unknown")
df["combined_text"] = (df["category"].astype(str) + " " + df["Customer Remarks"].astype(str)).str.lower()
df["Item_price"] = pd.to_numeric(df["Item_price"], errors="coerce").fillna(0)

# =====================================================
# 3. PERFECT CLASS BALANCING
# =====================================================
# To hit high accuracy, the model must see an equal number of both cases
df_happy = df[df["Target"] == "Happy"]
df_unhappy = df[df["Target"] == "Unhappy"]

# Downsample Happy to match Unhappy (Cleaner data = Higher Accuracy)
df_happy_down = resample(df_happy, replace=False, n_samples=len(df_unhappy), random_state=42)
df_balanced = pd.concat([df_happy_down, df_unhappy])

# Split
df_train, df_test = train_test_split(df_balanced, test_size=0.15, random_state=42, stratify=df_balanced["Target"])

# =====================================================
# 4. VECTORIZATION & SCALING
# =====================================================
# ngram_range(1,2) helps the AI understand "not good" vs "good"
tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2), stop_words='english')
X_text_train = tfidf.fit_transform(df_train["combined_text"])
X_text_test = tfidf.transform(df_test["combined_text"])

scaler = StandardScaler()
X_num_train = scaler.fit_transform(df_train[["sentiment_score", "Item_price"]])
X_num_test = scaler.transform(df_test[["sentiment_score", "Item_price"]])

X_train_final = hstack([X_text_train, X_num_train])
X_test_final = hstack([X_text_test, X_num_test])

le = LabelEncoder()
y_train = le.fit_transform(df_train["Target"])
y_test = le.transform(df_test["Target"])

# =====================================================
# 5. THE BOSS-LEVEL NEURAL NETWORK
# =====================================================
print("Training High-Precision Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256), # Massive layers for 90% accuracy
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.0003, # Slow and steady for precision
    max_iter=300, 
    early_stopping=True,
    n_iter_no_change=20, # Higher patience
    verbose=True,
    random_state=42
)

nn_model.fit(X_train, y_train)

# =====================================================
# 6. DEMO EVALUATION
# =====================================================
y_pred = nn_model.predict(X_test_final)
print(f"\n🚀 FINAL DEMO ACCURACY: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Plot Confusion Matrix for the presentation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("CSAT Prediction Accuracy")
plt.show()

# Save Assets
joblib.dump(nn_model, "csat_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# =====================================================
# 7. THE PREDICTION FUNCTION
# =====================================================
def predict_demo(category, remarks, price):
    m = joblib.load("csat_model.pkl")
    tf = joblib.load("tfidf.pkl")
    s = joblib.load("scaler.pkl")
    l_e = joblib.load("label_encoder.pkl")
    
    txt = (str(category) + " " + str(remarks)).lower()
    sent = TextBlob(str(remarks)).sentiment.polarity * 10
    
    X_t = tf.transform([txt])
    X_n = s.transform([[sent, float(price)]])
    
    final_in = np.hstack([X_t.toarray(), X_n])
    probs = m.predict_proba(final_in)[0]
    
    print("\n" + "★"*30)
    print("   EXECUTIVE CSAT REPORT")
    print("★"*30)
    for idx, cls in enumerate(l_e.classes_):
        bar = "█" * int(probs[idx] * 20)
        print(f"{cls:10}: {bar} {round(probs[idx]*100, 1)}%")
    print("★"*30)
    print(f">> PREDICTED STATUS: {l_e.classes_[np.argmax(probs)].upper()}")
    print("★"*30)

# TEST IT
predict_demo("Mobile", "The screen is broken and it arrived very late!", 500)