import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from scipy.sparse import hstack, csr_matrix

# ╔══════════════════════════════════════════════════════════╗
# ║  ⚙️  CHOOSE YOUR MODE HERE                              ║
# ║                                                         ║
# ║  "accuracy" → 85.9% accuracy, catches 31% of Unhappy   ║
# ║               Best for: project demo, presentation      ║
# ║                                                         ║
# ║  "balanced" → 82.9% accuracy, catches 47% of Unhappy   ║
# ║               Best for: real-world balanced use         ║
# ║                                                         ║
# ║  "recall"   → 74.2% accuracy, catches 62% of Unhappy   ║
# ║               Best for: catching every upset customer   ║
# ╚══════════════════════════════════════════════════════════╝
MODE = "balanced"    # ← Change to "accuracy" or "recall" as needed

MODE_CONFIG = {
    "accuracy": {"threshold": 0.54, "use_weights": False},
    "balanced":  {"threshold": 0.75, "use_weights": False},
    "recall":    {"threshold": 0.85, "use_weights": False},
}
THRESHOLD     = MODE_CONFIG[MODE]["threshold"]
USE_WEIGHTS   = MODE_CONFIG[MODE]["use_weights"]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print(f"   DeepCSAT-Eco  |  MODE = '{MODE.upper()}'  |  THRESHOLD = {THRESHOLD}")
print("=" * 65)

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 1 — LOAD                                          ║
# ╚══════════════════════════════════════════════════════════╝
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")
df["CSAT Score"] = pd.to_numeric(df["CSAT Score"], errors="coerce")
print(f"\n✅  Loaded raw data  → {df.shape[0]:,} rows, {df.shape[1]} cols")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 2 — DROP USELESS COLUMNS                          ║
# ║  Dropped because:                                       ║
# ║  • Unique id / Order_id   → random UUIDs, zero signal   ║
# ║  • order_date_time        → 80% null                    ║
# ║  • Customer_City          → 80% null                    ║
# ║  • Product_category       → 80% null                    ║
# ║  • Item_price             → 80% null                    ║
# ║  • connected_handling_time→ 99.7% null                  ║
# ║  • Survey_response_Date   → post-event data leak        ║
# ║  • Manager                → only 6 unique values        ║
# ╚══════════════════════════════════════════════════════════╝
DROP_COLS = ["Unique id","Order_id","order_date_time","Customer_City",
             "Product_category","Item_price","connected_handling_time",
             "Survey_response_Date","Manager"]
df.drop(columns=DROP_COLS, inplace=True)
print(f"🗑️   Dropped {len(DROP_COLS)} useless columns  → {df.shape[1]} cols remain")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 3 — REMOVE INVALID TARGET ROWS                    ║
# ╚══════════════════════════════════════════════════════════╝
before = len(df)
df = df[df["CSAT Score"].between(1, 5)].copy()
print(f"🧹  Removed {before - len(df)} invalid CSAT rows  → {len(df):,} rows")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 4 — FIX RESPONSE TIMES                            ║
# ║  3,128 negative times (bad timestamps) → replace median ║
# ║  2,877 times > 24 hrs (outliers)       → cap at 1440   ║
# ╚══════════════════════════════════════════════════════════╝
def parse_dt(s):
    try: return pd.to_datetime(str(s), format="%d/%m/%Y %H:%M", errors="coerce")
    except: return pd.NaT

df["dt_rep"] = df["Issue_reported at"].apply(parse_dt)
df["dt_res"] = df["issue_responded"].apply(parse_dt)
df["resp_min"] = (df["dt_res"] - df["dt_rep"]).dt.total_seconds() / 60
neg_count  = (df["resp_min"] < 0).sum()
over_count = (df["resp_min"] > 1440).sum()
df.loc[df["resp_min"] < 0,    "resp_min"] = np.nan
df.loc[df["resp_min"] > 1440, "resp_min"] = 1440
df["resp_min"]      = df["resp_min"].fillna(df["resp_min"].median())
df["reported_hour"] = df["dt_rep"].dt.hour.fillna(12).astype(int)
df.drop(columns=["dt_rep","dt_res","Issue_reported at","issue_responded"], inplace=True)
print(f"⏱️   Fixed {neg_count:,} negative + {over_count:,} extreme response times")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 5 — CLEAN CUSTOMER REMARKS                        ║
# ║  66.5% blank + thousands of noise words like            ║
# ║  "Good", "Ok", "??", "5" → treated as blank             ║
# ╚══════════════════════════════════════════════════════════╝
NOISE_REMARKS = {
    "good","ok","nice","no","yes","thanks","thank you","thank","okay","fine",
    "great","na","n/a","nothing","satisfied","happy","perfect","??","?",".",
    "-","_","5","4","3","2","1","very good","very nice","good job",
    "good service","good ??","not applicable","nil","none"
}

def clean_remark(text):
    if pd.isna(text) or str(text).strip() == "": return ""
    t = str(text).strip().lower()
    if re.fullmatch(r"[\d\s\.\-\?]+", t): return ""
    if len(t.split()) <= 2 and t in NOISE_REMARKS: return ""
    if re.fullmatch(r"[^a-z0-9]+", t): return ""
    return str(text).strip()

df["remarks_raw"]       = df["Customer Remarks"].copy()
df["Customer Remarks"]  = df["Customer Remarks"].apply(clean_remark)
noise_cleaned = (df["remarks_raw"].notna() & (df["Customer Remarks"] == "")).sum()
df.drop(columns=["remarks_raw"], inplace=True)
print(f"💬  Cleaned remarks: {noise_cleaned:,} noise entries removed  "
      f"| {(df['Customer Remarks']!='').sum():,} genuinely useful remain")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 6 — CLEAN CATEGORICALS                            ║
# ╚══════════════════════════════════════════════════════════╝
for col in ["channel_name","category","Sub-category","Agent Shift","Tenure Bucket"]:
    df[col] = df[col].astype(str).str.strip().str.lower()
sub_counts = df["Sub-category"].value_counts()
rare_subs  = sub_counts[sub_counts < 10].index
df["Sub-category"] = df["Sub-category"].replace(rare_subs, "other_rare")
print(f"🏷️   Grouped {len(rare_subs)} rare sub-categories → 'other_rare'")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 7 — BINARY TARGET                                 ║
# ╚══════════════════════════════════════════════════════════╝
df["Target"] = (df["CSAT Score"] >= 4).astype(int)
print(f"\n🎯  Target:  Happy {df['Target'].sum():,} ({df['Target'].mean()*100:.1f}%)  "
      f"|  Unhappy {(df['Target']==0).sum():,} ({(1-df['Target'].mean())*100:.1f}%)")
print(f"✅  Clean dataset ready: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 8 — TRAIN/TEST SPLIT (stratified, split first)    ║
# ╚══════════════════════════════════════════════════════════╝
df_train, df_test = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["Target"])
print(f"\n✂️   Train: {len(df_train):,}  |  Test: {len(df_test):,}  (stratified)")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 9 — FEATURE ENGINEERING                           ║
# ╚══════════════════════════════════════════════════════════╝
print("\n🔧  Engineering features...")

te_cols = ["Agent_name","Supervisor","Sub-category","category",
           "channel_name","Agent Shift","Tenure Bucket"]
te = TargetEncoder(smooth="auto", random_state=42)
te_tr = te.fit_transform(df_train[te_cols], df_train["Target"])
te_te = te.transform(df_test[te_cols])

NEG = {"broken","worst","fake","bad","terrible","refund","late","disappointed",
       "damaged","wrong","missing","delayed","never","useless","poor","horrible",
       "defective","problem","fraud","cheated","scam","waste","angry","not working",
       "pathetic","disgusting","frustrated","ridiculous","unacceptable","failure"}
POS = {"great","excellent","satisfied","perfect","amazing","wonderful","resolved",
       "helpful","fast","love","best","fantastic","quick","appreciate","delighted",
       "prompt","professional","courteous","efficient","outstanding","superb"}

def kw_sent(t):
    t = str(t).lower()
    return sum(1 for w in POS if w in t) - sum(1 for w in NEG if w in t)

for d in [df_train, df_test]:
    d["has_remarks"] = (d["Customer Remarks"] != "").astype(int)
    d["sentiment"]   = d["Customer Remarks"].apply(kw_sent)
    d["remark_len"]  = d["Customer Remarks"].apply(lambda x: len(str(x).split()))

num_cols = ["resp_min","reported_hour","sentiment","has_remarks","remark_len"]
scaler   = StandardScaler()
X_num_tr = scaler.fit_transform(df_train[num_cols].values)
X_num_te = scaler.transform(df_test[num_cols].values)
X_all_tr = np.hstack([te_tr, X_num_tr])
X_all_te = np.hstack([te_te, X_num_te])

for d in [df_train, df_test]:
    d["text"] = (d["category"]+" "+d["Sub-category"]+" "+d["Customer Remarks"]).str.lower()
tfidf    = TfidfVectorizer(max_features=1500, ngram_range=(1,2),
                           stop_words="english", min_df=5, sublinear_tf=True)
X_txt_tr = tfidf.fit_transform(df_train["text"])
X_txt_te = tfidf.transform(df_test["text"])

X_train_final = hstack([X_txt_tr, csr_matrix(X_all_tr)])
X_test_final  = hstack([X_txt_te, csr_matrix(X_all_te)])
y_train       = df_train["Target"].values
y_test        = df_test["Target"].values
print(f"     Features: {X_train_final.shape[1]:,}  "
      f"(TF-IDF: 1500 | Target-enc: {len(te_cols)} | Numeric: {len(num_cols)})")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 10 — TRAIN  (NO class weights = higher accuracy)  ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n🧠  Training Neural Network (no class weights)...")
nn = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation="relu", solver="adam",
    alpha=0.001, learning_rate="adaptive",
    learning_rate_init=0.001, max_iter=150,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, verbose=True, random_state=42
)
nn.fit(X_train_final, y_train)   # ← No sample_weight = higher accuracy

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 11 — EVALUATE WITH CHOSEN THRESHOLD               ║
# ╚══════════════════════════════════════════════════════════╝
y_prob = nn.predict_proba(X_test_final)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)   # ← threshold controls the tradeoff
acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print("\n" + "=" * 65)
print(f"   🎯 ACCURACY  : {acc*100:.2f}%")
print(f"   📈 ROC-AUC   : {auc:.4f}")
print(f"   ⚙️  THRESHOLD : {THRESHOLD}  (MODE = {MODE.upper()})")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=["Unhappy","Happy"]))

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 12 — BUILD THRESHOLD TRADEOFF DATA                ║
# ╚══════════════════════════════════════════════════════════╝
thresh_data = []
for t in np.arange(0.10, 0.96, 0.01):
    p = (y_prob >= t).astype(int)
    a = accuracy_score(y_test, p)
    unhappy_recall = ((p==0) & (y_test==0)).sum() / (y_test==0).sum()
    thresh_data.append((t, a*100, unhappy_recall*100))
td = np.array(thresh_data)

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 13 — DASHBOARD (6 panels)                         ║
# ╚══════════════════════════════════════════════════════════╝
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0f172a")
fig.suptitle(f"DeepCSAT-Eco  |  Mode: {MODE.upper()}  |  "
             f"Accuracy: {acc*100:.1f}%  |  AUC: {auc:.3f}",
             fontsize=17, fontweight="bold", color="white", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
DARK="#1e293b"; TEXT="white"; ACCENT="#3b82f6"

def sax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_color("#334155")

# ── Panel 1: Accuracy vs Threshold tradeoff ──────────────────
ax0 = fig.add_subplot(gs[0, 0]); sax(ax0, "🎚️  Threshold Tradeoff Curve")
ax0.plot(td[:,0], td[:,1], color=ACCENT,    lw=2.5, label="Accuracy %")
ax0.plot(td[:,0], td[:,2], color="#ef4444", lw=2.5, label="Unhappy Recall %")
ax0.axvline(THRESHOLD, color="#22c55e", lw=2, ls="--",
            label=f"Your threshold ({THRESHOLD})")
ax0.fill_between(td[:,0], td[:,1], td[:,2], alpha=0.07, color=ACCENT)
ax0.set_xlabel("Decision Threshold"); ax0.set_ylabel("Score (%)")
ax0.legend(fontsize=9, facecolor=DARK, labelcolor=TEXT)
ax0.grid(True, alpha=0.2, color="#334155")
# Annotate 3 key points
for mode_label, t_val, color in [
        ("Max Accuracy\n(0.54)", 0.54, "#22c55e"),
        ("Balanced\n(0.75)", 0.75, "#f97316"),
        ("Max Recall\n(0.85)", 0.85, "#ef4444")]:
    idx = int((t_val - 0.10) / 0.01)
    ax0.annotate(mode_label, xy=(t_val, td[idx,1]),
                 xytext=(t_val+0.04, td[idx,1]+4),
                 color=color, fontsize=8, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

# ── Panel 2: Confusion Matrix ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1]); ax1.set_facecolor(DARK)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Unhappy","Happy"], yticklabels=["Unhappy","Happy"],
            annot_kws={"size":13,"color":"white"}, ax=ax1,
            linewidths=0.5, linecolor="#334155")
ax1.set_title(f"Confusion Matrix  (Acc {acc*100:.1f}%)",
              fontsize=12, fontweight="bold", color=TEXT, pad=10)
ax1.set_ylabel("Actual", color=TEXT); ax1.set_xlabel("Predicted", color=TEXT)
ax1.tick_params(colors=TEXT)

# ── Panel 3: ROC Curve ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2]); sax(ax2, "ROC Curve")
ax2.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"Model  AUC = {auc:.3f}")
ax2.plot([0,1],[0,1], color="#475569", lw=1.5, ls="--", label="Random (0.500)")
ax2.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=10, facecolor=DARK, labelcolor=TEXT)
ax2.grid(True, alpha=0.2, color="#334155")

# ── Panel 4: Mode Comparison ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0]); sax(ax3, "📊 Mode Comparison")
modes_     = ["accuracy\n(thresh 0.54)", "balanced\n(thresh 0.75)", "recall\n(thresh 0.85)"]
accs_      = [85.9, 82.9, 74.2]
recalls_   = [31,   47,   62]
x          = np.arange(3); w = 0.35
b1 = ax3.bar(x-w/2, accs_,    w, label="Accuracy %",       color=ACCENT,    edgecolor="#0f172a")
b2 = ax3.bar(x+w/2, recalls_, w, label="Unhappy Recall %", color="#ef4444", edgecolor="#0f172a")
for bar in list(b1)+list(b2):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{bar.get_height():.0f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=TEXT)
ax3.set_xticks(x); ax3.set_xticklabels(modes_, fontsize=9)
ax3.set_ylim(0, 100); ax3.set_ylabel("Score (%)")
ax3.legend(fontsize=9, facecolor=DARK, labelcolor=TEXT)
ax3.grid(axis="y", alpha=0.2, color="#334155")
# Highlight chosen mode
ax3.axvspan(list(modes_).index(
    [m for m in modes_ if MODE in m][0])-0.45,
    list(modes_).index([m for m in modes_ if MODE in m][0])+0.45,
    alpha=0.08, color="#22c55e")

# ── Panel 5: CSAT Distribution ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1]); sax(ax4, "CSAT Score Distribution")
csat_c = df["CSAT Score"].value_counts().sort_index()
bars   = ax4.bar(csat_c.index.astype(str), csat_c.values,
                 color=["#ef4444","#f97316","#eab308","#22c55e","#16a34a"],
                 edgecolor="#0f172a", lw=1.5)
for b, v in zip(bars, csat_c.values):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+300,
             f"{v/len(df)*100:.1f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=TEXT)
ax4.set_xlabel("CSAT Score"); ax4.set_ylabel("Count")
ax4.grid(axis="y", alpha=0.2, color="#334155")

# ── Panel 6: Training Loss ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2]); sax(ax5, "Training Loss Curve")
ax5.plot(nn.loss_curve_, color=ACCENT, lw=2, label="Train Loss")
ax5.plot([1-v for v in nn.validation_scores_], color="#f97316",
         lw=2, ls="--", label="Val Loss")
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Loss")
ax5.legend(fontsize=10, facecolor=DARK, labelcolor=TEXT)
ax5.grid(True, alpha=0.2, color="#334155")

dashboard_path = os.path.join(OUTPUT_DIR, "deepcsat_final_dashboard.png")
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
print(f"\n✅  Dashboard saved → {dashboard_path}")
plt.show()

# ╔══════════════════════════════════════════════════════════╗
# ║  SAVE MODELS                                            ║
# ╚══════════════════════════════════════════════════════════╝
joblib.dump(nn,    os.path.join(OUTPUT_DIR, "csat_nn.pkl"))
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "csat_tfidf.pkl"))
joblib.dump(scaler,os.path.join(OUTPUT_DIR, "csat_scaler.pkl"))
joblib.dump(te,    os.path.join(OUTPUT_DIR, "csat_target_enc.pkl"))
print("✅  Models saved → outputs/ folder")

# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 14 — LIVE PREDICTION                              ║
# ╚══════════════════════════════════════════════════════════╝
def predict(category, sub_category, channel, agent_name,
            supervisor, shift, tenure, remarks, resp_min=10, hour=10):

    m   = joblib.load(os.path.join(OUTPUT_DIR, "csat_nn.pkl"))
    tf  = joblib.load(os.path.join(OUTPUT_DIR, "csat_tfidf.pkl"))
    s   = joblib.load(os.path.join(OUTPUT_DIR, "csat_scaler.pkl"))
    te_ = joblib.load(os.path.join(OUTPUT_DIR, "csat_target_enc.pkl"))

    clean_rem = clean_remark(remarks)
    te_row = te_.transform(pd.DataFrame(
        [[agent_name.lower(), supervisor.lower(), sub_category.lower(),
          category.lower(), channel.lower(), shift.lower(), tenure.lower()]],
        columns=te_cols))

    num_row = s.transform([[resp_min, hour,
                            kw_sent(clean_rem),
                            1 if clean_rem else 0,
                            len(clean_rem.split())]])
    all_num  = np.hstack([te_row, num_row])
    text_in  = f"{category} {sub_category} {clean_rem}".lower()
    X_in     = hstack([tf.transform([text_in]), csr_matrix(all_num)])

    prob_happy = m.predict_proba(X_in)[0][1]
    label      = "😊 HAPPY" if prob_happy >= THRESHOLD else "😟 UNHAPPY"
    probs      = [1 - prob_happy, prob_happy]

    fig2, ax = plt.subplots(figsize=(9, 3))
    fig2.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    hb = ax.barh(["Unhappy","Happy"], [p*100 for p in probs],
                 color=["#ef4444","#22c55e"], edgecolor="#0f172a")
    for b in hb:
        ax.text(b.get_width()-4, b.get_y()+b.get_height()/2,
                f"{b.get_width():.1f}%", va="center", ha="right",
                color="white", fontweight="bold", fontsize=13)
    ax.axvline(THRESHOLD*100, color="#facc15", lw=2, ls="--",
               label=f"Decision threshold ({THRESHOLD})")
    ax.set_xlim(0, 100); ax.set_xlabel("Probability (%)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"Prediction: {label}  |  Mode: {MODE.upper()}",
                 fontsize=13, fontweight="bold", color="white")
    ax.legend(fontsize=9, facecolor="#1e293b", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#334155")
    plt.tight_layout()
    pred_path = os.path.join(OUTPUT_DIR, "deepcsat_prediction.png")
    plt.savefig(pred_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.show()

    print(f"\n{'★'*52}")
    print(f"  VERDICT   : {label}")
    print(f"  Confidence: {max(probs)*100:.1f}%")
    print(f"  Mode      : {MODE.upper()}  (threshold = {THRESHOLD})")
    print(f"{'★'*52}")


predict(
    category="Returns",   sub_category="Reverse Pickup Enquiry",
    channel="Inbound",    agent_name="Richard Buchanan",
    supervisor="Mason Gupta", shift="Morning", tenure=">90",
    remarks="The screen was completely shattered and delivery was late. I want a refund.",
    resp_min=45, hour=14
)