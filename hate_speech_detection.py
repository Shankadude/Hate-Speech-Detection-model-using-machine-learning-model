import pandas as pd
import re, string, json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("labeled_data.csv")

df["label_hate"] = (df["class"] == 0).astype(int)


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)      # remove URLs
    s = re.sub(r"@\w+", " ", s)                 # remove @mentions
    s = re.sub(r"#\w+", " ", s)                 # remove hashtags
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["clean_tweet"] = df["tweet"].apply(clean_text)

X = df["clean_tweet"]
y = df["label_hate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("vec", CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of hate


acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Not Hate", "Hate"])

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)


preds_out = pd.DataFrame({
    "tweet": df.loc[X_test.index, "tweet"].values,
    "clean_tweet": X_test.values,
    "actual_hate": y_test.values,
    "predicted_hate": y_pred,
    "hate_probability": y_proba
})
preds_out.to_csv("hate_speech_predictions.csv", index=False)

cm_df = pd.DataFrame(cm, index=["Actual_NotHate", "Actual_Hate"],
                     columns=["Pred_NotHate", "Pred_Hate"]).reset_index()
cm_long = cm_df.melt(id_vars=["index"], var_name="Predicted", value_name="Count")
cm_long = cm_long.rename(columns={"index": "Actual"})
cm_long.to_csv("confusion_matrix_long.csv", index=False)

metrics = {"accuracy": acc, "confusion_matrix": cm.tolist()}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, "hate_speech_lr_pipeline.joblib")

print("\nSaved: hate_speech_predictions.csv, confusion_matrix_long.csv, model_metrics.json, hate_speech_lr_pipeline.joblib")
