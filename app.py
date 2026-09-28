import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------- Load & prepare data ----------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

feature_names = list(X.columns)
mean_features = [f for f in feature_names if f.startswith('mean')]
other_features = [f for f in feature_names if f not in mean_features]
other_defaults = X[other_features].mean()

malignant_avg = df[df['target'] == 0][mean_features].mean()
benign_avg = df[df['target'] == 1][mean_features].mean()
mins, maxs = X[mean_features].min(), X[mean_features].max()

# ---------- Train models ----------
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': CalibratedClassifierCV(SVC(random_state=42), ensemble=False)
}

trained_models = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    trained_models[name] = m

results = []
for name, m in trained_models.items():
    preds = m.predict(X_test)
    probs = m.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

# Final model: Logistic Regression (tied for best, chosen for interpretability)
model = trained_models['Logistic Regression']

# ---------- Helper functions ----------
def make_radar(patient_values):
    patient_values = [float(v) for v in patient_values]
    patient = pd.Series(patient_values, index=mean_features)
    mins_f, maxs_f = mins.astype(float), maxs.astype(float)
    mal_f, ben_f = malignant_avg.astype(float), benign_avg.astype(float)
    norm = lambda s: (s - mins_f) / (maxs_f - mins_f)
    labels = [f.replace('mean ', '') for f in mean_features]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for values, name, color in [(norm(patient), 'This Sample', 'blue'),
                                  (norm(mal_f), 'Avg Malignant', 'red'),
                                  (norm(ben_f), 'Avg Benign', 'green')]:
        vals = values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=name, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title('Sample vs Typical Profiles', fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def confidence_gauge_html(prob, label):
    color = '#2ecc71' if label == 'Benign' else '#e74c3c'
    pct = prob * 100
    return f"""
    <div style='font-family:sans-serif; padding:16px; background:#fafafa; border-radius:10px;'>
      <div style='display:flex; justify-content:space-between;'>
        <span style='font-weight:bold; font-size:15px;'>Model Confidence</span>
        <span style='font-weight:bold; font-size:15px;'>{pct:.1f}%</span>
      </div>
      <div style='background:#e0e0e0; border-radius:20px; height:18px; width:100%; overflow:hidden; margin-top:6px;'>
        <div style='background:{color}; width:{pct}%; height:100%;'></div>
      </div>
    </div>
    """


def predict_cancer(*mean_values):
    mean_values = [float(v) for v in mean_values]
    full_input = dict(zip(mean_features, mean_values))
    full_input.update(other_defaults.to_dict())
    arr_df = pd.DataFrame([full_input])[feature_names]
    arr_scaled = pd.DataFrame(scaler.transform(arr_df), columns=feature_names)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0][pred]
    label = "Benign" if pred == 1 else "Malignant"

    if pred == 1:
        result_html = ("<div style='padding:20px; background:#e8f8ef; border-left:6px solid #2ecc71; "
                        "border-radius:8px;'><h2 style='color:#27ae60; margin:0;'>✅ Benign</h2></div>")
    else:
        result_html = ("<div style='padding:20px; background:#fdecea; border-left:6px solid #e74c3c; "
                        "border-radius:8px;'><h2 style='color:#c0392b; margin:0;'>⚠️ Malignant</h2></div>")

    gauge = confidence_gauge_html(prob, label)
    radar_img = make_radar(mean_values)
    return result_html, gauge, radar_img


def load_example(kind):
    row = df[df['target'] == (0 if kind == "malignant" else 1)][mean_features].iloc[0]
    return list(row.values)


# ---------- Build interface ----------
custom_css = """
.gradio-container {max-width: 1000px !important; margin: auto;}
footer {visibility: hidden}
"""

with gr.Blocks(title="Breast Cancer Clinical Decision Support") as demo:
    gr.Markdown("# 🩺 Breast Cancer Risk Classifier")
    gr.Markdown("*Clinical decision-support tool using FNA biopsy imaging measurements — "
                "for lab/clinical use, not patient self-assessment.*")

    with gr.Tabs():
        with gr.Tab("🔍 Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Cell Nuclei Measurements (from FNA imaging)")
                    inputs = [gr.Number(label=f.replace('mean ', '').title(),
                                         value=round(float(X[f].mean()), 3))
                              for f in mean_features]
                    with gr.Row():
                        example_malignant_btn = gr.Button("Load Malignant Example", size="sm")
                        example_benign_btn = gr.Button("Load Benign Example", size="sm")
                    predict_btn = gr.Button("Run Classification", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    result_output = gr.HTML(
                        value="<div style='padding:20px; text-align:center; color:#999;'>Awaiting input...</div>")
                    gauge_output = gr.HTML()
                    radar_output = gr.Image(label="Profile Comparison", show_label=True)

        with gr.Tab("📊 Model Performance"):
            gr.Markdown("### Test Set Results (114 held-out samples)")
            gr.Dataframe(value=results_df.round(4), interactive=False)
            gr.Markdown(
                "**Final model: Logistic Regression** — tied for best accuracy (98.2%) and "
                "ROC-AUC (0.995) with SVM, chosen for interpretability. Missed only 1 malignant "
                "case out of 42 in testing (false negative), the fewest of any model tested."
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                "**Data source:** Wisconsin Breast Cancer dataset — 569 samples, 30 features "
                "extracted from digitized FNA (Fine Needle Aspiration) biopsy images.\n\n"
                "**Intended users:** lab technicians / pathologists reviewing biopsy imaging "
                "output — not patients, who do not measure these values themselves.\n\n"
                "**⚠️ Disclaimer:** Educational SIWES project only. Not a certified medical "
                "device. Not for real clinical decisions."
            )

    predict_btn.click(fn=predict_cancer, inputs=inputs,
                       outputs=[result_output, gauge_output, radar_output])
    example_malignant_btn.click(fn=lambda: load_example("malignant"), outputs=inputs)
    example_benign_btn.click(fn=lambda: load_example("benign"), outputs=inputs)

    gr.Markdown("<p style='text-align:center; color:#999; font-size:12px;'>"
                "Built by Gift Christopher — SIWES ML Project</p>")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(css=custom_css, server_name="0.0.0.0", server_port=port)
