import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix
)
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Fonction pour afficher la distribution des classes
def show_class_distribution(file, title):
    df = pd.read_csv(file)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title(f"Distribution des classes - {title}")
    plt.xlabel("Classe (0 = légitime, 1 = fraude)")
    plt.ylabel("Nombre de transactions")
    plt.tight_layout()
    plt.show()

# Visualisation de la distribution des classes
file1 = '/Users/floriskezimana/Downloads/Projets INF5103/projet#05/creditcard.csv'
file2 = '/Users/floriskezimana/Downloads/Projets INF5103/projet#05/transactions_balanced2.csv'

show_class_distribution(file1, "Fichier 1")
show_class_distribution(file2, "Fichier 2")

# Chargement et prétraitement
def load_and_preprocess(file, sample_frac=1.0):
    data = pd.read_csv(file)
    data = data.sample(frac=sample_frac, random_state=42)
    y = data['Class']
    X = data.drop(columns=['Class'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Prétraitement
X_train1, X_test1, y_train1, y_test1 = load_and_preprocess(file1)
X_train2, X_test2, y_train2, y_test2 = load_and_preprocess(file2)

# Modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=30, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=30, eval_metric='logloss', n_jobs=-1),
    "SVM": SVC(probability=True, kernel='linear'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(20,), max_iter=100)
}

# Évaluation
def evaluate_models(models, X_train, X_test, y_train, y_test, dataset_label=""):
    results = []
    model_probs = {}
    model_preds = {}

    for name, model in models.items():
        print(f"🔄 Entraînement du modèle : {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_preds[name] = y_pred

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

        model_probs[name] = y_prob

        results.append([
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        ])

    # Matrices de confusion
    print(f"\n📊 Matrices de confusion - {dataset_label}")
    for name, y_pred in model_preds.items():
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice de confusion - {name} ({dataset_label})")
        plt.xlabel("Prédiction")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results, columns=["Modèle", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]), model_probs

# Courbes ROC
def plot_roc_curves(model_probs, y_test, title):
    plt.figure(figsize=(10, 8))
    for name, y_prob in model_probs.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbes ROC - {title}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Évaluation + ROC
results_df1, probs1 = evaluate_models(models, X_train1, X_test1, y_train1, y_test1, "Fichier 1")
results_df2, probs2 = evaluate_models(models, X_train2, X_test2, y_train2, y_test2, "Fichier 2")

# Barplots F1-score
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='F1-score', y='Modèle', data=results_df1.sort_values(by='F1-score', ascending=False), ax=axes[0])
axes[0].set_title("Modèles - Fichier 1")

sns.barplot(x='F1-score', y='Modèle', data=results_df2.sort_values(by='F1-score', ascending=False), ax=axes[1])
axes[1].set_title("Modèles - Fichier 2")
plt.tight_layout()
plt.show()

# Courbes ROC
plot_roc_curves(probs1, y_test1, "Fichier 1")
plot_roc_curves(probs2, y_test2, "Fichier 2")

# Sauvegarde CSV
results_df1.to_csv('/Users/floriskezimana/Downloads/Projets INF5103/projet#05/resultats_modeles_creditcard_rapide.csv', index=False)
results_df2.to_csv('/Users/floriskezimana/Downloads/Projets INF5103/projet#05/resultats_modeles_transactions_balanced_rapide.csv', index=False)

# Affichage
print("✅ Résultats fichier 1 :")
print(results_df1)
print("\n✅ Résultats fichier 2 :")
print(results_df2)
