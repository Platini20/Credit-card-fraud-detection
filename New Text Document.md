# Credit Card Fraud Detection 💳⚠️

Ce projet vise à détecter les fraudes dans les transactions de cartes bancaires à l'aide d'algorithmes de machine learning.

## 📊 Dataset

Le dataset utilisé provient de Kaggle :
[Credit Card Fraud Detection | Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Il contient des transactions anonymisées effectuées par des porteurs de cartes en septembre 2013.

- **Transactions :** 284,807
- **Fraudes :** 492 (très déséquilibré !)

## 🧠 Modèles utilisés

- Régression logistique
- Random Forest
- XGBoost
- SVM
- Réseaux de neurones (MLP)
- Méthodes de suréchantillonnage (SMOTE)

## ⚙️ Techniques

- Analyse exploratoire
- Détection d’anomalies
- Équilibrage de classes
- Matrice de confusion, précision, rappel, F1-score
- ROC-AUC,

## 🔍 Objectif

Optimiser la précision de détection des fraudes tout en minimisant les faux positifs.

## 📁 Structure du projet

credit-card-fraud-detection/ ├── data/ │ └── creditcard.csv ├── models/ │ └── modèle_random_forest.pkl ├── notebooks/ │ └── fraud_detection.ipynb ├── README.md └── requirements.txt

## ✅ Résultats

Les meilleurs résultats ont été obtenus avec XGBoost + SMOTE :
- **Recall fraude :** 99.9%
- **Précision fraude :** 99.9%
- **F1-score :** 99.9%

## 🔗 Auteurs

Projet réalisé par Franklin dans le cadre de sa formation en data science.