# Intrusion Detection on ERENO-2.0-100K Dataset

# 1. Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import warnings
warnings.filterwarnings("ignore")

# 2. Load Data
file_path = 'C:\\Mestrado\\Materias\\2025-01\\PGC308D - Tópicos Especiais em Sistemas de Computação2 - Detecção de Intrusões\\Trabalhos\\20-05-2025\\data\\ERENO-2.0-100K.csv'
df = pd.read_csv(file_path)
print("Shape:", df.shape)
df.head()

# 3. Analyze Class Distribution
plt.figure(figsize=(10, 4))
df['class'].value_counts().plot(kind='bar')
plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Frequência')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4. Preprocessing
# Remover colunas irrelevantes ou com dados de texto redundantes
drop_cols = ['Time', 'GooseTimestamp', 'ethDst', 'ethSrc', 'ethType', 'gocbRef', 'datSet', 'goID', 'protocol']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Codificar colunas booleanas ou categóricas
categorical_cols = ['test', 'ndsCom', 'cbStatus']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).map({'FALSE': 0, 'TRUE': 1})

# Separar features e rótulo
X = df.drop('class', axis=1)
y = df['class']

# Codificar o rótulo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Garantir que só há colunas numéricas
X = X.select_dtypes(include=[np.number])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# 6. Modelo Base: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nRelatório de Classificação (Random Forest):")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

# Matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão - Random Forest")
plt.tight_layout()
plt.show()

# 7. Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Top 15 Features mais Importantes - Random Forest')
plt.tight_layout()
plt.show()

# 8. SHAP para Explicabilidade
explainer = shap.Explainer(rf, X_train)
# Converta X_test[:100] para DataFrame para SHAP
X_test_df = pd.DataFrame(X_test[:100], columns=X.columns)
shap_values = explainer(X_test_df, check_additivity=False)  # Desativa o check_additivity
shap.summary_plot(shap_values, X_test_df, feature_names=X.columns, class_names=label_encoder.classes_)

# 9. Conclusão
print("\nConclusão:")
print("- O modelo Random Forest teve bom desempenho em classificar eventos.")
print("- As features mais importantes incluem valores RMS e trap areas.")
print("- A explicação com SHAP mostra como cada feature influencia cada decisão.")
