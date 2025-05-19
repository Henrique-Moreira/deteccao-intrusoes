import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# ------------------ 1. Leitura e correção de tipos ------------------

df = pd.read_csv(
    r'C:\Mestrado\Materias\2025-01\PGC308D - Tópicos Especiais em Sistemas de Computação2 - Detecção de Intrusões\Trabalhos\20-05-2025\data\ERENO-2.0-100K.csv',
    header=0,
    low_memory=False
)

# elimina eventuais linhas sem label
df = df.dropna(subset=['class'])

# tenta converter cada coluna para numérico; se não der, deixa como está
for col in df.columns:
    if col == 'class':
        continue
    df[col] = pd.to_numeric(df[col], errors='ignore')

print("DTypes finais:\n", df.dtypes.value_counts())

# ------------------ 2. Separação de X e y ------------------

y = df['class'].astype(str)  # string para o label encoder
X = df.drop(columns=['class'])

# ------------------ 3. Identificação de numérico vs categórico ------------------

numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"{len(numeric_cols)} colunas numéricas e {len(categorical_cols)} colunas categóricas")

# ------------------ 4. Encoding ------------------

# 4.1 label-encode y
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4.2 ordinal-encode as categóricas (cada valor vira um inteiro)
ord_enc = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
X_cat = X[categorical_cols].fillna("##MISSING##")
X_cat_enc = pd.DataFrame(
    ord_enc.fit_transform(X_cat),
    columns=categorical_cols,
    index=X.index
)

# 4.3 junta tudo num DataFrame só
X_num = X[numeric_cols]
X_proc = pd.concat([X_num, X_cat_enc], axis=1)
print("Dimensão de X após encoding:", X_proc.shape)

# ------------------ 5. Train/Test split ------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_enc,
    test_size=0.3,
    random_state=42,
    stratify=y_enc
)

# ------------------ 6. Treinamento ------------------

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ------------------ 7. Avaliação ------------------

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score (weighted): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão")
plt.show()

# ------------------ 8. XAI (SHAP) ------------------

explainer   = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_p

# ------------------ 9. Validação Cruzada ------------------

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores    = cross_val_score(model, X_proc, y_enc, cv=cv, scoring='f1_weighted')
print("CV F1-scores por fold:", scores)
print("F1-Score médio:", np.mean(scores))
