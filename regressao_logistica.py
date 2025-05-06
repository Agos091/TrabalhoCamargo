import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Simulando base de dados financeira
np.random.seed(42)
df_log = pd.DataFrame({
    'Idade': np.random.randint(18, 70, 100),
    'Renda': np.random.randint(1000, 10000, 100),
    'Divida': np.random.randint(0, 5000, 100),
    'NumCartoes': np.random.randint(1, 5, 100),
    'Default': np.random.choice([0, 1], 100)
})

X = df_log[['Idade', 'Renda', 'Divida', 'NumCartoes']]
y = df_log['Default']

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.2f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
