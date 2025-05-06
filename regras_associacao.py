import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Simulando um dataset semelhante ao do aeroporto
np.random.seed(42)
df_cluster = pd.DataFrame({
    'NETPRO': np.random.randint(0, 10, 200),
    'Q20Age': np.random.randint(18, 70, 200),
    'Q21Gender': np.random.choice(['Male', 'Female'], 200),
    'Q22Income': np.random.randint(1000, 10000, 200),
    'Q23FLY': np.random.randint(0, 10, 200),
    'Q5TIMESFLOW': np.random.randint(1, 10, 200),
    'Q6LONGUSE': np.random.randint(0, 20, 200)
})

# Pré-processamento
numeric_features = ['NETPRO', 'Q20Age', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOW', 'Q6LONGUSE']
categorical_features = ['Q21Gender']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

X = preprocessor.fit_transform(df_cluster)

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(X)

# Avaliar Silhouette Score
score = silhouette_score(X, df_cluster['Cluster'])
print(f"Silhouette Score: {score:.2f}")

# Exibir proporção dos clusters
print(df_cluster['Cluster'].value_counts(normalize=True) * 100)
