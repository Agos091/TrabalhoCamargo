import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Simulando dataset de sintomas
df_assoc = pd.DataFrame({
    'Febre': [1, 0, 1, 1, 0],
    'Tosse': [1, 1, 1, 0, 1],
    'DorCabeça': [0, 1, 1, 0, 0],
    'Fadiga': [1, 1, 0, 1, 1]
})

# Executar FP-Growth
frequent_itemsets = fpgrowth(df_assoc, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

# Exibir regras
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
