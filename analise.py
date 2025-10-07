import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

try:
    df = pd.read_csv('housingData.csv', sep=",", names=column_names, header=0)
    df.dropna(inplace=True) 
except FileNotFoundError:
    print("Erro: Arquivo 'housingData.csv' não encontrado.")
    exit()

print("--- 2.1 Análise Numérica Descritiva de TODAS as colunas ---")
print(df.describe().round(2))

columns_to_plot = df.columns
n_cols = 4
n_rows = (len(columns_to_plot) - 1) // n_cols + 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))

for i, col in enumerate(columns_to_plot):
    ax = axes[i // n_cols, i % n_cols]
    sns.histplot(data=df, x=col, kde=True, ax=ax)
    ax.set_title(f'Distribuição de {col}')
    ax.set_xlabel('')
    ax.set_ylabel('')

for i in range(len(columns_to_plot), n_rows * n_cols):
    fig.delaxes(axes.flatten()[i])

plt.suptitle('Histogramas de Distribuição de Todas as Variáveis', fontsize=20, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

plt.figure(figsize=(16, 6))
sns.boxplot(data=df, orient='h')
plt.title('Análise de Quartis e Outliers de Todas as Variáveis')
plt.show()

print("\n--- 2.3 Análise da Variável Categórica (CHAS) ---")
print("Moda da variável CHAS (0 = Não Margeia o Rio, 1 = Margeia o Rio):")
print(df['CHAS'].mode())
print("\nContagem de cada categoria para CHAS:")
print(df['CHAS'].value_counts())

print("\n--- 3.1 Foco na Variável-Alvo (MEDV) ---")
plt.figure(figsize=(8, 6))
sns.histplot(df['MEDV'], bins=20, kde=True)
plt.title('Distribuição do Valor dos Imóveis (MEDV)')
plt.xlabel('Valor Mediano do Imóvel ($1000s)')
plt.ylabel('Frequência')
plt.savefig('histograma_medv.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['MEDV'])
plt.title('Boxplot do Valor dos Imóveis (MEDV)')
plt.xlabel('Valor Mediano do Imóvel ($1000s)')
plt.savefig('boxplot_medv.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- 3.2 Análise de Correlação ---")
corr_matrix = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title('Mapa de Calor da Correlação entre as Variáveis', fontsize=16)
plt.savefig('heatmap_correlacao.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Análise Detalhada: RM vs. MEDV ---")
corr_rm, p_value_rm = stats.pearsonr(df['RM'], df['MEDV'])
print(f"Coeficiente de Pearson (r): {corr_rm:.4f}, P-valor: {p_value_rm:.10f}")
plt.figure(figsize=(8, 6))
sns.regplot(x='RM', y='MEDV', data=df, scatter_kws={'alpha':0.6})
plt.title('Relação entre Número de Quartos (RM) e Valor do Imóvel (MEDV)', fontsize=14)
plt.xlabel('Número Médio de Quartos por Habitação')
plt.ylabel('Valor Mediano do Imóvel ($1000s)')
plt.savefig('scatter_rm_vs_medv.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Análise Detalhada: LSTAT vs. MEDV ---")
corr_lstat, p_value_lstat = stats.pearsonr(df['LSTAT'], df['MEDV'])
print(f"Coeficiente de Pearson (r): {corr_lstat:.4f}, P-valor: {p_value_lstat:.10f}")
plt.figure(figsize=(8, 6))
sns.regplot(x='LSTAT', y='MEDV', data=df, scatter_kws={'alpha':0.6}, color='indianred')
plt.title('Relação entre % Pop. Baixa Renda (LSTAT) e Valor do Imóvel (MEDV)', fontsize=14)
plt.xlabel('% de Status Inferior da População')
plt.ylabel('Valor Mediano do Imóvel ($1000s)')
plt.savefig('scatter_lstat_vs_medv.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- 3.3 Demonstração de Hipóteses ---")
print("\n--- Hipótese 1: Imóveis que margeiam o rio Charles são mais caros ---")
tabela_comparativa_chas = df.groupby('CHAS')['MEDV'].describe()
print(tabela_comparativa_chas.round(2))
plt.figure(figsize=(8, 6))
sns.boxplot(x='CHAS', y='MEDV', data=df, palette='viridis')
plt.title('Comparação do Valor dos Imóveis pela Proximidade ao Rio (CHAS)', fontsize=14)
plt.ylabel('Valor Mediano do Imóvel ($1000s)')
plt.xticks([0, 1], ['Não Margeia o Rio', 'Margeia o Rio'])
plt.xlabel('')
plt.savefig('boxplot_rio_chas.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Hipótese 2: Imóveis com mais quartos (>6.5) são mais caros ---")
df['TIPO_RM'] = np.where(df['RM'] > 6.5, 'Muitos Quartos (>6.5)', 'Poucos Quartos (<=6.5)')
tabela_comparativa_rm = df.groupby('TIPO_RM')['MEDV'].describe()
print(tabela_comparativa_rm.round(2))
plt.figure(figsize=(8, 6))
sns.boxplot(x='TIPO_RM', y='MEDV', data=df, palette='plasma', order=['Poucos Quartos (<=6.5)', 'Muitos Quartos (>6.5)'])
plt.title('Comparação do Valor dos Imóveis por Número de Quartos (RM)', fontsize=14)
plt.ylabel('Valor Mediano do Imóvel ($1000s)')
plt.xlabel('Categoria de Imóvel por Número de Quartos')
plt.savefig('boxplot_quartos_rm.png', dpi=300, bbox_inches='tight')
plt.show()