import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)


"""
Vamos criar um conjunto de dados com 500 usuários usando o numpy. Cada usuário terá 4 métricas descritas abaixo:

- visitas: Número de vezes que o usuário visitou o site no mês.

- tempo_no_site: tempo total em minutos que o usuário passou no site.

- itens_no_carrinho: Número de itens adicionados ao carrinho.

- valor_compra: O valor total em R$ da compra realizada pelo usuário no mês
"""


# Definindo o número de usuários
num_usuarios = 500


# 1. Gerar o número de vistas
visitas = np.random.randint(1, 51, size=num_usuarios)


# 2.Gerar o tempo no site (distribuição normal, correlacionado com as vistas)
# Média de 20 min, desvio padrão de 5, com um bônus por visita
tempo_no_site = np.random.normal(loc=20, scale=5, size=num_usuarios) + (visitas * 0.5)
tempo_no_site = np.round(tempo_no_site, 2)


# 3. Gerar número de itens no carrinho (dependente das visitas e do tempo)
# usuários que visitam mais e passam mais tempo no site, tendem a pôr mais itens no carrinho
itens_no_carrinho = np.random.randint(0, 8, size=num_usuarios) + (visitas // 10)


# garante que o tempo no site também influencie positivamente
itens_no_carrinho = (itens_no_carrinho + (tempo_no_site // 15)).astype(int)


# 4. Gerar o valor da compra (correlacionado com os itens no carrinho)
# preço médio por item de R$ 35,00 com alguma variação aleatória
valor_compra = (itens_no_carrinho * 35) + np.random.normal(loc=0, scale=10, size=num_usuarios)


# senão houver itens no carrinho, o valor da compradeve ser igual a 0
valor_compra[itens_no_carrinho == 0] = 0

# corringindo valores qeue podem ser negativos
valor_compra[valor_compra < 0] = 0
valor_compra = np.round(valor_compra, 2)


# Unindo tudo numa só matriz (ndarray)
# cada linha representa um usuário e cada coluna uma métrica
dados_ecommerce = np.column_stack((visitas, tempo_no_site, itens_no_carrinho, valor_compra))


print("\nShape da massa de dados: ", dados_ecommerce.shape)
print("\n5 primeiros usuários (linhas)")
print("\nColunas: [Visitas, Tempo no site (min), Itens no Carrinho, Valor da Compra (R$)]\n")
print(dados_ecommerce[:5])


# ----- CALCULANDO AS PRINCIPAIS MÉTRICAS ESTATÍSTICAS PARA TER UMA VISÃO GERAL DO PERFIL DOS USUÁRIOS -----
# Separando as colunas para facilitar a leitura
visitas_col = dados_ecommerce[:, 0]
tempo_col = dados_ecommerce[:, 1]
itens_col = dados_ecommerce[:, 2]
valor_col = dados_ecommerce[:, 3]


print("\n--- ANÁLISE ESTATÍSTICA GERAL ---")

df = pd.read_csv("DadosEcommerce.csv", sep=";")
print(df.head(5))



# Média
media_visitas = np.mean(visitas_col)
media_tempo = np.mean(tempo_col)
media_itens = np.mean(itens_col)
media_valor = np.mean(valor_col)

print(f"\nMédia de visitas: {media_visitas:.2f}")
print(f"Média de tempo no site: {media_tempo:.2f}")
print(f"Média de itens no carrinho: {media_itens:.2f}")
print(f"Média de compras em R$: {media_valor:.2f}\n")


# Mediana (menos sensível a outliers)
mediana_visitas = np.median(visitas_col)
mediana_tempo = np.median(tempo_col)
mediana_itens = np.median(itens_col)
mediana_valor = np.median(valor_col)

print(f"\nMediana das visitas: {mediana_visitas:.2f}")
print(f"Mediana do tempo no site: {mediana_tempo:.2f}")
print(f"Mediana dos itens no carrinho: {mediana_itens:.2f}")
print(f"Mediana do valor de compras R$: {mediana_valor:.2f}\n")


# Desvio Padrão (dispersão dos dados)
std_visitas = np.std(visitas_col)
std_tempo = np.std(tempo_col)
std_itens = np.std(itens_col)
std_valor = np.std(valor_col)

print(f"\nDesvio padrão das vistas: {std_visitas:.2f}")
print(f"Desvio padrão do tempo no site: {std_tempo:.2f}")
print(f"Desvio padrão dos itens no carrinho: {std_itens:.2f}")
print(f"Desvio padrão do valor de compras R$: {std_valor:.2f}\n")


# Valores máximos e mínimos
max_valor = np.max(valor_col)
min_valor_positivo = np.min(valor_col[valor_col > 0])       # Mínimo apenas entre quem comprou
print(f"Maior valor de compra R$: {max_valor:.2f}")
print(f"Menor valor de compra (de quem comprou) R$: {min_valor_positivo:.2f}\n")


# Gráfico para a análise da distribuição dos dados
plt.figure(figsize=(12, 5))
plt.hist(valor_col, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(media_valor, color='red', linestyle='--', linewidth=2, label=f"Média = R${media_valor:.2f}")
plt.axvline(mediana_valor, color='orange', linestyle='--', linewidth=2, label=f"Mediana = R${mediana_valor:.2f}")
plt.axvline(media_valor + std_valor, color='green', linestyle=':',
            linewidth=2, label=f"+1 STD = R${media_valor + std_valor:.2f}")
plt.axvline(media_valor - std_valor, color='green', linestyle=':',
            linewidth=2, label=f"-1 STD = R${media_valor - std_valor:.2f}")
plt.title("Distribuição dos Valores de Compra")
plt.xlabel("Valor de compra R$")
plt.ylabel("Frequência")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ----- SEGMENTAÇÃO E ANÁLISE DOS CLIENTES -----
# Descobrindo as características e comportamento dos clientes de 'alto valor'
clientes_alto_valor = dados_ecommerce[dados_ecommerce[:, 3] > 250]

print("\n--- CLIENTES DE ALTO VALOR (Compras > R$ 250) ---\n")
print(f"Número de Clientes de Alto Valor: {clientes_alto_valor.shape[0]}")

# Estatísticas desse segmento
media_visitas_alto_valor = np.mean(clientes_alto_valor[:, 0])
media_tempo_alto_valor = np.mean(clientes_alto_valor[:, 1])

print(f"Média de visitas desses clientes: {media_visitas_alto_valor:.2f}")
print(f"Média de tempo no site desses clientes: {media_tempo_alto_valor:.2f}")


# Comportamentos dos usuários que não realizaram nenhuma compra
visitantes_sem_compras = dados_ecommerce[dados_ecommerce[:, 3] == 0]

print("\n--- VISITANTES QUE NÃO COMPRAM ---\n")
print(f"Número de visitantes que não compraram: {visitantes_sem_compras.shape[0]}")

# Estatísticas para esse segmento
media_tempo_sem_compras = np.mean(visitantes_sem_compras[:, 1])
media_visitas_sem_compras = np.mean(visitantes_sem_compras[:, 0])

print(f"Média de acessos desses visitantes: {media_visitas_sem_compras:.2f}")
print(f"Média de tempo no site desses visitantes: {media_tempo_sem_compras:.2f}")


# ----- ANÁLISE DE CORRELAÇÃO -----
# Investigando de existe uma relação entre as variáveis


# a função np.corrcoef calcula a matriz de correlação, rowvar=False indica que as colunas são as variáveis
matriz_correlacao = np.corrcoef(dados_ecommerce, rowvar=False)

print("\n--- ATRIZ DE CORRELAÇÃO ---\n")
print("\n[Visitas, Tempo, Itens, Valor]\n")
print(np.round(matriz_correlacao, 2))


nome_variaveis = ["Visitas", "Tempo no Site", "Itens no Carrinho", "Valor da Compra"]

df_correlacao = pd.DataFrame(matriz_correlacao, index=nome_variaveis, columns=nome_variaveis)

# Heatmap da matriz de correlação
plt.figure(figsize=(12, 7))
sns.heatmap(df_correlacao, annot=True, cmap='Blues', fmt='.2f')
plt.title("Matriz de Correlação")
plt.show()
