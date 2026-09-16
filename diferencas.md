# Comparação de estratégias

`codigo_final.ipynb` (0.8834 CV) vs. notebook em cascata do colega (0.78 no Kaggle).

---

## 1. Arquitetura

**Colega — cascata de dois estágios:**

- STAGE 1: binário, `Spondylolisthesis` vs. resto → acurácia 0.98
- STAGE 2: binário, `Hernia` vs. `Normal`, só sobre o que o estágio 1 disse ser "resto" → acurácia 0.70

**Meu código — um classificador de 3 classes direto.**

A cascata não é uma ideia ruim: ela isola o problema fácil do difícil. O problema é que isolar não
resolve. O gargalo de `Hernia` vs. `Normal` continua existindo e continua valendo 0.70. Além disso a
cascata acumula erro — um paciente classificado errado no estágio 1 nunca chega no estágio 2, e o
erro vira definitivo.

No meu código, o mesmo gargalo existe (a `Hernia` é onde erro se concentra), mas o modelo vê as três
classes de uma vez e pode usar a evidência inteira em cada decisão.

---

## 2. Modelos usados

**Colega:** apenas `DecisionTreeClassifier`, nos dois estágios.

**Meu código:** KNN, Naïve Bayes e Árvore de Decisão, os três com `GridSearchCV`.

Dois problemas na escolha dele:

1. **Requisito da atividade** — o enunciado pede a implementação dos três algoritmos. Só árvore não
   cumpre.
2. **Árvore é o pior dos três nestes dados.** Medido sobre as mesmas 15 dobras:

```
KNN           0.8834
NaiveBayes    0.8773
DecisionTree  0.8541
```

Ele construiu toda a arquitetura em cima do modelo mais fraco.

---

## 3. Pré-processamento

**Colega:**

```python
train_df['degree_spondylolisthesis'] = train_df['degree_spondylolisthesis'].clip(lower=0, upper=110)
train_df['pelvic_incidence']         = train_df['pelvic_incidence'].clip(lower=30, upper=95)
train_df['sacral_slope']             = train_df['sacral_slope'].clip(lower=10, upper=70)
```

**Meu código:**

```python
def preparar(df):
    df = df.copy()
    ds = df['degree_spondylolisthesis'].clip(-20, 120)
    df['ds_clip'] = ds
    df['ds_log']  = np.sign(ds) * np.log1p(np.abs(ds))
    return df.drop(columns=['pelvic_incidence'])
```

Divergências:

- **`clip(lower=0)` apaga informação.** `degree_spondylolisthesis` vai até -11 nos dados. Valor
  negativo é deslizamento *posterior* (retrolistese), clinicamente diferente de zero. Zerar isso
  joga fora justamente um sinal que ajuda a separar `Normal`. Meu clip usa `-20`, que preserva os
  negativos e ainda trunca o outlier de 418,5.
- **Clipar `pelvic_incidence` quebra a geometria.** Vale a identidade exata
  `pelvic_incidence = pelvic_tilt + sacral_slope` (verificado: erro máximo `1e-8` nas 217 linhas).
  Clipar essa coluna cria linhas onde a soma não fecha mais — dado inconsistente. O que a coluna é
  de fato é **redundante**, e o tratamento certo é remover, não truncar.
- **Nenhuma feature derivada.** Ele não cria nada. A coluna `ds_log` (log simétrico) é o que aproxima
  a distribuição de uma normal e melhora tanto o KNN quanto o Naïve Bayes.

---

## 4. Coluna duplicada no notebook dele

A saída de `feature_importances_` mostra as duas:

```
lumbar_lordosis_angle    0.000000
lombar_lordosis_angle    0.107474
```

São a mesma medida, uma com o nome escrito errado. A árvore escolheu a duplicata com o nome
incorreto e ignorou a original. Isso indica dataframe sujo — em algum ponto uma coluna foi renomeada
ou recriada sem apagar a anterior. O modelo está treinando com uma feature fantasma.

---

## 5. Validação

**Colega:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` → **5 medições**.

**Meu código:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)` → **15 medições**.

Com 217 amostras, cada dobra de validação tem só 44 exemplos. Uma única medição varia cerca de 8
pontos percentuais só trocando a semente. Isso explica diretamente o resultado dele: o notebook
imprime

```
Accuracy: 0.8636
```

e o Kaggle devolve **0.78**. A diferença de ~8 pontos é exatamente a amplitude do ruído de uma
medição única. O número impresso nunca foi uma estimativa confiável.

Meu 0.8834 é média de 15 dobras, e se sustenta em 3 sementes independentes:

```
seed1   KNN=0.8817   NB=0.8779   DT=0.8449
seed7   KNN=0.8718   NB=0.8770   DT=0.8509
seed99  KNN=0.8800   NB=0.8809   DT=0.8496
```

---

## 6. SMOTE

```python
smote = SMOTE(random_state=42, sampling_strategy=0.8, k_neighbors=3)
X_train_stage2_bal, y_train_stage2_bal = smote.fit_resample(X_train_stage2, y_train_stage2)
```

O SMOTE gera amostras sintéticas interpolando entre vizinhos da classe minoritária. Com apenas 42
exemplos de `Hernia`, e `k_neighbors=3`, os pontos criados ficam em cima das poucas amostras reais
que existem. A árvore então memoriza essas regiões artificiais, que não correspondem a pacientes
reais — sobe a métrica de treino e não sobe a de teste. `min_samples_leaf=3` não é restrição
suficiente para conter isso.

Meu código não rebalanceia. Usa `Stratified` na validação (mantém a proporção 48/32/19% em cada
dobra) e deixa o modelo trabalhar com a distribuição real.

---

## 7. Bug: o GridSearch do estágio 2 é descartado

No notebook dele:

```python
grid_stage2 = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_stage2,
                           cv=cv, scoring='accuracy')
grid_stage2.fit(X_train_stage2, y_train_stage2)   # roda, imprime CV = 0.7869
```

E depois, na função que de fato gera as previsões:

```python
def predict_hierarquico(X):
    pred_stage1 = model_stage1.predict(X)
    final_preds = np.where(pred_stage1 == 1, 'Spondylolisthesis', None)
    idx_resto = np.where(pred_stage1 == 0)[0]
    if len(idx_resto) > 0:
        pred_stage2 = model_stage2_v3.predict(X.iloc[idx_resto])   # <- usa v3, não o grid
        final_preds[idx_resto] = pred_stage2
    return final_preds
```

`grid_stage2` é treinado e nunca usado. Quem prevê é `model_stage2_v3`, uma árvore com
hiperparâmetros fixados na mão (`max_depth=4, min_samples_leaf=3`) e treinada nos dados com SMOTE.

Consequência prática: **o relatório impresso avalia um modelo diferente do que gera o
`submission.csv`.** Toda a busca de hiperparâmetros do estágio 2 foi trabalho jogado fora, e as
métricas exibidas não descrevem o que foi submetido.

---

## Resumo

A cascata não é uma má ideia. O problema é o que foi combinado com ela: o modelo mais fraco dos três,
uma validação instável demais para o tamanho do dataset, clipping que apaga informação e quebra a
geometria dos dados, uma coluna duplicada com nome errado, e um bug que desconecta a busca de
hiperparâmetros do modelo final.

O 0.8636 impresso nunca foi real — era uma medição única de 44 amostras. Do outro lado, 0.8834 é
média de 15 dobras e se mantém em 3 sementes independentes.