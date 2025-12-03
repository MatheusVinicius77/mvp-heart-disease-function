# Implementação de SMOTE - TabM Heart Disease

## 📋 Resumo do Dataset Cleveland

**Fonte**: UCI Machine Learning Repository  
**Instâncias**: 303 pacientes (297 após limpeza)  
**Features**: 13 atributos clínicos + 1 target  
**Target**: Diagnóstico de doença coronariana (binário: 0=ausência, 1=presença)  
**Distribuição**: ~54% saudáveis (160), ~46% doentes (137)

### Features Selecionadas no Experimento

- **Numéricas**: `oldpeak` (ST depression induzida por exercício)
- **Categóricas**: `cp` (tipo de dor no peito), `exang` (angina induzida), `slope` (inclinação do ST), `ca` (vasos coronários), `thal` (talassemia)

---

## 🔍 Problema Identificado

Análise dos 10 folds revelou:

- **Fold 6**: 30% de erro (9/30 amostras)
- **Fold 4**: 16.7% de erro (5/30 amostras)
- **Folds 5, 7**: 16.7% de erro cada
- **Padrão**: Predominância de **Falsos Negativos** (predição conservadora)

**Causa raiz**: Dataset pequeno (297 amostras) com classe minoritária (137 doentes) leva a:

- Variabilidade alta entre folds
- Modelo subestima a probabilidade de doença em alguns folds
- Falta de representação de padrões minoritários em certos folds

---

## ✨ Solução: SMOTE Híbrido

### O que é SMOTE?

**SMOTE** (Synthetic Minority Over-sampling Technique) gera amostras sintéticas da classe minoritária interpolando entre vizinhos próximos, melhorando a generalização.

### Implementação Híbrida

A função `smote_hybrid()` foi desenvolvida para dados mistos (numéricos + categóricos):

```python
def smote_hybrid(X_num, X_cat, y, sampling_strategy=0.5, k_neighbors=5, random_state=42)
```

**Algoritmo**:

1. Identifica classe minoritária (doentes) e majoritária (saudáveis)
2. Calcula número de amostras sintéticas: `n_synthetic = max(0, int(n_majority * sampling_strategy) - n_minority)`
3. Para cada amostra sintética:
   - Seleciona aleatoriamente uma amostra minoritária
   - Encontra um vizinho próximo via KNN (usando features numéricas)
   - **Features numéricas**: Interpolação linear com peso aleatório
   - **Features categóricas**: Seleção aleatória entre as duas amostras
4. Concatena dados originais com sintéticos

### Parâmetros

- `sampling_strategy`: Razão de oversampling (0.5 = 50% da classe majoritária)
  - Valor padrão: 0.8 (gera ~80% de amostras sintéticas)
  - Otimizado via Optuna durante treinamento
- `k_neighbors`: Número de vizinhos para KNN (padrão: 5)
- `random_state`: Seed para reprodutibilidade

---

## 🔧 Integração no Pipeline

### Célula 5: Teste de SMOTE

- Demonstra funcionamento com `sampling_strategy=0.8`
- Mostra distribuição antes/depois

### Célula 7: Otimização com SMOTE

- Novo parâmetro: `smote_sampling_strategy` (otimizado entre 0.5 e 1.0)
- SMOTE aplicado em cada fold da validação cruzada interna
- Melhora a robustez da busca de hiperparâmetros

### Célula 8: Experimento Final

- SMOTE aplicado em cada fold do experimento final
- Seed diferente por fold: `RANDOM_STATE + fold` (garante reprodutibilidade com variação)
- Rastreamento de informações: tamanho original → tamanho com SMOTE

---

## 📊 Impacto Esperado

### Antes (sem SMOTE)

- Folds problemáticos com alta variância
- Falsos Negativos predominantes
- Acurácia média: ~84% (com threshold 0.5)

### Depois (com SMOTE)

- Melhor representação da classe minoritária
- Modelo aprende padrões mais robustos
- Redução de Falsos Negativos esperada
- Acurácia esperada: ~87-90% (com threshold otimizado)

---

## 🚀 Como Usar

### Executar Experimento Completo

1. Execute **Célula 1**: Instalação
2. Execute **Célula 2**: Imports
3. Execute **Célula 3**: Carregamento de dados
4. Execute **Célula 3.1**: Seleção de features
5. Execute **Célula 5**: Teste de SMOTE (opcional, para verificação)
6. Execute **Célula 7**: Otimização com SMOTE (~30-60 min)
7. Execute **Célula 8**: Experimento final com SMOTE

### Ajustar Parâmetros de SMOTE

```python
# Na célula 5 (teste)
X_num_test, X_cat_test, y_test = smote_hybrid(
    X_num, X_cat, y,
    sampling_strategy=0.9,  # Aumentar para mais amostras sintéticas
    k_neighbors=7            # Aumentar para maior suavidade
)

# Na célula 7 (otimização)
# Modificar range de busca:
'smote_sampling_strategy': trial.suggest_float('smote_sampling_strategy', 0.6, 1.0)
```

---

## 📈 Monitoramento

Cada fold imprime:

```
Fold 1
  Aplicando SMOTE...
  SMOTE: Gerando 45 amostras sintéticas
    Classe minoritária: 1 (45 amostras)
    Classe majoritária: 0 (60 amostras)
    Dataset original: 105 amostras
    Dataset com SMOTE: 150 amostras
    Nova distribuição: [60 90]
  AUC: 0.8234 | Acc (0.5): 0.8667 | Acc (Opt 0.62): 0.9000
```

Resumo final:

```
=== INFORMAÇÕES DE SMOTE ===
Fold 1: 105 → 150 (+45 sintéticas)
Fold 2: 105 → 150 (+45 sintéticas)
...
```

---

## 🔗 Referências

- **SMOTE Original**: Chawla et al., 2002 - "SMOTE: Synthetic Minority Over-sampling Technique"
- **Dados**: Detrano et al., 1989 - "International application of a new probability algorithm for the diagnosis of coronary artery disease"
- **TabM**: Gorishniy et al., 2021 - "TabM: An Empirical Study of Supervised Learning for Tabular Data"

---

## ✅ Checklist de Implementação

- [x] Função `smote_hybrid()` para dados mistos
- [x] Integração na célula de otimização (Optuna)
- [x] Integração no experimento final
- [x] Rastreamento de informações por fold
- [x] Seed diferente por fold para variação
- [x] Documentação completa
- [ ] Comparação antes/depois (executar experimento)
- [ ] Análise de impacto em folds problemáticos (Fold 4, 6)
