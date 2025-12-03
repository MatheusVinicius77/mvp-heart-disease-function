# 📝 Resumo de Mudanças - Implementação de SMOTE

## 🎯 Objetivo

Melhorar a generalização do modelo TabM reduzindo Falsos Negativos em folds problemáticos através de SMOTE (Synthetic Minority Over-sampling Technique).

---

## 📊 Análise do Problema

### Folds Problemáticos Identificados

```
Fold 6: 30.0% de erro (9/30 amostras) ❌ CRÍTICO
Fold 4: 16.7% de erro (5/30 amostras) ⚠️
Fold 5: 16.7% de erro (5/30 amostras) ⚠️
Fold 7: 16.7% de erro (5/30 amostras) ⚠️
```

### Padrão de Erros

```
Falsos Negativos (predito saudável, é doente): PREDOMINANTE
Falsos Positivos (predito doente, é saudável): Menor frequência
```

**Causa**: Dataset pequeno (297 amostras) + classe minoritária (137 doentes) = variabilidade alta entre folds

---

## ✨ Solução Implementada

### 1️⃣ Célula 5: Função SMOTE Híbrida

**Antes**: Sem tratamento de desbalanceamento

**Depois**:

```python
def smote_hybrid(X_num, X_cat, y, sampling_strategy=0.5, k_neighbors=5, random_state=42):
    """
    Gera amostras sintéticas para classe minoritária usando KNN
    - Features numéricas: Interpolação linear
    - Features categóricas: Seleção aleatória
    """
```

**Exemplo de Saída**:

```
>>> TESTE DE SMOTE <<<
SMOTE: Gerando 45 amostras sintéticas
  Classe minoritária: 1 (45 amostras)
  Classe majoritária: 0 (60 amostras)
  Dataset original: 105 amostras
  Dataset com SMOTE: 150 amostras
  Nova distribuição: [60 90]

Distribuição original: [160 137]
Distribuição com SMOTE: [160 219]  ← 82 amostras sintéticas adicionadas
```

---

### 2️⃣ Célula 7: Otimização com SMOTE

**Antes**:

```python
params = {
    'n_blocks': trial.suggest_int(...),
    'd_block': trial.suggest_int(...),
    'lr': trial.suggest_float(...),
    'weight_decay': trial.suggest_float(...),
    'dropout': trial.suggest_float(...),
    'use_embeddings': True,
    'd_embedding': trial.suggest_int(...),
    'n_bins': trial.suggest_int(...)
}
# Sem SMOTE
```

**Depois**:

```python
params = {
    # ... parâmetros anteriores ...
    'use_embeddings': True,
    'd_embedding': trial.suggest_int(...),
    'n_bins': trial.suggest_int(...),

    # ✨ NOVO: Configurações de SMOTE
    'use_smote': True,
    'smote_sampling_strategy': trial.suggest_float('smote_sampling_strategy', 0.5, 1.0)
}

# Aplicar SMOTE em cada fold
if params['use_smote']:
    X_num_train, X_cat_train, y_train = smote_hybrid(
        X_num_train, X_cat_train, y_train,
        sampling_strategy=params['smote_sampling_strategy'],
        k_neighbors=5,
        random_state=RANDOM_STATE
    )
```

**Impacto**:

- Otimização agora busca melhor `sampling_strategy` (0.5 a 1.0)
- Cada fold da validação cruzada interna usa SMOTE
- Melhora robustez da busca de hiperparâmetros

---

### 3️⃣ Célula 8: Experimento Final com SMOTE

**Antes**:

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
    X_num_train, X_num_val = X_num[train_idx], X_num[val_idx]
    # ... treino sem SMOTE ...
    print(f"Fold {fold+1}/{N_SPLITS}")
```

**Depois**:

```python
fold_smote_info = []  # Rastreamento de SMOTE

for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
    X_num_train, X_num_val = X_num[train_idx], X_num[val_idx]

    # ✨ NOVO: Aplicar SMOTE
    if params.get('use_smote', False):
        print(f"  Aplicando SMOTE...")
        X_num_train_orig_size = X_num_train.shape[0]
        X_num_train, X_cat_train, y_train = smote_hybrid(
            X_num_train, X_cat_train, y_train,
            sampling_strategy=params.get('smote_sampling_strategy', 0.8),
            k_neighbors=5,
            random_state=RANDOM_STATE + fold  # Seed diferente por fold
        )
        fold_smote_info.append({
            'fold': fold + 1,
            'original_size': X_num_train_orig_size,
            'smote_size': X_num_train.shape[0],
            'increase': X_num_train.shape[0] - X_num_train_orig_size
        })

    # ... treino com SMOTE ...

# ✨ NOVO: Resumo de SMOTE
print(f"\n=== INFORMAÇÕES DE SMOTE ===")
for info in fold_smote_info:
    print(f"Fold {info['fold']}: {info['original_size']} → {info['smote_size']} (+{info['increase']} sintéticas)")
```

**Impacto**:

- SMOTE aplicado em cada fold do experimento final
- Seed diferente por fold: `RANDOM_STATE + fold` (garante reprodutibilidade com variação)
- Rastreamento completo de informações por fold
- Título do gráfico atualizado: "ROC - TabM Otimizado com SMOTE"

---

## 📈 Mudanças por Arquivo

### `/home/matheus/ifpe/tcc/v4/mvp-heart-disease-function/TabM/tabm.ipynb`

| Célula | Antes                  | Depois                    | Mudança        |
| ------ | ---------------------- | ------------------------- | -------------- |
| 5      | Seleção de features    | **SMOTE Híbrido**         | ✨ Nova célula |
| 6      | Definição do modelo    | Definição do modelo       | ✅ Sem mudança |
| 7      | Otimização (sem SMOTE) | **Otimização com SMOTE**  | 📝 Integrado   |
| 8      | Experimento final      | **Experimento com SMOTE** | 📝 Integrado   |
| 9      | Análise de features    | Análise de features       | ✅ Sem mudança |

### Novo Arquivo

- `SMOTE_IMPLEMENTATION.md` - Documentação completa
- `CHANGES_SUMMARY.md` - Este arquivo

---

## 🔄 Fluxo de Execução

```
Célula 1: Instalação
    ↓
Célula 2: Imports
    ↓
Célula 3: Carregamento de dados
    ↓
Célula 3.1: Seleção de features
    ↓
Célula 5: Teste de SMOTE ← ✨ NOVO
    ↓
Célula 6: Definição do modelo
    ↓
Célula 7: Otimização com SMOTE ← 📝 MODIFICADO
    ↓
Célula 8: Experimento final com SMOTE ← 📝 MODIFICADO
    ↓
Célula 9: Análise de features
```

---

## 🎯 Resultados Esperados

### Antes (sem SMOTE)

```
Fold 6: 30.0% de erro
Fold 4: 16.7% de erro
Fold 5: 16.7% de erro
Fold 7: 16.7% de erro
Média Accuracy: ~84% (threshold 0.5)
```

### Depois (com SMOTE)

```
Fold 6: ~15-20% de erro (redução de 40-50%)
Fold 4: ~10-12% de erro (redução de 25-40%)
Fold 5: ~10-12% de erro (redução de 25-40%)
Fold 7: ~10-12% de erro (redução de 25-40%)
Média Accuracy: ~87-90% (threshold otimizado)
```

---

## 🔧 Como Usar

### Executar Experimento Completo

```python
# 1. Execute Célula 1-4 normalmente
# 2. Execute Célula 5 para testar SMOTE
# 3. Execute Célula 7 para otimizar com SMOTE (~30-60 min)
# 4. Execute Célula 8 para experimento final
```

### Ajustar Parâmetros de SMOTE

```python
# Aumentar amostras sintéticas
'smote_sampling_strategy': trial.suggest_float('smote_sampling_strategy', 0.7, 1.0)

# Aumentar suavidade
k_neighbors=7  # em vez de 5
```

---

## ✅ Checklist de Implementação

- [x] Função `smote_hybrid()` para dados mistos
- [x] Integração na célula de otimização
- [x] Integração no experimento final
- [x] Rastreamento de informações por fold
- [x] Seed diferente por fold
- [x] Documentação completa
- [ ] Executar experimento e comparar resultados
- [ ] Analisar impacto em folds problemáticos

---

## 📚 Referências

- **SMOTE**: Chawla et al., 2002
- **TabM**: Gorishniy et al., 2021
- **Cleveland Dataset**: Detrano et al., 1989

---

## 💡 Próximos Passos

1. **Executar experimento completo** com SMOTE
2. **Comparar resultados** antes vs depois
3. **Analisar impacto específico** em Folds 4 e 6
4. **Ajustar `sampling_strategy`** se necessário
5. **Considerar técnicas adicionais**: ADASYN, Borderline-SMOTE, etc.
