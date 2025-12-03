# 🏥 TabM para Diagnóstico de Doença Cardíaca: Exploração de Transformers em Dados Tabulares Clínicos

## 📋 Resumo Executivo

Este trabalho explora a aplicação de **TabM** (Transformer para Dados Tabulares) no diagnóstico de doença cardíaca coronariana usando o dataset Cleveland. TabM é uma arquitetura moderna que combina mecanismos de atenção com embeddings numéricos para capturar relações complexas em dados tabulares.

**Objetivo Principal**: Investigar a efetividade de TabM em diagnóstico de doença cardíaca, explorando como transformers podem capturar relações entre features clínicas em dados tabulares médicos.

**Diferencial**: Primeira aplicação de TabM em contexto de diagnóstico cardíaco - gap na literatura.

---

## 🎯 Problema de Pesquisa

### Questão Principal

**"TabM é efetivo para diagnóstico de doença cardíaca em dados tabulares clínicos?"**

### Contexto

- Transformers revolucionam IA, mas são pouco explorados em dados tabulares médicos
- TabM é uma arquitetura recente (2021) com poucos estudos em medicina
- Dataset Cleveland: 297 pacientes (160 saudáveis, 137 doentes)
- Alguns folds apresentam alta taxa de erro (até 30%)

### Gap na Literatura

- **Nenhum estudo anterior aplicou TabM em diagnóstico de doença cardíaca**
- Falta de benchmarks de transformers em dados médicos tabulares
- Necessidade de validação em contextos clínicos reais

### Desafios

- Dataset pequeno (297 amostras) - TabM foi desenvolvido para datasets maiores
- Desbalanceamento de classes (54% vs 46%)
- Variabilidade entre folds - alguns com distribuição atípica
- Necessidade de capturar relações complexas entre features clínicas

### Hipóteses (Benchmarking Descritivo)

**H1 (Hipótese Alternativa)**:
TabM é viável para diagnóstico de doença cardíaca no dataset Cleveland, demonstrando desempenho competitivo com o estado da arte reportado na literatura, alcançando:

- **Acurácia** dentro da faixa esperada (≥ 0.85)
- **F1-Score** competitivo (≥ 0.82)
- **AUC-ROC** comparável (≥ 0.85)
- **Precision** adequada para contexto clínico (≥ 0.83)
- **Recall** elevado para minimizar falsos negativos (≥ 0.85)

**H0 (Hipótese Nula)**:
TabM não alcança desempenho competitivo com o estado da arte em diagnóstico de doença cardíaca, apresentando resultados significativamente inferiores em múltiplas métricas (Acurácia < 0.85, F1-Score < 0.82, AUC-ROC < 0.85, Precision < 0.83, Recall < 0.85).

---

## Solução Proposta

### 0. Feature Selection

Seleção cuidadosa de features clínicas relevantes:

- **1 Feature Numérica**: `oldpeak` (ST depression induzida por exercício)
- **5 Features Categóricas**: `cp`, `exang`, `slope`, `ca`, `thal`
- **Redução**: De 13 para 6 features (54% redução)
- **Benefício**: Reduz ruído, melhora interpretabilidade, acelera treinamento

### 1. Arquitetura TabM

TabM combina múltiplas técnicas para dados tabulares:

**Componentes Principais**:

- **Embeddings Numéricos**: PiecewiseLinearEmbeddings para features numéricas
- **Mecanismo de Atenção**: Multi-head attention para capturar relações entre features
- **Blocos Transformer**: Múltiplos blocos para aprender representações hierárquicas
- **Ensemble**: k modelos independentes para robustez

**Hiperparâmetros Chave**:

- `n_blocks`: Número de blocos transformer (1-4)
- `d_block`: Dimensionalidade de cada bloco (64-512)
- `d_embedding`: Dimensionalidade dos embeddings numéricos (8-32)
- `n_bins`: Número de bins para discretização (2-64)

### 2. SMOTE Híbrido (Técnica Complementar)

Implementação que trata dados numéricos e categóricos para balanceamento:

**Features Numéricas**: Interpolação linear entre vizinhos próximos
**Features Categóricas**: Seleção aleatória

**Parâmetro Otimizado**: `smote_sampling_strategy` (0.86-1.0)

### 3. Otimização com Optuna

- 100 trials para encontrar melhores hiperparâmetros
- Validação cruzada interna (3 folds)
- Critério: Maximizar AUC-ROC

### 4. Validação Rigorosa

- 10-Fold Stratified Cross-Validation
- SMOTE aplicado apenas no conjunto de treinamento
- Validação em dados originais (sem SMOTE)
- Threshold otimizado por fold (Youden's J)

---

## 📊 Dataset: Cleveland Heart Disease

### Características

- **Fonte**: UCI Machine Learning Repository
- **Instâncias**: 303 originais → 297 após limpeza
- **Features**: 13 atributos clínicos
- **Target**: Diagnóstico de doença coronariana (binário)

### Features Selecionadas

| Tipo        | Features                                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| Numéricas   | `oldpeak` (ST depression induzida por exercício)                                                          |
| Categóricas | `cp` (tipo de dor), `exang` (angina induzida), `slope` (inclinação ST), `ca` (vasos), `thal` (talassemia) |

### Distribuição de Classes

```
Classe 0 (Saudáveis):  160 (54%)
Classe 1 (Doentes):    137 (46%)
Razão:                 1.17:1
```

### Arquitetura do Modelo

```
Input (1 numérica + 5 categóricas)
    ↓
PiecewiseLinearEmbeddings (features numéricas)
    ↓
TabM Blocks (múltiplos blocos transformer)
    ↓
Multi-Head Attention (captura relações entre features)
    ↓
Dropout (regularização)
    ↓
Output (probabilidade de doença)
```

### Hiperparâmetros Otimizados (Optuna)

```json
{
  "n_blocks": 2,
  "d_block": 384,
  "lr": 0.00401,
  "weight_decay": 0.00309,
  "dropout": 0.0698,
  "d_embedding": 28,
  "n_bins": 17,
  "smote_sampling_strategy": 0.906,
  "use_embeddings": true,
  "use_smote": true
}
```

### Processo de Otimização

1. **Optuna**: 100 trials com 3-fold CV interna
2. **Critério**: Maximizar AUC-ROC
3. **Pruning**: Early stopping se performance não melhora
4. **Tempo**: ~30-60 minutos

### Validação Final

1. **10-Fold Stratified Cross-Validation**
2. **SMOTE** aplicado em cada fold (balanceamento)
3. **Threshold otimizado** por fold (Youden's J)
4. **Métricas**: AUC-ROC, Accuracy, Precision, Recall, F1-score

---

## 📈 Resultados

### SMOTE - Teste

```
>>> TESTE DE SMOTE <<<
Usando sampling_strategy=1.0 (balanceamento perfeito)
SMOTE: Gerando 23 amostras sintéticas
  Classe minoritária: 1 (137 amostras)
  Classe majoritária: 0 (160 amostras)
  Dataset original: 297 amostras
  Dataset com SMOTE: 320 amostras
  Nova distribuição: [160 160]
```

### Otimização com Optuna

- **Melhor AUC-ROC**: ~0.85 (validação interna)
- **Trials completados**: 100
- **Melhor sampling_strategy**: 0.894

### Impacto Esperado em Folds Problemáticos

| Fold | Antes      | Depois       | Melhoria |
| ---- | ---------- | ------------ | -------- |
| 6    | 30.0% erro | ~12-15% erro | -50-60%  |
| 4    | 16.7% erro | ~8-10% erro  | -40-50%  |
| 5    | 16.7% erro | ~8-10% erro  | -40-50%  |
| 7    | 16.7% erro | ~8-10% erro  | -40-50%  |

---

## 📝 Questões de Pesquisa

### Questão Principal

**"Qual é o desempenho de TabM em diagnóstico de doença cardíaca no dataset Cleveland e como se compara com o estado da arte reportado na literatura?"**

### Questões Secundárias

1. **"TabM alcança métricas competitivas com a literatura?"**

   - Métricas: Acurácia, F1-Score, AUC-ROC, Precision, Recall
   - Análise: Comparação com benchmarks da literatura

2. **"TabM generaliza bem em folds com diferentes distribuições de dados?"**

   - Métrica: Desvio padrão e variância de performance entre folds
   - Análise: Robustez em folds problemáticos vs normais

3. **"Qual é o impacto de SMOTE na performance de TabM em dados desbalanceados?"**

   - Análise: Performance com/sem SMOTE
   - Métricas: Recall (minimizar falsos negativos), Precision

4. **"Como o mecanismo de atenção do TabM captura relações entre features clínicas?"**
   - Análise: Visualização de attention weights
   - Métrica: Correlação com importância clínica esperada

---

## 🔧 Como Executar

### Pré-requisitos

```bash
pip install tabm rtdl_num_embeddings optuna scikit-learn torch pandas numpy matplotlib
```

### Execução Passo a Passo

#### 1. Célula 1-4: Setup e Carregamento

```python
# Instalação, imports, carregamento de dados, seleção de features
# Tempo: ~30 segundos
```

#### 2. Célula 5: Teste de SMOTE

```python
# Verifica funcionamento de SMOTE com sampling_strategy=1.0
# Tempo: ~5 segundos
# Esperado: 23 amostras sintéticas geradas
```

#### 3. Célula 7: Otimização com Optuna

```python
# Busca melhores hiperparâmetros (100 trials)
# Tempo: 30-60 minutos
# Salva em: best_params_tabm.json
```

#### 4. Célula 8: Experimento Final

```python
# 10-fold CV com melhores parâmetros
# Tempo: 20-40 minutos
# Gera gráfico ROC com intervalo de confiança
```

#### 5. Célula 9: Análise de Features

```python
# Incerteza e importância de features
# Tempo: ~5 minutos
```

---

## 📁 Estrutura de Arquivos

```
TabM/
├── tabm.ipynb                          # Notebook principal
├── best_params_tabm.json               # Hiperparâmetros otimizados
├── README.md                           # Este arquivo
├── SMOTE_IMPLEMENTATION.md             # Detalhes técnicos de SMOTE
├── SMOTE_LITERATURE_REVIEW.md          # Revisão de literatura
└── CHANGES_SUMMARY.md                  # Resumo de mudanças
```

---

## 🔍 Análise Detalhada de Folds

### Fold 6 (Mais Problemático)

```
Tamanho: 30 amostras
Classe 0: 16 (53.3%)
Classe 1: 14 (46.7%)
Erros: 9/30 (30.0%)
Padrão: 6 Falsos Negativos, 3 Falsos Positivos

Características de erro:
- oldpeak: média=-0.073 (ligeiramente abaixo da média)
- cp: distribuição desbalanceada (2 tipo 0, 14 tipo 3)
- thal: distribuição desbalanceada (15 tipo 0, 14 tipo 2)
```

### Fold 4

```
Tamanho: 30 amostras
Classe 0: 16 (53.3%)
Classe 1: 14 (46.7%)
Erros: 5/30 (16.7%)
Padrão: 2 Falsos Negativos, 3 Falsos Positivos

Características:
- ca: distribuição muito concentrada (23 tipo 0)
- slope: distribuição desbalanceada (18 tipo 0)
```

---

## 🎓 Contribuições Técnicas

### 1. Primeira Aplicação de TabM em Diagnóstico Cardíaco

- **Gap na Literatura**: Nenhum estudo anterior aplicou TabM em contexto de doença cardíaca
- Benchmarking de TabM contra estado da arte reportado na literatura
- Validação rigorosa com 10-fold stratified cross-validation
- Mapeamento de desempenho para futuros pesquisadores

### 2. Avaliação com Múltiplas Métricas

- Análise abrangente: Acurácia, F1-Score, AUC-ROC, Precision, Recall
- Foco em Recall para minimizar falsos negativos (crítico em diagnóstico médico)
- Comparação com estado da arte usando múltiplas métricas

### 3. SMOTE Híbrido para Dados Mistos

- Implementação que trata numéricos e categóricos simultaneamente
- KNN para encontrar vizinhos representativos
- Interpolação inteligente para features numéricas

### 4. Otimização Integrada de TabM + SMOTE

- Otimização conjunta de hiperparâmetros via Optuna
- SMOTE aplicado em cada fold (não apenas no dataset completo)
- Melhora robustez em datasets pequenos e desbalanceados

### 5. Análise de Generalização em Folds Problemáticos

- Identificação e análise detalhada de folds com alta taxa de erro
- Avaliação de robustez em diferentes distribuições de dados
- Insights sobre comportamento de TabM em dados heterogêneos

---

## 📚 Referências Bibliográficas

### SMOTE

1. **Chawla et al., 2002** - "SMOTE: Synthetic Minority Over-sampling Technique"

   - Técnica original de oversampling

2. **Sowjanya & Mrudula, 2021** - "Effective treatment of imbalanced datasets in health care using modified SMOTE"

   - D-SMOTE e BP-SMOTE para dados médicos
   - Resultado: +3% em acurácia

3. **El-Sofany et al., 2024** - "A proposed technique for predicting heart disease using machine learning algorithms"
   - SMOTE + SHAP para diagnóstico de doença cardíaca
   - Dataset: Cleveland Heart Disease

### TabM

1. **Gorishniy et al., 2024** - "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (ICLR 2025)
   - https://arxiv.org/abs/2410.24210
   - Transformer para dados tabulares com ensemble parameter-efficient
   - Embeddings numéricos (PiecewiseLinearEmbeddings)
   - Multi-head attention para capturar relações entre features
   - Ensemble de k modelos independentes para robustez

### Heart Disease Dataset

1. **Detrano et al., 1989** - "International application of a new probability algorithm for the diagnosis of coronary artery disease"
   - Dataset original Cleveland
   - 303 pacientes, 76 atributos

---

## 🔍 Análise Detalhada de Folds

### Fold 6 (Mais Problemático)

```
Tamanho: 30 amostras
Classe 0: 16 (53.3%)
Classe 1: 14 (46.7%)
Erros: 9/30 (30.0%)
Padrão: 6 Falsos Negativos, 3 Falsos Positivos

Características de erro:
- oldpeak: média=-0.073 (ligeiramente abaixo da média)
- cp: distribuição desbalanceada (2 tipo 0, 14 tipo 3)
- thal: distribuição desbalanceada (15 tipo 0, 14 tipo 2)
```

### Fold 4

```
Tamanho: 30 amostras
Classe 0: 16 (53.3%)
Classe 1: 14 (46.7%)
Erros: 5/30 (16.7%)
Padrão: 2 Falsos Negativos, 3 Falsos Positivos

Características:
- ca: distribuição muito concentrada (23 tipo 0)
- slope: distribuição desbalanceada (18 tipo 0)
```

1. **Variantes de SMOTE**: Implementar D-SMOTE e BP-SMOTE
2. **Ensemble**: Combinar TabM com XGBoost e Random Forest
3. **Feature Engineering**: Adicionar interações entre features
4. **Explicabilidade**: Integrar SHAP para interpretabilidade
5. **Validação Externa**: Testar em datasets adicionais (Framingham, Hungarian)

### Pesquisa Adicional

1. Analisar impacto de diferentes `k_neighbors` em SMOTE
2. Comparar com undersampling e hybrid sampling
3. Estudar efeito de SMOTE em diferentes tamanhos de dataset
4. Investigar borderline-SMOTE para casos ambíguos

---

## 📝 Notas Importantes

### Limitações

- Dataset pequeno (297 amostras)
- Features limitadas (1 numérica, 5 categóricas)
- Validação apenas em dados Cleveland
- Sem validação externa em outros datasets

### Considerações Médicas

- Falso Negativo é mais crítico que Falso Positivo
- SMOTE melhora recall (reduz FN)
- Threshold otimizado por fold pode variar clinicamente
- Necessário validação clínica antes de deployment

### Reprodutibilidade

- Seed fixo: `RANDOM_STATE = 42`
- Seed diferente por fold: `RANDOM_STATE + fold`
- Todos os hiperparâmetros salvos em `best_params_tabm.json`
- Código totalmente determinístico

---

## 📞 Contato e Dúvidas

Para dúvidas sobre implementação, resultados ou metodologia, consulte:

- Documentação técnica: `SMOTE_IMPLEMENTATION.md`
- Revisão de literatura: `SMOTE_LITERATURE_REVIEW.md`
- Resumo de mudanças: `CHANGES_SUMMARY.md`

---

**Última atualização**: Dezembro 2025
**Status**: Experimento em progresso
**Versão**: 1.0
