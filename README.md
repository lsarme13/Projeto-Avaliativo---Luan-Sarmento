# Situação de Aprendizagem (Projeto Avaliativo) - Módulo 1 - Semana 14

Curso: Fundamentos de Programação, Dados e Machine Learning
Aluno: Luan Sarmento Orsi da Silva


# Pipeline Preditivo de Risco de Crédito

Previsão de inadimplência em empréstimos com **KNN** e **Árvore de Decisão**, passando por um
pipeline completo de análise exploratória, tratamento de dados, feature engineering, escolha de
hiperparâmetros com validação cruzada e veredito de negócio.

---

## O Problema

Um banco precisa decidir, no momento da concessão, se um cliente vai quitar o empréstimo
(`loan_status = 0`) ou se tornará inadimplente (`loan_status = 1`). O erro tem dois lados, e cada
um custa dinheiro de forma diferente:

| Erro | O que aconteceu | Custo para o banco |
|---|---|---|
| **Falso Negativo (FN)** | Liberou crédito para quem **não** paga | Todo o valor do contrato |
| **Falso Positivo (FP)** | Recusou crédito para quem **paga** | A margem que deixou de ganhar |

Responder "paga" para todo mundo já acerta 78% da base. O objetivo do modelo é quebrar esse atalho.

---

## Resumo Executivo

### Principais insights da EDA

- **32.581 linhas × 12 variáveis**, com apenas 1,03% de células nulas. 165 linhas duplicadas
  foram removidas, resultando em 32.416 linhas.
- **Desbalanceamento de 3,58:1** — 78,18% de pagos contra 21,82% de inadimplentes.
- **Nulos** concentrados em `loan_int_rate` (9,56%) e `person_emp_length` (2,75%).
- **Sinais mais fortes:** `loan_percent_income` (r = 0,379) e `loan_int_rate` (r = 0,335). Nenhuma
  variável isolada tem correlação forte com o alvo, o que indica que o risco é multicausal.
- **Multicolinearidade:** `person_age` e `cb_person_cred_hist_length` têm r = 0,859 — praticamente
  a mesma informação.
- **Erros de registro:** idade de 144 anos e 123 anos de estabilidade no emprego.
- **Outliers legítimos:** `person_income` tem assimetria de 32,86 e 1.478 clientes acima do limite
  de Tukey. São extremos estatísticos, mas clientes de alta renda reais.

### Veredito

A Árvore de Decisão vai para produção. Acerta **90,10%** da base de teste, tem F1 de 76,86% na
classe dos inadimplentes e é mais barata em **14 dos 16 cenários financeiros** simulados. O KNN
acerta um pouco mais dos inadimplentes (recall de 77,50% contra 75,18%), mas custa US$ 4,8 milhões
a mais em crédito indevidamente barrado para evitar 33 defaults.

---

## Dicionário de Dados

Base: `credit_risk_dataset.csv` — 32.581 linhas × 12 variáveis.

| Variável | Tipo | Descrição |
|---|---|---|
| `person_age` | Numérica | Idade do cliente (anos) |
| `person_income` | Numérica | Renda anual do cliente (USD) |
| `person_emp_length` | Numérica | Tempo de estabilidade no emprego (anos) |
| `person_home_ownership` | Categórica | Situação do imóvel do cliente |
| `loan_intent` | Categórica | Finalidade do empréstimo |
| `loan_grade` | Categórica | Faixa de risco do empréstimo (A–G) |
| `loan_amnt` | Numérica | Valor solicitado do empréstimo (USD) |
| `loan_int_rate` | Numérica | Taxa de juros anual (%) |
| `loan_status` | **Alvo** | `0` = Pago, `1` = Inadimplente |
| `loan_percent_income` | Numérica | Percentual da renda comprometida (percent) |
| `cb_person_default_on_file` | Categórica | Default registrado no histórico de crédito |
| `cb_person_cred_hist_length` | Numérica | Tamanho do histórico de crédito (anos) |
| `comprometimento_renda` | Numérica (criada) | `(loan_amnt / person_income) × 100` |

### Colunas auxiliares geradas no pipeline

| Coluna | Finalidade |
|---|---|
| `loan_status_rotulo` | Rótulo textual do alvo, usado apenas nos gráficos e relatórios |
| `loan_int_rate_ausente` | Indicador binário de ausência, preservado porque a ausência em si carrega sinal |

### Sobre a coluna calculada

`comprometimento_renda` traduz a comparação entre valor solicitado e renda em um único número
interpretável — algo que a soma ou a subtração não permitiria, por serem grandezas de unidades
diferentes. Ela é a variável com maior correlação com o alvo de toda a base (r = 0,3862).

Sua correlação com `loan_percent_income` é de 0,9989: as duas carregam a mesma informação em escalas
diferentes. Manter ambas é uma decisão consciente, com um custo e um benefício de cada lado — no
KNN o mesmo sinal entra pesado duas vezes na distância euclidiana, e na Árvore a coluna duplicada é
simplesmente ignorada em um dos ramos, sem prejuízo.

---

## Metodologia

| Fase | O que foi feito |
|---|---|
| **1 — EDA** | Tamanho, tipos, `.describe()`, categorias, nulos, duplicatas, **4 figuras** (histogramas das variáveis numéricas, boxplot de idade por classe, desbalanceamento do alvo e mapa de calor de Pearson), ranking de correlações com o alvo e leitura analítica que guiou todo o pipeline. |
| **2 — Data Prep** | Remoção das 165 duplicatas, clipping de valores impossíveis (idade, estabilidade, histórico), **imputação pela média** em `loan_int_rate` (simétrica, r = 0,20) e **pela mediana** em `person_emp_length` (assimétrica, r = 1,40), ambas com indicador binário de ausência. Outliers diagnosticados por Tukey e **mantidos**, por serem plausíveis. |
| **3 — Feature Engineering** | Criação de `comprometimento_renda = (loan_amnt / person_income) × 100`, após a imputação, para não gerar `NaN` nem divisão por zero. |
| **4 — Encoding, Balanceamento e Escalonamento** | One-Hot Encoding com `drop='first'` e `handle_unknown='ignore'` (15 categorias → 24 features); split estratificado 80/20; **Random Under Sampling apenas no treino** (25.932 → 11.342 linhas, 50/50); `StandardScaler` exclusivamente nas 8 variáveis contínuas, com `fit_transform` no treino e `transform` no teste. |
| **5 — Modelagem e Overfitting** | Varredura de K = 3, 5, 7, 9, 11, 15, 21 e `max_depth` = 3, 5, 7, 10, ilimitada. Cada configuração é medida em três bases: **validação cruzada de 5 dobras** (critério de escolha), treino inteiro e teste. |
| **6 — Veredito** | `ClassificationReport` e matrizes de confusão das duas configurações vencedoras, seguida de análise financeira em 16 cenários de perda e margem. |

### Decisões que sustentam o rigor do pipeline

- O split estratificado acontece antes da imputação. Remover duplicatas e aplicar limites de
  domínio são regras que não usam o alvo, então valem para a base inteira. Imputação é diferente:
  ela *calcula* um parâmetro e o aplica a todas as linhas. Se esse cálculo incluir as linhas que
  depois serão o teste, o modelo passa a conocer a resposta que deveria medir.
- O teste nunca participa de nenhuma escolha. Nem da imputação, nem do balanceamento, nem da
  seleção de hiperparâmetros. Por isso a métrica da Fase 6 mede generalização, não memória.
- A grade de busca vai além do mínimo de quatro valores. Com K = 3, 5, 7 e 9, o F1 de teste
  subia a cada passo e a curva terminava no fim da busca, sem mostrar onde o ganho acaba. A grade
  estendida revelou que o ponto ideal é **interior** nos dois modelos.
- O balanceamento fica só no treino. O teste preserva a proporção real (78,13% / 21,87%), que é
  contra ela que o modelo será julgado em produção.
- Uma única semente (`42`) é reutilizada na separação, no balanceamento e na validação cruzada,
  de modo que qualquer reexecução reproduz os mesmos números.

---

## Resultados

Base de teste: 6.484 clientes, com 1.418 inadimplentes.

| Métrica | KNN (K = 11) | Árvore (profundidade 10) |
|---|---|---|
| Acurácia | 82,60% | **90,10%** |
| Precisão dos inadimplentes | 57,60% | **78,61%** |
| Recall dos inadimplentes | **77,50%** | 75,18% |
| F1 dos inadimplentes | 66,09% | **76,86%** |
| F1 médio na validação cruzada | 79,60% | **82,23%** |
| Falsos negativos | **319** | 352 |
| Falsos positivos | 809 | **290** |

### Diagnóstico do overfitting

O **gap** é a métrica de treino menos a de teste: quanto maior, maior a chance de o modelo ter apenas
decorado os dados. Para a Árvore, a acurácia não serve a esse fim, porque o treino foi balanceado
(50/50) e o teste não (78/22) — o gap de acurácia sai negativo em quatro das cinco configurações, e
isso é o comportamento esperado, não um problema. O diagnóstico foi feito pelo **gap de F1** e pela
evolução das curvas.

| KNN | F1 treino | F1 teste | Gap | | Árvore | F1 treino | F1 teste | Gap |
|---|---|---|---|---|---|---|---|---|
| K = 3 | 88,67% | 62,27% | **26,40** | | Prof. 3 | 77,69% | 72,04% | 5,65 |
| **K = 11** | 82,75% | 66,09% | **16,66** | | **Prof. 10** | 86,11% | 76,86% | **9,25** |
| K = 21 | 80,92% | 66,65% | 14,27 | | Ilimitada | 100,00% | 65,58% | **34,42** |

O caso mais evidente de overfitting é a árvore **sem limite de profundidade**: 100% de acerto no
treino e 65,58% de F1 no teste. A árvore também mostra que gap pequeno não é sinônimo de bom
desempenho — a profundidade 3 tem o menor gap (5,65) e o pior F1 de validação (77,75%), porque
deixa de recuperar cerca de um terço dos inadimplentes: isso é subajuste.

No KNN, o contraste entre treino e teste é o ponto central: o **F1 de teste continua subindo até
K = 21**, mas a validação cruzada mostra que a partir de K = 9 já é platô. Se o teste participasse da
escolha, K = 21 seria selecionado — e ele erra justamente a classe que o banco precisa acertar.

---

## Análise Financeira e Veredito

O recall dos inadimplentes é a métrica que mais pesa na matriz de confusão, mas não é o único fator
devisivo: FN e FP precisam ser convertidos em dinheiro. Foram simulados 16 cenários combinando taxa 
de perda (40% a 100%) e margem anual (5% a 20%).

| | KNN (K = 11) | Árvore (profundidade 10) |
|---|---|---|
| Capital em risco (defaults liberados) | **US$ 3.436.826** | US$ 3.792.359 |
| Crédito barrado de bons clientes | US$ 7.501.183 | **US$ 2.688.929** |

**A Árvore é mais barata em 14 dos 16 cenários.** O KNN só vence nos dois cenários mais fechados —
perda de 80% ou 100% contra margem de 5%, isto é, um default custando 16 a 20 vezes a margem de um
cliente recusado. Ele só compensaria se o custo de um default superasse **13,54 vezes** a margem
perdida com um cliente recusado, relação que não se sustenta em nenhum dos outros 14 cenários.

**Por que a Árvore de Decisão foi escolhida:**

1. **Custa menos na esmagadora maioria dos cenários.** A diferença de recall é de 2,32 pontos contra
   uma economia de US$ 4,8 milhões em crédito barrado.
2. **É explicável.** A decisão sai de cortes em atributos, e é possível apontar exatamente qual regra
   recusou o pedido. Um KNN que pondera 11 vizinhos não permite explicar nada.
3. **É mais barata de operar.** O KNN exige padronização, guarda a base de treino inteira e faz 11
   comparações de distância por cliente; a árvore percorre no máximo 10 cortes.

### Contraponto ao modelo de Árvore de Decisão

- A Árvore ainda deixa passar 352 dos 1.418 inadimplentes da carteira de teste. Com recall de
  75,18%, esse é o problema real que sobra.
---