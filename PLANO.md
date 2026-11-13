# Plano: representação de estados tabulares em RL offline (projeto da matéria)

Este plano é executado **de uma vez só**, numa única sessão. Ele substitui
todo o planejamento anterior (`docs/planos/`, ROTEIRO, ESPECIFICACAO,
DECISOES), que deixa de valer.

---

## 0. Regras para quem executa

1. **Não** fazer commit, push, merge nem PR. Não mexer em outras branches.
   Tudo fica como mudança local não commitada para o Pedro revisar.
2. O plano já está aprovado. Não parar para pedir aprovação entre etapas.
   Executar E0 → E12 em ordem.
3. **O agente não roda o experimento final.** Ele roda só testes, smoke tests
   (tier `dev`) e medições curtas de tempo, e entrega uma **estimativa** e
   **um comando único** que o Pedro roda depois. Proibido gerar os datasets
   `final`, rodar `baselines` no `final` ou rodar a matriz `final`, mesmo que
   pareça rápido.
4. Parar e perguntar **só** se: (a) um simulador não rodar no teste de
   viabilidade (E0) mesmo com correções pequenas de compatibilidade; (b) um
   teste contra verdade conhecida falhar e a correção exigir mudar o que este
   plano especifica; (c) a política de comportamento de um ambiente não for
   melhor que a aleatória depois dos ajustes permitidos.
   Qualquer outra dúvida: escolher o caminho mais simples coerente com este
   plano e registrar no resumo final.
5. Regras de código que continuam valendo (copiadas para o novo `CLAUDE.md`
   em E10):
   - toda conta nova (normalização, representação, estatística, montagem de
     estado e recompensa dos adaptadores) tem teste contra verdade conhecida;
   - nada falha em silêncio: pré-condição quebrada levanta erro claro;
   - hiperparâmetros, ε, tamanhos e limiares vêm de config;
   - seeds explícitas, nada de RNG global (`np.random.*`, `random.*` e
     `torch.manual_seed` globais proibidos, inclusive no código vendorizado,
     que é portado para geradores por instância);
   - o simulador nunca é usado em treino nem em seleção de checkpoint;
   - a representação é ajustada **só** no split `train`;
   - código vendorizado mantém a licença, ganha `NOTICE` e entra em
     `THIRD_PARTY_NOTICES.md`; toda mudança nele fica listada no `NOTICE`;
   - fato externo (API, licença, colunas) é conferido no código da fonte antes
     de virar código;
   - código, docstrings e docs em inglês; este plano em português;
   - sem shims, aliases, código morto ou comentado; código que fica morto sai.
6. Antes de escrever um helper, procurar se já existe no repositório.
7. Código baixado de fora fica numa pasta temporária fora do repositório até
   ser vendorizado. Rodar código de terceiros pode pedir permissão ao Pedro:
   pedir e esperar.

---

## 1. Objetivo

Pergunta: **uma representação vetorial melhor dos estados tabulares
(estruturados) melhora o retorno de políticas aprendidas offline?**

Protocolo: para cada ambiente, um dataset offline fixo, coletado no simulador
por uma política de comportamento. Cada método clássico (`bc`, `dqn`, `iql`,
`cql`; nenhum método novo) é treinado nesse dataset duas vezes, com as mesmas
seeds e os mesmos hiperparâmetros:

- com a observação crua (`representation = "raw"`);
- com a observação transformada por uma representação (`representation = "<nome>"`).

A métrica é o **retorno médio no simulador online** do checkpoint escolhido
pela loss de validação offline, mais o **score normalizado**
`100 · (R − R_random) / (R_behavior − R_random)`: 0 = política aleatória,
100 = política que gerou o dataset, > 100 = melhor que ela. **Sem OPE.**

As técnicas de representação de verdade (one-hot, quantis, PLE, embeddings
periódicos, autoencoders etc.) entram **depois**, pelo Pedro. Este plano
entrega a infraestrutura e duas representações mínimas: `raw` (identidade) e
`standardize` (z-score ajustado no `train`), que provam que o encaixe funciona
de ponta a ponta.

---

## 2. Ambientes e decisões

### 2.1 Os 4 ambientes

Escolhidos entre os ambientes da v1 do plano original com **simulador
verdadeiro vindo da fonte** (avaliação online sem ajustar simulador nosso), que
passam no contrato de tabularidade do plano original e que exigem menos
trabalho. Fontes conferidas em 09/11/2026. O plano original é apagado em
E1 e não faz parte deste repositório.

| Ambiente | Domínio | Colunas | Ações | H | Fonte e licença | Trabalho |
|---|---|---|---|---|---|---|
| `epicare` | saúde | 8 sintomas em [0, 1] (homogêneo) | 16 | ≤ 8 | já vendorizado (Apache-2.0) | só tirar o que não usa |
| `industrial-benchmark` | operações | 6 variáveis atuais: setpoint, velocity, gain, shift, fatigue, consumption (homogêneo) | 27 (`IBGym` discreto) | 250 | `github.com/siemens/industrialbenchmark`, `industrial_benchmark_python/` (~700 linhas numpy; `LICENCE.txt` Apache-2.0) | vendorizar; trocar `gym` por gymnasium, `np.float` por `float`, `np.random` global por `Generator` |
| `simbank` | cliente (crédito) | atributos do caso de empréstimo: valor, qualidade estimada e incerteza, custo acumulado, tempo decorrido, taxa, contagens, **atividade atual (categórica)** e agregados do prefixo (heterogêneo) | 2 por ponto de decisão (`time_contact_HQ`) | ≤ 4 | Zenodo 10.5281/zenodo.15574272, v1.0.0, **CC BY 4.0** (snapshot, não o GitHub, que não tem licença) | vendorizar `SimBank/` (~1100 linhas); API online `start_simulation_inference` → `continue_simulation_inference(ação)` → `end_simulation_inference`; depende de `simpy` e `pm4py` |
| `sales-promotion` | cliente (varejo) | total de pedidos, média de pedidos/dia, ticket médio/dia, **dia da semana (cíclico)**, dia do episódio (heterogêneo) | 48 (6 cupons × 8 descontos) | 30 | NeoRL, branch `benchmark`, `neorl/neorl_envs/SalesPromotion/` (`user_model.pt`, estados iniciais de 10 000 usuários; Apache-2.0) | vendorizar modelo + lógica de transição sem `gym`; patch para episódio e recompensa **por usuário** |

Fora, e por quê:

- **ICU-Sepsis:** é o controle **não tabular** do plano original (estado =
  centróide de cluster; reprova o contrato). Não é estruturado. Sai.
- **Oberst & Sontag:** não está no plano original. Passa nos estágios 1–4 do
  funil (escopo, acesso MIT, contrato com 7 colunas categóricas, simulador
  verdadeiro), mas foi **excluído no estágio 5** ("toy sintético com política
  inventada"; ver o funil no commit `9f03cd2`). Sai.
- **MABIM:** heterogêneo e com simulador verdadeiro, mas multiagente, com
  182 MB de dados e porte para episódio por SKU com agregados: mais caro que os
  quatro acima. É o primeiro substituto se algum falhar em E0.
- **RetailSynth, CityLearn:** precisam de ponte com outro Python/JAX ou
  dependências pesadas. **KDD98, RL4RS, Health Gym HIV:** exigem ajustar um
  simulador nosso. **HeartSteps, Drink Less:** só OPE, sem simulador.

`epicare` e `industrial-benchmark` são o lado **homogêneo** (só colunas
numéricas); `simbank` e `sales-promotion`, o lado **heterogêneo**. A resposta
final compara os dois lados.

### 2.2 Decisões

| # | Decisão | Motivo |
|---|---|---|
| 1 | Métodos: `bc`, `dqn`, `iql`, `cql`. **Dreamer sai inteiro** | pedido do Pedro: só os clássicos |
| 2 | Um dataset por ambiente, **gerado no simulador** por uma política de comportamento avaliável online; tiers `dev` e `final` | o score normalizado precisa avaliar a política do dataset no mesmo simulador |
| 3 | Comportamento: `epicare` = `soc`; `industrial-benchmark` = regra de operador mediano (o repositório não traz as políticas originais, só scripts de stable-baselines); `simbank` = regra do banco do guia do SimBank; `sales-promotion` = heurística por faixa de ticket. Todas misturadas com **ε = 0,1** uniforme (`behavior_epsilon` no config) | política da prática quando existe na fonte, como no plano original; ε dá cobertura para o RL offline (os 5% do plano original existiam para a OPE, que saiu) |
| 4 | `sales-promotion` **não usa os logs humanos** do NeoRL | os logs vêm do mundo real, e a política deles não pode ser avaliada no simulador para normalizar; desvio do plano original, registrado no README |
| 5 | `simbank` usa só a intervenção `time_contact_HQ` (binária, até 4 pontos): a única sequencial do guia | H > 1 com o mínimo de trabalho. A mistura com aleatório é **por decisão** (ε), não por caso como o δ do SimBank; registrado |
| 6 | Recompensa do `simbank` = lucro do caso **em milhares**, entregue no último passo, 0 nos outros | escala de valores de empréstimo derruba o DQN; a unidade é do ambiente, então crua e escalada não se misturam |
| 7 | Sem OPE: remover `ope.py`, o split `ope_test`, o comando `ope` e tudo que só servia a isso | a métrica é o simulador |
| 8 | Representação = pré-processamento ajustado no `train` e **congelado**, aplicado às observações antes do algoritmo | comparação justa: mesmo dataset, método e rede; só muda a entrada |
| 9 | 5 seeds (0–4); 1000 episódios de avaliação no `final`, 100 no `dev` | poder estatístico razoável para comparar pares de runs |
| 10 | Remover `research tune`, `research collect`, `tuning_space` e `optuna` | não serão usados |
| 11 | `pm4py` (AGPL-3.0) e `simpy` entram como dependência instalada, **não vendorizada** | o código do SimBank importa só a semântica de rede de Petri do pm4py; registrar a licença no README |
| 12 | O pacote se chama `projeto_rl` e o comando `projeto-rl` | nome do projeto da matéria |

---

## 3. Etapas

### E0. Ambiente e viabilidade dos simuladores

- Ambiente conda `projeto-rl` (`python -c "import projeto_rl, torch, numpy, pytest"`);
  `pytest -m "not slow"` passa **antes** de qualquer mudança.
- `nvidia-smi`: anotar a GPU livre (na última conferência a GPU 1 tinha 29 GB
  ocupados por outro processo; usar a 0).
- Baixar as três fontes novas numa pasta temporária fora do repositório:
  - `git clone https://github.com/siemens/industrialbenchmark`;
  - `git clone -b benchmark https://github.com/polixir/NeoRL`;
  - SimBank:
    `https://zenodo.org/api/records/15574272/files/JakobDeMoorKULstudent/SimBank-v1.0.0.zip/content`
    (o guia de uso está em `SimBank/SimBank_Generator_Guide.ipynb`: os
    `dataset_params`, a regra do banco `get_bank_best_action` e o laço online).
- `pip install simpy pm4py` no ambiente `projeto-rl` (já conferido: instalam no
  Python 3.13 com numpy 2.4).
- **Teste de viabilidade** (antes de qualquer adaptador): rodar 50 episódios de
  cada simulador com política aleatória, já com as correções mínimas de
  compatibilidade, e medir passos por segundo. No SimBank, rodar o laço online
  do guia com `time_contact_HQ` e a regra do banco. Se um deles não rodar:
  parar (regra 0.4a); o substituto sugerido é o MABIM.
- Estimar (sem coletar) o tempo de coleta do tier `final` de cada ambiente a
  partir dos passos por segundo medidos; se o SimBank passar de 1 h (o modo
  online simula um ramo por ação), reduzir o `final` dele para 5000/500 casos
  e registrar.

### E1. Limpeza: tirar o que é do projeto principal

**Apagar:**

- `docs/planos/`, `docs/research/`, `docs/paper/`, `docs/provenance.md`,
  `docs/datasets.md`, `docs/extending.md`, `docs/README.md`,
  `docs/dreamer-fidelity.md`, `CHANGELOG.md`, `examples/` (inteiro),
  `configs/research/` (recriado como `configs/experiments/` em E7),
  `configs/overrides/` (inteiro);
- **ICU-Sepsis:** `projeto_rl/envs/icu_sepsis.py`, `third_party/icu-sepsis/`,
  `configs/envs/icu_sepsis.json`, `configs/datasets/icu_sepsis.json`, as
  entradas dele no `pyproject.toml` (`packages.find`, `package-data`,
  `pythonpath`), a exceção do `dynamics.npz` no `.gitignore`, os fixtures e
  testes que o citam;
- **Dreamer:** `projeto_rl/algorithms/dreamer/`, `projeto_rl/data/sequences.py`,
  `configs/algorithms/dreamer.json`, `tests/test_dreamer.py`,
  `tests/test_dreamer_fidelity.py`, `tests/fixtures/`, e o que só existia
  para ele (família `model_based`, `MODEL_BASED`, `family` nos relatórios se
  ficar com valor único);
- **OPE:** `projeto_rl/ope.py`, `tests/test_ope.py`, o split `ope_test` e
  `num_ope_episodes` em `data/collect.py` e nos configs; `ope_enabled`,
  `ope_dataset_path`, `ope_bootstrap_replicates`, `ope_deterministic` em
  `config.py`, `protocol.json`, `runner.py`, `cli/*`, `operations/*`; o
  `ope_probability_contract`/`EXACT_PROBABILITIES` em `policies.py` se só OPE
  os usa; a coluna OPE de `dataset list`;
- `research tune`, `research collect`, `tuning_space` em todos os
  algoritmos, `optuna` no `pyproject.toml` e no `requirements-lock.txt`;
- `planning_gain` em `operations/summary.py` (substituído em E9).

**Manter:** `third_party/epicare/`, `bc`, `dqn`, `iql`, `cql`, o loop de treino,
avaliação no simulador, fila de jobs (`suite`, `research sweep`,
`research matrix`, `research reevaluate`), `baselines`, CI, `ruff`/pre-commit.
`behavior_logprobs` continua no dataset (custa nada e é dado legítimo).

Depois da limpeza, `grep -rniw "ope\|ope_test\|optuna\|planning_gain\|tabsynth\|dreamer\|icu\|sepsis"`
em `projeto_rl/ tests/ configs/ docs/ README.md pyproject.toml` não pode achar resto
(conferir também `grep -rn "ope_"`).

### E2. Tipos de coluna (`FeatureSpec`)

Em `projeto_rl/envs/base.py`:

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "numeric" | "binary" | "ordinal" | "categorical" | "cyclic"
    cardinality: int | None = None  # obrigatório para ordinal, categorical, cyclic
```

- `EnvSpec.features: tuple[FeatureSpec, ...]` passa a ser a fonte única;
  `feature_names` vira propriedade derivada (não duplicar).
- Validação no `__post_init__`: `kind` válido; `cardinality` presente e ≥ 2
  quando exigido.
- `to_dict()` grava `features`; `collect.py` grava em
  `metadata["observation"]["features"]`.
- `epicare`: 8 × `numeric`.
- Teste de contrato (`tests/test_envs.py`): para todo ambiente, toda
  observação de um rollout curto respeita os tipos declarados (binária ∈ {0,1};
  ordinal/categórica/cíclica inteira em `0..cardinality-1`; tudo finito).

### E3. Mistura ε na coleta

- Um helper só em `projeto_rl/policies.py`: `EpsilonMixture(policy, epsilon)`, que
  devolve `(1 − ε)·p + ε·uniforme_mascarada` e repassa `bind_env`,
  `reset_episode` e `observe` à política interna (a `soc` do EpiCare se adapta
  às recompensas dentro do episódio). `collect.py` aplica quando o config do
  dataset tem `behavior_epsilon > 0`; o metadata registra ε.
- `baselines` avalia a política de comportamento **com o mesmo ε** (é ela que
  gerou o dataset e que normaliza o score).
- Teste: a mistura bate com a fórmula num vetor à mão e respeita a máscara.

### E4. Ambiente `industrial-benchmark`

Fonte: Hein et al., *A Benchmark Environment Motivated by Industrial Control
Problems*, IEEE SSCI 2017 (arXiv:1709.09480); código em
`industrial_benchmark_python/`. Citar no docstring.

- Vendorizar `IDS.py`, `EffectiveAction.py` e `goldstone/` em
  `third_party/industrial-benchmark/` (conferir licença nos cabeçalhos de cada
  arquivo; o repositório declara Apache-2.0), com `NOTICE` listando as
  mudanças: `np.random` global → `np.random.Generator` passado no construtor;
  `np.float` → `float`. Não vendorizar `IBGym.py` (depende de `gym`): o
  adaptador `projeto_rl/envs/industrial_benchmark.py` reproduz a parte dele que
  usamos (as 27 ações discretas = {−1, 0, +1}³ em velocity, gain, shift;
  recompensa = −custo, como o `reward_type="classic"`).
- Episódio: reset com setpoint sorteado por episódio entre
  `{10, 20, …, 100}` (config), 250 passos, truncamento, `gamma = 0.99` no
  config do ambiente.
- Observação: as 6 variáveis atuais, todas `numeric`, sem janela de histórico.
- Políticas: `random`; `mediocre`: para cada uma das 3 alavancas, ação na
  direção de um alvo fixo (`mediocre_target`, padrão 50) com zona morta
  (`mediocre_deadband`, padrão 5), ambos no config. Exigir
  `R_mediocre > R_random` em 200 episódios; se falhar, ajustar alvo e zona
  morta (até 3 tentativas) e registrar.
- Testes: os índices de ação mapeiam exatamente para {−1, 0, +1}³; com a mesma
  seed, o adaptador reproduz a sequência de custos do `IDS` vendorizado
  chamado diretamente; reset com seed é determinístico.

### E5. Ambiente `simbank`

Fonte: De Moor et al., *SimBank: from Simulation to Solution in Prescriptive
Process Monitoring* (arXiv:2506.14772); Zenodo 10.5281/zenodo.15574272 v1.0.0,
CC BY 4.0. Citar no docstring.

- Vendorizar `SimBank/simulation.py`, `activity_execution.py`,
  `extra_flow_conditions.py`, `petri_net_generator.py` em
  `third_party/simbank/` (sem `confounding_level.py`, que usa `np.random`
  global e não é necessário), com `LICENSE` CC BY 4.0 e `NOTICE`. Trocar os
  imports relativos ao diretório por imports de pacote; o RNG já é
  `random.Random` por instância (conferir que nada usa o módulo `random`
  global).
- Adaptador `projeto_rl/envs/simbank.py`, montado sobre o modo online do guia:
  `reset(seed)` → `start_simulation_inference(seed_to_add=seed)`;
  `step(a)` → `continue_simulation_inference(a)`; fim quando
  `int_points_available` vira falso; o caso completo vem de
  `end_simulation_inference()`. **Cuidado:** `start_simulation_inference` faz
  `self.random_seed = self.random_seed + seed_to_add`, ou seja, acumula as
  seeds entre resets; o adaptador guarda a seed base e a restaura antes de
  cada `reset`, senão o mesmo `reset(seed)` dá casos diferentes. A observação
  sai do prefixo como faz `get_bank_best_action` (`prefix_list[0][0:-1]`, sem
  o evento candidato da intervenção). Os `dataset_params` (com `time_contact_HQ`,
  `policies_info` e colunas do log) vêm do guia, copiados para
  `configs/envs/simbank.json`.
- Recompensa: o `outcome` (lucro) do caso, em milhares, no último passo; 0 nos
  outros. Conferir no código como o `outcome` é calculado e documentar.
- Observação: do último evento do prefixo, as colunas do log que variam
  (`amount`, `est_quality`, `unc_quality`, `cum_cost`, `elapsed_time`,
  `interest_rate`, `discount_factor`, `noc`, `nor`, `min_interest_rate`) como
  `numeric`; `activity` como `categorical` (cardinalidade = número de
  atividades da rede de Petri, com dicionário fixo); contagem de eventos por
  atividade no prefixo e o índice do ponto de decisão. Colunas constantes nos
  dados saem e o motivo vai para o docstring.
- Máscara: as 2 ações sempre válidas; se o guia mostrar ponto em que só uma
  vale, mascarar.
- Políticas: `random` (uniforme por decisão) e `bank` (`get_bank_best_action`
  do guia, portada).
- Testes: o adaptador reproduz, com a mesma seed e as mesmas ações, o log que o
  laço online do guia produz chamado diretamente; a recompensa é o `outcome`
  do caso dividido por 1000 e só aparece no último passo; reset com seed é
  determinístico; `R_bank > R_random`.

### E6. Ambiente `sales-promotion`

Fonte: Qin et al., *NeoRL: A Near Real-World Benchmark for Offline
Reinforcement Learning*, NeurIPS 2022 Datasets and Benchmarks; NeoRL branch
`benchmark`, Apache-2.0. Citar no docstring.

- Vendorizar em `third_party/sales-promotion/`: `env/marketing.py`,
  `models/env_model.py`, `user_model.pt`, `test_initial_states_10000_people.npy`
  e `evaluation_init_obs.npy`, com `NOTICE`: sem `gym`; RNG global
  (`np.random.seed`, `torch.manual_seed`) → geradores por instância; `print`
  removidos; passo **por usuário** devolvendo GMV e custo individuais (o
  original só devolve o total).
- Adaptador `projeto_rl/envs/sales_promotion.py`: um episódio = um usuário,
  30 dias, fim `terminal` (horizonte real da tarefa); a dinâmica das colunas
  observáveis segue `SalesPromotion_v0._get_next_state` (`sp_env.py`).
  Recompensa por passo = GMV do usuário − custo dos cupons dele no dia.
  Modelo na CPU.
- Usuários: os 10 000 estados iniciais são divididos de forma fixa: 0–7999
  para coletar `train`/`val`, 8000–9999 para avaliação. `reset(seed)` escolhe o
  usuário dentro do grupo do modo. Para o ambiente saber o modo, `make_env`
  ganha o argumento `purpose` (`"collect"` ou `"evaluate"`), passado por
  `data/collect.py` e por `evaluation.py` (inclui `baselines`) e repassado a
  `from_config`; os outros ambientes o ignoram. Sem `purpose`, erro claro.
- Ações: 48 = 6 quantidades de cupom (0–5) × 8 descontos igualmente espaçados
  em [0,6; 1,0]; rótulos legíveis.
- Observação: `total_orders`, `avg_orders`, `avg_fee` (`numeric`),
  `weekday` (`cyclic`, 7), `day` (dia do episódio, `numeric`).
- Políticas: `random` (uniforme nas 48) e `ticket_band`: 3 faixas de
  `avg_fee` (limiares no config, tirados dos quantis 33/66% dos estados
  iniciais de coleta) → (cupons, desconto) fixos por faixa, mais cupons e
  desconto maior para ticket baixo. Exigir `R_ticket_band > R_random`; se
  falhar, ajustar as ações por faixa (até 3 tentativas) e registrar.
- Testes: a atualização das colunas bate com a fórmula de `_get_next_state`
  em números à mão; com o mesmo estado e ação, a soma das recompensas
  individuais de um lote é igual à recompensa total do `MarketingEnv` original
  vezes o número de usuários; usuários de coleta e de avaliação são disjuntos;
  reset com seed é determinístico.

### E7. Configs

- `configs/envs/<env>.json` (4 arquivos): identidade, `run` (como hoje, com
  `gamma`) e `"normalization": {"low": "random", "high": "<comportamento>"}`.
- `configs/datasets/<env>.json`: **um** comportamento por ambiente, com
  `behavior_epsilon: 0.1`:
  `epicare/soc`, `industrial-benchmark/mediocre`, `simbank/bank`,
  `sales-promotion/ticket-band`. Tiers:
  - `dev`: 500 episódios de treino / 100 de validação (IB: 50 / 10);
  - `final`: `epicare` 10 000 / 1000; `industrial-benchmark` 1000 / 100
    (250 000 transições); `simbank` 10 000 / 1000; `sales-promotion`
    7000 / 1000 (usuários de coleta).
- `protocol.json`: sem OPE; `"representation": "raw"`; `eval_episodes` 100 no
  `dev` e 1000 no `final` (via `tiers`).
- `configs/experiments/dev.json` e `configs/experiments/final.json`: os 4
  datasets do tier, os 4 métodos, `"representations": ["raw", "standardize"]`,
  seeds `[0]` no dev e `[0,1,2,3,4]` no final. `research matrix` passa a ler de
  `configs/experiments/`.
- `tests/conftest.py`: `TINY_DATASETS` com os 4 ambientes (o teste
  `test_every_environment_has_tiny_fixtures` exige isso), em tamanhos mínimos;
  o que ficar lento (SimBank, SalesPromotion) vai para `slow`.

### E8. Encaixe de representações

Arquivos novos: `projeto_rl/representations/__init__.py` (registro, mesmo padrão de
`algorithms/__init__.py`), `projeto_rl/representations/base.py`,
`projeto_rl/representations/raw.py`, `projeto_rl/representations/standardize.py`.

Contrato:

```python
class Representation:
    name: ClassVar[str]

    def __init__(self, features: tuple[FeatureSpec, ...], config: dict): ...
    def fit(self, train: OfflineTransitionDataset, *, seed: int) -> None: ...
    def transform(self, obs: np.ndarray) -> np.ndarray: ...  # [N, d] -> [N, k] float32
    @property
    def output_dim(self) -> int: ...
    def state_dict(self) -> dict: ...  # inclui as FeatureSpec e o que foi ajustado
    def load_state_dict(self, state: dict) -> None: ...
```

- `fit` recebe o dataset de treino inteiro (observações, ações, recompensas,
  transições), para que representações futuras possam usar dinâmica. Só o
  `train`; nunca `val` nem o simulador.
- `raw`: identidade, `output_dim = d`.
- `standardize`: média e desvio do `train`; coluna constante → erro claro.
- `transform` levanta erro se `obs.shape[1]` não bate com as `features`.

Onde entra:

- `training.py`: depois de `load_splits`, construir a representação de
  `config["representation"]` com as `features` do metadata do `train`,
  `fit(train, seed=config["seed"])`, aplicar em `train` e `val` (novo método
  `OfflineTransitionDataset.apply_representation(rep)`, que substitui
  `observations` e `next_observations` em memória). `DataSpec.obs_dim =
  rep.output_dim`. `best_checkpoint` e `last_checkpoint` ganham
  `"representation_state": rep.state_dict()`.
- `algorithms.load_policy`: reconstruir a representação a partir do
  `representation_state` do checkpoint (sem reajustar), montar o `DataSpec`
  com `output_dim` e devolver `RepresentationPolicy(inner, rep)`.
- `RepresentationPolicy` (em `projeto_rl/representations/base.py`): delega
  `bind_env`, `reset_episode`, `observe`, `act` e `action_probabilities` à
  política interna, transformando `obs` antes.
- Dataset no disco é sempre cru; a representação é por run.
- `protocol.json`: `"representation": "raw"`. Trocar por run com
  `--set representation='"standardize"'`.
- `run_id` e diretório da run incluem a representação
  (`<dataset>:<method>:<representation>:seed-<n>:<hash>`).
- `research matrix`: o JSON da matriz aceita `"representations": [...]`
  (padrão `["raw"]`); `queue_jobs` cria um job por representação.

Testes (`tests/test_representations.py`):

- `raw` devolve exatamente a entrada;
- `standardize`: média 0 e desvio 1 no `train` (tolerância 1e-5), valores
  conhecidos num array à mão, e o `val` não influencia o ajuste;
- coluna constante e dimensão errada levantam erro;
- `state_dict` → `load_state_dict` reproduz o `transform`;
- `RepresentationPolicy` chama a política interna com a observação
  transformada;
- todo algoritmo treina e avalia com `standardize` num dataset minúsculo de
  todo ambiente (estender `tests/test_algorithms.py`, parametrizado).

### E9. Resultado: normalização e comparação

- `projeto-rl baselines <dataset>` avalia `random` e a política de comportamento
  (com o ε do dataset), com as mesmas seeds de avaliação (`eval_seed_base`) e o
  mesmo `eval_episodes` das runs, e grava também em lugar fixo
  `artifacts/reports/baselines/<env_id>/baseline_summary.json` (sobrescreve).
- `operations/summary.py` (`research summarize`) produz, em
  `artifacts/reports/summary/<timestamp>/`:
  - `runs.csv`: uma linha por run (ambiente, método, representação, seed,
    retorno, sucesso, score normalizado);
  - `by_method.csv`: média ± desvio por (ambiente, método, representação);
  - `comparison.csv`: para cada (ambiente, método, representação ≠ raw), a
    diferença **pareada por seed** contra `raw` no score normalizado: média,
    IC 95% (t de Student), vitórias/seeds;
  - `summary.md`: as tabelas acima e uma tabela agregada (score normalizado
    médio por representação: por ambiente, nos homogêneos, nos heterogêneos e
    no total).
- Score: `100 · (R − R_low) / (R_high − R_low)` com as referências do config;
  baseline ausente ou `R_high == R_low` → erro claro.
- Testes: fórmula do score em números à mão; diferença pareada e IC num caso
  calculado à mão; seed sem par levanta erro.

### E10. Documentação, dependências e testes

- `README.md` reescrito: pergunta, os 4 ambientes (tabela gerada a partir dos
  configs/registro, não redigitada), métodos, protocolo, comandos do começo ao
  fim, como adicionar uma representação, onde ficam os resultados, licenças
  (vendorizados com a licença de origem; `pm4py` AGPL-3.0 como
  dependência).
- `docs/`: `environments.md` (os 4, tipos de coluna, fontes, desvios),
  `representations.md` (contrato e passo a passo para adicionar uma),
  `experiments.md` (protocolo, normalização, estatística, como rodar, incorpora
  o atual `evaluation.md`), `algorithms.md`, `configuration.md`,
  `data-schema.md`, `operations.md` e `development.md` atualizados.
- `CLAUDE.md` reescrito para este projeto (em português, curto): objetivo,
  onde está o plano (`PLANO.md`), regras de corretude da seção 0.5, comandos;
  sem commit por etapa e sem `docs/planos/`.
- `THIRD_PARTY_NOTICES.md`: EpiCare, Industrial Benchmark, SimBank, NeoRL
  SalesPromotion.
- `pyproject.toml`: `description` nova; pacotes vendorizados novos em
  `packages.find` e `pythonpath`; `package-data` para `user_model.pt` e os
  `.npy`; `simpy` e `pm4py` nas dependências; sem `optuna`.
  `requirements-lock.txt` com as versões instaladas. `.gitignore`: exceção
  para os `.npy`/`.pt` vendorizados do SalesPromotion.
- `pytest` inteiro na GPU livre (`CUDA_VISIBLE_DEVICES=0 pytest`); tudo passa.
  `pre-commit run --all-files` limpo. Marcar como `slow` o que passar de 10 s.

### E11. Comando único com contagem de tempo: `projeto-rl run-all`

Um subcomando novo, `projeto-rl run-all <experimento.json>`, que faz o experimento
inteiro de ponta a ponta e termina imprimindo os resultados. Ele só encadeia o
que já existe (sem lógica nova de treino):

1. `[1/4] datasets`: `dataset build` de cada dataset do experimento (pula os
   que já existem e estão completos);
2. `[2/4] baselines`: `baselines` de cada ambiente (pula se o
   `baseline_summary.json` já existe para o mesmo dataset, `eval_episodes` e
   `eval_seed_base`);
3. `[3/4] training`: a matriz do experimento pela fila existente, com
   `--jobs` e `--gpus` repassados (retoma: runs completas são puladas);
4. `[4/4] summary`: `research summarize` e impressão no terminal da tabela
   agregada do `summary.md` e do caminho dos arquivos.

Contagem de tempo no terminal (o Pedro vai acompanhar):

- uma **linha geral** sempre visível: etapa atual, itens feitos/total da
  etapa, tempo decorrido da etapa, **tempo decorrido total** e **tempo
  restante estimado (ETA)** da etapa de treino, calculado pela duração média
  das runs já terminadas de cada método ÷ número de workers;
- na etapa de treino, barras só para as runs **ativas** (as terminadas somem
  da tela e entram na contagem); hoje a fila desenha uma barra por job, o que
  com 160 jobs polui a tela;
- na coleta e nos baselines, a barra Rich que já existe (com decorrido e
  restante);
- sem terminal interativo (por exemplo `nohup ... > log`), imprimir a linha
  geral a cada `progress_interval_seconds` (config, padrão 30) em vez das
  barras;
- no fim: tempo total e tempo de cada etapa.

Interromper com Ctrl+C e rodar o mesmo comando de novo **continua** de onde
parou (datasets, baselines e runs completos são reaproveitados). `--dry-run`
lista o que falta fazer em cada etapa, sem rodar nada.

Testes: o encadeamento roda inteiro num experimento minúsculo (2 ambientes,
1 método, 1 seed, tamanhos mínimos) num diretório temporário; a segunda
chamada não refaz nada; o ETA bate com a fórmula num caso à mão (durações
conhecidas, número de workers conhecido).

### E12. Smoke test, medição de tempo e estimativa (sem rodar o final)

1. **Smoke de ponta a ponta:** `projeto-rl run-all configs/experiments/dev.json
   --gpus 0`. Tem que terminar sem erro e imprimir a tabela. O `dev` deve
   levar poucos minutos; se passar de 20 min, diminuir o `dev` e registrar.
   Rodar de novo e conferir que a segunda chamada pula tudo.
2. **Medições curtas** (não são o experimento, só cronometragem):
   - coleta: episódios por segundo de cada ambiente com a política de
     comportamento (já medido no smoke);
   - treino: segundos por 500 passos de gradiente de cada método, no batch do
     `final`, na GPU 0, com 1 e com N jobs simultâneos (escolher o N
     recomendado, por exemplo 4 ou 8, pelo que der mais runs por hora);
   - avaliação: segundos por episódio de avaliação de cada ambiente.
3. **Estimativa do `final`**, escrita na seção 4, com a conta aberta:
   - coleta = Σ ambientes (episódios `train`+`val` ÷ episódios/s);
   - baselines = Σ ambientes (2 políticas × `eval_episodes` × s/episódio);
   - treino = Σ runs (`offline_train_steps` × s/passo do método +
     `eval_episodes` × s/episódio do ambiente) ÷ N workers, como **teto**
     (o early stopping só encurta);
   - total, com a faixa (melhor caso com early stopping na metade, pior caso
     sem early stopping).
4. Preencher a seção 4 deste plano. **Não** gerar datasets `final` nem rodar a
   matriz `final`.

---

## 4. Resumo final (09/11/2026)

### Arquivos novos

| Arquivo | Para que serve |
|---|---|
| `projeto_rl/representations/__init__.py` | registro de representações (`register_representation`, `build_representation`, `load_representation`) |
| `projeto_rl/representations/base.py` | contrato `Representation` e `RepresentationPolicy` (aplica a representação na avaliação) |
| `projeto_rl/representations/raw.py` | representação identidade |
| `projeto_rl/representations/standardize.py` | z-score ajustado no `train` |
| `projeto_rl/envs/industrial_benchmark.py` | adaptador do Industrial Benchmark (27 ações, política `mediocre`) |
| `projeto_rl/envs/simbank.py` | adaptador do SimBank (`time_contact_HQ`, políticas `bank` e `random` do guia) |
| `projeto_rl/envs/sales_promotion.py` | adaptador do SalesPromotion (episódio por usuário, política `ticket_band`) |
| `projeto_rl/operations/baselines.py` | avalia `random` e o comportamento (com ε) que definem o score 0 e 100 |
| `projeto_rl/cli/experiment.py` | comando `projeto-rl run-all` (4 etapas, contagem de tempo, retomada) |
| `configs/envs/{industrial_benchmark,simbank,sales_promotion}.json` | identidade, `run`, `feature_mix`, `normalization` |
| `configs/datasets/{industrial_benchmark,simbank,sales_promotion}.json` | comportamento, ε e tamanhos `dev`/`final` |
| `configs/experiments/{dev,final}.json` | smoke (1 seed, datasets pequenos) e o experimento (5 seeds) |
| `third_party/industrial-benchmark/` | `IDS.py`, `EffectiveAction.py`, `goldstone/` (MIT), `LICENSE`, `NOTICE` |
| `third_party/simbank/` | `simulation.py` e afins do Zenodo v1.0.0 (CC BY 4.0), `LICENSE`, `NOTICE` |
| `third_party/sales-promotion/` | `marketing.py`, `models/`, `assets/` (modelo de usuário e estados iniciais, Apache-2.0), `LICENSE`, `NOTICE` |
| `docs/representations.md` | contrato e passo a passo para adicionar uma representação |
| `docs/experiments.md` | protocolo, normalização, estatística, arquivos de experimento |
| `tests/test_representations.py` | identidade, z-score à mão, ajuste só no `train`, ida e volta do estado, política |
| `tests/test_industrial_benchmark.py` | tabela de ações, reprodução passo a passo do `IDS`, regra `mediocre` |
| `tests/test_simbank.py` | reprodução do laço online do guia, seed base, regra do banco, `random` do guia |
| `tests/test_sales_promotion.py` | atualização de `_get_next_state` à mão, zeragem conjunta, horizonte, usuários separados, soma por usuário |
| `tests/test_policies.py` | mistura ε pela fórmula e repasse dos ganchos do episódio |

### O que foi apagado

`docs/planos/`, `docs/research/`, `docs/paper/`, `docs/provenance.md`,
`docs/datasets.md`, `docs/extending.md`, `docs/README.md`,
`docs/dreamer-fidelity.md`, `docs/evaluation.md` (virou `experiments.md`),
`CHANGELOG.md`, `examples/`, `configs/research/`, `configs/overrides/`; o
ICU-Sepsis inteiro (`projeto_rl/envs/icu_sepsis.py`, `third_party/icu-sepsis/`, configs);
o Dreamer inteiro (`projeto_rl/algorithms/dreamer/`, `projeto_rl/data/sequences.py`, config,
testes e fixtures); a OPE (`projeto_rl/ope.py`, `tests/test_ope.py`, o split `ope_test`,
o comando `ope`, as chaves `ope_*`); `research tune`, `research collect`,
`tuning_space`, `optuna`; `planning_gain`; o conceito de `family`; a coleta a partir
de checkpoint treinado; `reference_policies`; as políticas `smart` e `oracle` do
adaptador do EpiCare (o código vendorizado não mudou).

### `git diff --stat`

Arquivos rastreados: 112 arquivos, +1497 −13393 (61 apagados, 51 modificados).
Arquivos novos (fora do git, como pedido): 5066 linhas de texto, das quais 2667 são
código vendorizado, mais os assets binários do SalesPromotion (4 MB).

### Desvios do plano e por quê

1. **SimBank, referência 0 (decidido pelo Pedro):** `random` é a política
   aleatória do guia do SimBank (um ponto de decisão sorteado, uma ação mantida),
   não a uniforme por decisão: a regra do banco perde da uniforme (3361 ± 151 contra
   3701 ± 180 em 2000 casos) e ganha da do guia (2618 ± 150).
2. **SimBank, horizonte:** os casos têm de 1 a 8 pontos de decisão (não ≤ 4); cerca
   de 7% não têm nenhum e o `reset` os pula de forma determinística. Coleta e
   avaliação usam as duas seeds base do guia (82 e 130).
3. **SalesPromotion, descontos:** os 8 níveis são a lista da fonte (0,95 … 0,60),
   não 8 valores igualmente espaçados em [0,6; 1,0] como o plano dizia.
4. **SalesPromotion, verificação contra o original:** feita uma vez, fora dos testes
   do repositório (o código original importa `gym` e `neorl`; usei stubs no
   scratchpad): diferença máxima 0,0 em 5 dias × 200 usuários. Os testes do
   repositório conferem a fórmula de `_get_next_state` à mão e a soma por usuário.
5. **Industrial Benchmark, licença:** os arquivos Python são MIT (cabeçalho de cada
   arquivo); o `LICENCE.txt` Apache-2.0 do repositório cobre a versão Java.
   O episódio começa no `reset` (não é janela de um processo de 1000 passos) e não
   tem evento de sucesso (`success` sempre falso).
6. **Baselines:** gravados por dataset (`artifacts/reports/baselines/<dataset>/`),
   não por ambiente, porque `dev` e `final` têm números de episódios diferentes e se
   sobrescreveriam; avaliados por amostragem (não gulosos), como geraram os dados.
7. **Fila:** saiu o limite de `--jobs` ≤ número de GPUs, para rodar N jobs na mesma GPU.
8. **`progress_interval_seconds`** fica no arquivo de experimento, não no protocolo
   (no protocolo entraria no hash de todas as runs sem mudar resultado).
9. **`save_trajectories: false`** no protocolo, para as 160 runs não gravarem um
   CSV por passo de avaliação (o IB teria 250 000 linhas por run).
10. **`research matrix`** perdeu `--seeds` e os blocos `jobs` (as seeds vêm do arquivo).
11. **`petri_net_generator.py`** do SimBank perdeu os imports do visualizador do
    pm4py e a função de desenho que não é chamada (exigiria Graphviz).
12. Nenhuma política de comportamento precisou de recalibração: `mediocre` −51 mil
    contra −71 mil da `random` (IB, 50 episódios); `ticket_band` 867 ± 33 contra
    633 ± 23 (SalesPromotion, 300 usuários); `bank` 2,26 ± 0,12 contra 1,64 ± 0,12
    (SimBank, 2000 casos). Nenhum tamanho foi reduzido.

### Onde olhar o diff com atenção

- `projeto_rl/training.py` e `projeto_rl/algorithms/__init__.py` (`load_policy`): ajuste da
  representação só no `train`, estado e `data_spec` no checkpoint, reconstrução sem
  reajuste.
- `projeto_rl/envs/simbank.py`: restauração da seed base, pulo de casos sem decisão,
  observação a partir de `prefix_list[0][0:-1][-1]`, recompensa no último passo.
- `projeto_rl/envs/sales_promotion.py` e `third_party/sales-promotion/sales_promotion/`:
  RNG portado (geradores por instância) e episódio por usuário.
- `projeto_rl/operations/summary.py`: score normalizado, diferença pareada com IC t.
- `projeto_rl/operations/queue.py`: ETA e tela de progresso nova.

### Testes e smoke

- `pytest`: **118 passaram** em 27 s na GPU 0; `CUDA_VISIBLE_DEVICES="" pytest -m "not slow"`:
  117 passaram em 17 s. `pre-commit run --all-files` (ruff check, ruff format,
  arquivos grandes): limpo.
- Smoke `projeto-rl run-all configs/experiments/dev.json --gpus 0 --jobs 4`: terminou sem
  erro em **5m12s** (datasets 34 s, baselines 16 s, treino de 32 runs 4m22s, resumo
  < 1 s). A segunda chamada pulou tudo e terminou em 0 s.
- Tabela do smoke (**`dev`, 1 seed, datasets de 500 episódios: não vale como
  resultado, só mostra que o pipeline funciona**):

  | Grupo | `raw` | `standardize` |
  |---|---:|---:|
  | epicare | 259,2 | 220,5 |
  | industrial-benchmark | −101,5 | −15,0 |
  | sales-promotion | 81,4 | 139,3 |
  | simbank | −45,1 | 159,5 |
  | homogêneos | 78,9 | 102,7 |
  | heterogêneos | 18,1 | 149,4 |
  | todos | 48,5 | 126,1 |

### Tempos medidos e estimativa do `final`

Medições (GPU 0, 64 núcleos com carga ~24 de outros usuários):

- coleta, segundos por episódio com a política de comportamento: epicare 0,0008;
  IB 0,0135; SalesPromotion 0,0467; SimBank 0,0038;
- treino com os hiperparâmetros do `final` (batch 256), segundos por passo de cada
  processo com 8 processos simultâneos: bc 0,0081; dqn 0,0058; iql 0,0116;
  cql 0,0087 (vazão com 8 jobs 30–45% maior que com 1, e ~10% maior que com 4);
- avaliação gulosa, segundos por episódio (smoke, 4 jobs): epicare 0,0041;
  IB 0,204; SalesPromotion 0,062; SimBank 0,0076;
- partida de um worker e carga dos dados: ~3 s por run.

Conta:

- **Coleta** = 11 000 × 0,0008 + 1100 × 0,0135 + 8000 × 0,0467 + 11 000 × 0,0038
  = 9 + 15 + 374 + 41 ≈ **7 min**.
- **Baselines** = 2 políticas × 1000 episódios × (0,0008 + 0,0135 + 0,0467 + 0,0038)
  ≈ **2 min**.
- **Treino**, por run = 5000 passos × s/passo do método + validação (20 checagens
  do `val`) + 1000 episódios × s/episódio do ambiente + 3 s. Somando os 4 métodos de
  um ambiente (uma seed, uma representação): epicare 205 s, IB 1020 s
  (a avaliação de 250 000 passos domina), SalesPromotion 457 s, SimBank 219 s, total
  1901 s; × 2 representações × 5 seeds = 19 010 s de run; ÷ 8 workers =
  **≈ 40 min no pior caso** (sem early stopping). Com o early stopping parando na
  metade, o treino cai para ≈ 32 min (a avaliação não encolhe).
- **Total: ≈ 41 min (melhor caso) a ≈ 49 min (pior caso)**; contando com a carga de
  outros usuários na CPU, **reserve 1 hora**. Jobs recomendados: **8**.

### O comando

```bash
cd /scratch/pedrobacelar/projeto-rl
conda activate projeto-rl
projeto-rl run-all configs/experiments/final.json --dry-run          # confere o que falta (4 datasets, 4 baselines, 160 runs)
CUDA_VISIBLE_DEVICES=0 projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 8
```

Em background, para fechar o terminal:

```bash
CUDA_VISIBLE_DEVICES=0 nohup projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 8 > artifacts/run-all.log 2>&1 &
tail -f artifacts/run-all.log
```

Se cair ou for interrompido (Ctrl+C), rodar o mesmo comando de novo: datasets e
baselines prontos e runs completas são reaproveitados, e runs com falha são
refeitas.

### Onde ficam os resultados

`artifacts/reports/summary/<data-hora>/summary.md`, que é o mesmo conteúdo que o
comando imprime no fim. Primeira tabela: score normalizado médio por
representação, por ambiente, nos homogêneos, nos heterogêneos e no total
(0 = aleatória, 100 = política que gerou os dados). Segunda: para cada ambiente e
método, a diferença pareada `standardize − raw` com IC 95% sobre as 5 seeds e
quantas seeds ganharam. Terceira: médias e desvios por método. Os mesmos números
estão em `runs.csv`, `by_method.csv` e `comparison.csv`.

### Próxima representação

1. Criar `projeto_rl/representations/<nome>.py` (ver `docs/representations.md`) e
   importá-lo no fim de `projeto_rl/representations/__init__.py`.
2. Acrescentar `"<nome>"` em `representations` de `configs/experiments/final.json`
   (o `"standardize"`, que é a `reference_representation`, tem que ficar).
3. `CUDA_VISIBLE_DEVICES=0 projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 32`:
   as runs `standardize` já feitas são reaproveitadas, só a nova treina, e o
   resumo compara a nova com o `standardize` nas mesmas seeds.

(Atualizado na rodada 2, seção 6: a referência deixou de ser o `raw`.)

---

## 5. Fora do escopo

- Rodar o experimento `final` (quem roda é o Pedro, com o comando de E11).
- Métodos novos de RL; Dreamer; OPE; ajuste de hiperparâmetros.
- Representações além de `raw` e `standardize` (vêm depois).
- Representações treinadas ponta a ponta junto com a política.
- Mais de um dataset por ambiente; logs reais.

---

## 6. Rodada 2 (09/11/2026): correções depois do primeiro `final`

### O que o primeiro `final` mostrou (160 runs, 5 seeds, `raw` e `standardize`)

- Rodou sem falha em ≈ 2h05 (a estimativa de 41–49 min errou ≈ 3×: avaliação
  medida com 1 processo só e otimização medida só no EpiCare).
- `standardize` ganhou de `raw` em 8 das 16 comparações (SimBank e
  SalesPromotion, sobretudo DQN e CQL); `raw` colapsa onde as colunas têm
  escalas muito diferentes (SimBank: desvio de 1,1 a 19 000).
- Industrial Benchmark inconclusivo: DQN e CQL instáveis, CQL e IQL com
  checkpoint no passo 250 e loss de validação explodindo.
- SimBank com score > 400: checado (F) com os checkpoints; as políticas
  aprendidas escolhem caso a caso quando contatar a matriz (retorno 3,7–4,5)
  e superam a regra do banco (2,16), "sempre contatar" (2,25) e "nunca
  contatar" (−0,26, que é o que o `raw` aprendeu). Não é uma ação constante.
- BC no EpiCare longe do piso de entropia (1,63 × 0,57): a SoC usa histórico,
  então a observação não determina a ação. Não é bug.

### Problemas encontrados e correções (A–E aprovadas pelo Pedro)

| # | Problema | Correção |
|---|---|---|
| A | Fim por limite de tempo gravado como terminal: sem bootstrap no alvo TD, e o tempo não está na observação (IB: 100% dos episódios, γ = 0,99; EpiCare: 39,8%) | Schema v5: `terminals` só `terminated`, novo `timeouts`; `td_target` compartilhado bootstrapa no timeout (Pardo et al., 2018). Datasets antigos viram "stale" e o `run-all` recoleta; a versão do schema entra na identidade da run |
| B | Recompensa sem escala (média de \|r\| por passo: IB 199, SP 27, EpiCare 11, SimBank 0,7): Q enorme, CQL divergindo (10¹² no SP `raw`) | Recompensa dividida pela média de \|r\| do `train`, só no treino; `reward_scale` no `train_summary`; avaliação no retorno original |
| C | IB e SP com 5–6 épocas, melhor checkpoint no fim | `offline_train_steps` 5000 → 10 000 |
| D | Comparação fixa contra `raw`; média dos agregados dominada por outliers; o resumo lia todas as runs da pasta (misturaria protocolos) | `reference_representation` no arquivo do experimento; agregados por IQM (Agarwal et al., 2021); o resumo lê só as runs do experimento |
| E | — | `final.json`: só `standardize`, seeds 0–2 (48 runs); `dev.json`: `raw` e `standardize`, referência `standardize`; protocolo com `standardize` por padrão |
| extra | Pedido do Pedro | Gráficos de loss (treino e validação por método, uma linha por seed, ponto no checkpoint escolhido) em `reports/summary/<data-hora>/loss_curves/<dataset>.png` |

Arquivo novo: `projeto_rl/operations/plots.py` (gráficos de loss). Alterados:
`data/collect.py`, `data/dataset.py`, `algorithms/{base,dqn,cql,iql}.py`,
`training.py`, `config.py`, `operations/summary.py`, `cli/{common,experiment,research}.py`,
`configs/protocol.json`, `configs/experiments/{dev,final}.json`, docs
(`experiments`, `algorithms`, `data-schema`, `representations`, `operations`,
`configuration`, `development`), README, `pyproject.toml` e
`requirements-lock.txt` (matplotlib 3.10.8). `research summarize` agora recebe o
arquivo do experimento.

### Testes e smoke

- `pytest`: 126 testes passam (incluindo o lento); `pre-commit` limpo. Novos
  contra verdade conhecida: alvo TD à mão com timeout × terminal; escala =
  média de \|r\| do train (+2/−2 → 2) ignorando o val (100); cada ambiente
  grava limite de tempo como timeout e fim real como terminal; dataset de
  schema antigo detectado; IQM à mão.
- Smoke `dev` (32 runs, 10 000 passos, 8 jobs): sem falha em 5m50s; no IB o
  CQL deixou de divergir (checkpoint no passo 8750, score 99,8, antes −423 no
  `final`). Tamanho `dev` é pequeno demais para concluir algo.

### Estimativa do novo `final` (48 runs, 32 jobs)

Por run, a partir da rodada 1 (otimização × 2 pelos 10 000 passos; o early
stopping pode encurtar): EpiCare ≈ 90 s, IB ≈ 620 s, SP ≈ 690 s, SimBank ≈ 410 s;
× 12 runs cada = ≈ 21 700 s. Com 32 jobs numa máquina de 64 núcleos já com
carga ≈ 60–90 de outros usuários, o paralelismo efetivo fica entre 16 e 32:
treino ≈ 12–25 min, mais a run mais longa (≈ 12–23 min) como piso. Coleta dos 4
datasets ≈ 10 min (SalesPromotion domina, em série). Baselines reaproveitados.
**Total ≈ 25–45 min; reservar 1 h.** Memória da GPU 0: ≈ 0,65 GB por worker ×
32 ≈ 21 GB, mais ≈ 6 GB de outro processo, de 32,6 GB: cabe, sem muita folga.

### Comando

```bash
cd /scratch/pedrobacelar/projeto-rl && conda activate projeto-rl
CUDA_VISIBLE_DEVICES=0 projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs 32
```

Se der falta de memória na GPU, rodar o mesmo comando com `--jobs 24`: as runs
completas são reaproveitadas.

## 7. Rodada 3 (12–13/11/2026): dois ambientes largos, `citylearn` e `grid2op`

### Por quê

Com 5 a 8 colunas, as representações do PR #1 (`spr`, `tjepa`, `tabpfn`,
`tdjepa`) ficaram iguais ou um pouco abaixo de `standardize`. Hipótese do
Pedro: há pouco a ganhar além da normalização com tão poucas colunas. Para
testá-la sem mudar nada nos métodos nem nas representações, entram dois
ambientes com muito mais colunas, avaliados do mesmo jeito (simulador online,
política aleatória = 0 e comportamento com ε = 0,1 = 100).

### Escolha (aprovada pelo Pedro)

- Do catálogo do plano principal e de uma busca na web, só dois candidatos
  com simulador pronto, ações discretas, política de comportamento e
  instalação viável em Python 3.13 / numpy 2 / pandas 3: CityLearn 3.2 e
  Grid2Op case 14. Sinergym (precisa do EnergyPlus), KDD98 e RL4RS (exigem
  ajustar um simulador nosso) e os demais ficaram de fora.
- `citylearn` (28 colunas, dados medidos de 17 casas): a regra de
  comportamento é `time_of_use` (carregar 1 kWh/h nas horas 10–15, descarregar
  2 kWh/h nas horas 16–20). As regras do próprio CityLearn foram escritas para
  outra unidade de ação e perdem para a bateria parada; a de autoconsumo é
  forte demais (o ótimo com informação perfeita fica só ~8% acima dela).
  Medido em 40 semanas por casa, com `time_of_use` = 100 e aleatória = 0:
  bateria parada 77–130, autoconsumo 108–135.
- `grid2op` (153 colunas): os cenários são **sintéticos** e o download **não
  declara licença**; o Pedro aceitou porque é o único candidato pronto com
  centenas de colunas de tipos mistos. Recompensa: a do L2RPN por linha;
  blackout é terminal. A regra (`rule`, limiar 0,9 com `obs.simulate`) nunca
  perdeu a rede em 30 episódios de um dia; com ε = 0,1 os dias do dataset terminam
  em blackout depois de ~90 passos em média (a metodologia de ε foi mantida).

### O que mudou

Arquivos novos: `projeto_rl/envs/citylearn.py`, `projeto_rl/envs/grid2op.py`,
`configs/envs/{citylearn,grid2op}.json`, `configs/datasets/{citylearn,grid2op}.json`,
`configs/experiments/wide.json` (os dois ambientes, 144 runs) e
`wide-dev.json` (seu smoke), `requirements-nodeps.txt`,
`tests/test_citylearn.py`, `tests/test_grid2op.py`. Alterados:
`projeto_rl/envs/__init__.py`, `projeto_rl/cli/info.py` (`doctor` lista os
pacotes novos), `configs/experiments/{dev,final}.json` (6 ambientes; o
`final` vira 432 runs), `tests/{conftest,test_envs,test_config,test_operations}.py`,
`requirements-lock.txt`, `README.md`, `THIRD_PARTY_NOTICES.md`,
`docs/environments.md`, `docs/representations.md`, `pyproject.toml`,
`CLAUDE.md`.

- CityLearn e Grid2Op são **instalados, não vendorizados**:
  `pip install --no-deps -r requirements-nodeps.txt`, porque os pinos
  declarados (numpy < 2, pandas < 3, scikit-learn ≤ 1.2.2) não resolvem em
  Python 3.13. As trajetórias do Grid2Op saíram idênticas bit a bit em pandas
  3.0.2 e 2.3.3. Os dados são baixados no primeiro uso
  (`~/.cache/citylearn`, ~3 MB; `~/data_grid2op`, ~300 MB).
- O `tabpfn` (no lock desde o PR #1) não estava instalado no ambiente
  `projeto-rl`; foi instalado.

### Testes e smoke

- `pytest`: 304 testes passam. Novos contra verdade conhecida: observação do
  CityLearn igual à linha do CSV da casa; +0,2 a partir da bateria vazia
  compra exatamente 1 kWh e a recompensa é −preço × max(consumo líquido, 0);
  energia exportada vale 0; recompensa do Grid2Op × 20 linhas igual ao
  `L2RPNReward` da fonte; máscara igual ao que as regras da fonte aceitam
  (`is_illegal` do `simulate`); blackout terminal e fim do dia truncado; a
  regra escolhe a menor carga simulada.
- Smoke `wide-dev.json` (48 runs): passou em todas as runs, menos nas
  `tabpfn` (abaixo). Três runs de Grid2Op deram falta de memória na GPU 0
  (outro usuário com 19 GB) e passaram ao rodar de novo na GPU 1.

### Limitação conhecida: `tabpfn` sem a licença do TabPFN-3.5

Sem a licença aceita, o `tabpfn` cai para o TabPFN-2, que aceita no máximo 10
classes; as classes são as ações registradas. Nesta máquina, as runs `tabpfn`
de todos os ambientes com mais de 10 ações (todos menos o SimBank) falham
com `TabPFNValidationError`. Para rodar: aceitar a licença em
https://ux.priorlabs.ai e exportar `TABPFN_TOKEN` antes do `run-all`.

### Tempos medidos e estimativa do `wide.json` (144 runs)

Smoke `wide-dev` com 8 jobs, GPU 0 a 100% por outro usuário: CityLearn ≈
750 s por run, Grid2Op ≈ 1050 s. O gargalo do Grid2Op é a avaliação: a
política gulosa faz uma chamada à GPU por episódio e por passo (≈ 5 ms cada
com a GPU disputada), e o `final` avalia 1000 episódios em vez de 100.
Estimativa por run no `final`: CityLearn ≈ 1100 s, Grid2Op ≈ 2700 s. Coleta
dos dois datasets ≈ 25 min (Grid2Op domina) e baselines ≈ 15 min. Com 8 jobs:
**≈ 8 h sem `tabpfn` e ≈ 10 h com**. Cada run de `tjepa` no Grid2Op ocupa
6–8 GB de GPU e ≈ 15 GB de RAM.

### Comando

```bash
cd /scratch/pedrobacelar/projeto-rl && conda activate projeto-rl
export TABPFN_TOKEN=<chave da conta em ux.priorlabs.ai>   # sem ela, as 24 runs tabpfn falham
nohup projeto-rl run-all configs/experiments/wide.json --gpus 1 --jobs 8 > artifacts/run-all-wide.log 2>&1 &
```

Se alguma run falhar por falta de memória na GPU, rodar o mesmo comando de
novo: as runs completas são reaproveitadas.
