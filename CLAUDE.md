# CLAUDE.md

Regras para trabalhar neste repositório:
um projeto de matéria que pergunta se uma representação vetorial melhor dos
estados tabulares melhora o retorno de RL offline. Seis simuladores
(`epicare`, `industrial-benchmark`, `simbank`, `sales-promotion`, `citylearn`,
`grid2op`), quatro métodos clássicos (`bc`, `dqn`, `iql`, `cql`), avaliação só
no simulador online (sem OPE).

## Onde está o plano

`PLANO.md` (na raiz): o que foi construído, as decisões e o resumo final com a
estimativa de tempo e o comando do experimento. Ler só a parte que a tarefa
pedir.

## Ritmo de trabalho

1. Ler a tarefa e as partes do `PLANO.md` e de `docs/` que ela toca.
2. Para mudança não trivial, mostrar um plano curto (arquivos a criar, alterar e
   apagar; decisões em aberto) e esperar aprovação.
3. Implementar só o pedido; o que aparecer fora do escopo vai para o resumo.
4. Rodar os testes.
5. Resumo final: arquivos novos (uma linha cada), o que foi apagado,
   `git diff --stat`, desvios e onde olhar o diff com atenção.

Commit, push e PR só quando o Pedro pedir. Nunca mexer em outras branches.

Se algo estiver ambíguo ou contraditório: parar e perguntar, nunca decidir em
silêncio.

## Corretude

- Toda conta (dinâmica e recompensa de adaptador, representação, normalização,
  estatística) tem teste contra verdade conhecida: forma fechada, valor calculado
  à mão ou reprodução do código da fonte. Teste que só confere formato não conta.
- Nada falha em silêncio: pré-condição quebrada levanta erro com mensagem clara.
  Fallback só quando documentado.
- Todo valor que afeta uma nota (hiperparâmetro, ε, tamanho, limiar) vem de
  config, não fica fixo no código.
- Seeds explícitas; nada de RNG global (`np.random.*`, `random.*`,
  `torch.manual_seed` globais), nem no código vendorizado.
- O simulador nunca é usado em treino nem em seleção de checkpoint.
- A representação é ajustada só no split `train`; o `val` e o simulador nunca
  entram no ajuste.
- Código vendorizado mantém a licença, tem `NOTICE` com cada mudança e entra em
  `THIRD_PARTY_NOTICES.md`.
- Método ou ambiente vindo de um paper cita o paper no docstring, com cada
  adaptação ou desvio.
- Fato externo (licença, API, colunas, número publicado) é conferido na fonte
  antes de virar código ou doc.

## Sem complexidade à toa

- Nenhuma opção de config, flag ou abstração que ninguém usa.
- Antes de escrever um helper, procurar se já existe.
- Sem aliases, shims, código legado ou comentado; código morto sai junto.
- Validar na borda (config, dataset), não em toda função interna.
- Comentários explicam o porquê. Docstring de uma linha por módulo.
- Código, docstrings e docs em inglês; `PLANO.md` em português.
- Tabelas dos docs (colunas, ações, políticas) saem dos configs e do registro.

## Arquitetura

- `envs/` e `data/` não importam `algorithms/`; algoritmos só leem `DataSpec` e
  batches e devolvem uma `Policy`; `representations/` só lê `FeatureSpec` e o
  dataset de treino; `runner.py` e `training.py` juntam tudo.
- IDs de ambientes e datasets usam `-` (`sales-promotion/ticket-band/final`);
  pacotes Python usam `_`.

## Comandos

```bash
conda activate projeto-rl
pip install -r requirements-lock.txt && pip install --no-deps -r requirements-nodeps.txt && pip install -e . --no-deps   # instalação
pytest                                                              # testes (GPU 0 se livre)
pytest -m "not slow"                                                # verificação rápida
pre-commit run --all-files                                          # ruff
projeto-rl run-all configs/experiments/dev.json --gpus 0 --jobs 4        # smoke de ponta a ponta
projeto-rl run-all configs/experiments/final.json --gpus 0 --jobs <N>    # o experimento (roda o Pedro)
```

O ambiente conda é o `projeto-rl` (Python 3.13); não instalar nada no `base`. A GPU 1
desta máquina costuma estar ocupada por outro processo: usar a 0.

## Dados

Dados nunca entram no git: datasets, runs e relatórios ficam em `artifacts/`
(ignorado). Só os assets dos simuladores vendorizados entram (modelo de usuário
e estados iniciais do SalesPromotion).
