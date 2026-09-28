# Reverse-Statistics

Scripts e notebooks para **calibração de parâmetros de escoamento bifásico** (modelo de permeabilidade relativa) no solver **IMPESFoam** (OpenFOAM), ajustando os resultados da simulação a dados experimentais por otimização (mínimos quadrados).

Os parâmetros calibrados são: `a_exp`, `b_exp`, `kra_max` e `krb_max`.

## Arquivos do repositório

| Arquivo | Função |
|---|---|
| `extrair_dados_minimos.py` | Calibração **usando apenas a pressão**. Compara o delta de pressão simulado (`p` no primeiro ponto − `p` no último ponto, ao longo do tempo) com o delta de pressão experimental (`delta_p_exp.csv`). |
| `extrair_parametros_p_w.py` | Calibração **usando pressão + saturação**. Além do delta de pressão, compara o perfil de saturação simulado `Sb(x,t)` com o experimental (`Sw_exp.csv`), com pesos ajustáveis entre os dois blocos de resíduo. |
| `extrair_dados.ipynb` | Notebook de extração/tratamento dos dados.|
| `exemplos_bifasico.ipynb` | Notebook com exemplos/análises do caso bifásico.

## Como os scripts funcionam

A cada avaliação do otimizador (`scipy.optimize.least_squares`):

1. Copia o caso base do OpenFOAM para `parametric_runs/run_<n>`.
2. Edita os 4 parâmetros em `constant/transportProperties`.
3. Executa o `impesFoam2ph`. A *function object* `minhaAmostra` (definida no `controlDict`) amostra os campos ao longo da linha `linha1`.
4. Monta as matrizes de `p(x,t)` (e `Sb(x,t)`, no script de pressão + saturação).
5. Calcula os resíduos em relação aos dados experimentais.
6. Devolve os resíduos ao otimizador, que propõe novos parâmetros.

## Como usar

1. Coloque na mesma pasta dos scripts o caso base do OpenFOAM e os arquivos experimentais:
   - `delta_p_exp.csv`: delta de pressão experimental (índice sequencial + valor)
   - `Sw_exp.csv`: perfil de saturação experimental (apenas para `extrair_parametros_p_w.py`)
2. Ajuste a seção `CONFIGURAÇÃO` no topo do script: caso base, chute inicial, limites dos parâmetros, tempo final experimental etc.
3. Execute:

```bash
python extrair_dados_minimos.py      # só pressão
# ou
python extrair_parametros_p_w.py     # pressão + saturação
```

## Saídas geradas

- `parametric_runs/`: cópias do caso de cada avaliação e o log `parametric_log.csv`
- `results/`: matrizes de campo (`run_<n>_p.csv`, `run_<n>_Sb.csv`)
- `calibration_result.csv`: parâmetros calibrados, custo final e número de avaliações

- `parametric_runs/`: cópias do caso de cada avaliação e o log `parametric_log.csv`
- `results/`: matrizes de campo (`run_<n>_p.csv`, `run_<n>_Sb.csv`)
- `calibration_result.csv`: parâmetros calibrados, custo final e número de avaliações
