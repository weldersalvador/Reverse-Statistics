"""
Calibração de parâmetros do IMPESFoam (a_exp, b_exp, kra_max, krb_max)
(Atualmente utilizando minimize do scipy)

A cada avaliação do algorítmo:
  1. Copia o caso base para parametric_runs/run_<n>
  2. Edita os 4 parâmetros com os valores propostos pelo otimizador
  3. Roda o IMPESFoam (a function object "sets" gera a amostragem)
  4. Monta p_mod(x,t), calcula delta_p_mod(t) = p(primeiro x) - p(ultimo x)
  5. devolve os resíduos

REQUISITOS: pip install numpy scipy  (--break-system-packages se necessário)
"""

import os
import re
import glob
import shutil
import subprocess
import csv
import numpy as np
from datetime import datetime
from scipy.optimize import least_squares, minimize

# ---------------------------------------------------------------
# CONFIGURAÇÃO — ajuste aqui
# ---------------------------------------------------------------

BASE_CASE = "Kc200"          # pasta do seu caso já funcional

# Nome de cada propriedade -> arquivo onde ela está.
PROPERTY_FILES = {
    "a_exp":   "constant/transportProperties",
    "b_exp":   "constant/transportProperties",
    "kra_max": "constant/transportProperties",
    "krb_max": "constant/transportProperties",
}

# Chute inicial da calibração (a ordem define a ordem do vetor de parâmetros)
INITIAL_GUESS = {"a_exp": 1.50, "b_exp": 1.50, "kra_max": 0.50, "krb_max": 0.50}
PARAM_NAMES = list(INITIAL_GUESS.keys())

# Limites, na mesma ordem de PARAM_NAMES
LOWER_BOUNDS = [1, 1, 0.01, 0.01]
UPPER_BOUNDS = [6, 6, 3, 3]

BOUNDS = [[1,6],[1,6],[0.01,3],[0.01,3]]

DIFF_STEP = [0.001, 0.001, 0.001, 0.001]

MAX_NFEV = 500

RUN_COMMANDS = [
    ["impesFoam2ph"]
    # ["blockMesh"],
    # ["setFields"],
]

# Devem bater com system/controlDict
SAMPLE_FUNCTION_NAME = "minhaAmostra"
SET_NAME = "linha1"
FIELDS = ["p", "Sb"]

# Dados experimentais
EXP_FILE = "delta_p_exp.csv"
T_FINAL_EXP = 150      # tempo (s) correspondente ao último índice do CSV

OUTPUT_ROOT = "parametric_runs"
RESULTS_DIR = "results"
LOG_FILE = "parametric_log.csv"
RESULT_FILE = "calibration_result.csv"

# Apaga a pasta de cada rodada bem-sucedida depois de ler os resultados
KEEP_RUN_DIRS = False


# ---------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------

def edit_property(case_dir, rel_path, prop_name, new_value):
    """Troca só o número de uma propriedade num dicionário do OpenFOAM."""
    file_path = os.path.join(case_dir, rel_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Não encontrei {file_path}. Confira PROPERTY_FILES."
        )

    with open(file_path, "r") as f:
        content = f.read()

    pattern = re.compile(
        rf"^(\s*{re.escape(prop_name)}\s+(?:{re.escape(prop_name)}\s+\[[^\]]*\]\s+)?)"
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(\s*;)",
        re.MULTILINE,
    )

    new_content, n_subs = pattern.subn(rf"\g<1>{new_value}\g<2>", content)

    if n_subs == 0:
        raise ValueError(
            f"Não encontrei a propriedade '{prop_name}' em {file_path}. "
            f"Confira o nome exato dentro do arquivo."
        )

    with open(file_path, "w") as f:
        f.write(new_content)


def run_commands(case_dir, commands, label):
    """Roda uma lista de comandos dentro da pasta do caso, salvando log de cada um."""
    for cmd in commands:
        log_name = f"{label}_{cmd[0]}.log"
        log_path = os.path.join(case_dir, log_name)
        with open(log_path, "w") as logf:
            result = subprocess.run(
                cmd, cwd=case_dir, stdout=logf, stderr=subprocess.STDOUT
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Comando {cmd} falhou em {case_dir}. Veja o log em {log_path}."
            )


def extract_field_matrices(case_dir):
    """
    Lê postProcessing/<SAMPLE_FUNCTION_NAME>/<tempo>/ e monta, para cada
    campo em FIELDS, uma matriz campo(x, t) (linhas = x, colunas = tempo).
    Aceita um arquivo .xy por campo (linha1_p.xy, ...) ou um arquivo
    combinado (linha1_p_Sa_Sb.xy).
    """
    sample_dir = os.path.join(case_dir, "postProcessing", SAMPLE_FUNCTION_NAME)

    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(
            f"Não encontrei {sample_dir}. Confira se a function object "
            f"'{SAMPLE_FUNCTION_NAME}' está no controlDict do caso base "
            f"e se o IMPESFoam rodou até salvar pelo menos um tempo."
        )

    time_folders = sorted(
        (d for d in os.listdir(sample_dir)
         if os.path.isdir(os.path.join(sample_dir, d))),
        key=lambda t: float(t),
    )

    if not time_folders:
        raise RuntimeError(f"Nenhuma pasta de tempo encontrada em {sample_dir}.")

    x_ref = None
    times = []
    columns_per_field = {field: [] for field in FIELDS}

    for t in time_folders:
        folder = os.path.join(sample_dir, t)
        times.append(float(t))

        for field in FIELDS:
            per_field = glob.glob(os.path.join(folder, f"{SET_NAME}_{field}.xy"))
            if per_field:
                data = np.genfromtxt(per_field[0], comments="#")
                col = 1
            else:
                combined = [
                    m for m in glob.glob(os.path.join(folder, f"{SET_NAME}_*.xy"))
                    if field in os.path.basename(m)[len(SET_NAME) + 1:-3].split("_")
                ]
                if not combined:
                    raise FileNotFoundError(
                        f"Não achei arquivo do campo '{field}' em {folder}. "
                        f"Arquivos: {os.listdir(folder)}"
                    )
                names = os.path.basename(combined[0])[len(SET_NAME) + 1:-3].split("_")
                data = np.genfromtxt(combined[0], comments="#")
                col = 1 + names.index(field)

            if data.ndim == 1:
                data = data.reshape(1, -1)
            if col >= data.shape[1]:
                raise ValueError(
                    f"Campo '{field}' esperado na coluna {col}, mas o arquivo "
                    f"tem só {data.shape[1]} colunas."
                )

            if x_ref is None:
                x_ref = data[:, 0]
            columns_per_field[field].append(data[:, col])

    times = np.array(times)
    matrices = {
        field: np.array(columns_per_field[field]).T
        for field in FIELDS
    }
    return x_ref, times, matrices


def save_field_matrix(path, x, times, matrix):
    """Salva campo(x,t) em CSV: 1ª linha = tempos, 1ª coluna = x."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x/t"] + [f"{t:g}" for t in times])
        for i, xv in enumerate(x):
            writer.writerow([f"{xv:g}"] + list(matrix[i, :]))


def load_field_matrix(path):
    """Lê o CSV gerado por save_field_matrix() -> (x, times, matrix)."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    times = np.array([float(t) for t in header[1:]])

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, 0], times, data[:, 1:]


def prepare_and_run(i, param_set):
    """Clona o caso, edita os parâmetros, roda o solver e salva os CSVs."""
    case_name = f"run_{i}"
    case_dir = os.path.join(OUTPUT_ROOT, case_name)

    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(BASE_CASE, case_dir)

    status = "OK"
    try:
        for prop_name, value in param_set.items():
            edit_property(case_dir, PROPERTY_FILES[prop_name], prop_name, value)

        run_commands(case_dir, RUN_COMMANDS, label="solver")

        x, times, matrices = extract_field_matrices(case_dir)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        for field, matrix in matrices.items():
            field_path = os.path.join(RESULTS_DIR, f"{case_name}_{field}.csv")
            save_field_matrix(field_path, x, times, matrix)

    except Exception as e:
        status = f"ERRO: {e}"
        print(f"  {case_name} -> {status}")

    row = {
        "run": case_name,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    row.update(param_set)
    return row


def append_log_row(row):
    """Acrescenta uma linha ao log (cria o arquivo com cabeçalho se preciso)."""
    log_path = os.path.join(OUTPUT_ROOT, LOG_FILE)
    fieldnames = ["run", "status", "timestamp"] + PARAM_NAMES
    new_file = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


_run_counter = 0   # numera as rodadas: run_0, run_1, ...


def simulate(param_vector):
    """
    Roda UMA simulação com o vetor de parâmetros dado.
    Devolve (t_mod, p_mod) ou None se a simulação falhou.
    """
    global _run_counter
    i = _run_counter
    _run_counter += 1

    param_set = {name: float(v) for name, v in zip(PARAM_NAMES, param_vector)}
    row = prepare_and_run(i, param_set)
    append_log_row(row)

    case_dir = os.path.join(OUTPUT_ROOT, f"run_{i}")

    if row["status"] != "OK":
        return None

    _, t_mod, p_mod = load_field_matrix(
        os.path.join(RESULTS_DIR, f"run_{i}_p.csv")
    )

    return t_mod, p_mod


def residuals(param_vector, times_exp, p_exp):
    times_exp = np.asarray(times_exp, dtype=float)
    p_exp = np.asarray(p_exp, dtype=float)

    out = simulate(param_vector)

    t_mod, p_mod = out
    t_mod = np.asarray(t_mod, dtype=float)
    p_mod = np.asarray(p_mod, dtype=float)

    t_mod = t_mod[1:]
    p_mod = p_mod[:, 1:]

    delta_p_mod = p_mod[0, :] - p_mod[-1, :]

    res = (p_exp - delta_p_mod)

    params_str = ", ".join(f"{n}={v}" for n, v in zip(PARAM_NAMES, param_vector))
    print(f"[avaliação {_run_counter - 1}] {params_str} | soma dos quadrados = {np.sum(res ** 2):.6g}")

    return res


def load_experimental_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    indices = data[:, 0]
    p_target = data[:, 1]

    if not np.all(np.isfinite(indices)) or not np.all(np.isfinite(p_target)):
        raise ValueError("CSV experimental contém valores não finitos (NaN/Inf).")

    n = len(indices)

    if not np.allclose(indices, np.arange(n)):
        raise ValueError(
            "A primeira coluna do CSV deveria ser o índice sequencial 0..N-1, "
            f"mas os valores lidos foram: {indices[:5]}...{indices[-5:]}."
        )

    times_exp = indices * (T_FINAL_EXP / (n - 1))

    print("=" * 60)
    print(f"Dados experimentais carregados de: {file_path}")
    print(f"Pontos: {n} -> tempo de 0 s a {times_exp[-1]:.6g} s "
          f"(passo uniforme de {T_FINAL_EXP / (n):.6g} s)")
    print("=" * 60)

    return times_exp, p_target


def main():
    x0 = [INITIAL_GUESS[n] for n in PARAM_NAMES]

    print("=" * 60)
    print("CALIBRAÇÃO DE PARÂMETROS COM DELTA DE PRESSÃO")
    print(f"Chute inicial: {INITIAL_GUESS}")
    print(f"Máximo de avaliações: {MAX_NFEV}")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # começa um log novo
    log_path = os.path.join(OUTPUT_ROOT, LOG_FILE)
    if os.path.isfile(log_path):
        os.remove(log_path)

    times_exp, p_exp = load_experimental_data(EXP_FILE)

    result = minimize(
        residuals,
        x0=x0,
        bounds= BOUNDS,
        args=(times_exp, p_exp),
        options = {"maxiter": 500}
    )

    print("\n" + "=" * 60)
    print("CALIBRAÇÃO FINALIZADA")
    print(f"Status: {result.status} - {result.message}")
    print(f"Avaliações da função: {result.nfev}")
    print(f"Custo final: {result.cost}")
    for name, v in zip(PARAM_NAMES, result.x):
        print(f"  {name} = {v:.6g}")
    print("=" * 60)

    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PARAM_NAMES + ["cost", "nfev", "status"])
        writer.writerow(list(result.x) + [result.fun, result.nfev, result.status])
    print(f"Resultado salvo em: {RESULT_FILE}")


if __name__ == "__main__":
    main()