"""
Calibração de parâmetros do IMPESFoam (a_exp, b_exp, kra_max, krb_max)
usando pressão (delta_p) + perfil de saturação Sw(x,t), via least_squares.

A cada avaliação:
  1. Copia o caso base para parametric_runs/run_<n>
  2. Edita os 4 parâmetros com os valores propostos pelo otimizador
  3. Roda o IMPESFoam (a function object "sets" gera a amostragem de p e Sw)
  4. Monta p_mod(x,t) e Sw_mod(x,t)
  5. Calcula delta_p_mod(t) e interpola Sw_mod nos pontos experimentais (x,t)
  6. Devolve o vetor de resíduos combinado (pressão + saturação)

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
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------
# CONFIGURAÇÃO — ajuste aqui
# ---------------------------------------------------------------

BASE_CASE = "Kc200_Sb"          # pasta do seu caso já funcional

PROPERTY_FILES = {
    "a_exp":   "constant/transportProperties",
    "b_exp":   "constant/transportProperties",
    "kra_max": "constant/transportProperties",
    "krb_max": "constant/transportProperties",
}

INITIAL_GUESS = {"a_exp": 1.90, "b_exp": 1.90, "kra_max": 0.90, "krb_max": 0.90}
PARAM_NAMES = list(INITIAL_GUESS.keys())

LOWER_BOUNDS = [1, 1, 0.01, 0.01]
UPPER_BOUNDS = [6, 6, 3, 3]

DIFF_STEP = [0.05, 0.05, 0.05, 0.05]

MAX_NFEV = 500

RUN_COMMANDS = [
    ["impesFoam2ph"]
]

# Devem bater com system/controlDict
SAMPLE_FUNCTION_NAME = "minhaAmostra"
SET_NAME = "linha1"

# Nome do campo de saturação da água NOS ARQUIVOS .xy DO OPENFOAM.
FIELDS = ["p", "Sb"]

# ---- dados experimentais: pressão ----
EXP_FILE = "delta_p_exp.csv"
T_FINAL_EXP = 150      # tempo (s) do último índice do CSV de pressão

# ---- dados experimentais: saturação ----
SAT_EXP_FILE = "Sw_exp.csv"
# tempo (s) do último índice de tempo do Sw_exp.csv
T_FINAL_SAT_EXP = 150.0   

# Comprimento real do domínio experimental em x (o "T_FINAL_EXP" da posição).
X_TOTAL_EXP = 0.051 

# Pesos relativos entre os dois blocos de resíduos (pressão vs. saturação).
# Comece com 1.0/1.0 e ajuste depois de olhar a magnitude de cada SSQ
# separadamente (ver print de diagnóstico dentro de residuals_combined).
WEIGHT_PRESSURE = 1.0
WEIGHT_SATURATION = 0.5

OUTPUT_ROOT = "parametric_runs"
RESULTS_DIR = "results"
LOG_FILE = "parametric_log.csv"
RESULT_FILE = "calibration_result.csv"

KEEP_RUN_DIRS = False

PENALTY_RESIDUAL = 1e3


def edit_property(case_dir, rel_path, prop_name, new_value):
    """Troca só o número de uma propriedade num dicionário do OpenFOAM."""
    file_path = os.path.join(case_dir, rel_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Não encontrei {file_path}. Confira PROPERTY_FILES.")

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
    for cmd in commands:
        log_name = f"{label}_{cmd[0]}.log"
        log_path = os.path.join(case_dir, log_name)
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, cwd=case_dir, stdout=logf, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Comando {cmd} falhou em {case_dir}. Veja o log em {log_path}.")


def extract_field_matrices(case_dir):
    """
    Lê postProcessing/<SAMPLE_FUNCTION_NAME>/<tempo>/ e monta, para cada
    campo em FIELDS, uma matriz campo(x, t) (linhas = x, colunas = tempo).
    """
    sample_dir = os.path.join(case_dir, "postProcessing", SAMPLE_FUNCTION_NAME)

    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(
            f"Não encontrei {sample_dir}. Confira se a function object "
            f"'{SAMPLE_FUNCTION_NAME}' está no controlDict e se o solver "
            f"rodou até salvar pelo menos um tempo."
        )

    time_folders = sorted(
        (d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))),
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
    matrices = {field: np.array(columns_per_field[field]).T for field in FIELDS}
    return x_ref, times, matrices


def save_field_matrix(path, x, times, matrix):
    """Salva campo(x,t) em CSV: 1ª linha = tempos, 1ª coluna = x."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x/t"] + [f"{t:g}" for t in times])
        for i, xv in enumerate(x):
            writer.writerow([f"{xv:g}"] + list(matrix[i, :]))


def load_field_matrix(path):
    """Lê o CSV gerado por save_field_matrix() -> (x, times, matrix), formato
    com valores REAIS no cabeçalho (usado para os resultados da simulação)."""
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
    Devolve (x_mod, t_mod, matrices) com matrices = {"p": ..., "Sw": ...},
    ou None se a simulação falhou.
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

    matrices = {}
    x_mod = t_mod = None
    for field in FIELDS:
        x_mod, t_mod, matrices[field] = load_field_matrix(
            os.path.join(RESULTS_DIR, f"run_{i}_{field}.csv")
        )

    if not KEEP_RUN_DIRS:
        shutil.rmtree(case_dir, ignore_errors=True)

    return x_mod, t_mod, matrices


# ---------------------------------------------------------------
# LEITURA DOS DADOS EXPERIMENTAIS
# ---------------------------------------------------------------

def load_experimental_data(file_path):
    """Delta de pressão experimental: índice sequencial 0..N-1 -> tempo real."""
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
    print(f"Dados experimentais de pressão carregados de: {file_path}")
    print(f"Pontos: {n} -> tempo de 0 s a {times_exp[-1]:.6g} s")
    print("=" * 60)

    return times_exp, p_target


def load_saturation_experimental(file_path, t_final_exp, x_total_exp):
    """
    Lê o Sw_exp.csv

    Converte os dois eixos de índice para valores reais usando
    t_final_exp e x_total_exp, e devolve no formato (x, t, matrix)
    já com x nas LINHAS e t nas COLUNAS -- mesma convenção usada para
    p_mod/Sw_mod ao longo do script (facilita reaproveitar a interpolação).
    """
    if x_total_exp is None:
        raise ValueError(
            "X_TOTAL_EXP não foi definido. Preencha o comprimento real do "
            "domínio experimental em x no topo do script antes de rodar."
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    with open(file_path, newline="") as f:
        header = next(csv.reader(f))
    x_idx = np.array([float(v) for v in header[1:]])

    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t_idx = data[:, 0]
    S_data = data[:, 1:]   # forma (n_t, n_x): linhas = tempo, colunas = x

    if not np.all(np.isfinite(t_idx)) or not np.all(np.isfinite(S_data)):
        raise ValueError(f"{file_path} contém valores não finitos (NaN/Inf).")

    n_t = len(t_idx)
    n_x = len(x_idx)

    if not np.allclose(t_idx, np.arange(n_t)):
        raise ValueError(
            "A 1a coluna do Sw_exp.csv deveria ser o índice de tempo 0..n_t-1, "
            f"mas veio: {t_idx[:5]}...{t_idx[-5:]}."
        )
    if not np.allclose(x_idx, np.arange(n_x)):
        raise ValueError(
            "O cabeçalho do Sw_exp.csv deveria ser o índice de posição 0..n_x-1, "
            f"mas veio: {x_idx[:5]}...{x_idx[-5:]}."
        )

    t_exp = t_idx * (t_final_exp / (n_t - 1))
    x_exp = x_idx * (x_total_exp / (n_x - 1))

    # transpõe para (n_x, n_t): x nas linhas, t nas colunas
    S_exp = S_data.T

    print("=" * 60)
    print(f"Perfil de saturação experimental carregado de: {file_path}")
    print(f"Pontos em x: {n_x} (0 a {x_exp[-1]:.6g}) | "
          f"pontos em t: {n_t} (0 a {t_exp[-1]:.6g} s)")
    print("=" * 60)

    return x_exp, t_exp, S_exp


# ---------------------------------------------------------------
# RESÍDUOS COMBINADOS: PRESSÃO + SATURAÇÃO
# ---------------------------------------------------------------

def _relative_residual(exp, mod_interp):
    """Resíduo relativo com piso no denominador."""
    exp = np.asarray(exp, dtype=float)
    mod_interp = np.asarray(mod_interp, dtype=float)
    ref = np.max(np.abs(exp))
    piso = 1e-2 * ref if ref > 0 else 1.0
    denom = np.maximum(np.abs(exp), piso)
    return (exp - mod_interp) / denom


def residuals_combined(param_vector, times_exp_p, p_exp, x_exp_sat, t_exp_sat, S_exp):
    n_p = len(p_exp)
    n_sat = S_exp.size
    penalty = np.full(n_p + n_sat, PENALTY_RESIDUAL, dtype=float)

    out = simulate(param_vector)

    x_mod, t_mod, matrices = out
    p_mod = matrices["p"]
    S_mod = matrices["Sb"]

    # ---- bloco de pressão ----
    t_mod_p = np.asarray(t_mod[1:], dtype=float)
    delta_p_mod = p_mod[0, 1:] - p_mod[-1, 1:]

    delta_p_interp = np.interp(times_exp_p, t_mod_p, delta_p_mod)
    res_p = _relative_residual(p_exp, delta_p_interp)

    # ---- bloco de saturação (interpolação 2D em x e t) ----
    order_x = np.argsort(x_mod)
    x_mod_sorted = x_mod[order_x]
    S_mod_sorted = S_mod[order_x, :]

    t_mod_full = np.asarray(t_mod, dtype=float)

    try:
        interp_S = RegularGridInterpolator(
            (x_mod_sorted, t_mod_full), S_mod_sorted,
            bounds_error=False, fill_value=None,
        )
    except ValueError as e:
        print(f"  saturação: grade inválida para interpolação ({e}) -> penalidade")
        return penalty

    Xg, Tg = np.meshgrid(x_exp_sat, t_exp_sat, indexing="ij")   # (n_x, n_t)
    pontos = np.column_stack([Xg.ravel(), Tg.ravel()])
    S_mod_interp = interp_S(pontos).reshape(S_exp.shape)

    if not np.all(np.isfinite(S_mod_interp)):
        print("  saturação: interpolação não finita -> penalidade")
        return penalty

    res_sat = _relative_residual(S_exp.ravel(), S_mod_interp.ravel())

    # ---- combinação ponderada ----
    res_p_w = WEIGHT_PRESSURE * res_p / np.sqrt(n_p)
    res_sat_w = WEIGHT_SATURATION * res_sat / np.sqrt(n_sat)
    res = np.concatenate([res_p_w, res_sat_w])

    params_str = ", ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, param_vector))
    print(f"[avaliação {_run_counter - 1}] {params_str} | "
          f"SSQ_p={np.sum(res_p_w**2):.4g}  SSQ_sat={np.sum(res_sat_w**2):.4g}  "
          f"total={np.sum(res**2):.6g}")

    return res


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    x0 = [INITIAL_GUESS[n] for n in PARAM_NAMES]

    print("=" * 60)
    print("CALIBRAÇÃO DE PARÂMETROS COM PRESSÃO + SATURAÇÃO")
    print(f"Chute inicial: {INITIAL_GUESS}")
    print(f"Máximo de avaliações: {MAX_NFEV}")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_ROOT, LOG_FILE)
    if os.path.isfile(log_path):
        os.remove(log_path)

    times_exp, p_exp = load_experimental_data(EXP_FILE)
    x_exp_sat, t_exp_sat, S_exp = load_saturation_experimental(
        SAT_EXP_FILE, T_FINAL_SAT_EXP, X_TOTAL_EXP
    )

    result = least_squares(
        residuals_combined,
        x0=x0,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        args=(times_exp, p_exp, x_exp_sat, t_exp_sat, S_exp),
        max_nfev=MAX_NFEV,
        diff_step=DIFF_STEP,
        xtol=1e-12,
    )

    print("\n" + "=" * 60)
    print("CALIBRAÇÃO FINALIZADA")
    print(f"Status: {result.status} - {result.message}")
    print(f"Avaliações da função: {result.nfev}")
    print(f"Custo final: {result.cost}")
    for name, v in zip(PARAM_NAMES, result.x):
        print(f"  {name} = {v:.6g}")
    print("=" * 60)

    try:
        J = result.jac
        _, s, _ = np.linalg.svd(J)
        cov = np.linalg.inv(J.T @ J)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        print("valores singulares de J:", s)
        print("cond(J) =", s[0] / s[-1])
        print("matriz de correlação entre parâmetros:")
        print(corr)
    except np.linalg.LinAlgError:
        print("Não foi possível calcular a covariância (J^T J singular).")

    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PARAM_NAMES + ["cost", "nfev", "status"])
        writer.writerow(list(result.x) + [result.cost, result.nfev, result.status])
    print(f"Resultado salvo em: {RESULT_FILE}")


if __name__ == "__main__":
    main()