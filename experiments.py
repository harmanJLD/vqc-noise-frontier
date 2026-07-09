"""
Noise-Optimal Depth Frontier: joint encoding-depth study for VQCs under hardware noise.
Submission 1551 -- WKS51 QCE26.

Protocol: train noiseless (statevector), evaluate frozen weights under scaled
depolarizing noise (density matrix), anchored to device-calibration medians.

Usage:
    python experiments.py                 # full grid (~hours)
    python experiments.py --smoke         # tiny grid to verify install (~1 min)
    python experiments.py --workers 8     # parallelize across processes

Output: results/runs.csv  (one row per (encoding, depth, dataset, seed, noise) point)
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
N_QUBITS = 4
DEPTHS = [1, 2, 3, 4, 5, 6, 8]
SEEDS = [0, 1, 2, 3, 4]
NOISE_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # lambda; 1.0 = anchored device point
SHOTS = 64              # finite-shot readout: deployment-realistic evaluation
MAX_ITER = 200          # fixed optimizer budget for ALL configs (design decision)
N_TRAIN, N_TEST = 200, 100
BASIS_GATES = ["cx", "rz", "sx", "x"]

# Anchored device medians. Attempt to pull from a fake-backend calibration
# snapshot; fall back to representative IBM Eagle/Heron medians if the
# qiskit-ibm-runtime package is unavailable. Record which path was used.
def anchored_error_rates():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        target = FakeBrisbane().target
        p1s, p2s = [], []
        for op_name in target.operation_names:
            try:
                qarg_map = target[op_name]
            except Exception:
                continue
            for qargs, iprops in qarg_map.items():
                err = getattr(iprops, "error", None)
                if err is None or qargs is None:
                    continue
                if len(qargs) == 2:
                    p2s.append(err)
                elif len(qargs) == 1 and op_name in ("sx", "x"):
                    p1s.append(err)
        if not p1s or not p2s:
            raise RuntimeError("no calibration data found in target")
        return float(np.median(p1s)), float(np.median(p2s)), "FakeBrisbane calibration snapshot"
    except Exception:
        # Representative medians for IBM 127q devices (cite device docs in paper)
        return 2.5e-4, 7.0e-3, "hardcoded IBM-Eagle-class medians (fallback)"

P1_BASE, P2_BASE, NOISE_SOURCE = anchored_error_rates()

# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
def get_dataset(name, seed):
    rng = np.random.RandomState(seed)
    if name == "moons":
        X, y = make_moons(n_samples=N_TRAIN + N_TEST, noise=0.15, random_state=seed)
        X = np.hstack([X, X**2])  # lift 2D -> 4 features
    elif name == "circles":
        X, y = make_circles(n_samples=N_TRAIN + N_TEST, noise=0.10, factor=0.5,
                            random_state=seed)
        X = np.hstack([X, X**2])
    elif name == "cancer":
        data = load_breast_cancer()
        idx = rng.choice(len(data.data), N_TRAIN + N_TEST, replace=False)
        X = PCA(n_components=N_QUBITS, random_state=seed).fit_transform(data.data[idx])
        y = data.target[idx]
    else:
        raise ValueError(name)
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    return train_test_split(X, y, train_size=N_TRAIN, test_size=N_TEST,
                            random_state=seed, stratify=y)

# ----------------------------------------------------------------------------
# Encodings (three distinct two-qubit-gate profiles)
# ----------------------------------------------------------------------------
def build_encoding(name):
    if name == "angle":                    # 0 two-qubit gates
        x = ParameterVector("x", N_QUBITS)
        qc = QuantumCircuit(N_QUBITS)
        for i in range(N_QUBITS):
            qc.ry(x[i], i)
        return qc
    if name == "zz_linear":                # O(n) two-qubit gates
        return zz_feature_map(N_QUBITS, reps=1, entanglement="linear")
    if name == "zz_full2":                 # O(n^2) two-qubit gates
        return zz_feature_map(N_QUBITS, reps=2, entanglement="full")
    raise ValueError(name)

ENCODINGS = ["angle", "zz_linear", "zz_full2"]
DATASETS = ["moons", "circles", "cancer"]

def full_circuit(encoding_name, depth):
    enc = build_encoding(encoding_name)
    ansatz = real_amplitudes(N_QUBITS, reps=depth, entanglement="linear")
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(enc, inplace=True)
    qc.compose(ansatz, inplace=True)
    enc_params = list(enc.parameters)
    var_params = list(ansatz.parameters)
    return qc, enc_params, var_params

def cx_count(qc):
    t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    return t.count_ops().get("cx", 0), t

# ----------------------------------------------------------------------------
# Model: P(class=1) = prob qubit 0 measured in |1>
# ----------------------------------------------------------------------------
def predict_proba_statevector(qc, enc_params, var_params, theta, X):
    probs = np.empty(len(X))
    for k, x in enumerate(X):
        bind = dict(zip(enc_params, x)) | dict(zip(var_params, theta))
        sv = Statevector(qc.assign_parameters(bind))
        p = sv.probabilities([0])          # marginal on qubit 0
        probs[k] = p[1]
    return probs

def bce_loss(theta, qc, enc_params, var_params, X, y):
    p = np.clip(predict_proba_statevector(qc, enc_params, var_params, theta, X),
                1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def train(qc, enc_params, var_params, X, y, seed):
    rng = np.random.RandomState(seed)
    theta0 = rng.uniform(-np.pi, np.pi, len(var_params))
    res = minimize(bce_loss, theta0, args=(qc, enc_params, var_params, X, y),
                   method="COBYLA", options={"maxiter": MAX_ITER, "rhobeg": 0.5})
    return res.x

# ----------------------------------------------------------------------------
# Noisy evaluation: scaled depolarizing model, density-matrix simulation
# ----------------------------------------------------------------------------
def noise_model(scale):
    if scale == 0.0:
        return None
    nm = NoiseModel(basis_gates=BASIS_GATES)
    p1, p2 = min(P1_BASE * scale, 0.75), min(P2_BASE * scale, 0.9375)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm

def eval_noisy(transpiled, enc_params, var_params, theta, X, y, scale, eval_seed):
    """Finite-shot readout under scaled depolarizing noise (deployment-realistic).

    Predict class 1 iff the fraction of shots with qubit 0 measured |1>
    exceeds 0.5. All test circuits are batched into a single simulator job.
    """
    nm = noise_model(scale)
    sim = AerSimulator(noise_model=nm) if nm is not None else AerSimulator()
    circuits = []
    for x in X:
        bind = dict(zip(enc_params, x)) | dict(zip(var_params, theta))
        qc = transpiled.assign_parameters(bind)
        qc.measure_all()
        circuits.append(qc)
    result = sim.run(circuits, shots=SHOTS, seed_simulator=eval_seed).result()
    correct = 0
    for i, label in enumerate(y):
        counts = result.get_counts(i)
        n1 = sum(v for k, v in counts.items() if k.replace(" ", "")[-1] == "1")
        correct += int((n1 / SHOTS > 0.5) == bool(label))
    return correct / len(y)

# ----------------------------------------------------------------------------
# One training config = (encoding, depth, dataset, seed) -> 1 training + all noise evals
# ----------------------------------------------------------------------------
def run_config(cfg):
    enc_name, depth, ds_name, seed, noise_scales = cfg
    t0 = time.time()
    Xtr, Xte, ytr, yte = get_dataset(ds_name, seed)
    qc, ep, vp, = full_circuit(enc_name, depth)
    ncx, transpiled = cx_count(qc)

    theta = train(qc, ep, vp, Xtr, ytr, seed)
    train_acc = float(np.mean(
        (predict_proba_statevector(qc, ep, vp, theta, Xtr) > 0.5).astype(int) == ytr))

    rows = []
    for j, lam in enumerate(noise_scales):
        acc = eval_noisy(transpiled, ep, vp, theta, Xte, yte, lam,
                         eval_seed=1000 * seed + j)
        rows.append(dict(encoding=enc_name, depth=depth, dataset=ds_name, seed=seed,
                         noise_scale=lam, test_acc=acc, train_acc=train_acc,
                         shots=SHOTS,
                         cx_count=ncx, n_var_params=len(vp),
                         p1_base=P1_BASE, p2_base=P2_BASE,
                         wall_s=round(time.time() - t0, 1)))
    print(f"done {enc_name} d={depth} {ds_name} s={seed} "
          f"(cx={ncx}, train={train_acc:.2f}, {time.time()-t0:.0f}s)", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid to verify setup")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="results/runs.csv")
    args = ap.parse_args()

    if args.smoke:
        encs, depths, dss, seeds, noises = ["angle"], [1, 3], ["moons"], [0], [0.0, 8.0]
        global MAX_ITER, N_TRAIN, N_TEST
        MAX_ITER, N_TRAIN, N_TEST = 30, 40, 20
    else:
        encs, depths, dss, seeds, noises = ENCODINGS, DEPTHS, DATASETS, SEEDS, NOISE_SCALES

    configs = [(e, d, ds, s, noises)
               for e, d, ds, s in itertools.product(encs, depths, dss, seeds)]
    print(f"Noise anchor: p1={P1_BASE:.2e}, p2={P2_BASE:.2e} ({NOISE_SOURCE})")
    print(f"{len(configs)} training configs x {len(noises)} noise levels "
          f"on {args.workers} workers")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.workers) as pool:
        all_rows = [r for rows in pool.imap_unordered(run_config, configs) for r in rows]

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
