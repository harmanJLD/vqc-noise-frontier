# The Noise-Optimal Depth Frontier

Code and data for the paper **"The Noise-Optimal Depth Frontier: Joint Encoding-Depth Selection for Variational Quantum Classifiers under Hardware Noise"**, presented at the 4th International Workshop on Quantum Machine Learning: From Research to Practice (WKS51), IEEE Quantum Week 2026 (QCE26), Toronto, Canada.

## What this repository contains

| File | Description |
|---|---|
| `experiments.py` | Full experiment runner: trains 315 VQC configurations (3 encodings x 7 depths x 3 datasets x 5 seeds) noiselessly, then evaluates each frozen model under 8 depolarizing-noise scales with 64-shot readout. Writes one CSV row per (configuration, noise level). |
| `analyze.py` | Reads `runs.csv` and produces the accuracy-landscape heatmaps, the noise-optimal depth frontier figure, and the selection-rule table used in the paper. |
| `runs.csv` | Per-run results of the main campaign (finite-shot readout, 2,520 rows). Columns: encoding, depth, dataset, seed, noise_scale, test_acc, train_acc, shots, cx_count, n_var_params, p1_base, p2_base, wall_s. |
| `runs_exact_readout.csv` | Per-run results of the first campaign using exact expectation-value readout (2,205 rows). Retained as the cautionary baseline discussed in Section IV-A of the paper: under exact readout, thresholded accuracy is nearly invariant to depolarizing noise. |
| `frontier.pdf` | Figure 1 of the paper: the noise-optimal depth frontier per encoding. |
| `heatmap_circles.pdf`, `heatmap_cancer.pdf` | Figures 2 and 3 of the paper: accuracy landscapes. |
| `heatmap_moons.pdf` | The moons accuracy landscape, referenced in the paper but not printed for space. |

## Study design in brief

Three encodings with distinct post-transpilation two-qubit-gate profiles (angle: 0 CX, ZZ linear reps=1: 6 CX, ZZ full reps=2: 24 CX), RealAmplitudes ansatz at depths {1,2,3,4,5,6,8}, three binary tasks (moons, circles, breast cancer via PCA to 4 features). Noise is per-gate depolarizing, anchored to the FakeBrisbane calibration snapshot (median p1 = 2.43e-4, median p2 = 7.72e-3) and scaled by lambda in {0, 0.25, 0.5, 1, 2, 4, 8, 16}. Training is noiseless (COBYLA, fixed 200 iterations for every configuration); evaluation uses frozen parameters with 64 measurement shots per test point.

## Reproducing the results

Python 3.10+ recommended (3.9 works but is deprecated by Qiskit).
python -m venv venv && source venv/bin/activate
pip install qiskit qiskit-aer qiskit-ibm-runtime scikit-learn scipy pandas matplotlib
python experiments.py --smoke        # ~1 minute sanity check
python experiments.py --workers 8    # full grid, a few hours on a laptop
python analyze.py                    # figures and selection rule from results/runs.csv

`qiskit-ibm-runtime` is required for the calibration-anchored noise model; without it the script falls back to hardcoded rates and says so.

## Citation

If you use this code or data, please cite the paper:
H. Kaur, "The Noise-Optimal Depth Frontier: Joint Encoding-Depth Selection for
Variational Quantum Classifiers under Hardware Noise," in Proc. 4th Int.
Workshop on Quantum Machine Learning: From Research to Practice, IEEE Quantum
Week (QCE26), 2026.

## License

MIT
