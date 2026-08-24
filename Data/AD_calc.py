import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
NBITS = 2048
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
_POOL = {}


def make_fp(smi):
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return None
    return _MORGAN.GetFingerprint(mol)


def fingerprints(smiles_list):
    fps, bad = [], 0
    for s in smiles_list:
        f = make_fp(s)
        if f is None:
            bad += 1
        else:
            fps.append(f)
    if bad:
        print(f'  [warn] {bad} SMILES could not be parsed and were skipped')
    return fps


def _init(pool_fps):
    _POOL['fps'] = pool_fps


def _train_nn(i):
    fps = _POOL['fps']
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    sims[i] = 0.0  # exclude self
    return 1.0 - max(sims)


def _query_nn(args):
    f, = args
    fps = _POOL['fps']
    sims = DataStructs.BulkTanimotoSimilarity(f, fps)
    return 1.0 - max(sims)


def load_smiles(path):
    df = pd.read_csv(path)
    col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    return df[col].astype(str).tolist(), col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='training set CSV with a SMILES column')
    ap.add_argument('--query', default=None, help='optional query CSV with a SMILES column')
    ap.add_argument('--sample', type=int, default=10000, help='# training molecules sampled for the distance distribution')
    ap.add_argument('--percentile', type=float, default=10.0, help='AD threshold percentile of within-training NN distances')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_jobs', type=int, default=mp.cpu_count())
    ap.add_argument('--out', default='AD_results')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    t0 = time.time()
    print('[1/4] Loading training set and computing Morgan fingerprints ...')
    train_smiles, _ = load_smiles(args.train)
    train_fps = fingerprints(train_smiles)
    print(f'       {len(train_fps)} valid training molecules (of {len(train_smiles)}) '
          f'in {time.time()-t0:.1f}s')

    print('[2/4] Computing within-training nearest-neighbor distance distribution ...')
    sample_size = min(args.sample, len(train_fps))
    sample_idx = sorted(random.sample(range(len(train_fps)), sample_size))
    sample_fps = [train_fps[i] for i in sample_idx]
    with mp.Pool(args.n_jobs, initializer=_init, initargs=(sample_fps,)) as pool:
        dists = np.array(pool.map(_train_nn, range(sample_size)))
    p = args.percentile
    T = float(np.percentile(dists, p))
    print(f'       sampled n={sample_size}  |  10th percentile = {T:.4f}')
    for q in (5, 10, 25, 50):
        print(f'         {q}th pct: {np.percentile(dists, q):.4f}   '
              f'(sim {1-np.percentile(dists, q):.4f})')
    print(f'       distance-based AD threshold T = {T:.4f}  '
          f'(equiv. min nearest-neighbor Tanimoto sim >= {1-T:.4f})')

    # Save distribution + threshold
    np.save(os.path.join(args.out, 'AD_training_nn_distances.npy'), dists)
    with open(os.path.join(args.out, 'AD_threshold.txt'), 'w') as fh:
        fh.write(f'percentile={p}\nthreshold_distance={T:.6f}\n'
                 f'threshold_min_sim={1-T:.6f}\nsample_n={sample_size}\n')
    try:
        _plot_histogram(dists, T, os.path.join(args.out, 'AD_training_distribution.png'))
    except Exception as e:
        print(f'  [warn] figure not generated: {e}')

    # Optional: flag query compounds
    if args.query:
        print('[3/4] Evaluating query compounds ...')
        q_smiles, q_col = load_smiles(args.query)
        q_fps = fingerprints(q_smiles)
        with mp.Pool(args.n_jobs, initializer=_init, initargs=(train_fps,)) as pool:
            q_dists = np.array(pool.map(_query_nn, [(f,) for f in q_fps]))
        n_in = int(np.sum(q_dists <= T))
        print(f'       {len(q_fps)} valid queries | inside AD: {n_in} '
              f'({n_in/len(q_fps)*100:.1f}%) | outside AD: {len(q_fps)-n_in}')
        out_df = pd.DataFrame({
            'SMILES': q_smiles,
            'nearest_sim': 1.0 - q_dists,
            'nn_distance': q_dists,
            'in_AD': q_dists <= T,
        })
        out_df.to_csv(os.path.join(args.out, 'query_AD.csv'), index=False)
        print(f'       results saved to {os.path.join(args.out, "query_AD.csv")}')

    print(f'[4/4] Done in {time.time()-t0:.1f}s. Outputs in {os.path.abspath(args.out)}/')


def _plot_histogram(dists, T, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dists, bins=60, color='#4C72B0', edgecolor='white', alpha=0.85)
    ax.axvline(T, color='#C44E52', linestyle='--', linewidth=1.5,
               label=f'AD threshold (P10) = {T:.3f}')
    ax.set_xlabel('Distance to nearest neighbor in training set (1 - Tanimoto)')
    ax.set_ylabel('Number of training molecules')
    ax.set_title('Within-training nearest-neighbor distance distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
