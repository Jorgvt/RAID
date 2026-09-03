#!/usr/bin/env python3
"""
benchmark_metrics.py
--------------------
Utility script to benchmark and calibrate Image Quality Assessment (IQA) and perceptual
distance metrics against human Just-Objectionable-Difference (JOD) datasets (RAID & TID2013).

Uses standard ITU-R BT.500 / VQEG 4-parameter logistic psychometric calibration.

Usage:
    python benchmark_metrics.py --help
    python benchmark_metrics.py --demo
    python benchmark_metrics.py --dataset raid_jod.csv --metric_col your_metric_dist
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. 4-Parameter Logistic Psychometric Model
# ---------------------------------------------------------

def logistic_4p(d, b1, b2, b3, b4):
    """
    Standard VQEG 4-Parameter Logistic Mapping function.
    Maps raw distance d monotonically to target JOD scale.
    """
    d = np.asarray(d, dtype=float)
    b4_safe = np.maximum(np.abs(b4), 1e-6)
    z = np.clip(-(d - b3) / b4_safe, -50.0, 50.0)
    return b2 + (b1 - b2) / (1.0 + np.exp(z))


def fit_logistic_4p(x, y, max_iter=200):
    """
    Fits 4-parameter logistic mapping parameters [b1, b2, b3, b4] using Levenberg-Marquardt.
    Falls back to scipy.optimize.curve_fit if available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Try scipy curve_fit if available and compatible
    try:
        from scipy.optimize import curve_fit
        # If x and y positively correlate, b1 > b2, else b2 > b1
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 1.0
        if corr >= 0:
            p0 = [np.max(y), np.min(y), float(np.median(x)), float(max(np.std(x), 0.1))]
        else:
            p0 = [np.min(y), np.max(y), float(np.median(x)), float(max(np.std(x), 0.1))]
        popt, _ = curve_fit(logistic_4p, x, y, p0=p0, maxfev=10000)
        return popt
    except Exception:
        pass

    # Pure NumPy Levenberg-Marquardt optimizer
    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 1.0
    if corr >= 0:
        b1, b2 = float(np.max(y)), float(np.min(y))
    else:
        b1, b2 = float(np.min(y)), float(np.max(y))
    b3 = float(np.median(x))
    b4 = float(max(np.std(x), 0.1))
    p = np.array([b1, b2, b3, b4], dtype=float)
    
    lam = 1e-2
    for _ in range(max_iter):
        b1, b2, b3, b4 = p
        b4_safe = max(abs(b4), 1e-6)
        z = np.clip(-(x - b3) / b4_safe, -50.0, 50.0)
        exp_z = np.exp(z)
        sig = 1.0 / (1.0 + exp_z)
        y_pred = b2 + (b1 - b2) * sig
        r = y_pred - y
        cost = np.sum(r**2)
        
        d_sig = sig * (1.0 - sig)
        J1 = sig
        J2 = 1.0 - sig
        J3 = (b1 - b2) * d_sig / b4_safe
        J4 = (b1 - b2) * d_sig * (x - b3) * np.sign(b4) / (b4_safe**2)
        J = np.stack([J1, J2, J3, J4], axis=1)
        
        JT_J = J.T @ J
        JT_r = J.T @ r
        diag_JT_J = np.diag(np.diag(JT_J) + 1e-5)
        
        try:
            dp = np.linalg.solve(JT_J + lam * diag_JT_J, -JT_r)
        except np.linalg.LinAlgError:
            dp = -1e-3 * JT_r
            
        p_new = p + dp
        r_new = logistic_4p(x, *p_new) - y
        cost_new = np.sum(r_new**2)
        
        if cost_new < cost:
            p = p_new
            lam /= 5.0
            if abs(cost - cost_new) < 1e-7:
                break
        else:
            lam *= 5.0
            if lam > 1e8:
                break
                
    return p


# ---------------------------------------------------------
# 2. Evaluation Metrics (SROCC, KROCC, PLCC, RMSE, MAE)
# ---------------------------------------------------------

def compute_metrics(y_true, y_pred, raw_d):
    """
    Computes all standard IQA benchmark performance metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    raw_d = np.asarray(raw_d, dtype=float)
    
    # PLCC (Pearson Linear Correlation Coefficient)
    if np.std(y_pred) > 1e-7 and np.std(y_true) > 1e-7:
        plcc = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        plcc = 0.0
        
    # SROCC (Spearman Rank Order Correlation Coefficient)
    rank_d = np.argsort(np.argsort(raw_d))
    rank_true = np.argsort(np.argsort(y_true))
    srocc = float(np.corrcoef(rank_d, rank_true)[0, 1])
    
    # KROCC (Kendall Rank Order Correlation Coefficient)
    n = len(raw_d)
    if n > 1:
        # Concordant / Discordant pairs calculation
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                sign_d = np.sign(raw_d[i] - raw_d[j])
                sign_t = np.sign(y_true[i] - y_true[j])
                prod = sign_d * sign_t
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        krocc = (concordant - discordant) / (0.5 * n * (n - 1)) if n > 1 else 0.0
    else:
        krocc = 0.0
        
    # Errors in physical JOD units
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    
    return {
        'SROCC': srocc,
        'KROCC': krocc,
        'PLCC': plcc,
        'RMSE_JOD': rmse,
        'MAE_JOD': mae
    }


# ---------------------------------------------------------
# 3. K-Fold Cross-Validated Benchmark Engine
# ---------------------------------------------------------

def evaluate_metric(raw_distances, y_true_jod, n_splits=5, seed=42):
    """
    Runs K-fold cross-validated logistic psychometric evaluation.
    """
    raw_d = np.asarray(raw_distances, dtype=float)
    y_true = np.asarray(y_true_jod, dtype=float)
    n = len(raw_d)
    
    if n < n_splits or n_splits <= 1:
        # Train on full dataset
        popt = fit_logistic_4p(raw_d, y_true)
        y_pred = logistic_4p(raw_d, *popt)
    else:
        # K-Fold split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        folds = np.array_split(indices, n_splits)
        y_pred = np.zeros(n, dtype=float)
        
        for fold_idx in range(n_splits):
            test_idx = folds[fold_idx]
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            
            popt = fit_logistic_4p(raw_d[train_idx], y_true[train_idx])
            y_pred[test_idx] = logistic_4p(raw_d[test_idx], *popt)
            
    metrics = compute_metrics(y_true, y_pred, raw_d)
    metrics['y_pred'] = y_pred
    return metrics


def print_results_table(title, metrics):
    """Helper to format benchmark summary."""
    print("=" * 60)
    print(f"🎯  {title}")
    print("=" * 60)
    print(f"  • Spearman Rho  (SROCC) : {metrics['SROCC']:>7.4f}  (Monotonic rank order)")
    print(f"  • Kendall Tau   (KROCC) : {metrics['KROCC']:>7.4f}  (Pairwise consistency)")
    print(f"  • Pearson r     (PLCC)  : {metrics['PLCC']:>7.4f}  (Linear calibration)")
    print(f"  • RMSE (in JOD units)   : {metrics['RMSE_JOD']:>7.3f} JOD (Average physical error)")
    print(f"  • MAE  (in JOD units)   : {metrics['MAE_JOD']:>7.3f} JOD (Mean absolute error)")
    print("=" * 60)


# ---------------------------------------------------------
# 4. Main CLI
# ---------------------------------------------------------

def run_demo():
    """Runs a demonstration on RAID dataset using a mock perceptual metric."""
    print("\n🚀 Running Demo: Benchmarking a simulated perceptual metric on RAID dataset...")
    df = pd.read_csv('raid_jod.csv')
    
    # Simulate a deep perceptual metric (e.g. LPIPS-like distance with non-linear saturation and noise)
    true_jod = df['jod_distortion'].values
    np.random.seed(42)
    simulated_raw_distance = np.tanh(true_jod / 3.0) + np.random.normal(0, 0.05, len(true_jod))
    
    # 1. Overall evaluation
    results = evaluate_metric(simulated_raw_distance, true_jod, n_splits=5)
    print_results_table("OVERALL PERFORMANCE ACROSS ALL RAID CONDITIONS (960 pairs)", results)
    
    # 2. Per-distortion breakdown
    print("\n📊 PER-DISTORTION BREAKDOWN:")
    for dist in ['rotation', 'translation', 'scaling', 'additive_gaussian_noise']:
        sub = df[df['distortion_type'] == dist]
        sub_d = simulated_raw_distance[sub.index]
        sub_y = sub['jod_distortion'].values
        res_sub = evaluate_metric(sub_d, sub_y, n_splits=5)
        print(f"  [{dist.upper():<25}] SROCC: {res_sub['SROCC']:.4f} | PLCC: {res_sub['PLCC']:.4f} | RMSE: {res_sub['RMSE_JOD']:.3f} JOD")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark and calibrate IQA metrics into human JOD units.")
    parser.add_argument('--dataset', type=str, default='raid_jod.csv', help='Path to dataset CSV (raid_jod.csv, tid2013_jod.csv, or raid_tid2013_aligned.csv)')
    parser.add_argument('--metric_col', type=str, default=None, help='Column name containing raw metric distances/scores')
    parser.add_argument('--target_col', type=str, default='jod_distortion', help='Ground truth JOD column (default: jod_distortion)')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation splits (default: 5)')
    parser.add_argument('--by_distortion', action='store_true', help='Report per-distortion breakdown')
    parser.add_argument('--demo', action='store_true', help='Run self-contained demonstration')
    
    args = parser.parse_args()
    
    if args.demo or args.metric_col is None:
        run_demo()
        return

    df = pd.read_csv(args.dataset)
    if args.metric_col not in df.columns:
        print(f"❌ Error: Column '{args.metric_col}' not found in '{args.dataset}'.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    raw_d = df[args.metric_col].values
    y_true = df[args.target_col].values
    
    # Overall evaluation
    res = evaluate_metric(raw_d, y_true, n_splits=args.n_splits)
    print_results_table(f"EVALUATION RESULTS FOR '{args.metric_col}' ON '{args.dataset}'", res)
    
    # Optional breakdown
    if args.by_distortion and 'distortion_type' in df.columns:
        print("\n📊 PER-DISTORTION BREAKDOWN:")
        for dist in df['distortion_type'].unique():
            sub = df[df['distortion_type'] == dist]
            sub_d = df.loc[sub.index, args.metric_col].values
            sub_y = df.loc[sub.index, args.target_col].values
            res_sub = evaluate_metric(sub_d, sub_y, n_splits=args.n_splits)
            print(f"  [{dist:<30}] SROCC: {res_sub['SROCC']:.4f} | PLCC: {res_sub['PLCC']:.4f} | RMSE: {res_sub['RMSE_JOD']:.3f} JOD")
        print()


if __name__ == '__main__':
    main()
