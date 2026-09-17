#!/usr/bin/env python3
"""
benchmark_metrics.py
--------------------
Utility CLI to benchmark and calibrate perceptual distance metrics
against human Just-Objectionable-Difference (JOD) datasets.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from raid.psychometrics import (
    logistic_4p,
    fit_logistic_4p,
    evaluate_predictions
)


def evaluate_metric_cv(raw_distances, y_true_jod, n_splits=5, seed=42):
    """Runs K-fold cross-validated logistic psychometric evaluation."""
    raw_d = np.asarray(raw_distances, dtype=float)
    y_true = np.asarray(y_true_jod, dtype=float)
    n = len(raw_d)

    if n < n_splits or n_splits <= 1:
        popt = fit_logistic_4p(raw_d, y_true)
        y_pred = logistic_4p(raw_d, *popt)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_pred = np.zeros(n, dtype=float)
        for train_idx, test_idx in kf.split(raw_d):
            popt = fit_logistic_4p(raw_d[train_idx], y_true[train_idx])
            y_pred[test_idx] = logistic_4p(raw_d[test_idx], *popt)

    metrics = evaluate_predictions(y_true, y_pred, raw_d)
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


def run_demo():
    """Runs a demonstration on RAID dataset using a mock perceptual metric."""
    print("\n🚀 Running Demo: Benchmarking a simulated perceptual metric on RAID dataset...")
    df = pd.read_csv('raid_jod.csv')
    true_jod = df['jod_distortion'].values
    np.random.seed(42)
    simulated_raw_distance = np.tanh(true_jod / 3.0) + np.random.normal(0, 0.05, len(true_jod))

    results = evaluate_metric_cv(simulated_raw_distance, true_jod, n_splits=5)
    print_results_table("OVERALL PERFORMANCE ACROSS ALL RAID CONDITIONS (960 pairs)", results)

    print("\n📊 PER-DISTORTION BREAKDOWN:")
    for dist in ['rotation', 'translation', 'scaling', 'additive_gaussian_noise']:
        sub = df[df['distortion_type'] == dist]
        sub_d = simulated_raw_distance[sub.index]
        sub_y = sub['jod_distortion'].values
        res_sub = evaluate_metric_cv(sub_d, sub_y, n_splits=5)
        print(f"  [{dist.upper():<25}] SROCC: {res_sub['SROCC']:.4f} | PLCC: {res_sub['PLCC']:.4f} | RMSE: {res_sub['RMSE_JOD']:.3f} JOD")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark and calibrate IQA metrics into human JOD units.")
    parser.add_argument('--dataset', type=str, default='raid_jod.csv', help='Path to dataset CSV')
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

    res = evaluate_metric_cv(raw_d, y_true, n_splits=args.n_splits)
    print_results_table(f"EVALUATION RESULTS FOR '{args.metric_col}' ON '{args.dataset}'", res)

    if args.by_distortion and 'distortion_type' in df.columns:
        print("\n📊 PER-DISTORTION BREAKDOWN:")
        for dist in df['distortion_type'].unique():
            sub = df[df['distortion_type'] == dist]
            sub_d = df.loc[sub.index, args.metric_col].values
            sub_y = df.loc[sub.index, args.target_col].values
            res_sub = evaluate_metric_cv(sub_d, sub_y, n_splits=args.n_splits)
            print(f"  [{dist:<30}] SROCC: {res_sub['SROCC']:.4f} | PLCC: {res_sub['PLCC']:.4f} | RMSE: {res_sub['RMSE_JOD']:.3f} JOD")
        print()


if __name__ == '__main__':
    main()
