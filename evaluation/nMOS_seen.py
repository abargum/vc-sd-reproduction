import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import scikit_posthocs as sp
from itertools import combinations

np.random.seed(42)

N_BOOTSTRAP = 10000  
CI_ALPHA = 0.05     

INPUT_FILE = "evaluation/seen_survey_mos.csv"

def bootstrap_ci(data, statistic='mean', n_bootstrap=10000, alpha=0.05):
    """
    Calculate bootstrap confidence interval for a given statistic.
    
    Parameters:
    -----------
    data : array-like
        The data to bootstrap
    statistic : str
        The statistic to calculate ('mean' or 'median')
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    data = np.array(data)
    data = data[~np.isnan(data)] 
    
    if len(data) == 0:
        return (np.nan, np.nan)
    
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        
        if statistic == 'mean':
            bootstrap_stats.append(np.mean(sample))
        elif statistic == 'median':
            bootstrap_stats.append(np.median(sample))
    
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return (ci_lower, ci_upper)


df = pd.read_csv(INPUT_FILE)

print(f"\nData shape: {df.shape}")
print(f"Total participants: {len(df)}")

categories = [
    "source",
    "fake",
    "quickvc",
    "diffvc",
    "vcsd",
    "triaanvc"
]

results = {}
category_data = {}

print("\n" + "=" * 80)
print("\nCalculating statistics and confidence intervals...")
print("(This may take a moment due to bootstrap calculations)")
print("=" * 80)

for category in categories:
    cols = [c for c in df.columns if c.startswith(category)]
    values = df[cols].apply(pd.to_numeric, errors="coerce").stack()
    
    category_data[category] = values.dropna().values
    
    mean_ci = bootstrap_ci(values.dropna(), statistic='mean', n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA)

    results[category] = {
        "mean": values.mean(),
        "mean_ci_lower": mean_ci[0],
        "mean_ci_upper": mean_ci[1],
        "std": values.std(),
        "median": values.median(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "n": values.count()
    }

print("\n" + "=" * 80)
print("\nDescriptive Statistics:\n")
print(f"{'Category':<12} {'Mean ± 95% CI':<20} {'SD':<8} {'Median':<8} {'Q25':<8} {'Q75':<8} {'N':<8}")
print("-" * 80)

for cat, stats_dict in results.items():
    mean_margin = (stats_dict['mean_ci_upper'] - stats_dict['mean_ci_lower']) / 2
    mean_str = f"{stats_dict['mean']:.3f} ± {mean_margin:.3f}"
    
    print(
        f"{cat:<12} "
        f"{mean_str:<20} "
        f"{stats_dict['std']:>7.3f} "
        f"{stats_dict['median']:>7.3f} "
        f"{stats_dict['q25']:>7.3f} "
        f"{stats_dict['q75']:>7.3f} "
        f"{stats_dict['n']:>7.0f}"
    )

total_participants = len(df)
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isna().sum().sum()
missing_rate = missing_cells / total_cells

print("\n" + "="*70)
print("Survey Summary:")
print("="*70)
print(f"Total participants: {total_participants}")
print(f"Missing response rate: {missing_rate:.2%}")

print()
if results["fake"]["mean"] > 0.3:
    print("⚠️ ALERT: The 'fake' category mean is high.")
    print(f"Fake mean = {results['fake']['mean']:.3f} (threshold = 0.3)")
    print("This may indicate scam or low-quality responses.")
else:
    print("✅ The 'fake' category is within the expected low range.")

# -------------------------
# STATISTICAL ANALYSIS
# -------------------------
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Exclude 'fake' from statistical comparison (it's a control)
test_categories = [cat for cat in categories if cat != "fake"]

groups = [category_data[cat] for cat in test_categories]

# -------------------------
# Kruskal-Wallis H Test
# -------------------------
print("\n--- Kruskal-Wallis H Test ---")
h_stat, p_value = kruskal(*groups)

print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_value:.4e}")
print(f"Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# ε² = (H - k + 1) / (N - k)
# where k = number of groups, N = total sample size
k = len(groups)
N = sum(len(g) for g in groups)
epsilon_squared = (h_stat - k + 1) / (N - k)

print(f"\nEffect Size (ε²): {epsilon_squared:.4f}")
print(f"Interpretation: ", end="")
if epsilon_squared < 0.01:
    print("negligible")
elif epsilon_squared < 0.06:
    print("small")
elif epsilon_squared < 0.14:
    print("medium")
else:
    print("large")

# -------------------------
# Post-hoc Pairwise Comparisons
# -------------------------
if p_value < 0.05:
    print("\n" + "="*70)
    print("POST-HOC PAIRWISE COMPARISONS")
    print("="*70)
    
    print("\n--- Dunn's Test (Bonferroni correction) ---\n")
    
    data_list = []
    group_list = []
    for cat in test_categories:
        data_list.extend(category_data[cat])
        group_list.extend([cat] * len(category_data[cat]))
    
    dunn_df = pd.DataFrame({'data': data_list, 'group': group_list})
    
    dunn_results = sp.posthoc_dunn(dunn_df, val_col='data', group_col='group', p_adjust='bonferroni')
    
    print("Dunn's Test p-values (Bonferroni corrected):")
    print(dunn_results.round(4))
    
    print("\n--- Mann-Whitney U Tests (Bonferroni correction) ---\n")
    
    n_comparisons = len(list(combinations(test_categories, 2)))
    bonferroni_alpha = 0.05 / n_comparisons
    
    print(f"Number of pairwise comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}\n")
    
    print(f"{'Comparison':<30} {'U-stat':<12} {'p-value':<12} {'Sig':<8} {'Effect Size (r)':<15}")
    print("-" * 85)
    
    pairwise_results = []
    
    for cat1, cat2 in combinations(test_categories, 2):
        group1 = category_data[cat1]
        group2 = category_data[cat2]
        
        u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
        
        n1, n2 = len(group1), len(group2)
        r = 1 - (2*u_stat) / (n1 * n2) 
        
        if p_val < bonferroni_alpha:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
        else:
            sig = "ns"
        
        comparison = f"{cat1} vs {cat2}"
        print(f"{comparison:<30} {u_stat:>11.1f} {p_val:>11.4e} {sig:<8} {abs(r):>14.4f}")
        
        pairwise_results.append({
            'cat1': cat1,
            'cat2': cat2,
            'u_stat': u_stat,
            'p_val': p_val,
            'effect_size': abs(r),
            'significant': p_val < bonferroni_alpha
        })
    
    print("\n--- Summary of Significant Differences (Bonferroni-corrected) ---\n")
    
    sig_pairs = []
    for result in pairwise_results:
        if result['significant']:
            cat1 = result['cat1']
            cat2 = result['cat2']
            
            mean1 = results[cat1]['mean']
            mean2 = results[cat2]['mean']
            median1 = results[cat1]['median']
            median2 = results[cat2]['median']
            
            sig_pairs.append({
                'cat1': cat1,
                'cat2': cat2,
                'mean1': mean1,
                'mean2': mean2,
                'median1': median1,
                'median2': median2,
                'p_val': result['p_val'],
                'effect_size': result['effect_size']
            })
    
    if sig_pairs:
        for pair in sig_pairs:
            direction = ">" if pair['mean1'] > pair['mean2'] else "<"
            print(f"{pair['cat1']} (Mean={pair['mean1']:.3f}) {direction} {pair['cat2']} (Mean={pair['mean2']:.3f}), p={pair['p_val']:.4e}, r={pair['effect_size']:.3f}")
    else:
        print("No significant pairwise differences found after Bonferroni correction.")

else:
    print("\nKruskal-Wallis test was not significant. Post-hoc tests not performed.")

print("\n" + "="*70)
print("\nNote: Bootstrap CIs calculated using {} iterations".format(N_BOOTSTRAP))
print("="*70)