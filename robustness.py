"""
Robustness analysis: run simulation under alternative calibrations
and compute dGDP/d(fusion_year) at window edges.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Core simulation (extracted from sim.py, parameterised)
# ---------------------------------------------------------------------------

def run_sim(n_years=75, n_paths=10000,
            fusion_mu=2044, fusion_sigma=9,
            csens_mu=3.0, csens_sigma=0.4,
            brit_lo=0.1, brit_hi=0.25,
            growth_mu=0.028, growth_sigma=0.006,
            damage_exp=2.6, eroi_decay=0.12,
            inst_coeff=0.035, gdp_ceiling=2000,
            adoption_k=0.5):
    """Return dict of summary statistics for one calibration."""
    start_year = 2026
    years = np.arange(start_year, start_year + n_years)
    collapse_threshold = 1.05

    final_gdps = []
    depths = []
    durations = []
    no_downturn = 0
    fusion_years_drawn = []

    for _ in range(n_paths):
        T, E, Y, I = 1.3, 15.0, 105.0, 1.0
        initial_y = Y

        fusion_year = np.random.normal(fusion_mu, fusion_sigma)
        c_sensitivity = np.random.normal(csens_mu, csens_sigma)
        conflict_brittleness = np.random.uniform(brit_lo, brit_hi)
        base_growth = np.random.normal(growth_mu, growth_sigma)

        fusion_years_drawn.append(fusion_year)
        path_gdp = []
        recovered_year = None
        collapsed = False
        always_above = True

        for i, y in enumerate(years):
            prev_y = Y
            dT = (0.04 * (1.0 - 0.018 * (y - start_year)) * (Y / 100)
                  * (c_sensitivity / 3.0) * (15.0 / E)) + np.random.normal(0, 0.04)
            T += max(0, dT)

            damage_coeff = 0.003 * (T ** damage_exp)
            damages = Y * damage_coeff

            if y >= fusion_year:
                fusion_mid = fusion_year + np.log(999) / adoption_k
                fusion_share = 1.0 / (1.0 + np.exp(-adoption_k * (y - fusion_mid)))
                E = min(100, E + 3.5 * I * fusion_share)
            else:
                E = max(1.0, E - eroi_decay - 0.03 * T + np.random.normal(0, 0.04))

            velocity_impact = max(0, prev_y - Y) / (Y + 10)
            stability_loss = (damages / (Y + 5)) + (6.0 / E) + 8.0 * velocity_impact
            I = np.clip(I - inst_coeff * stability_loss + 0.006, 0.0, 1.2)

            if I < conflict_brittleness:
                risk = 0.02 + 0.2 * (conflict_brittleness - I)
                if np.random.rand() < risk:
                    Y = 0.5
                    collapsed = True

            if not collapsed:
                growth_dampener = 1.0 / (1.0 + (Y / gdp_ceiling) ** 2)
                investment = Y * base_growth * I * (E / 15) * growth_dampener
                energy_eff = max(0.5, 1.0 - E / 200)
                maintenance = Y * 0.022 * (T / 1.3) * energy_eff
                Y += investment - damages - maintenance
                if Y < collapse_threshold:
                    Y = 0.5
                    collapsed = True

            Y = max(0, Y)
            path_gdp.append(Y)
            if Y < initial_y:
                always_above = False
            if recovered_year is None and i > 5 and Y > initial_y:
                recovered_year = y
            if collapsed:
                break

        if always_above:
            no_downturn += 1

        path_gdp.extend([path_gdp[-1]] * (n_years - len(path_gdp)))
        arr = np.array(path_gdp)
        final_gdps.append(arr[-1])
        depths.append(np.min(arr))
        if recovered_year and not collapsed:
            durations.append(recovered_year - start_year)
        else:
            durations.append(n_years + 25)

    final_gdps = np.array(final_gdps)
    depths = np.array(depths)
    durations = np.array(durations)

    abundance = (final_gdps > 500).sum()
    collapse = (depths < 1.05).sum()
    never_rec = (durations > 75).sum()
    valid_dur = durations[durations <= 75]

    return {
        'no_downturn_pct': no_downturn * 100 / n_paths,
        'abundance_pct': abundance * 100 / n_paths,
        'collapse_pct': collapse * 100 / n_paths,
        'never_rec_pct': never_rec * 100 / n_paths,
        'median_nadir': int(np.median([
            2026 + np.argmin(np.array(path_gdp))  # approximate
            for path_gdp in [final_gdps]  # placeholder
        ])) if len(final_gdps) > 0 else 0,
        'median_recovery': float(np.median(valid_dur)) if len(valid_dur) > 0 else -1,
        'median_gdp_2100': float(np.median(final_gdps)),
        'mean_gdp_2100': float(np.mean(final_gdps)),
        'final_gdps': final_gdps,
        'fusion_years': np.array(fusion_years_drawn),
    }


# ---------------------------------------------------------------------------
# 1. Robustness table: vary each parameter while holding others at baseline
# ---------------------------------------------------------------------------

BASELINE = dict(
    fusion_mu=2035, fusion_sigma=7,
    csens_mu=3.0, csens_sigma=0.4,
    brit_lo=0.1, brit_hi=0.25,
    growth_mu=0.028, growth_sigma=0.006,
    damage_exp=2.6, eroi_decay=0.12,
    inst_coeff=0.035, gdp_ceiling=2000,
)

VARIATIONS = [
    ("Baseline",                {}),
    ("Damage exp = 2.0",        {"damage_exp": 2.0}),
    ("Damage exp = 3.2",        {"damage_exp": 3.2}),
    ("EROI decay = 0.06",       {"eroi_decay": 0.06}),
    ("EROI decay = 0.18",       {"eroi_decay": 0.18}),
    ("Inst. coeff = 0.020",     {"inst_coeff": 0.020}),
    ("Inst. coeff = 0.050",     {"inst_coeff": 0.050}),
    ("GDP ceiling = 1000T",     {"gdp_ceiling": 1000}),
    ("GDP ceiling = 4000T",     {"gdp_ceiling": 4000}),
    ("Fusion $\\mu$ = 2028",    {"fusion_mu": 2028}),
    ("Fusion $\\mu$ = 2045",    {"fusion_mu": 2045}),
]

N = 10000
np.random.seed(42)

print("Running robustness analysis...")
rows = []
all_results = {}
for label, overrides in VARIATIONS:
    params = {**BASELINE, **overrides}
    res = run_sim(n_paths=N, **params)
    rows.append({
        'Calibration': label,
        'Abundance': f"{res['abundance_pct']:.1f}\\%",
        'Never Rec.': f"{res['never_rec_pct']:.1f}\\%",
        'Collapse': f"{res['collapse_pct']:.1f}\\%",
        'Med. GDP 2100': f"{res['median_gdp_2100']:.0f}T",
        'No Downturn': f"{res['no_downturn_pct']:.1f}\\%",
    })
    all_results[label] = res
    print(f"  {label}: abundance={res['abundance_pct']:.1f}%, "
          f"never_rec={res['never_rec_pct']:.1f}%, "
          f"median_gdp={res['median_gdp_2100']:.0f}T")

# ---------------------------------------------------------------------------
# 2. Causal intervention: fix fusion year, randomize everything else
# ---------------------------------------------------------------------------

print("\nRunning causal intervention on fusion timing...")

FIXED_FUSION_YEARS = [2030, 2035, 2040, 2045, 2050, 2055, 2060]
N_CAUSAL = 10000
np.random.seed(123)

causal_results = {}
for fy_fixed in FIXED_FUSION_YEARS:
    res = run_sim(n_paths=N_CAUSAL, fusion_mu=fy_fixed, fusion_sigma=0.001,
                  **{k: v for k, v in BASELINE.items()
                     if k not in ('fusion_mu', 'fusion_sigma')})
    causal_results[fy_fixed] = res
    print(f"  do(fusion={fy_fixed}): mean GDP 2100 = {res['mean_gdp_2100']:.0f}T, "
          f"median = {res['median_gdp_2100']:.0f}T, "
          f"P(abundance) = {res['abundance_pct']:.1f}%, "
          f"P(collapse) = {res['collapse_pct']:.1f}%")

# Compute causal gradient via finite differences
fy_arr = np.array(FIXED_FUSION_YEARS)
gdp_arr = np.array([causal_results[y]['mean_gdp_2100'] for y in FIXED_FUSION_YEARS])
causal_gradient = np.diff(gdp_arr) / np.diff(fy_arr)
grad_centers = (fy_arr[:-1] + fy_arr[1:]) / 2

print("\nCausal gradient dGDP/d(fusion_year):")
for c, g in zip(grad_centers, causal_gradient):
    print(f"  {c:.0f}: {g:.1f} T/yr")

# ---------------------------------------------------------------------------
# 3. Output LaTeX table
# ---------------------------------------------------------------------------

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Robustness table
with open(os.path.join(fig_dir, 'robustness_table.tex'), 'w') as f:
    f.write("\\begin{table}[H]\n\\centering\n\\small\n")
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("\\textbf{Calibration} & \\textbf{Abundance} & \\textbf{Never Rec.} & "
            "\\textbf{Collapse} & \\textbf{Med.\\ GDP} & \\textbf{No Downturn} \\\\\n")
    f.write("\\midrule\n")
    for row in rows:
        f.write(f"{row['Calibration']} & {row['Abundance']} & {row['Never Rec.']} & "
                f"{row['Collapse']} & {row['Med. GDP 2100']} & {row['No Downturn']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\caption{Robustness of key outcomes to alternative parameter calibrations "
            "(10,000 paths each). Baseline calibration shown in first row.}\n")
    f.write("\\label{tab:robustness}\n")
    f.write("\\end{table}\n")

# Gradient table (causal intervention)
with open(os.path.join(fig_dir, 'gradient_table.tex'), 'w') as f:
    f.write("\\begin{table}[H]\n\\centering\n\\small\n")
    f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrr}\n\\toprule\n")
    f.write("\\textbf{$\\text{do}(\\tau_f)$} & \\textbf{Mean GDP 2100 (T)} & "
            "\\textbf{P(Abund.)} & \\textbf{P(Collapse)} & "
            "\\textbf{$\\Delta$GDP/$\\Delta$yr (T/yr)} \\\\\n")
    f.write("\\midrule\n")
    for i, target in enumerate(FIXED_FUSION_YEARS):
        res = causal_results[target]
        mean_gdp = res['mean_gdp_2100']
        abund = res['abundance_pct']
        collap = res['collapse_pct']
        if i < len(causal_gradient):
            g_str = f"{causal_gradient[i]:.1f}"
        else:
            g_str = "---"
        f.write(f"{target} & {mean_gdp:.0f} & {abund:.1f}\\% & {collap:.1f}\\% & {g_str} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
    # Find gradient near 2035 and 2053 for caption
    grad_2035 = causal_gradient[0]  # 2030-2035 interval
    grad_2050 = causal_gradient[4]  # 2050-2055 interval
    f.write(f"\\caption{{Causal effect of fusion timing on mean terminal GDP, probability of "
            f"abundance ($>$500T), and probability of collapse. "
            f"Each row fixes $\\tau_f$ to the indicated year and draws all other "
            f"parameters from their baseline distributions ({N_CAUSAL:,} paths per intervention). "
            f"Gradient at 2032: {grad_2035:.1f} T/yr; "
            f"at 2052: {grad_2050:.1f} T/yr.}}\n")
    f.write("\\label{tab:gradient}\n")
    f.write("\\end{table}\n")

print(f"\nTables written to {fig_dir}/")

# ---------------------------------------------------------------------------
# 4. Gradient plot
# ---------------------------------------------------------------------------

fy_plot = np.array(FIXED_FUSION_YEARS)
gdp_plot = np.array([causal_results[y]['mean_gdp_2100'] for y in FIXED_FUSION_YEARS])
abund_plot = np.array([causal_results[y]['abundance_pct'] for y in FIXED_FUSION_YEARS])
collap_plot = np.array([causal_results[y]['collapse_pct'] for y in FIXED_FUSION_YEARS])

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis: Mean GDP
color_gdp = '#2563eb'
ax1.set_xlabel('Fusion Commercialization Year', fontsize=12)
ax1.set_ylabel(r'Mean GDP 2100 (Trillions \$)', color=color_gdp, fontsize=12)
ax1.plot(fy_plot, gdp_plot, 'o-', color=color_gdp, lw=2.5, markersize=8, label='Mean GDP 2100')
ax1.fill_between(fy_plot, 0, gdp_plot, color=color_gdp, alpha=0.08)
ax1.tick_params(axis='y', labelcolor=color_gdp)
ax1.set_ylim(bottom=0)
ax1.set_xticks(FIXED_FUSION_YEARS)

# Right axis: Probabilities
ax2 = ax1.twinx()
color_abund = '#16a34a'
color_collap = '#dc2626'
ax2.set_ylabel('Probability (%)', fontsize=12)
ax2.plot(fy_plot, abund_plot, 's--', color=color_abund, lw=2, markersize=7, label='P(Abundance)')
ax2.plot(fy_plot, collap_plot, '^--', color=color_collap, lw=2, markersize=7, label='P(Collapse)')
ax2.set_ylim(-2, 100)
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10,
           framealpha=0.9)

# Annotation: transition zone
ax1.axvspan(2035, 2045, alpha=0.08, color='orange', label='_nolegend_')
ax1.text(2040, max(gdp_plot) * 0.85, 'Transition\nZone', ha='center', va='top',
         fontsize=9, fontstyle='italic', color='#92400e')

ax1.set_title('Causal Effect of Fusion Timing', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'gradient_plot.pdf'), bbox_inches='tight', dpi=300)
print(f"Gradient plot written to {fig_dir}/gradient_plot.pdf")
print("Done.")
