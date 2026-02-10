import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from scipy.signal import savgol_filter

def run_simulation(n_years=75, n_paths=1000, adoption_k=0.5, clean_ceiling=0.08):
    start_year = 2026
    years = np.arange(start_year, start_year + n_years)
    gdp_results = []
    sensitivity_data = []

    depths = []
    speeds = []
    durations = []
    no_downturn_count = 0

    # Systemic collapse: < 1% of initial baseline
    collapse_threshold = 1.05

    for path in range(n_paths):
        # Initial States
        T, E, Y, I = 1.3, 15.0, 105.0, 1.0
        initial_y = Y

        # Latent Parameters - Recalibrated for wider tails
        fusion_year = np.random.normal(2035, 7)
        c_sensitivity = np.random.normal(3.0, 0.4)
        conflict_brittleness = np.random.uniform(0.1, 0.25)
        base_growth = np.random.normal(0.028, 0.006)

        path_gdp = []
        recovered_year = None
        collapsed = False
        always_above_initial = True
        fusion_delay = 0.0

        for i, y in enumerate(years):
            prev_y = Y

            # 1. Temperature Evolution
            dT = (0.04 * (1.0 - (0.018 * (y - start_year))) * (Y / 100) * (c_sensitivity / 3.0) * (15.0 / E)) + np.random.normal(0, 0.04)
            T += max(0, dT)

            # 2. Damage & Energy Dynamics
            damage_coeff = 0.003 * (T**2.6)
            damages = Y * damage_coeff

            # Endogenous fusion delay: degraded Y or I slow R&D progress
            effective_fusion_year = fusion_year + fusion_delay
            if y < effective_fusion_year:
                fusion_delay += max(0, 1 - I) * 0.5 + max(0, 1 - Y / 105) * 0.5

            if y >= effective_fusion_year:
                fusion_mid = effective_fusion_year + np.log(999) / adoption_k
                fusion_share = 1.0 / (1.0 + np.exp(-adoption_k * (y - fusion_mid)))
                E = min(100, E + 3.5 * I * fusion_share)
            else:
                # Pre-fusion clean energy (solar, wind, fission, geothermal)
                clean_share = clean_ceiling / (1.0 + np.exp(-0.15 * (y - 2030)))
                clean_contribution = clean_share * I
                E = max(1.0, E - 0.12 - (0.03 * T) + clean_contribution + np.random.normal(0, 0.04))

            # 3. Institutional Stability
            velocity_impact = max(0, prev_y - Y) / (Y + 10)
            stability_loss = (damages / (Y + 5)) + (6.0 / E) + (8.0 * velocity_impact)
            I = np.clip(I - (0.035 * stability_loss) + 0.006, 0.0, 1.2)

            # 4. Collapse Trigger
            if I < conflict_brittleness:
                risk_of_collapse = 0.02 + (0.2 * (conflict_brittleness - I))
                if np.random.rand() < risk_of_collapse:
                    Y = 0.5
                    collapsed = True

            # 5. GDP Evolution
            if not collapsed:
                growth_dampener = 1.0 / (1.0 + (Y / 2000) ** 2)
                investment = (Y * base_growth * I * (E/15)) * growth_dampener
                energy_efficiency = max(0.5, 1.0 - (E / 200))
                maintenance = (Y * 0.022 * (T/1.3) * energy_efficiency)
                Y += investment - damages - maintenance

                if Y < collapse_threshold:
                    Y = 0.5
                    collapsed = True

            Y = max(0, Y)
            path_gdp.append(Y)

            # Downturn tracking
            if Y < initial_y:
                always_above_initial = False

            # Recovery Tracking
            if recovered_year is None and i > 5 and Y > initial_y:
                recovered_year = y

            if collapsed: break

        if always_above_initial:
            no_downturn_count += 1

        path_gdp.extend([path_gdp[-1] if path_gdp else 0] * (n_years - len(path_gdp)))
        final_path_arr = np.array(path_gdp)
        gdp_results.append(final_path_arr)

        # Logging
        depths.append(np.min(final_path_arr))
        speeds.append(years[np.argmin(final_path_arr)])

        if recovered_year and not collapsed:
            durations.append(recovered_year - start_year)
        else:
            durations.append(n_years + 25)

        sensitivity_data.append([fusion_year + fusion_delay, c_sensitivity, conflict_brittleness, base_growth, final_path_arr[-1]])

    return years, np.array(gdp_results), sensitivity_data, depths, speeds, durations, no_downturn_count

N_PATHS = 10000

# Execution
years, gdp_matrix, sens_raw, depths, speeds, durations, no_downturn_count = run_simulation(n_paths=N_PATHS)
df = pd.DataFrame(sens_raw, columns=['Fusion Year', 'Climate Sens.', 'Brittleness', 'Growth', 'Final GDP'])

# Analysis
corrs = {col: spearmanr(df[col], df['Final GDP'])[0] for col in df.columns[:-1]}
sorted_corrs = dict(sorted(corrs.items(), key=lambda item: item[1]))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Percentiles
axes[0,0].plot(years, np.percentile(gdp_matrix, 50, axis=0), color='blue', lw=3, label='Median')
axes[0,0].fill_between(years, np.percentile(gdp_matrix, 10, axis=0), np.percentile(gdp_matrix, 90, axis=0), color='blue', alpha=0.1, label='10-90th Pctl')
# Fusion frontier: for each percentile of final GDP, find mean fusion year of nearby paths
# and mark it on the corresponding percentile GDP curve
fusion_years = df['Fusion Year'].values
final_gdp = gdp_matrix[:, -1]
pctl_levels = np.arange(2, 99, 2)
fusion_curve_x, fusion_curve_y = [], []
for p in pctl_levels:
    lo = np.percentile(final_gdp, max(0, p - 3))
    hi = np.percentile(final_gdp, min(100, p + 3))
    mask = (final_gdp >= lo) & (final_gdp <= hi)
    if mask.sum() == 0:
        continue
    mean_fy = np.mean(fusion_years[mask])
    fy_year_idx = int(np.clip(round(mean_fy) - years[0], 0, len(years) - 1))
    gdp_at_fy = np.percentile(gdp_matrix[:, fy_year_idx], p)
    fusion_curve_x.append(mean_fy)
    fusion_curve_y.append(gdp_at_fy)
sort_order = np.argsort(fusion_curve_x)
fusion_curve_x = np.array(fusion_curve_x)[sort_order]
fusion_curve_y = np.array(fusion_curve_y)[sort_order]
window = min(15, len(fusion_curve_x) // 2 * 2 + 1)
if window >= 5:
    fusion_curve_x = savgol_filter(fusion_curve_x, window, 3)
    fusion_curve_y = savgol_filter(fusion_curve_y, window, 3)
# Enforce monotonically decreasing y (earlier fusion → higher GDP)
for i in range(len(fusion_curve_y) - 2, -1, -1):
    fusion_curve_y[i] = max(fusion_curve_y[i], fusion_curve_y[i + 1])
axes[0,0].plot(fusion_curve_x, fusion_curve_y, color='red', lw=2, label='Fusion Frontier')
axes[0,0].set_title("Global GDP Percentiles (Log Scale)"); axes[0,0].set_yscale('log'); axes[0,0].legend()

# 2. Sensitivity
axes[0,1].barh(list(sorted_corrs.keys()), list(sorted_corrs.values()), color=['red' if x < 0 else 'green' for x in sorted_corrs.values()])
axes[0,1].set_title("Spearman Rank Correlation (2100 GDP)")

# 3. Depth
axes[0,2].hist(depths, bins=40, color='purple', alpha=0.7, density=True)
axes[0,2].set_title("Min GDP Recorded (Depth)")
axes[0,2].set_xlabel("Trillions $")

# 4. Speed
axes[1,0].hist(speeds, bins=np.arange(min(speeds), max(speeds) + 2) - 0.5, color='orange', alpha=0.7, density=True)
axes[1,0].set_title("Year of Maximum Crisis (Nadir)")

# 5. Recovery
valid_durations = [d for d in durations if d <= 75]
axes[1,1].hist(valid_durations, bins=np.arange(min(valid_durations), max(valid_durations) + 2) - 0.5, color='green', alpha=0.7, density=True)
axes[1,1].set_title("Years to Recovery (Baseline 105T)")

# 6. GDP in 2100 Histogram
axes[1,2].hist(gdp_matrix[:, -1], bins=40, color='teal', alpha=0.7, density=True)
axes[1,2].set_title("GDP in 2100 Distribution")
axes[1,2].set_xlabel("Trillions $")
axes[1,2].set_ylabel("Density")

# Stats Summary (printed)
collapse_count = (np.array(depths) < 1.05).sum()
never_rec = (np.array(durations) > 75).sum()
abundance_count = (gdp_matrix[:, -1] > 500).sum()
print(f"\nSTOCHASTIC DYNAMICS ({N_PATHS} Paths)")
print(f"------------------------------")
print(f"Uninterrupted Growth: {no_downturn_count} ({no_downturn_count*100/N_PATHS:.1f}%)")
print(f"Abundance (>500T): {abundance_count} ({abundance_count*100/N_PATHS:.1f}%)")
print(f"Systemic Collapses (<1%): {collapse_count} ({collapse_count*100/N_PATHS:.1f}%)")
print(f"Never Recovered (>2100): {never_rec} ({never_rec*100/N_PATHS:.1f}%)")
print(f"Median Nadir Year: {int(np.median(speeds))}")
print(f"Median Recovery Time: {np.median(valid_durations) if valid_durations else 'N/A':.1f} yrs")

plt.tight_layout()
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, 'simulation.pdf'), bbox_inches='tight', dpi=300)
print(f"Figure saved to {os.path.join(fig_dir, 'simulation.pdf')}")
plt.show()