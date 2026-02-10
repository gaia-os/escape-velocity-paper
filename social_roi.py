"""
Compute the cumulative GDP gain from fusion acceleration and the implied
social discount rate of current fusion investment levels.

For each do(τ_f) intervention, we run N paths and store the full GDP
trajectory year-by-year.  The cumulative benefit of accelerating fusion
by Δ years is the integral (sum) of the GDP difference across all years,
not just the terminal value.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Core simulation returning full trajectories
# ---------------------------------------------------------------------------

def run_trajectories(n_years=75, n_paths=10000, fusion_year_fixed=2044,
                     csens_mu=3.0, csens_sigma=0.4,
                     brit_lo=0.1, brit_hi=0.25,
                     growth_mu=0.028, growth_sigma=0.006,
                     damage_exp=2.6, eroi_decay=0.12,
                     inst_coeff=0.035, gdp_ceiling=2000,
                     adoption_k=0.5, clean_ceiling=0.08):
    """Return (n_paths, n_years) array of GDP trajectories."""
    start_year = 2026
    years = np.arange(start_year, start_year + n_years)
    collapse_threshold = 1.05
    all_trajectories = np.zeros((n_paths, n_years))

    for p in range(n_paths):
        T, E, Y, I = 1.3, 15.0, 105.0, 1.0
        c_sensitivity = np.random.normal(csens_mu, csens_sigma)
        conflict_brittleness = np.random.uniform(brit_lo, brit_hi)
        base_growth = np.random.normal(growth_mu, growth_sigma)
        collapsed = False

        for i, y in enumerate(years):
            prev_y = Y
            dT = (0.04 * (1.0 - 0.018 * (y - start_year)) * (Y / 100)
                  * (c_sensitivity / 3.0) * (15.0 / E)) + np.random.normal(0, 0.04)
            T += max(0, dT)

            damage_coeff = 0.003 * (T ** damage_exp)
            damages = Y * damage_coeff

            if y >= fusion_year_fixed:
                fusion_mid = fusion_year_fixed + np.log(999) / adoption_k
                fusion_share = 1.0 / (1.0 + np.exp(-adoption_k * (y - fusion_mid)))
                E = min(100, E + 3.5 * I * fusion_share)
            else:
                # Pre-fusion clean energy (solar, wind, fission, geothermal)
                clean_share = clean_ceiling / (1.0 + np.exp(-0.15 * (y - 2030)))
                clean_contribution = clean_share * I
                E = max(1.0, E - eroi_decay - 0.03 * T + clean_contribution + np.random.normal(0, 0.04))

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
            all_trajectories[p, i] = Y

            if collapsed:
                all_trajectories[p, i:] = Y
                break

    return years, all_trajectories


# ---------------------------------------------------------------------------
# Run causal interventions and compute cumulative GDP
# ---------------------------------------------------------------------------

FIXED_FUSION_YEARS = [2030, 2035, 2040, 2045, 2050, 2055, 2060]
N_PATHS = 10000
np.random.seed(123)

print("Running causal interventions with full trajectories...")
mean_trajectories = {}
for fy in FIXED_FUSION_YEARS:
    years, trajs = run_trajectories(n_paths=N_PATHS, fusion_year_fixed=fy)
    mean_traj = trajs.mean(axis=0)  # mean GDP at each year across paths
    mean_trajectories[fy] = mean_traj
    cumulative = mean_traj.sum()  # sum of annual GDP = cumulative GDP (T·yr)
    print(f"  do(fusion={fy}): terminal GDP = {mean_traj[-1]:.0f}T, "
          f"cumulative GDP = {cumulative:,.0f} T·yr")

# ---------------------------------------------------------------------------
# Compute cumulative GDP gain per year of acceleration
# ---------------------------------------------------------------------------

print("\n--- Cumulative GDP gain from 1 year of acceleration ---")
print(f"{'Interval':<15} {'Terminal ΔGDP (T)':<20} {'Cumulative ΔGDP (T·yr)':<25} {'Per year (T·yr/yr)':<20}")
print("-" * 80)

for i in range(len(FIXED_FUSION_YEARS) - 1):
    fy_early = FIXED_FUSION_YEARS[i]
    fy_late = FIXED_FUSION_YEARS[i + 1]
    delta_years = fy_late - fy_early

    terminal_diff = mean_trajectories[fy_early][-1] - mean_trajectories[fy_late][-1]
    cumulative_diff = mean_trajectories[fy_early].sum() - mean_trajectories[fy_late].sum()
    per_year = cumulative_diff / delta_years

    print(f"{fy_early}-{fy_late}      {terminal_diff:>12.0f}T       "
          f"{cumulative_diff:>14,.0f} T·yr      {per_year:>10,.0f} T·yr/yr")

# ---------------------------------------------------------------------------
# Implied social discount rate calculation
# ---------------------------------------------------------------------------

print("\n--- Implied social discount rate ---")
print("")
print("The right comparison: total cumulative fusion capex (all programs, all")
print("countries, all time) vs. PV of cumulative GDP gain from 1 year of acceleration.")
print("")

# Total cumulative global fusion capex (generous upper bound):
# - ITER: $22-65B (use $65B DOE upper bound)
# - US public fusion R&D: ~$40B cumulative since 1950s
# - EU/Japan/Russia/China/Korea public programs: ~$40B cumulative
# - Private sector: ~$10B cumulative through 2025
# Total: ~$100-150B.  Use $100B as round number.
TOTAL_FUSION_CAPEX_T = 0.1  # $100B = 0.1T

print(f"Total cumulative global fusion capex (upper bound): ${TOTAL_FUSION_CAPEX_T*1000:.0f}B")
print()

# For the critical 2040-2050 window, compute the cumulative gain per year
# of acceleration.
fy_early, fy_late = 2040, 2050
delta_years = fy_late - fy_early
cumul_diff = mean_trajectories[fy_early].sum() - mean_trajectories[fy_late].sum()
gain_per_year_accel = cumul_diff / delta_years  # T·yr of cumulative GDP per year of acceleration

# The gain is a stream of annual GDP increments from ~fusion_year through 2100.
# The annual GDP increment in year t from 1 year of acceleration:
annual_increment = (mean_trajectories[fy_early] - mean_trajectories[fy_late]) / delta_years

print(f"Critical window: {fy_early}-{fy_late}")
print(f"Cumulative GDP gain per year of acceleration: {gain_per_year_accel:,.0f} T·yr")
print(f"  (vs. terminal-only: {(mean_trajectories[fy_early][-1] - mean_trajectories[fy_late][-1]) / delta_years:.0f} T)")
print(f"  Cumulative gain is {gain_per_year_accel / ((mean_trajectories[fy_early][-1] - mean_trajectories[fy_late][-1]) / delta_years):.0f}x the terminal-only figure")
print()

# For a range of discount rates, compute PV of the annual increment stream
current_year = 2026

print(f"{'Discount rate':<16} {'PV of gain (T)':<18} {'Ratio to capex':<20} {'Framework'}")
print("-" * 75)

for r, label in [(0.014, "Stern Review (1.4%)"),
                 (0.03, "Ramsey moderate (3%)"),
                 (0.05, "Nordhaus (5%)"),
                 (0.10, "High (10%)"),
                 (0.15, "Very high (15%)"),
                 (0.20, "Extreme (20%)"),
                 (0.30, "Absurd (30%)"),
                 (0.50, "Inconceivable (50%)")]:
    pv = 0.0
    for t_idx, y in enumerate(years):
        dt = y - current_year
        pv += annual_increment[t_idx] * np.exp(-r * dt)
    print(f"  r = {r*100:5.1f}%      {pv:>10.1f}T       {pv/TOTAL_FUSION_CAPEX_T:>12,.0f}x          {label}")

# Compute PV at key rates for later use
from scipy.optimize import brentq

pv_at_stern = sum(annual_increment[t] * np.exp(-0.014 * (years[t] - current_year)) for t in range(len(years)))
pv_at_5 = sum(annual_increment[t] * np.exp(-0.05 * (years[t] - current_year)) for t in range(len(years)))

# Find the break-even discount rate where PV = total capex
def pv_minus_capex(r):
    pv = 0.0
    for t_idx, y in enumerate(years):
        dt = y - current_year
        pv += annual_increment[t_idx] * np.exp(-r * dt)
    return pv - TOTAL_FUSION_CAPEX_T

try:
    r_breakeven = brentq(pv_minus_capex, 0.01, 5.0)
    print(f"\nBreak-even discount rate: r = {r_breakeven*100:.1f}%")
    print(f"  At this rate, PV of cumulative gain from 1 year of acceleration")
    print(f"  = total cumulative global fusion capex (${TOTAL_FUSION_CAPEX_T*1000:.0f}B)")
    print(f"")
    print(f"  For comparison:")
    print(f"    Stern Review:    1.4%")
    print(f"    Nordhaus:        5.0%")
    print(f"    Typical private: 10-15%")
    print(f"")
    print(f"  Even at Nordhaus's 5%, the return is {pv_at_5/TOTAL_FUSION_CAPEX_T:,.0f}x the investment.")
except Exception as e:
    print(f"\nCould not find break-even rate: {e}")
    r_breakeven = float('nan')

term_gain = (mean_trajectories[fy_early][-1] - mean_trajectories[fy_late][-1]) / delta_years
print(f"\n=== SUMMARY FOR PAPER ===")
print(f"Cumulative GDP gain per year of acceleration (2040-2050): {gain_per_year_accel:,.0f} T·yr")
print(f"This is {gain_per_year_accel / term_gain:.0f}x the terminal-only value ({term_gain:.0f}T)")
print(f"Total cumulative fusion capex (all time, all programs): ${TOTAL_FUSION_CAPEX_T*1000:.0f}B")
print(f"PV at Stern 1.4%: {pv_at_stern:.0f}T = {pv_at_stern/TOTAL_FUSION_CAPEX_T:,.0f}x capex")
print(f"PV at Nordhaus 5%: {pv_at_5:.0f}T = {pv_at_5/TOTAL_FUSION_CAPEX_T:,.0f}x capex")
print(f"Break-even discount rate: {r_breakeven*100:.1f}%")
