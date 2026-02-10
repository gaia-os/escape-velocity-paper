"""
Diagnose why do(fusion=2030) produces LOWER GDP than do(fusion=2040).
Trace a single representative path under each intervention.
"""
import numpy as np

def trace_path(fusion_year, seed=42):
    """Run one deterministic path and return yearly state."""
    np.random.seed(seed)
    start_year = 2026
    n_years = 75
    years = np.arange(start_year, start_year + n_years)
    
    T, E, Y, I = 1.3, 15.0, 105.0, 1.0
    c_sensitivity = 3.0  # fixed at mean
    conflict_brittleness = 0.175  # fixed at mean
    base_growth = 0.028  # fixed at mean
    
    trace = []
    collapsed = False
    
    for i, y in enumerate(years):
        prev_y = Y
        
        # Temperature
        dT = (0.04 * (1.0 - 0.018 * (y - start_year)) * (Y / 100) * (c_sensitivity / 3.0) * (15.0 / E))
        T += max(0, dT)
        
        # Damage
        damage_coeff = 0.003 * (T ** 2.6)
        damages = Y * damage_coeff
        
        # Energy
        if y >= fusion_year:
            E = min(100, E + 3.5 * I)
        else:
            # Pre-fusion clean energy (solar, wind, fission, geothermal)
            clean_ceiling = 0.08
            clean_share = clean_ceiling / (1.0 + np.exp(-0.15 * (y - 2030)))
            clean_contribution = clean_share * I
            E = max(1.0, E - 0.12 - 0.03 * T + clean_contribution)
        
        # Institutional stability
        velocity_impact = max(0, prev_y - Y) / (Y + 10)
        stability_loss = (damages / (Y + 5)) + (6.0 / E) + 8.0 * velocity_impact
        I = np.clip(I - 0.035 * stability_loss + 0.006, 0.0, 1.2)
        
        # GDP
        if not collapsed:
            growth_dampener = 1.0 / (1.0 + (Y / 2000) ** 2)
            investment = Y * base_growth * I * (E / 15) * growth_dampener
            energy_eff = max(0.5, 1.0 - E / 200)
            maintenance = Y * 0.022 * (T / 1.3) * energy_eff
            Y += investment - damages - maintenance
            if Y < 1.05:
                Y = 0.5
                collapsed = True
        
        Y = max(0, Y)
        
        trace.append({
            'year': y, 'T': T, 'E': E, 'Y': Y, 'I': I,
            'damages': damages, 'investment': investment if not collapsed else 0,
            'maintenance': maintenance if not collapsed else 0,
            'growth_dampener': growth_dampener if not collapsed else 0,
            'E_over_15': E / 15,
        })
    
    return trace

# Compare fusion at 2030, 2035, 2040
for fy in [2030, 2035, 2040, 2045]:
    trace = trace_path(fy)
    print(f"\n=== do(fusion={fy}) ===")
    print(f"{'Year':>6} {'T':>6} {'E':>6} {'Y':>8} {'I':>6} {'Inv':>8} {'Dmg':>8} {'Mnt':>8} {'Damp':>6} {'E/15':>6}")
    for t in trace:
        if t['year'] in [2026, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]:
            print(f"{t['year']:>6} {t['T']:>6.2f} {t['E']:>6.1f} {t['Y']:>8.1f} {t['I']:>6.3f} "
                  f"{t['investment']:>8.1f} {t['damages']:>8.1f} {t['maintenance']:>8.1f} "
                  f"{t['growth_dampener']:>6.4f} {t['E_over_15']:>6.2f}")
    print(f"Final GDP: {trace[-1]['Y']:.1f}T")
