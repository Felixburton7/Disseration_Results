# /home/s_felix/drDataScience/scripts/generate_temp_analysis_figures.py

import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import warnings
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Input dataset path (contains base data + ALL model predictions/uncertainty)
INPUT_DIR = "../data" # Relative path from scripts/ to data/
INPUT_BASE_FILENAME = "analysis3_holdout_dataset" # Input file name

# Directory to save the output figures
FIGURE_OUTPUT_DIR = "/home/s_felix/drDataScience/analysis_outputs/figures/temperature_correlation/"

# --- File Paths ---
INPUT_CSV_PATH = os.path.join(INPUT_DIR, f"{INPUT_BASE_FILENAME}.csv")
INPUT_PARQUET_PATH = os.path.join(INPUT_DIR, f"{INPUT_BASE_FILENAME}.parquet")

# --- Column Names ---
TEMP_COL = "temperature"
RMSF_COL = "rmsf" # Ground truth RMSF column
DOMAIN_COL = "domain_id"
DSSP_COL = "dssp"

# --- Analysis Constants ---
MIN_POINTS_FOR_CORR = 30
MIN_TEMPS_FOR_SLOPE = 3
MIN_POINTS_FOR_SLOPE = 10
SCATTER_SAMPLE_SIZE = 50000 # Sample size for density scatter plot

# --- Plotting Style ---
sns.set_style("whitegrid")
sns.set_context("talk") # Use "talk" context for potentially larger labels/lines

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
root_logger = logging.getLogger()
if root_logger.hasHandlers(): root_logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Setting an item of incompatible dtype is deprecated")


# --- Helper Functions ---
def save_plot(fig, output_path: str, dpi: int = 300):
    """Saves a matplotlib figure, ensuring directory exists and closing plot."""
    if output_path is None:
        logger.warning("No output path provided for saving plot. Skipping save.")
        plt.close(fig) # Close the specific figure
        return
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        # Apply tight layout before saving
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Close the specific figure to free memory

def map_ss_group(ss_code):
    """Maps DSSP codes to broader Helix/Sheet/Loop groups."""
    if isinstance(ss_code, str):
        if ss_code in ('H', 'G', 'I'): return 'Helix'
        elif ss_code in ('E', 'B'): return 'Sheet'
    return 'Loop'

def calculate_domain_slope(group):
    """Calculates linear slope of RMSF vs Temp for a domain group."""
    if group[TEMP_COL].nunique() >= MIN_TEMPS_FOR_SLOPE and len(group) >= MIN_POINTS_FOR_SLOPE:
        group_clean = group[[TEMP_COL, RMSF_COL]].dropna()
        if len(group_clean) >= MIN_POINTS_FOR_SLOPE and group_clean[TEMP_COL].nunique() >= MIN_TEMPS_FOR_SLOPE:
            try:
                slope, _, _, _, _ = linregress(group_clean[TEMP_COL], group_clean[RMSF_COL])
                return slope
            except ValueError: return np.nan
    return np.nan

# --- Plotting Functions ---

def plot_rmsf_distribution(df_clean, output_path):
    """Generates Violin plots of RMSF distribution per temperature."""
    logger.info(f"Generating RMSF Distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_temps_sorted = sorted(df_clean[TEMP_COL].unique())
    sns.violinplot(x=TEMP_COL, y=RMSF_COL, data=df_clean, ax=ax,
                   order=unique_temps_sorted, palette="coolwarm", inner="quartile", cut=0) # Use quartiles inside
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("RMSF Distribution vs. Temperature")
    # Optional: Limit y-axis if outliers are extreme
    # ylim_upper = df_clean[RMSF_COL].quantile(0.99) # Example: show up to 99th percentile
    # ax.set_ylim(bottom=0, top=ylim_upper)
    save_plot(fig, output_path)

def plot_mean_trend(per_temp_stats, lin_reg_results, output_path):
    """Generates Line plot of Mean RMSF vs Temperature with Std Dev shading."""
    logger.info(f"Generating Mean RMSF Trend plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    temp_vals = per_temp_stats.index
    mean_vals = per_temp_stats['mean']
    std_vals = per_temp_stats['std']

    # Plot mean line
    ax.plot(temp_vals, mean_vals, marker='o', linestyle='-', color='black', label='Mean RMSF')
    # Add shaded region for +/- 1 Std Dev
    ax.fill_between(temp_vals, mean_vals - std_vals, mean_vals + std_vals, color='skyblue', alpha=0.3, label='Mean ± 1 SD')
    # Add linear regression line
    if lin_reg_results:
        slope, intercept, r_value, _, _ = lin_reg_results
        reg_line = intercept + slope * temp_vals
        ax.plot(temp_vals, reg_line, linestyle='--', color='red', label=f'Linear Fit (R²={r_value**2:.3f})')

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Mean RMSF Trend vs. Temperature (± Std Dev)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    save_plot(fig, output_path)

def plot_trend_by_ss(ss_trend_data, output_path):
    """Generates Line plot comparing Mean RMSF trend for Helix/Sheet/Loop."""
    logger.info(f"Generating Mean RMSF Trend by Secondary Structure plot...")
    if ss_trend_data.empty or len(ss_trend_data.columns) == 0:
        logger.warning("No secondary structure trend data available to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Helix': 'red', 'Sheet': 'blue', 'Loop': 'green'}
    for ss_type in ['Helix', 'Sheet', 'Loop']:
        if ss_type in ss_trend_data.columns:
            ax.plot(ss_trend_data.index, ss_trend_data[ss_type], marker='o', linestyle='-',
                    label=ss_type, color=colors.get(ss_type))

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean RMSF (Å)")
    ax.set_title("Mean RMSF Trend vs. Temperature by Secondary Structure")
    ax.legend(title="SS Type")
    ax.grid(True, linestyle=':', alpha=0.5)
    save_plot(fig, output_path)

def plot_domain_slope_distribution(domain_slopes, output_path):
    """Generates Histogram and KDE plot of per-domain RMSF vs Temp slopes."""
    logger.info(f"Generating Per-Domain Slope Distribution plot...")
    if domain_slopes.empty:
        logger.warning("No domain slope data available to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(domain_slopes, kde=True, ax=ax, stat='density', bins=30) # Use density for KDE overlay
    mean_slope = domain_slopes.mean()
    median_slope = domain_slopes.median()
    ax.axvline(mean_slope, color='red', linestyle='--', label=f'Mean Slope ({mean_slope:.4f} Å/K)')
    ax.axvline(median_slope, color='black', linestyle=':', label=f'Median Slope ({median_slope:.4f} Å/K)')
    ax.set_xlabel("Per-Domain Linear Slope (RMSF vs Temperature, Å/K)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Per-Domain Temperature Sensitivity Slopes")
    ax.legend()
    save_plot(fig, output_path)

def plot_temperature_rmsf_scatter_density(df_clean, output_path):
    """Generates a 2D density plot (hexbin or KDE) of RMSF vs Temperature."""
    logger.info(f"Generating RMSF vs Temperature Density Scatter plot...")
    # Sample data for plotting if it's very large
    if len(df_clean) > SCATTER_SAMPLE_SIZE:
        df_sample = df_clean.sample(SCATTER_SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df_clean

    fig, ax = plt.subplots(figsize=(10, 7))
    # Using hexbin for large datasets can be clearer than KDE scatter
    hb = ax.hexbin(df_sample[TEMP_COL], df_sample[RMSF_COL], gridsize=50, cmap='viridis', mincnt=1) # mincnt=1 shows all bins with data
    fig.colorbar(hb, ax=ax, label='Residue Count per Bin')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Density of RMSF vs. Temperature")
    # Add overall correlation as annotation
    try:
        r_value, _ = pearsonr(df_clean[TEMP_COL], df_clean[RMSF_COL])
        ax.text(0.05, 0.95, f'Overall Pearson R = {r_value:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    except Exception: pass # Ignore if correlation failed
    save_plot(fig, output_path)

def plot_relative_rmsf_change(per_temp_stats, output_path):
    """Calculates and plots the relative change in mean RMSF vs. the lowest temperature."""
    logger.info(f"Generating Relative RMSF Change plot...")
    if per_temp_stats.empty or len(per_temp_stats) < 2:
        logger.warning("Not enough temperature points to calculate relative change.")
        return

    stats_sorted = per_temp_stats.sort_index()
    baseline_temp = stats_sorted.index[0]
    baseline_rmsf = stats_sorted.loc[baseline_temp, 'mean']

    if baseline_rmsf <= 1e-6:
        logger.warning("Baseline RMSF at lowest temperature is near zero. Cannot calculate relative change reliably.")
        return

    relative_change = (stats_sorted['mean'] / baseline_rmsf) - 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(relative_change.index, relative_change * 100, marker='o', linestyle='-') # Plot as percentage change
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean RMSF Change (%) Relative to Baseline")
    ax.set_title(f"Relative Increase in Mean RMSF (Baseline = {baseline_temp:.0f}K)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format)) # Format y-axis as percentage
    save_plot(fig, output_path)


# --- Main Execution ---
def main():
    logger.info("--- Generating Temperature vs RMSF Analysis Figures ---")

    # --- 1. Load Data ---
    df = None; input_file_used = None
    if os.path.exists(INPUT_PARQUET_PATH):
        try: logger.info(f"Loading data from Parquet: {INPUT_PARQUET_PATH}"); df = pd.read_parquet(INPUT_PARQUET_PATH); input_file_used = INPUT_PARQUET_PATH
        except Exception as e: logger.warning(f"Could not load Parquet file ({e}). Falling back to CSV.")
    if df is None and os.path.exists(INPUT_CSV_PATH):
        try: logger.info(f"Loading data from CSV: {INPUT_CSV_PATH}"); df = pd.read_csv(INPUT_CSV_PATH); input_file_used = INPUT_CSV_PATH
        except Exception as e: logger.error(f"Failed to load data from CSV: {e}. Exiting."); return
    if df is None: logger.error(f"Could not find or load input data file '{INPUT_BASE_FILENAME}' in '{INPUT_DIR}'. Exiting."); return
    logger.info(f"Data loaded successfully from {input_file_used}. Shape: {df.shape}")

    # --- 2. Validate Required Columns & Clean ---
    required_cols = [TEMP_COL, RMSF_COL, DOMAIN_COL, DSSP_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: logger.warning(f"Input data is missing columns needed for full analysis: {missing_cols}.")
    df_clean = df.dropna(subset=[TEMP_COL, RMSF_COL])
    if len(df_clean) < len(df): logger.warning(f"Dropped {len(df) - len(df_clean)} rows with missing Temperature or RMSF values.")
    if df_clean.empty: logger.error("No valid data remaining after dropping NaNs. Exiting."); return

    # Add SS Group column if needed and possible
    if DSSP_COL in df.columns and 'ss_group' not in df_clean.columns:
        logger.info("Adding 'ss_group' column...")
        df_clean = df_clean.copy()
        df_clean['ss_group'] = df_clean[DSSP_COL].apply(map_ss_group)
    elif 'ss_group' not in df_clean.columns:
        logger.warning(f"'{DSSP_COL}' column not found, cannot perform secondary structure analysis.")

    # --- 3. Create Output Directory ---
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Figures will be saved to: {os.path.abspath(FIGURE_OUTPUT_DIR)}")

    # --- 4. Perform Calculations ---
    logger.info("Performing necessary calculations...")
    per_temp_stats = pd.DataFrame()
    ss_trend_data = pd.DataFrame()
    domain_slopes = pd.Series(dtype=float)
    lin_reg_results = None

    # Per-Temperature Stats
    if df_clean[TEMP_COL].nunique() > 0:
         try: per_temp_stats = df_clean.groupby(TEMP_COL)[RMSF_COL].agg(['mean', 'std']) # Need mean/std for Fig 2
         except Exception as e: logger.error(f"Error calculating per-temp stats: {e}")
    # Linear Regression on Means
    if len(per_temp_stats) > 1:
         try: lin_reg_results = linregress(per_temp_stats.index, per_temp_stats['mean'])
         except ValueError: pass # Handle cases with not enough points or variance
    # Per-SS Trend
    if 'ss_group' in df_clean.columns:
         try: ss_trend_data = df_clean.groupby([TEMP_COL, 'ss_group'])[RMSF_COL].mean().unstack(level='ss_group')
         except Exception as e: logger.error(f"Error calculating stats per SS type: {e}")
    # Per-Domain Slopes
    if DOMAIN_COL in df_clean.columns:
         try:
              logger.info("Applying slope calculation per domain (this may take time)...")
              domain_slopes = df_clean.groupby(DOMAIN_COL).apply(calculate_domain_slope).dropna()
              logger.info(f"Calculated slopes for {len(domain_slopes)} domains.")
         except Exception as e: logger.error(f"Error calculating per-domain slopes: {e}")

    # --- 5. Generate Figures ---
    logger.info("Generating figures...")

    # Figure 1: Distribution
    plot_rmsf_distribution(df_clean, os.path.join(FIGURE_OUTPUT_DIR, "fig1_rmsf_distribution_vs_temp.png"))

    # Figure 2: Mean Trend
    if not per_temp_stats.empty:
        plot_mean_trend(per_temp_stats, lin_reg_results, os.path.join(FIGURE_OUTPUT_DIR, "fig2_mean_rmsf_trend_vs_temp.png"))
    else: logger.warning("Skipping Figure 2 due to missing per-temperature stats.")

    # Figure 3: Trend by SS
    if 'ss_group' in df_clean.columns:
        plot_trend_by_ss(ss_trend_data, os.path.join(FIGURE_OUTPUT_DIR, "fig3_mean_rmsf_trend_by_ss.png"))
    else: logger.warning("Skipping Figure 3 due to missing secondary structure info.")

    # Figure 4: Domain Slope Distribution
    plot_domain_slope_distribution(domain_slopes, os.path.join(FIGURE_OUTPUT_DIR, "fig4_domain_slope_distribution.png"))

    # Figure 5: RMSF vs Temp Scatter Density
    plot_temperature_rmsf_scatter_density(df_clean, os.path.join(FIGURE_OUTPUT_DIR, "fig5_rmsf_vs_temp_density.png"))

    # Figure 6: Relative RMSF Change
    if not per_temp_stats.empty:
        plot_relative_rmsf_change(per_temp_stats, os.path.join(FIGURE_OUTPUT_DIR, "fig6_relative_rmsf_change_vs_temp.png"))
    else: logger.warning("Skipping Figure 6 due to missing per-temperature stats.")


    logger.info(f"--- Figure Generation Finished ---")

if __name__ == "__main__":
    main()