# analysis.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Configuration ---
# Assumes the script is run from ./home/s_felix/drDataScience/scripts
DATA_DIR = "../data"
OUTPUT_DIR = "../analysis_outputs" # Let's create a new directory for outputs
BASE_FILENAME = "analysis_holdout_dataset" # Name of the input file (without extension)
PARQUET_PATH = os.path.join(DATA_DIR, f"{BASE_FILENAME}.parquet")
CSV_PATH = os.path.join(DATA_DIR, f"{BASE_FILENAME}.csv")

TARGET_TEMPERATURE = 320.0 # Temperature for the detailed scatter plot

# Column Names
DOMAIN_COL = "domain_id"
TEMP_COL = "temperature"
ACTUAL_COL = "rmsf"
PREDICTED_COL = "deepflex_rmsf"

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a clean seaborn style

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def save_plot(fig, output_path: str, dpi: int = 300):
    """Saves the plot and closes the figure."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Try tight_layout
        try:
            fig.tight_layout()
        except ValueError:
             logger.warning(f"tight_layout failed for {os.path.basename(output_path)}. Proceeding without it.")
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Close the specific figure

def calculate_domain_pcc(df, domain_col, actual_col, predicted_col):
    """Calculates PCC per domain."""
    domain_metrics = []
    logger.info(f"Calculating PCC per domain between '{actual_col}' and '{predicted_col}'...")

    if actual_col not in df.columns or predicted_col not in df.columns:
        logger.error(f"Required columns missing for PCC calculation: {actual_col}, {predicted_col}")
        return pd.DataFrame(columns=[domain_col, 'pcc', 'count'])

    # Drop rows where either actual or predicted is NaN before grouping
    df_valid = df[[domain_col, actual_col, predicted_col]].dropna()

    grouped = df_valid.groupby(domain_col)

    for name, group in grouped:
        pcc = np.nan
        count = len(group)
        if count >= 2: # Need at least 2 points for correlation
            actual = group[actual_col]
            predicted = group[predicted_col]
            # Check for variance to avoid errors in pearsonr
            if actual.nunique() > 1 and predicted.nunique() > 1:
                try:
                    pcc, _ = pearsonr(actual, predicted)
                    if np.isnan(pcc): # Handle potential NaN output from pearsonr
                         pcc = 0.0
                         logger.debug(f"PCC was NaN for domain {name} (likely constant data after filtering?), setting to 0.0")
                except ValueError as e:
                    logger.warning(f"Could not calculate PCC for domain {name}: {e}")
                    pcc = np.nan # Indicate calculation failure
                except Exception as e:
                    logger.error(f"Unexpected error calculating PCC for domain {name}: {e}")
                    pcc = np.nan
            else:
                logger.debug(f"Skipping PCC for domain {name} due to insufficient variance (nunique <= 1).")
                pcc = np.nan # Not meaningful to calculate correlation on constant data
        else:
             logger.debug(f"Skipping PCC for domain {name} due to insufficient data points ({count}).")


        domain_metrics.append({domain_col: name, 'pcc': pcc, 'count': count})

    metrics_df = pd.DataFrame(domain_metrics)
    logger.info(f"Calculated PCC for {metrics_df['pcc'].notna().sum()} out of {len(metrics_df)} domains.")
    return metrics_df

# --- Main Execution ---
def main():
    logger.info("Starting analysis script...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data (Prefer Parquet)
    df = None
    if os.path.exists(PARQUET_PATH):
        try:
            logger.info(f"Loading data from Parquet: {PARQUET_PATH}")
            df = pd.read_parquet(PARQUET_PATH)
        except Exception as e:
            logger.warning(f"Could not load Parquet file ({e}). Falling back to CSV.")
            df = None

    if df is None and os.path.exists(CSV_PATH):
        try:
            logger.info(f"Loading data from CSV: {CSV_PATH}")
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            logger.error(f"Failed to load data from CSV: {e}. Exiting.")
            return
    elif df is None:
        logger.error(f"Could not find or load data file ({PARQUET_PATH} or {CSV_PATH}). Exiting.")
        return

    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    if PREDICTED_COL not in df.columns:
         logger.error(f"Predicted column '{PREDICTED_COL}' not found in the dataset. Exiting.")
         return
    if ACTUAL_COL not in df.columns:
         logger.error(f"Actual column '{ACTUAL_COL}' not found in the dataset. Exiting.")
         return


    # 2. Calculate and Plot Domain PCC
    domain_pcc_df = calculate_domain_pcc(df, DOMAIN_COL, ACTUAL_COL, PREDICTED_COL)

    if not domain_pcc_df.empty:
        # Sort by PCC for potentially better visualization
        domain_pcc_df_sorted = domain_pcc_df.sort_values('pcc', ascending=False, na_position='last')

        fig_pcc, ax_pcc = plt.subplots(figsize=(12, 7))
        # Use scatter plot - x can be rank or just index
        x_vals = range(len(domain_pcc_df_sorted))
        colors = plt.cm.viridis(domain_pcc_df_sorted['pcc'].fillna(0)) # Color by PCC value

        ax_pcc.scatter(x_vals, domain_pcc_df_sorted['pcc'], c=colors, alpha=0.8, s=20) # Use computed colors
        ax_pcc.set_xlabel("Domain Rank (Sorted by PCC)")
        ax_pcc.set_ylabel("Pearson Correlation Coefficient (PCC)")
        ax_pcc.set_title(f"Per-Domain PCC between Actual ({ACTUAL_COL}) and Predicted ({PREDICTED_COL})")
        ax_pcc.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
        ax_pcc.axhline(0.7, color='red', linestyle='--', linewidth=0.8)
        ax_pcc.set_ylim(-0.1, 1.05) # Sensible limits for PCC
        ax_pcc.grid(True, linestyle=':', alpha=0.6)

        # Add text annotation for average PCC
        avg_pcc = domain_pcc_df_sorted['pcc'].mean()
        ax_pcc.text(0.02, 0.02, f"Mean PCC: {avg_pcc:.3f}", transform=ax_pcc.transAxes,
                     fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

        pcc_plot_path = os.path.join(OUTPUT_DIR, "domain_pcc_scatter.png")
        save_plot(fig_pcc, pcc_plot_path)
    else:
        logger.warning("No domain PCC data to plot.")

    # 3. Filter for Target Temperature (e.g., 320K)
    logger.info(f"Filtering data for temperature = {TARGET_TEMPERATURE}K")
    df_temp = df[np.isclose(df[TEMP_COL], TARGET_TEMPERATURE)].copy()

    if df_temp.empty:
        logger.error(f"No data found for target temperature {TARGET_TEMPERATURE}K. Cannot create scatter plot.")
        return
    logger.info(f"Found {len(df_temp)} data points for T={TARGET_TEMPERATURE}K.")

    # 4. Calculate Overall Metrics for the Target Temperature
    # Drop NaNs specifically for this subset's actual/predicted columns
    df_temp_valid = df_temp[[ACTUAL_COL, PREDICTED_COL]].dropna()
    if len(df_temp_valid) < 2:
        logger.warning(f"Insufficient valid data points ({len(df_temp_valid)}) after dropping NaNs for T={TARGET_TEMPERATURE}K metrics.")
        metrics_text = "Metrics:\n(Insufficient data)"
    else:
        actual_vals = df_temp_valid[ACTUAL_COL].values
        predicted_vals = df_temp_valid[PREDICTED_COL].values

        try:
            pcc_temp, _ = pearsonr(actual_vals, predicted_vals)
        except ValueError: pcc_temp = np.nan
        try:
            r2_temp = r2_score(actual_vals, predicted_vals)
        except ValueError: r2_temp = np.nan
        rmse_temp = np.sqrt(mean_squared_error(actual_vals, predicted_vals))
        mae_temp = mean_absolute_error(actual_vals, predicted_vals)

        # Format metrics for annotation
        metrics_text = (
            f"Metrics (T={TARGET_TEMPERATURE}K):\n"
            f"  PCC: {pcc_temp:.3f}\n"
            f"  R²:   {r2_temp:.3f}\n"
            f"  RMSE: {rmse_temp:.3f}\n"
            f"  MAE:  {mae_temp:.3f}\n"
            f"  N:    {len(actual_vals)}"
        )
        logger.info(f"Overall metrics for T={TARGET_TEMPERATURE}K: PCC={pcc_temp:.3f}, R2={r2_temp:.3f}, RMSE={rmse_temp:.3f}, MAE={mae_temp:.3f}")


    # 5. Create Scatter Plot with Density for Target Temperature
    logger.info(f"Generating scatter plot with density for T={TARGET_TEMPERATURE}K...")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))

    # Use seaborn's kdeplot for density contours/shading
    sns.kdeplot(
        data=df_temp_valid, # Use the NaN-dropped data
        x=ACTUAL_COL,
        y=PREDICTED_COL,
        fill=True, # Add color shading
        cmap="Blues", # Colormap for density
        levels=8, # Number of contour levels
        alpha=0.6,
        ax=ax_scatter
    )

    # Overlay scatter points with low alpha
    ax_scatter.scatter(
        df_temp_valid[ACTUAL_COL],
        df_temp_valid[PREDICTED_COL],
        alpha=0.15, # Make points less prominent
        s=10,      # Smaller points
        color='darkblue',
        edgecolors='none' # Remove edges for dense plots
    )

    # Add y=x line
    lim_min = min(df_temp_valid[ACTUAL_COL].min(), df_temp_valid[PREDICTED_COL].min())
    lim_max = max(df_temp_valid[ACTUAL_COL].max(), df_temp_valid[PREDICTED_COL].max())
    diag_range = np.linspace(lim_min - 0.1*(lim_max-lim_min), lim_max + 0.1*(lim_max-lim_min), 10) # Extend line slightly
    ax_scatter.plot(diag_range, diag_range, 'r--', linewidth=1.5, label="Ideal (y=x)")

    # Add annotations and labels
    ax_scatter.text(0.03, 0.97, metrics_text, transform=ax_scatter.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    ax_scatter.set_xlabel(f"Actual {ACTUAL_COL.upper()}")
    ax_scatter.set_ylabel(f"Predicted {PREDICTED_COL.upper()}")
    ax_scatter.set_title(f"Actual vs. Predicted {ACTUAL_COL.upper()} at {TARGET_TEMPERATURE}K (with Density)")
    ax_scatter.set_xlim(diag_range[0], diag_range[-1])
    ax_scatter.set_ylim(diag_range[0], diag_range[-1])
    ax_scatter.legend(loc='lower right')
    ax_scatter.grid(True, linestyle=':', alpha=0.5)
    ax_scatter.set_aspect('equal', adjustable='box') # Make axes equal for easier comparison to y=x

    scatter_plot_path = os.path.join(OUTPUT_DIR, f"actual_vs_predicted_density_{TARGET_TEMPERATURE}K.png")
    save_plot(fig_scatter, scatter_plot_path)

    logger.info("Analysis script finished.")

if __name__ == "__main__":
    main()