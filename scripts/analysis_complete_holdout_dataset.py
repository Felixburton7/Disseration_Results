# /home/s_felix/drDataScience/scripts/add_bfactor_to_analysis_complete.py

import os
import logging
import pandas as pd
import numpy as np
import warnings

# Optional: Check if pyarrow is installed for Parquet loading/saving
try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# --- Configuration ---
# Input dataset path (contains results from RF, NN, LGBM etc.)
# This file will be READ FROM and OVERWRITTEN
ANALYSIS_DATASET_PATH = "/home/s_felix/drDataScience/data/analysis_complete_holdout_dataset.csv"

# Input B-Factor data path
BFACTOR_PARQUET_PATH = "/home/s_felix/inputs/noise_ceiling_analysis/bfactors_norm.parquet"

# B-Factor column name (in parquet AND the final merged file)
BFACTOR_COL_NAME = "bfactor_norm"

# Columns needed for merging
MERGE_COLS = ['domain_id', 'resid']

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
root_logger = logging.getLogger()
if root_logger.hasHandlers(): root_logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Setting an item of incompatible dtype is deprecated")


# --- Main Logic ---
def main():
    logger.info("--- Adding/Updating B-Factor Column in Analysis Dataset ---")
    logger.info(f"Analysis Dataset (Input/Output): {ANALYSIS_DATASET_PATH}")
    logger.info(f"B-Factor Source: {BFACTOR_PARQUET_PATH}")
    logger.info(f"B-Factor Column Name: {BFACTOR_COL_NAME}")

    # --- 1. Validate Inputs ---
    if not os.path.isfile(ANALYSIS_DATASET_PATH):
        logger.error(f"Analysis dataset file not found: {ANALYSIS_DATASET_PATH}. Exiting.")
        return
    if not os.path.isfile(BFACTOR_PARQUET_PATH):
        logger.error(f"B-Factor parquet file not found: {BFACTOR_PARQUET_PATH}. Exiting.")
        return

    # --- 2. Load Data ---
    try:
        logger.info("Loading analysis dataset...")
        # Check if parquet version exists and load that preferably
        parquet_version_path = ANALYSIS_DATASET_PATH.replace('.csv', '.parquet')
        if HAS_PYARROW and os.path.exists(parquet_version_path):
            df_analysis = pd.read_parquet(parquet_version_path)
            logger.info(f"  Loaded Parquet version: {parquet_version_path}")
        else:
            df_analysis = pd.read_csv(ANALYSIS_DATASET_PATH)
            logger.info(f"  Loaded CSV version: {ANALYSIS_DATASET_PATH}")
        logger.info(f"  Analysis dataset shape: {df_analysis.shape}.")

        logger.info("Loading B-Factor data...")
        df_bfactor = pd.read_parquet(BFACTOR_PARQUET_PATH)
        logger.info(f"  Loaded {len(df_bfactor)} B-factor entries.")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    # --- 3. Prepare B-Factor Data & Calculate Medians ---
    required_bfactor_cols = MERGE_COLS + [BFACTOR_COL_NAME]
    missing_b_cols = [col for col in required_bfactor_cols if col not in df_bfactor.columns]
    if missing_b_cols:
        logger.error(f"B-Factor file missing required columns: {missing_b_cols}. Exiting.")
        return

    # Select columns, drop NaNs IN BFACTOR column, drop duplicates
    df_bfactor_prep = df_bfactor[required_bfactor_cols].dropna(subset=[BFACTOR_COL_NAME]).drop_duplicates(subset=MERGE_COLS, keep='first')

    # Calculate GLOBAL median (fallback)
    global_bfactor_median = df_bfactor_prep[BFACTOR_COL_NAME].median()
    if pd.isna(global_bfactor_median):
        logger.warning(f"Could not calculate GLOBAL median for '{BFACTOR_COL_NAME}'. Fallback fill will use 0.0.")
        global_bfactor_median = 0.0
    else:
        logger.info(f"Calculated GLOBAL median '{BFACTOR_COL_NAME}' for fallback filling: {global_bfactor_median:.4f}")

    # Calculate PER-DOMAIN median (primary fill)
    logger.info("Calculating per-domain median B-factors...")
    domain_median_map = df_bfactor_prep.groupby(MERGE_COLS[0])[BFACTOR_COL_NAME].median() # Group by domain_id
    logger.info(f"Calculated medians for {len(domain_median_map)} domains from B-factor file.")

    # --- 4. Prepare Base Analysis Data (Drop existing B-factor if needed) ---
    if not all(col in df_analysis.columns for col in MERGE_COLS):
        logger.error(f"Analysis dataset missing merge columns: {MERGE_COLS}. Exiting.")
        return

    df_analysis_cleaned = df_analysis.copy()
    if BFACTOR_COL_NAME in df_analysis_cleaned.columns:
        logger.warning(f"Column '{BFACTOR_COL_NAME}' already exists in {ANALYSIS_DATASET_PATH}. It will be dropped and replaced.")
        df_analysis_cleaned = df_analysis_cleaned.drop(columns=[BFACTOR_COL_NAME])

    # --- 5. Merge Data ---
    logger.info(f"Merging analysis data with B-Factor data on {MERGE_COLS}...")
    df_merged = pd.merge(df_analysis_cleaned, df_bfactor_prep, on=MERGE_COLS, how='left')

    # Check merge integrity
    if len(df_merged) != len(df_analysis_cleaned):
        logger.warning(f"Row count changed after merge ({len(df_analysis_cleaned)} -> {len(df_merged)}). Check merge keys.")

    # --- 6. Hierarchical Fill for Missing B-Factor Values ---
    missing_initial = df_merged[BFACTOR_COL_NAME].isnull().sum()
    if missing_initial > 0:
        logger.info(f"{missing_initial} rows initially missing '{BFACTOR_COL_NAME}' after merge.")

        # Step 6.1: Fill using per-domain median
        logger.info("Attempting fill using per-domain median...")
        fill_values_domain = df_merged[MERGE_COLS[0]].map(domain_median_map) # Map using domain_id
        df_merged[BFACTOR_COL_NAME].fillna(fill_values_domain, inplace=True)

        missing_after_domain_fill = df_merged[BFACTOR_COL_NAME].isnull().sum()
        filled_by_domain = missing_initial - missing_after_domain_fill
        if filled_by_domain > 0:
            logger.info(f"Filled {filled_by_domain} missing values using their respective domain's median B-factor.")

        # Step 6.2: Fill remaining NaNs using global median
        if missing_after_domain_fill > 0:
            logger.info(f"Filling remaining {missing_after_domain_fill} missing values using GLOBAL median ({global_bfactor_median:.4f})...")
            df_merged[BFACTOR_COL_NAME].fillna(global_bfactor_median, inplace=True)
            if df_merged[BFACTOR_COL_NAME].isnull().any():
                 logger.error("!!! Still found NaNs after global median fill - check logic !!!")

    else:
        logger.info(f"No missing '{BFACTOR_COL_NAME}' values after merge.")

    # Ensure correct dtype
    df_merged[BFACTOR_COL_NAME] = df_merged[BFACTOR_COL_NAME].astype(float)

    # --- 7. Overwrite Original Analysis File ---
    output_path_to_write = ANALYSIS_DATASET_PATH # Overwrite the original CSV path
    logger.warning(f"!!! OVERWRITING ORIGINAL ANALYSIS FILE: {output_path_to_write} !!!")
    try:
        df_merged.to_csv(output_path_to_write, index=False)
        logger.info(f"Successfully overwrote {output_path_to_write} with updated '{BFACTOR_COL_NAME}' column.")
        logger.info(f"Final dataset shape: {df_merged.shape}")

        # Optionally overwrite parquet too if it existed
        if HAS_PYARROW and os.path.exists(parquet_version_path):
             try:
                 logger.info(f"Overwriting Parquet version: {parquet_version_path}")
                 df_merged.to_parquet(parquet_version_path, index=False, engine='pyarrow')
                 logger.info("Parquet version successfully overwritten.")
             except Exception as e:
                 logger.error(f"Error overwriting Parquet file: {e}")

    except Exception as e:
        logger.error(f"Error saving updated analysis file: {e}", exc_info=True)

if __name__ == "__main__":
    main()