# /home/s_felix/drDataScience/scripts/add_bfactor_to_aggregate_v2.py

import os
import logging
import pandas as pd
import numpy as np

# --- Configuration ---
AGG_DATASET_PATH = "/home/s_felix/packages/DeepFlex/data/aggregated_dataset.csv"
BFACTOR_PARQUET_PATH = "/home/s_felix/inputs/noise_ceiling_analysis/bfactors_norm.parquet"
BFACTOR_COL_NAME = "bfactor_norm" # Name in parquet and final output

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Main Logic ---
def main():
    logger.info("--- Adding B-Factor Column with Hierarchical Filling ---")
    logger.info(f"Aggregated Dataset: {AGG_DATASET_PATH}")
    logger.info(f"B-Factor Source: {BFACTOR_PARQUET_PATH}")
    logger.info(f"B-Factor Column Name: {BFACTOR_COL_NAME}")

    # --- 1. Validate Inputs ---
    if not os.path.isfile(AGG_DATASET_PATH): logger.error(f"Aggregated dataset not found: {AGG_DATASET_PATH}. Exiting."); return
    if not os.path.isfile(BFACTOR_PARQUET_PATH): logger.error(f"B-Factor parquet not found: {BFACTOR_PARQUET_PATH}. Exiting."); return

    # --- 2. Load Data ---
    try:
        logger.info("Loading aggregated dataset...")
        df_agg = pd.read_csv(AGG_DATASET_PATH)
        logger.info(f"  Loaded {len(df_agg)} rows.")

        logger.info("Loading B-Factor data...")
        df_bfactor = pd.read_parquet(BFACTOR_PARQUET_PATH)
        logger.info(f"  Loaded {len(df_bfactor)} B-factor entries.")
    except Exception as e: logger.error(f"Error loading data: {e}", exc_info=True); return

    # --- 3. Prepare B-Factor Data & Calculate Medians ---
    required_bfactor_cols = ['domain_id', 'resid', BFACTOR_COL_NAME]
    missing_b_cols = [col for col in required_bfactor_cols if col not in df_bfactor.columns]
    if missing_b_cols: logger.error(f"B-Factor file missing required columns: {missing_b_cols}. Exiting."); return

    # Keep only necessary columns and drop duplicates
    df_bfactor_prep = df_bfactor[required_bfactor_cols].dropna(subset=[BFACTOR_COL_NAME]).drop_duplicates(subset=['domain_id', 'resid'], keep='first')

    # Calculate GLOBAL median (fallback)
    global_bfactor_median = df_bfactor_prep[BFACTOR_COL_NAME].median()
    if pd.isna(global_bfactor_median):
        logger.warning(f"Could not calculate GLOBAL median for '{BFACTOR_COL_NAME}'. Fallback fill will use 0.0.")
        global_bfactor_median = 0.0
    else:
        logger.info(f"Calculated GLOBAL median '{BFACTOR_COL_NAME}' for fallback filling: {global_bfactor_median:.4f}")

    # Calculate PER-DOMAIN median (primary fill)
    logger.info("Calculating per-domain median B-factors...")
    domain_median_map = df_bfactor_prep.groupby('domain_id')[BFACTOR_COL_NAME].median()
    logger.info(f"Calculated medians for {len(domain_median_map)} domains.")


    # --- 4. Merge Data ---
    logger.info("Merging aggregated data with B-Factor data...")
    merge_cols = ['domain_id', 'resid']
    if not all(col in df_agg.columns for col in merge_cols): logger.error(f"Aggregated dataset missing merge columns: {merge_cols}. Exiting."); return

    # Drop existing bfactor column if present
    if BFACTOR_COL_NAME in df_agg.columns:
        logger.warning(f"Column '{BFACTOR_COL_NAME}' already exists in {AGG_DATASET_PATH}. It will be dropped and replaced.")
        df_agg = df_agg.drop(columns=[BFACTOR_COL_NAME])

    df_merged = pd.merge(df_agg, df_bfactor_prep, on=merge_cols, how='left')

    # Check merge integrity
    if len(df_merged) != len(df_agg): logger.warning(f"Row count changed after merge ({len(df_agg)} -> {len(df_merged)}). Check merge keys.")

    # --- 5. Hierarchical Fill for Missing B-Factor Values ---
    missing_initial = df_merged[BFACTOR_COL_NAME].isnull().sum()
    if missing_initial > 0:
        logger.info(f"{missing_initial} rows initially missing '{BFACTOR_COL_NAME}' after merge.")

        # Step 5.1: Fill using per-domain median
        logger.info("Attempting fill using per-domain median...")
        # Map domain medians to the 'domain_id' column of the merged dataframe
        fill_values_domain = df_merged['domain_id'].map(domain_median_map)
        df_merged[BFACTOR_COL_NAME].fillna(fill_values_domain, inplace=True)

        missing_after_domain_fill = df_merged[BFACTOR_COL_NAME].isnull().sum()
        filled_by_domain = missing_initial - missing_after_domain_fill
        if filled_by_domain > 0:
            logger.info(f"Filled {filled_by_domain} missing values using their respective domain's median B-factor.")

        # Step 5.2: Fill remaining NaNs using global median (for domains completely missing)
        if missing_after_domain_fill > 0:
            logger.info(f"Filling remaining {missing_after_domain_fill} missing values (likely from domains absent in B-factor file) using GLOBAL median ({global_bfactor_median:.4f})...")
            df_merged[BFACTOR_COL_NAME].fillna(global_bfactor_median, inplace=True)
            if df_merged[BFACTOR_COL_NAME].isnull().any(): # Should not happen now
                 logger.error("!!! Still found NaNs after global median fill - check logic !!!")

    else:
        logger.info(f"No missing '{BFACTOR_COL_NAME}' values after merge.")

    # Ensure correct dtype
    df_merged[BFACTOR_COL_NAME] = df_merged[BFACTOR_COL_NAME].astype(float)

    # --- 6. Overwrite Original Aggregated File ---
    logger.warning(f"!!! OVERWRITING ORIGINAL FILE: {AGG_DATASET_PATH} !!!")
    try:
        df_merged.to_csv(AGG_DATASET_PATH, index=False)
        logger.info(f"Successfully overwrote {AGG_DATASET_PATH} with added/filled '{BFACTOR_COL_NAME}' column.")
        logger.info(f"Final dataset shape: {df_merged.shape}")
    except Exception as e:
        logger.error(f"Error saving updated aggregated file: {e}", exc_info=True)

if __name__ == "__main__":
    main()