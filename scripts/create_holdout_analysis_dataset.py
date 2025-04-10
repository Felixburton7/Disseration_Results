# # # Example filename: scripts/aggregate_rf_predictions.py


# # Example filename: scripts/aggregate_nn_predictions.py

# import os
# import glob
# import re
# import logging
# import pandas as pd
# import numpy as np
# # Optional: Check if pyarrow is installed for Parquet saving
# try:
#     import pyarrow
#     HAS_PYARROW = True
# except ImportError:
#     HAS_PYARROW = False

# # --- Configuration (Define your paths and settings here) ---

# # Directory containing the **NEURAL NETWORK** holdout_predictions_*K.csv files
# # ASSUMPTION: Assuming NN predictions are in the same directory as RF ones. CHANGE IF DIFFERENT.
# PRED_DIR = "/home/s_felix/packages/DeepFlex/output/deepflex" #<-- VERIFY THIS FOR NN PREDICTIONS

# # Full path to the intermediate dataset (output from previous script, contains RF results)
# BASE_FILE = "/home/s_felix/drDataScience/data/analysis2_holdout_dataset.csv" #<-- UPDATED INPUT FILE

# # Base name for the NEW output files (without extension)
# OUTPUT_NAME = "analysis3_holdout_dataset" #<-- UPDATED OUTPUT NAME

# # Directory to save the final output files. If None, defaults to the directory of BASE_FILE.
# # Setting to ./output/deepflex/ for consistency, change if needed
# OUTPUT_DIR = "/home/s_felix/packages/DeepFlex/output/deepflex/" #<-- SET THIS (e.g., "/home/s_felix/analysis_output") or keep None

# # Names for the NEW columns containing the matched NN predictions and uncertainty
# NEW_NN_PRED_COL_NAME = "deepflex_NN_rmsf" #<-- ADDED for NN
# NEW_NN_UNC_COL_NAME = "deepflex_NN_rmsf_uncertainty" #<-- ADDED for NN

# # Source column names in the NN prediction files (ASSUMED to be same as RF)
# SOURCE_PRED_COL = "rmsf_predicted" #<-- VERIFY THIS in NN output files
# SOURCE_UNC_COL = "rmsf_uncertainty" #<-- VERIFY THIS in NN output files

# # --- End Configuration ---

# # --- Logging Setup ---
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# logger = logging.getLogger(__name__)

# # --- Regex Pattern ---
# # Regex to extract temperature (numeric) from prediction filename
# FILENAME_TEMP_PATTERN = re.compile(r"holdout_predictions_(\d+)K\.csv", re.IGNORECASE)

# # --- Helper Function ---
# def extract_temp_from_filename(filename):
#     """Extracts the numeric temperature from the prediction filename."""
#     match = FILENAME_TEMP_PATTERN.search(filename)
#     if match:
#         try:
#             return float(match.group(1))
#         except ValueError:
#             return None
#     return None

# # --- Main Logic ---
# def main():
#     logger.info("--- Starting Analysis Dataset Creation (Adding NN Predictions) ---")
#     logger.info(f"NN Prediction directory: {PRED_DIR}")
#     logger.info(f"Base dataset file (with RF results): {BASE_FILE}")
#     logger.info(f"Output file base name: {OUTPUT_NAME}")
#     logger.info(f"New NN prediction column name: {NEW_NN_PRED_COL_NAME}")
#     logger.info(f"New NN uncertainty column name: {NEW_NN_UNC_COL_NAME}")

#     # Validate input paths/files exist
#     if not os.path.isdir(PRED_DIR):
#         logger.error(f"NN Prediction directory does not exist: {PRED_DIR}. Exiting.")
#         return
#     if not os.path.isfile(BASE_FILE):
#         logger.error(f"Base dataset file ('{BASE_FILE}') does not exist. Run previous script first. Exiting.")
#         return

#     # Determine output directory
#     effective_output_dir = OUTPUT_DIR
#     if effective_output_dir is None:
#         effective_output_dir = os.path.dirname(BASE_FILE)
#         if not effective_output_dir: # Handle case where base file is in current dir
#             effective_output_dir = "."
#     os.makedirs(effective_output_dir, exist_ok=True)
#     logger.info(f"Output directory set to: {os.path.abspath(effective_output_dir)}")

#     # 1. Find NN prediction files
#     pred_file_pattern = os.path.join(PRED_DIR, "holdout_predictions_*K.csv")
#     pred_files = glob.glob(pred_file_pattern)
#     pred_files = [f for f in pred_files if "_metrics" not in os.path.basename(f)] # Exclude metrics files

#     if not pred_files:
#         logger.error(f"No NN prediction files found matching pattern in {PRED_DIR}. Exiting.")
#         return
#     logger.info(f"Found {len(pred_files)} potential NN prediction files.")

#     # 2. Extract target NN predictions and uncertainty
#     target_data_list = []
#     for file_path in sorted(pred_files):
#         filename = os.path.basename(file_path)
#         prediction_temp = extract_temp_from_filename(filename)

#         if prediction_temp is None:
#             logger.warning(f"Could not extract temperature from {filename}. Skipping.")
#             continue

#         logger.info(f"Processing {filename} for NN prediction temp {prediction_temp}K...")
#         try:
#             pred_df = pd.read_csv(file_path)

#             # --- Check for required source columns (assuming NN output is same as RF) ---
#             required_source_cols = ['domain_id', 'resid', 'temperature', SOURCE_PRED_COL, SOURCE_UNC_COL]
#             missing_source_cols = [col for col in required_source_cols if col not in pred_df.columns]
#             if missing_source_cols:
#                  logger.error(f"Missing expected columns {missing_source_cols} in NN prediction file {filename}. Skipping.")
#                  continue

#             # --- Filter rows where original temp matches prediction temp ---
#             target_rows = pred_df[np.isclose(pred_df['temperature'], prediction_temp)]

#             if target_rows.empty:
#                 logger.warning(f"No rows found in {filename} where original temperature matches prediction temperature ({prediction_temp}K).")
#                 continue

#             # --- Select relevant columns and rename to NN column names ---
#             processed_rows = target_rows[['domain_id', 'resid', 'temperature', SOURCE_PRED_COL, SOURCE_UNC_COL]].copy()
#             processed_rows.rename(columns={
#                 SOURCE_PRED_COL: NEW_NN_PRED_COL_NAME, # RENAME TO NN PREDICTION COL
#                 SOURCE_UNC_COL: NEW_NN_UNC_COL_NAME    # RENAME TO NN UNCERTAINTY COL
#             }, inplace=True)

#             # Simple validation
#             if not np.isclose(processed_rows['temperature'], prediction_temp).all():
#                  logger.error(f"Temperature mismatch error after filtering {filename}. Check logic.")
#                  continue # Skip this file due to error

#             target_data_list.append(processed_rows)
#             logger.info(f"  Extracted {len(processed_rows)} target NN predictions and uncertainties.")

#         except FileNotFoundError:
#             logger.error(f"File not found: {file_path}. Skipping.")
#         except pd.errors.EmptyDataError:
#             logger.warning(f"File is empty: {filename}. Skipping.")
#         except KeyError as e:
#             logger.error(f"Missing expected column {e} in {filename}. Skipping.")
#         except Exception as e:
#             logger.error(f"Error processing {filename}: {e}", exc_info=True)

#     # 3. Combine extracted NN data
#     if not target_data_list:
#         logger.error("No target NN predictions/uncertainties could be extracted. Cannot update analysis dataset.")
#         return

#     logger.info("Concatenating extracted target NN data...")
#     target_nn_data_df = pd.concat(target_data_list, ignore_index=True)
#     logger.info(f"Combined NN data shape: {target_nn_data_df.shape}")

#     # Optional: Check for duplicates in the NN data
#     id_cols = ['domain_id', 'resid', 'temperature']
#     duplicates = target_nn_data_df[target_nn_data_df.duplicated(subset=id_cols, keep=False)]
#     if not duplicates.empty:
#         logger.warning(f"Found {len(duplicates)} duplicate entries (same residue/temp) in combined NN data. Review NN prediction files/process.")
#         target_nn_data_df = target_nn_data_df.drop_duplicates(subset=id_cols, keep='first')
#         logger.info(f"Dropped duplicates from NN data, keeping first entry. Shape is now: {target_nn_data_df.shape}")

#     # 4. Load base data (Analysis 2 dataset, with RF results)
#     try:
#         logger.info(f"Loading base dataset (Analysis 2): {BASE_FILE}")
#         base_df = pd.read_csv(BASE_FILE)
#         logger.info(f"Base dataset (Analysis 2) shape: {base_df.shape}")
#     except FileNotFoundError:
#         logger.error(f"Base dataset file not found: {BASE_FILE}. Exiting.")
#         return
#     except Exception as e:
#         logger.error(f"Error loading base file {BASE_FILE}: {e}", exc_info=True)
#         return

#     # 5. Merge base data (with RF) and new NN data
#     logger.info(f"Merging Analysis 2 dataset with target NN data on {id_cols}...")
#     # --- Perform the merge ---
#     analysis_df = pd.merge(base_df, target_nn_data_df, on=id_cols, how='left')
#     logger.info(f"Merged dataset (Analysis 3) shape: {analysis_df.shape}")

#     # Validation after merge
#     if analysis_df.shape[0] != base_df.shape[0]:
#         logger.warning(f"Number of rows changed after merge ({base_df.shape[0]} -> {analysis_df.shape[0]}). Check merge keys/duplicates.")

#     # Check for missing NN predictions and uncertainties
#     missing_nn_preds = analysis_df[NEW_NN_PRED_COL_NAME].isnull().sum()
#     missing_nn_unc = analysis_df[NEW_NN_UNC_COL_NAME].isnull().sum()

#     if missing_nn_preds > 0:
#         logger.warning(f"Found {missing_nn_preds} rows in the base dataset with no matching NN prediction added (NaN in '{NEW_NN_PRED_COL_NAME}').")
#     else:
#         logger.info(f"Successfully added NN predictions to all rows (column '{NEW_NN_PRED_COL_NAME}').")

#     if missing_nn_unc > 0:
#         logger.warning(f"Found {missing_nn_unc} rows in the base dataset with no matching NN uncertainty added (NaN in '{NEW_NN_UNC_COL_NAME}').")
#     else:
#         logger.info(f"Successfully added NN uncertainties to all rows (column '{NEW_NN_UNC_COL_NAME}').")

#     # 6. Save final output (Analysis 3)
#     output_csv_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.csv")
#     output_parquet_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.parquet")

#     try:
#         logger.info(f"Saving final analysis dataset (Analysis 3) to CSV: {output_csv_path}")
#         analysis_df.to_csv(output_csv_path, index=False)
#         logger.info("CSV saved successfully.")
#     except Exception as e:
#         logger.error(f"Error saving final analysis dataset to CSV: {e}", exc_info=True)

#     if HAS_PYARROW:
#         try:
#             logger.info(f"Saving final analysis dataset (Analysis 3) to Parquet: {output_parquet_path}")
#             analysis_df.to_parquet(output_parquet_path, index=False, engine='pyarrow')
#             logger.info("Parquet saved successfully (using pyarrow engine).")
#         except Exception as e:
#             logger.error(f"Error saving final analysis dataset to Parquet: {e}", exc_info=True)
#     else:
#          logger.warning(f"Cannot save to Parquet format because 'pyarrow' is not installed. Skipping Parquet save. To enable, run: pip install pyarrow")

#     logger.info("--- Analysis Dataset Creation (NN Added) Finished ---")

# if __name__ == "__main__":
#     main()
    


# Example filename: scripts/replace_rf_predictions.py

import os
import glob
import re
import logging
import pandas as pd
import numpy as np
# Optional: Check if pyarrow is installed for Parquet saving
try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# --- Configuration ---

# Directory containing the NEW RF holdout_predictions_*K.csv files
PRED_DIR = "/home/s_felix/packages/DeepFlex/output/deepflex/"

# Full path to the base dataset CSV (contains the OLD/WRONG deepflex_RF columns)
BASE_FILE = "/home/s_felix/drDataScience/data/analysis3_holdout_dataset_wrong.csv" # Input file

# Base name for the CORRECTED output files
OUTPUT_NAME = "analysis_complete_holdout_dataset"

# Directory to save the CORRECTED output files.
OUTPUT_DIR = "/home/s_felix/drDataScience/data/"

# --- Column Names ---
# Names of the INCORRECT columns in BASE_FILE to be dropped/replaced
OLD_PRED_COL_TO_DROP = "deepflex_RF_rmsf"
OLD_UNC_COL_TO_DROP = "deepflex_RF_rmsf_uncertainty"

# These are the names the NEW RF data should have in the FINAL output file.
FINAL_RF_PRED_COL_NAME = "deepflex_RF_rmsf"
FINAL_RF_UNC_COL_NAME = "deepflex_RF_rmsf_uncertainty"

# Source column names in the NEW prediction files being read
SOURCE_PRED_COL = "rmsf_predicted" # Column name in holdout_predictions_*K.csv
SOURCE_UNC_COL = "rmsf_uncertainty" # Column name in holdout_predictions_*K.csv

# Generic names that might also exist and should be dropped from base if not the target
GENERIC_PRED_COL = "rmsf_predicted"
GENERIC_UNC_COL = "rmsf_uncertainty"

# --- End Configuration ---

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Regex Pattern ---
FILENAME_TEMP_PATTERN = re.compile(r"holdout_predictions_(\d+)K\.csv", re.IGNORECASE)

# --- Helper Function ---
def extract_temp_from_filename(filename):
    match = FILENAME_TEMP_PATTERN.search(filename)
    if match:
        try: return float(match.group(1))
        except ValueError: return None
    return None

# --- Main Logic ---
def main():
    logger.info("--- Starting Dataset Update (Replacing Incorrect RF Predictions/Uncertainty with Correct Values) ---")
    logger.info(f"Prediction directory (New Correct RF Values): {PRED_DIR}")
    logger.info(f"Base dataset file (To Be Updated): {BASE_FILE}")
    logger.info(f"Output file base name: {OUTPUT_NAME}")
    logger.info(f"Columns to drop from base: '{OLD_PRED_COL_TO_DROP}', '{OLD_UNC_COL_TO_DROP}' (and potentially generic names)")
    logger.info(f"Final RF prediction column name: '{FINAL_RF_PRED_COL_NAME}'")
    logger.info(f"Final RF uncertainty column name: '{FINAL_RF_UNC_COL_NAME}'")

    # Validate input paths/files exist
    if not os.path.isdir(PRED_DIR): logger.error(f"Prediction directory does not exist: {PRED_DIR}. Exiting."); return
    if not os.path.isfile(BASE_FILE): logger.error(f"Base dataset file does not exist: {BASE_FILE}. Exiting."); return

    # Determine output directory
    effective_output_dir = OUTPUT_DIR
    if effective_output_dir is None: effective_output_dir = os.path.dirname(BASE_FILE) or "."
    os.makedirs(effective_output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {os.path.abspath(effective_output_dir)}")

    # 1. Find prediction files containing NEW RF data
    pred_file_pattern = os.path.join(PRED_DIR, "holdout_predictions_*K.csv")
    pred_files = glob.glob(pred_file_pattern)
    pred_files = [f for f in pred_files if "_metrics" not in os.path.basename(f)]
    if not pred_files: logger.error(f"No prediction files found matching pattern in {PRED_DIR}. Exiting."); return
    logger.info(f"Found {len(pred_files)} prediction files to source new RF data.")

    # 2. Extract NEW target predictions and uncertainty
    target_data_list = []
    for file_path in sorted(pred_files):
        filename = os.path.basename(file_path)
        prediction_temp = extract_temp_from_filename(filename)
        if prediction_temp is None: logger.warning(f"Could not extract temperature from {filename}. Skipping."); continue

        logger.info(f"Processing {filename} for NEW prediction temp {prediction_temp}K...")
        try:
            pred_df = pd.read_csv(file_path)
            required_source_cols = ['domain_id', 'resid', 'temperature', SOURCE_PRED_COL, SOURCE_UNC_COL]
            missing_source_cols = [col for col in required_source_cols if col not in pred_df.columns]
            if missing_source_cols: logger.error(f"Missing expected columns {missing_source_cols} in {filename}. Skipping."); continue

            target_rows = pred_df[np.isclose(pred_df['temperature'], prediction_temp)]
            if target_rows.empty: logger.warning(f"No matching temp rows found in {filename}."); continue

            # --- Select relevant columns and RENAME to FINAL RF names ---
            processed_rows = target_rows[['domain_id', 'resid', 'temperature', SOURCE_PRED_COL, SOURCE_UNC_COL]].copy()
            processed_rows.rename(columns={
                SOURCE_PRED_COL: FINAL_RF_PRED_COL_NAME, # Rename to deepflex_RF_rmsf
                SOURCE_UNC_COL: FINAL_RF_UNC_COL_NAME    # Rename to deepflex_RF_rmsf_uncertainty
            }, inplace=True)

            if not np.isclose(processed_rows['temperature'], prediction_temp).all(): logger.error(f"Temperature mismatch after filtering {filename}. Skipping."); continue
            target_data_list.append(processed_rows)
            logger.info(f"  Extracted {len(processed_rows)} NEW target RF predictions and uncertainties.")

        except Exception as e: logger.error(f"Error processing {filename}: {e}", exc_info=True)

    # 3. Combine NEW extracted data
    if not target_data_list: logger.error("No NEW target RF data extracted. Update cannot proceed."); return
    logger.info("Concatenating NEW extracted target RF data...")
    target_data_df = pd.concat(target_data_list, ignore_index=True)
    logger.info(f"Combined NEW RF data shape: {target_data_df.shape}")

    # Optional: Check for duplicates in the NEW data
    id_cols = ['domain_id', 'resid', 'temperature']
    duplicates = target_data_df[target_data_df.duplicated(subset=id_cols, keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate entries in NEW RF data. Keeping first.")
        target_data_df = target_data_df.drop_duplicates(subset=id_cols, keep='first')
        logger.info(f"NEW RF data shape after dropping duplicates: {target_data_df.shape}")

    # 4. Load base data
    try:
        logger.info(f"Loading base dataset: {BASE_FILE}")
        base_df = pd.read_csv(BASE_FILE)
        logger.info(f"Base dataset shape: {base_df.shape}")
    except Exception as e: logger.error(f"Error loading base file {BASE_FILE}: {e}", exc_info=True); return

    # --- 5. Prepare for Replacement: Drop OLD/WRONG RF columns + generic names from base_df ---
    cols_to_drop_from_base = []
    # Explicitly drop the OLD incorrect RF columns
    if OLD_PRED_COL_TO_DROP in base_df.columns:
        cols_to_drop_from_base.append(OLD_PRED_COL_TO_DROP)
    else:
        logger.warning(f"Did not find column '{OLD_PRED_COL_TO_DROP}' to drop in base data.")
    if OLD_UNC_COL_TO_DROP in base_df.columns:
        cols_to_drop_from_base.append(OLD_UNC_COL_TO_DROP)
    else:
        logger.warning(f"Did not find column '{OLD_UNC_COL_TO_DROP}' to drop in base data.")

    # Also drop the standard generic names if they exist, to avoid conflict during merge
    if GENERIC_PRED_COL in base_df.columns and GENERIC_PRED_COL not in cols_to_drop_from_base:
        cols_to_drop_from_base.append(GENERIC_PRED_COL)
        logger.warning(f"Also dropping existing generic column '{GENERIC_PRED_COL}' to avoid merge conflicts.")
    if GENERIC_UNC_COL in base_df.columns and GENERIC_UNC_COL not in cols_to_drop_from_base:
        cols_to_drop_from_base.append(GENERIC_UNC_COL)
        logger.warning(f"Also dropping existing generic column '{GENERIC_UNC_COL}' to avoid merge conflicts.")

    base_df_cleaned = base_df.copy()
    if cols_to_drop_from_base:
        base_df_cleaned = base_df_cleaned.drop(columns=cols_to_drop_from_base)
        logger.info(f"Dropped existing columns from base data: {cols_to_drop_from_base}")
    else:
        logger.info("Did not find specified columns to drop in base data.")

    # --- 6. Merge base data (cleaned) with NEW RF data ---
    logger.info(f"Merging base dataset with NEW RF data (containing columns '{FINAL_RF_PRED_COL_NAME}', '{FINAL_RF_UNC_COL_NAME}')...")
    # The target_data_df now has the new data under the FINAL desired column names
    analysis_df = pd.merge(base_df_cleaned, target_data_df, on=id_cols, how='left')
    logger.info(f"Merged dataset shape: {analysis_df.shape}")

    # Validation after merge
    if analysis_df.shape[0] != base_df.shape[0]:
        logger.warning(f"Row count changed after merge ({base_df.shape[0]} -> {analysis_df.shape[0]}). Check merge keys/duplicates in input data.")

    # Check final columns
    missing_preds = analysis_df[FINAL_RF_PRED_COL_NAME].isnull().sum()
    missing_unc = analysis_df[FINAL_RF_UNC_COL_NAME].isnull().sum()
    if missing_preds > 0: logger.warning(f"{missing_preds} rows have missing values in the final '{FINAL_RF_PRED_COL_NAME}' column.")
    else: logger.info(f"Successfully added/updated '{FINAL_RF_PRED_COL_NAME}' for all rows.")
    if missing_unc > 0: logger.warning(f"{missing_unc} rows have missing values in the final '{FINAL_RF_UNC_COL_NAME}' column.")
    else: logger.info(f"Successfully added/updated '{FINAL_RF_UNC_COL_NAME}' for all rows.")

    # --- 7. Save FINAL output ---
    output_csv_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.csv")
    output_parquet_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.parquet")

    try:
        logger.info(f"Saving FINAL analysis dataset to CSV: {output_csv_path}")
        analysis_df.to_csv(output_csv_path, index=False)
        logger.info("CSV saved successfully.")
    except Exception as e: logger.error(f"Error saving FINAL dataset to CSV: {e}", exc_info=True)

    if HAS_PYARROW:
        try:
            logger.info(f"Saving FINAL analysis dataset to Parquet: {output_parquet_path}")
            analysis_df.to_parquet(output_parquet_path, index=False, engine='pyarrow')
            logger.info("Parquet saved successfully.")
        except Exception as e: logger.error(f"Error saving FINAL dataset to Parquet: {e}", exc_info=True)
    else: logger.warning("Cannot save to Parquet: 'pyarrow' not installed.")

    logger.info("--- Dataset Update Finished ---")

if __name__ == "__main__":
    main()

# import os
# import glob
# import re
# import logging
# import pandas as pd
# import numpy as np
# # Optional: Check if pyarrow is installed for Parquet saving
# try:
#     import pyarrow
#     HAS_PYARROW = True
# except ImportError:
#     HAS_PYARROW = False

# # --- Configuration (Define your paths and settings here) ---

# # Directory containing the holdout_predictions_*K.csv files
# PRED_DIR = "/home/s_felix/packages/DeepFlex/output/deepflex/" #<-- SET THIS

# # Full path to the base aggregated holdout dataset CSV
# # BASE_FILE = "/home/s_felix/packages/DeepFlex/data/aggregated_holdout_dataset.csv" #<-- SET THIS
# BASE_FILE = "/home/s_felix/drDataScience/data/analysis_holdout_dataset.csv" #<-- SET THIS

# # Base name for the output files (without extension)
# OUTPUT_NAME = "analysis2_holdout_dataset" #<-- SET THIS (e.g., "analysis_holdout_with_deepflex")

# # Directory to save the output files. If None, defaults to the directory of BASE_FILE.
# OUTPUT_DIR = None #<-- SET THIS (e.g., "/home/s_felix/analysis_output") or keep None

# # Name for the new column containing the matched predictions
# NEW_COL_NAME = "deepflex_rmsf" #<-- SET THIS

# # --- End Configuration ---

# # --- Logging Setup ---
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# logger = logging.getLogger(__name__)

# # --- Regex Pattern ---
# # Regex to extract temperature (numeric) from prediction filename
# # Example: holdout_predictions_320K.csv -> 320
# FILENAME_TEMP_PATTERN = re.compile(r"holdout_predictions_(\d+)K\.csv", re.IGNORECASE)

# # --- Helper Function ---
# def extract_temp_from_filename(filename):
#     """Extracts the numeric temperature from the prediction filename."""
#     match = FILENAME_TEMP_PATTERN.search(filename)
#     if match:
#         try:
#             return float(match.group(1))
#         except ValueError:
#             return None
#     return None

# # --- Main Logic ---
# def main():
#     logger.info("--- Starting Analysis Dataset Creation (Hardcoded Config) ---")
#     logger.info(f"Prediction directory: {PRED_DIR}")
#     logger.info(f"Base dataset file: {BASE_FILE}")
#     logger.info(f"Output file base name: {OUTPUT_NAME}")
#     logger.info(f"New prediction column name: {NEW_COL_NAME}")

#     # Validate input paths/files exist
#     if not os.path.isdir(PRED_DIR):
#         logger.error(f"Prediction directory does not exist: {PRED_DIR}. Exiting.")
#         return
#     if not os.path.isfile(BASE_FILE):
#         logger.error(f"Base dataset file does not exist: {BASE_FILE}. Exiting.")
#         return

#     # Determine output directory
#     effective_output_dir = OUTPUT_DIR
#     if effective_output_dir is None:
#         effective_output_dir = os.path.dirname(BASE_FILE)
#         if not effective_output_dir: # Handle case where base file is in current dir
#             effective_output_dir = "."
#     os.makedirs(effective_output_dir, exist_ok=True)
#     logger.info(f"Output directory set to: {os.path.abspath(effective_output_dir)}")


#     # 1. Find prediction files
#     pred_file_pattern = os.path.join(PRED_DIR, "holdout_predictions_*K.csv")
#     pred_files = glob.glob(pred_file_pattern)
#     pred_files = [f for f in pred_files if "_metrics" not in os.path.basename(f)] # Exclude metrics files

#     if not pred_files:
#         logger.error(f"No prediction files found matching pattern in {PRED_DIR}. Exiting.")
#         return
#     logger.info(f"Found {len(pred_files)} prediction files.")

#     # 2. Extract target predictions
#     target_predictions_list = []
#     for file_path in sorted(pred_files):
#         filename = os.path.basename(file_path)
#         prediction_temp = extract_temp_from_filename(filename)

#         if prediction_temp is None:
#             logger.warning(f"Could not extract temperature from {filename}. Skipping.")
#             continue

#         logger.info(f"Processing {filename} for prediction temp {prediction_temp}K...")
#         try:
#             pred_df = pd.read_csv(file_path)

#             # Filter rows where original temp matches prediction temp
#             # Use np.isclose for robust float comparison (adjust tolerance if needed)
#             target_rows = pred_df[np.isclose(pred_df['temperature'], prediction_temp)]

#             if target_rows.empty:
#                 logger.warning(f"No rows found in {filename} where original temperature matches prediction temperature ({prediction_temp}K).")
#                 continue

#             # Select relevant columns and rename prediction column
#             processed_rows = target_rows[['domain_id', 'resid', 'temperature', 'rmsf_predicted']].copy()
#             processed_rows.rename(columns={'rmsf_predicted': NEW_COL_NAME}, inplace=True)

#             # Simple validation
#             if not np.isclose(processed_rows['temperature'], prediction_temp).all():
#                  logger.error(f"Temperature mismatch error after filtering {filename}. Check logic.")
#                  continue # Skip this file due to error

#             target_predictions_list.append(processed_rows)
#             logger.info(f"  Extracted {len(processed_rows)} target predictions.")

#         except FileNotFoundError:
#             logger.error(f"File not found: {file_path}. Skipping.")
#         except pd.errors.EmptyDataError:
#             logger.warning(f"File is empty: {filename}. Skipping.")
#         except KeyError as e:
#             logger.error(f"Missing expected column {e} in {filename}. Skipping.")
#         except Exception as e:
#             logger.error(f"Error processing {filename}: {e}", exc_info=True)

#     # 3. Combine predictions
#     if not target_predictions_list:
#         logger.error("No target predictions could be extracted from any file. Cannot create analysis dataset.")
#         return

#     logger.info("Concatenating extracted target predictions...")
#     target_predictions_df = pd.concat(target_predictions_list, ignore_index=True)
#     logger.info(f"Combined predictions shape: {target_predictions_df.shape}")

#     # Optional: Check for duplicates in the combined predictions
#     # A duplicate means the same residue at the same temp was predicted in multiple files correctly
#     id_cols = ['domain_id', 'resid', 'temperature']
#     duplicates = target_predictions_df[target_predictions_df.duplicated(subset=id_cols, keep=False)]
#     if not duplicates.empty:
#         logger.warning(f"Found {len(duplicates)} duplicate entries (same residue/temp) in combined predictions. Review prediction files/process.")
#         # Decide how to handle: keep first, average, error out? Keeping first for now.
#         target_predictions_df = target_predictions_df.drop_duplicates(subset=id_cols, keep='first')
#         logger.info(f"Dropped duplicates, keeping first entry. Shape is now: {target_predictions_df.shape}")

#     # 4. Load base data
#     try:
#         logger.info(f"Loading base holdout dataset: {BASE_FILE}")
#         base_df = pd.read_csv(BASE_FILE)
#         logger.info(f"Base dataset shape: {base_df.shape}")
#     except FileNotFoundError:
#         logger.error(f"Base holdout file not found: {BASE_FILE}. Exiting.")
#         return
#     except Exception as e:
#         logger.error(f"Error loading base file {BASE_FILE}: {e}", exc_info=True)
#         return

#     # 5. Merge data
#     logger.info(f"Merging base dataset with target predictions on {id_cols}...")
#     analysis_df = pd.merge(base_df, target_predictions_df, on=id_cols, how='left')
#     logger.info(f"Merged dataset shape: {analysis_df.shape}")

#     # Validation after merge
#     if analysis_df.shape[0] != base_df.shape[0]:
#         logger.warning(f"Number of rows changed after merge ({base_df.shape[0]} -> {analysis_df.shape[0]}). Check merge keys/duplicates.")

#     missing_preds = analysis_df[NEW_COL_NAME].isnull().sum()
#     if missing_preds > 0:
#         logger.warning(f"Found {missing_preds} rows in the base dataset with no matching prediction added (NaN in '{NEW_COL_NAME}').")
#     else:
#         logger.info(f"Successfully added predictions to all rows (column '{NEW_COL_NAME}').")

#     # 6. Save output
#     output_csv_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.csv")
#     output_parquet_path = os.path.join(effective_output_dir, f"{OUTPUT_NAME}.parquet")

#     try:
#         logger.info(f"Saving analysis dataset to CSV: {output_csv_path}")
#         analysis_df.to_csv(output_csv_path, index=False)
#         logger.info("CSV saved successfully.")
#     except Exception as e:
#         logger.error(f"Error saving analysis dataset to CSV: {e}", exc_info=True)

#     if HAS_PYARROW:
#         try:
#             logger.info(f"Saving analysis dataset to Parquet: {output_parquet_path}")
#             # Specify the engine for clarity and reproducibility
#             analysis_df.to_parquet(output_parquet_path, index=False, engine='pyarrow')
#             logger.info("Parquet saved successfully (using pyarrow engine).")
#         except Exception as e:
#             logger.error(f"Error saving analysis dataset to Parquet: {e}", exc_info=True)
#     else:
#          logger.warning(f"Cannot save to Parquet format because 'pyarrow' is not installed. Skipping Parquet save. To enable, run: pip install pyarrow")

#     logger.info("--- Analysis Dataset Creation Finished ---")

# if __name__ == "__main__":
#     main()