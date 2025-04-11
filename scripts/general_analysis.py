# Example filename: scripts/general_analysis.py # Or your current filename
import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             explained_variance_score, median_absolute_error)
from collections import Counter, defaultdict
import re
import warnings

# --- Configuration ---
# Input dataset path
INPUT_DIR = "/home/s_felix/drDataScience/data" # Directory containing the file
INPUT_BASE_FILENAME = "holdout_with_AttentionESM_preds" # Filename WITHOUT extension

# Directory to save the output ANALYSIS REPORT ONLY
OUTPUT_DIR = "/home/s_felix/drDataScience/analysis_outputs" # Where the report will be saved

# Output analysis text file name
OUTPUT_TXT_FILENAME = "general_analysis3_report.txt"

# --- File Paths ---
INPUT_CSV_PATH = os.path.join(INPUT_DIR, f"{INPUT_BASE_FILENAME}.csv")
INPUT_PARQUET_PATH = os.path.join(INPUT_DIR, f"{INPUT_BASE_FILENAME}.parquet")
OUTPUT_TXT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_TXT_FILENAME)

# --- Column Names (Base Features & Target) ---
DOMAIN_COL = "domain_id"; TEMP_COL = "temperature"; RESID_COL = "resid"
RESNAME_COL = "resname"; ACTUAL_COL = "rmsf"; CORE_EXT_COL = "core_exterior"
DSSP_COL = "dssp"; NORM_RESID_COL = "normalized_resid"; REL_ACC_COL = "relative_accessibility"
SIZE_COL = "protein_size"

# --- Model/Prediction Column Patterns (for dynamic finding) ---
PREDICTION_PATTERN = re.compile(r"^(?!rmsf$)(?!.*_uncertainty$)(.+_rmsf)$", re.IGNORECASE)
UNCERTAINTY_PATTERN = re.compile(r"^(deepflex_(?:NN|RF)_rmsf_uncertainty)$", re.IGNORECASE)

# --- Analysis Constants ---
N_TOP_BOTTOM = 10; POS_BINS = 5
POS_BIN_LABELS = ['N-term (0-0.2)', 'Mid-N (0.2-0.4)', 'Middle (0.4-0.6)', 'Mid-C (0.6-0.8)', 'C-term (0.8-1.0)']
QUANTILE_BINS = 10; MIN_POINTS_FOR_METRICS = 10
# MODIFIED: Changed primary model for ranking
PRIMARY_MODEL_KEY_FOR_RANKING = 'ATTENTION-ESM' # Model to use for sorting/selecting best/worst/outliers

# --- Case Study Thresholds (Based on PRIMARY_MODEL_KEY_FOR_RANKING) ---
NAILED_IT_MIN_PCC = 0.88; NAILED_IT_MIN_R2 = 0.70; NAILED_IT_MAX_MAE = 0.12
NAILED_IT_MIN_ACTUAL_STD_FACTOR = 0.75; NAILED_IT_MIN_PRED_STD_FACTOR = 0.75
TEMP_MASTERED_MIN_ACTUAL_DELTA = 0.5; TEMP_MASTERED_MAX_MAE = 0.15 # Based on primary model overall MAE
TEMP_MASTERED_MAX_DELTA_MAE_ABS = 0.05 # Based on primary model delta MAE

# --- Logging Setup ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
root_logger = logging.getLogger()
if root_logger.hasHandlers(): root_logger.handlers.clear() # Avoid duplicate logs
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning) # Suppress PearsonR constant input warnings

# --- Helper Functions ---
# [Keep write_section, calculate_group_metrics, map_ss_group, get_top_bottom_indices, analyze_domain_characteristics functions as they were in the previous version]
# --- Helper Functions ---
def write_section(outfile, title, content, note=None):
    """Writes a formatted section to the output text file."""
    outfile.write(f"\n{'=' * 80}\n")
    outfile.write(f"## {title.upper()} ##\n")
    if note: outfile.write(f"   ({note})\n")
    outfile.write(f"{'-' * 80}\n")
    if isinstance(content, (pd.DataFrame, pd.Series)):
        # Adjust display width dynamically or set a larger fixed width
        pd.set_option('display.width', 1000) # Set a wider display
        outfile.write(content.to_string())
        pd.reset_option('display.width') # Reset to default if needed elsewhere
        outfile.write("\n")
    elif isinstance(content, dict):
         import json
         outfile.write(json.dumps(content, indent=2, default=str))
         outfile.write("\n")
    elif isinstance(content, (list, tuple)):
         for i, item in enumerate(content): outfile.write(f"  {i+1}. {item}\n")
    else:
        try: outfile.write(f"{content:.4f}\n") # Try formatting as float
        except: outfile.write(str(content) + "\n") # Fallback to string

def calculate_group_metrics(group, actual_col, pred_cols_config, uncertainty_cols_config=None):
    """Calculates performance metrics for multiple models within a data group."""
    results = {}; group_count = len(group); results['count'] = group_count
    metric_keys = ['rmse', 'mae', 'pcc', 'r2', 'explained_variance', 'median_abs_error']
    uncertainty_metrics = ['avg_uncertainty', 'uncertainty_error_corr', 'within_1std']

    # Initialize results with NaN for ALL detected prediction models
    for model_key in pred_cols_config: # Use keys from the config passed in
        for metric in metric_keys: results[f"{model_key}_{metric}"] = np.nan
        results[f"{model_key}_pred_stddev"] = np.nan
        # Initialize uncertainty metrics only if the model has a corresponding uncertainty column
        if uncertainty_cols_config and model_key in uncertainty_cols_config:
            for unc_metric in uncertainty_metrics: results[f"{model_key}_{unc_metric}"] = np.nan
    results['actual_stddev'] = np.nan

    if group_count < MIN_POINTS_FOR_METRICS: return pd.Series(results)
    if actual_col not in group: return pd.Series(results)
    actual_vals = group[actual_col].dropna()
    # Require minimum points *after* dropping NaNs in actual
    if actual_vals.nunique() <= 1 or len(actual_vals) < MIN_POINTS_FOR_METRICS: return pd.Series(results)
    results['actual_stddev'] = actual_vals.std()

    # Iterate through all models found (including new ones)
    for model_key, pred_col in pred_cols_config.items():
        if pred_col not in group: continue
        df_pair = group[[actual_col, pred_col]].dropna()
        # Require min points after dropping NaNs in pred too
        if len(df_pair) < MIN_POINTS_FOR_METRICS: continue

        y_true, y_pred = df_pair[actual_col].values, df_pair[pred_col].values
        if np.var(y_pred) > 1e-9: results[f'{model_key}_pred_stddev'] = np.std(y_pred)
        else: results[f'{model_key}_pred_stddev'] = 0.0

        # Calculate standard performance metrics
        try: results[f"{model_key}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception: pass
        try: results[f"{model_key}_mae"] = mean_absolute_error(y_true, y_pred)
        except Exception: pass
        try: results[f"{model_key}_median_abs_error"] = median_absolute_error(y_true, y_pred)
        except Exception: pass
        try:
            if np.var(y_true) > 1e-9 : results[f"{model_key}_explained_variance"] = explained_variance_score(y_true, y_pred)
        except Exception: pass

        if np.var(y_true) > 1e-9 and np.var(y_pred) > 1e-9:
            try:
                with warnings.catch_warnings(): # Suppress PearsonR constant input warnings specifically here
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pcc, _ = pearsonr(y_true, y_pred)
                    results[f"{model_key}_pcc"] = pcc if not np.isnan(pcc) else 0.0
            except Exception: pass
            try: results[f"{model_key}_r2"] = r2_score(y_true, y_pred)
            except Exception: pass

        # Uncertainty Metrics (only if uncertainty config exists for this model key)
        if uncertainty_cols_config and model_key in uncertainty_cols_config:
            unc_col = uncertainty_cols_config[model_key]
            if unc_col in group:
                df_unc_pair = group[[actual_col, pred_col, unc_col]].dropna()
                if len(df_unc_pair) >= MIN_POINTS_FOR_METRICS:
                    y_true_unc, y_pred_unc, y_std_unc = df_unc_pair[[actual_col, pred_col, unc_col]].values.T
                    y_std_unc_safe = np.maximum(y_std_unc, 1e-9); errors_unc = np.abs(y_true_unc - y_pred_unc)
                    results[f"{model_key}_avg_uncertainty"] = np.mean(y_std_unc)
                    results[f"{model_key}_within_1std"] = np.mean(errors_unc <= y_std_unc_safe)
                    if np.var(y_std_unc_safe) > 1e-9 and np.var(errors_unc) > 1e-9:
                        try:
                             with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                unc_err_corr, _ = pearsonr(y_std_unc_safe, errors_unc)
                                results[f"{model_key}_uncertainty_error_corr"] = unc_err_corr if not np.isnan(unc_err_corr) else 0.0
                        except Exception: pass

    return pd.Series(results)

def map_ss_group(ss_code):
    """Maps DSSP codes to broader Helix/Sheet/Loop groups."""
    if isinstance(ss_code, str):
        if ss_code in ('H', 'G', 'I'): return 'Helix'
        elif ss_code in ('E', 'B'): return 'Sheet'
    return 'Loop'

def get_top_bottom_indices(perf_data, metric_col, n=5):
    """Gets indices of top/bottom items based on a metric, handling NaNs & metric type."""
    if not isinstance(perf_data, (pd.DataFrame, pd.Series)):
        logger.debug(f"Invalid input type for perf_data: {type(perf_data)}")
        return [], []
    metric_exists = False
    if isinstance(perf_data, pd.DataFrame): metric_exists = metric_col in perf_data.columns
    elif isinstance(perf_data, pd.Series): metric_exists = metric_col == perf_data.name

    if not metric_exists:
        logger.debug(f"Metric column '{metric_col}' not found in performance data.")
        return [], []

    ascending_sort = True # Default (for MAE, RMSE)
    if any(k in metric_col.lower() for k in ['pcc', 'r2', 'explained_variance', 'within_1std', 'uncertainty_error_corr']):
         ascending_sort = False # Higher is better

    try:
         valid_perf_data = perf_data.dropna(subset=[metric_col] if isinstance(perf_data, pd.DataFrame) else None)
         if valid_perf_data.empty: return [], []
         if isinstance(perf_data, pd.DataFrame): sorted_data = valid_perf_data.sort_values(by=metric_col, ascending=ascending_sort)
         else: sorted_data = valid_perf_data.sort_values(ascending=ascending_sort)
         top_n_indices = list(sorted_data.head(n).index)
         bottom_n_indices = list(sorted_data.tail(n).index)
         # If sorting descending (higher is better), swap top/bottom
         if not ascending_sort: top_n_indices, bottom_n_indices = bottom_n_indices, top_n_indices
         return top_n_indices, bottom_n_indices
    except Exception as e: logger.error(f"Error sorting/ranking by '{metric_col}': {e}"); return [], []

def analyze_domain_characteristics(df, domain_list, pred_cols_config, uncertainty_cols_config):
    """Analyzes characteristics and performance of a list of domains."""
    if not domain_list: return pd.DataFrame()
    # Ensure domain_list contains unique values
    unique_domain_list = list(set(domain_list))
    subset_df = df[df[DOMAIN_COL].isin(unique_domain_list)].copy()
    if subset_df.empty: return pd.DataFrame()

    features = {}
    # --- Corrected Temperature aggregation ---
    if TEMP_COL in subset_df: features['Avg Temp'] = subset_df.groupby(DOMAIN_COL)[TEMP_COL].mean() # Use mean temp
    if SIZE_COL in subset_df: features['Size'] = subset_df.groupby(DOMAIN_COL)[SIZE_COL].first()
    if REL_ACC_COL in subset_df: features['Avg Rel Acc'] = subset_df.groupby(DOMAIN_COL)[REL_ACC_COL].mean()
    if ACTUAL_COL in subset_df: features['Avg Actual RMSF'] = subset_df.groupby(DOMAIN_COL)[ACTUAL_COL].mean()
    if CORE_EXT_COL in subset_df: features['% Core'] = subset_df.groupby(DOMAIN_COL)[CORE_EXT_COL].apply(lambda x: (x == 'core').mean() * 100 if not x.empty else 0, include_groups=False)
    if 'ss_group' in subset_df:
        ss_dist = subset_df.groupby(DOMAIN_COL)['ss_group'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
        for ss_type in ['Helix', 'Sheet', 'Loop']: features[f'% {ss_type}'] = ss_dist.get(ss_type, 0.0)

    # Calculate metrics for ALL models for these domains
    domain_perf = subset_df.groupby(DOMAIN_COL).apply(
        lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
        include_groups=False
    )

    # Combine features and performance metrics
    all_data_to_concat = [s for s in features.values() if isinstance(s, (pd.Series, pd.DataFrame))]
    if not domain_perf.empty:
        # Ensure domain_perf index matches features index before concatenating
        # Find a reliable feature index to align with
        align_index = None
        if 'Avg Temp' in features:
            align_index = features['Avg Temp'].index
        elif 'Size' in features:
             align_index = features['Size'].index
        # Add more fallbacks if necessary

        if align_index is not None:
             domain_perf = domain_perf.reindex(index=align_index) # Align index
             all_data_to_concat.append(domain_perf)
        else:
             logger.warning("Could not find a suitable index in features to align domain performance.")
             # Attempt concat anyway, but it might fail or produce NaNs
             all_data_to_concat.append(domain_perf)


    if not all_data_to_concat: return pd.DataFrame()
    try:
        # Use 'outer' join to preserve all domains even if some features/metrics are missing
        analysis = pd.concat(all_data_to_concat, axis=1, join='outer')
    except Exception as e:
        logger.error(f"Concat failed in analyze_domain_characteristics: {e}")
        # Debugging: Print shapes and indices
        logger.debug("Shapes and indices before concat:")
        for i, item in enumerate(all_data_to_concat):
            if isinstance(item, (pd.Series, pd.DataFrame)):
                logger.debug(f" Item {i}: shape={item.shape}, index={item.index.tolist()[:5] if item.index is not None else 'No Index'}")
        return pd.DataFrame() # Return empty if concat fails

    return analysis.round(3)


# --- Main Analysis Function ---
def main():
    logger.info(f"--- Starting Comprehensive General Analysis Report ({OUTPUT_TXT_FILENAME}) ---")

    # --- Section 1: Load Data ---
    df = None; input_file_used = None
    if os.path.exists(INPUT_PARQUET_PATH):
        try: logger.info(f"Loading data from Parquet: {INPUT_PARQUET_PATH}"); df = pd.read_parquet(INPUT_PARQUET_PATH); input_file_used = INPUT_PARQUET_PATH
        except Exception as e: logger.warning(f"Could not load Parquet file ({e}). Falling back to CSV.")
    if df is None and os.path.exists(INPUT_CSV_PATH):
        try: logger.info(f"Loading data from CSV: {INPUT_CSV_PATH}"); df = pd.read_csv(INPUT_CSV_PATH); input_file_used = INPUT_CSV_PATH
        except Exception as e: logger.error(f"Failed to load data from CSV: {e}. Exiting."); return
    if df is None: logger.error(f"Could not find or load input data file '{INPUT_BASE_FILENAME}' in '{INPUT_DIR}'. Exiting."); return
    logger.info(f"Data loaded. Shape: {df.shape}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Dynamically Find Models ---
    pred_cols_config = {}; uncertainty_cols_config = {}; model_keys_found = []
    logger.info("Identifying available model prediction/uncertainty columns...")
    for col in df.columns:
        pred_match = PREDICTION_PATTERN.match(col); unc_match = UNCERTAINTY_PATTERN.match(col)

        if pred_match:
            col_name = pred_match.group(1) # Full matched column name (e.g., deepflex_RF_rmsf, new_model_rmsf)
            model_key = None
            # --- MODIFIED: Logic to extract a cleaner key ---
            if col_name.lower().startswith('deepflex_'):
                try: model_key = col_name.split('_')[1].upper() # e.g., RF
                except IndexError: pass
            elif col_name.lower() == 'esm_rmsf': model_key = 'ESM'
            elif col_name.lower() == 'voxel_rmsf': model_key = 'Voxel'
            # Handle attention_esm_rmsf specifically
            elif col_name.lower() == 'attention_esm_rmsf': model_key = 'ATTENTION-ESM'
            elif col_name.lower().endswith('_rmsf'):
                 # Generic fallback for new models like 'my_model_rmsf' -> 'MY_MODEL'
                 key_base = col_name[:-len('_rmsf')]
                 # Avoid making keys too generic like '' if column was just '_rmsf' (should be caught by pattern)
                 if key_base:
                     model_key = key_base.upper().replace('_', '-') # Replace underscores for readability
                 else:
                     model_key = col_name # Fallback to full column name if base is empty

            # Fallback if no key assigned yet or if key is problematic
            if model_key is None or not model_key:
                model_key = col_name # Use the full column name as the key

            # Check for duplicate keys (if different columns map to same key)
            if model_key in pred_cols_config:
                logger.warning(f"Duplicate model key '{model_key}' detected! Column '{col_name}' might overwrite '{pred_cols_config[model_key]}'. Adjust naming or key extraction logic if needed.")

            pred_cols_config[model_key] = col_name # Store full column name
            if model_key not in model_keys_found: model_keys_found.append(model_key)
            logger.info(f"  Found Prediction: '{model_key}' -> {col_name}")

        elif unc_match:
            col_name = unc_match.group(1) # Full matched column name
            model_key = None
            # Extract key based on naming convention (currently only DeepFlex)
            if col_name.lower().startswith('deepflex_'):
                 try: model_key = col_name.split('_')[1].upper() # e.g., RF
                 except IndexError: pass

            if model_key is None or not model_key:
                 model_key = col_name # Fallback to full column name

            # Check for duplicate keys
            if model_key in uncertainty_cols_config:
                 logger.warning(f"Duplicate uncertainty key '{model_key}' detected for column '{col_name}'.")

            uncertainty_cols_config[model_key] = col_name
            # Add to model_keys_found if not already there from prediction discovery
            if model_key not in model_keys_found: model_keys_found.append(model_key)
            logger.info(f"  Found Uncertainty: '{model_key}' -> {col_name}")

    # --- Order keys consistently ---
    # Prioritize known/common models, then sort the rest alphabetically
    known_order = ['RF', 'NN', 'LGBM', 'ESM', 'Voxel', 'ATTENTION-ESM'] # Added ATTENTION-ESM here
    ordered_keys = [k for k in known_order if k in model_keys_found]
    other_keys = sorted([k for k in model_keys_found if k not in known_order])
    model_keys_ordered = ordered_keys + other_keys # Final ordered list of all model keys

    models_with_uncertainty = sorted([k for k in uncertainty_cols_config.keys() if k in model_keys_ordered]) # Use ordered keys

    logger.info(f"Detected models for analysis (ordered): {model_keys_ordered}")
    logger.info(f"Models with uncertainty: {models_with_uncertainty}")
    performance_possible = ACTUAL_COL in df.columns and bool(pred_cols_config)

    # --- Calculate Error Columns ---
    error_cols_available = {}; abs_error_cols_available = {}
    if performance_possible:
        for model_key, pred_col in pred_cols_config.items():
             error_col = f"{model_key}_error"; abs_error_col = f"{model_key}_abs_error"
             # Use unique but predictable names based on the potentially generic model_key
             if error_col not in df.columns:
                 logger.debug(f"Calculating error columns for {model_key} (column: {pred_col})...")
                 try:
                     # Important: Use .loc to avoid SettingWithCopyWarning if df is a slice
                     df.loc[:, error_col] = df[pred_col] - df[ACTUAL_COL]
                     df.loc[:, abs_error_col] = df[error_col].abs()
                 except Exception as e:
                     logger.error(f"Error calculating error for model {model_key} (col: {pred_col}): {e}")
                     continue # Skip adding to available if calculation failed
             if error_col in df.columns: error_cols_available[model_key] = error_col
             if abs_error_col in df.columns: abs_error_cols_available[model_key] = abs_error_col

    # Add SS Group column if needed
    if DSSP_COL in df.columns and 'ss_group' not in df.columns:
        df['ss_group'] = df[DSSP_COL].apply(map_ss_group)

    # --- Open Output Text File ---
    try:
        with open(OUTPUT_TXT_PATH, 'w') as outfile:
            logger.info(f"Writing analysis report to: {OUTPUT_TXT_PATH}")

            # --- Section 1: Basic Info ---
            num_unique_domains = df[DOMAIN_COL].nunique() if DOMAIN_COL in df.columns else "N/A"
            basic_info = {
                "Source Data File": input_file_used,
                "Total Rows": df.shape[0],
                "Total Columns": df.shape[1],
                "Unique Domains": num_unique_domains,
                "Memory Usage": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
                "Models Detected (Ordered)": model_keys_ordered, # Use ordered list
                "Uncertainty Models": models_with_uncertainty,
                "Prediction Cols Config": pred_cols_config, # Show key -> column mapping
                "Uncertainty Cols Config": uncertainty_cols_config,
                "Primary Model for Ranking": PRIMARY_MODEL_KEY_FOR_RANKING # Added this info
            }
            write_section(outfile, "1. BASIC INFORMATION", basic_info)

            # --- Section 2: Missing Values ---
            missing_values = df.isnull().sum(); missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            if not missing_values.empty: missing_summary = pd.DataFrame({'count': missing_values, 'percentage': (missing_values / df.shape[0]) * 100}); write_section(outfile, "2. MISSING VALUE SUMMARY", missing_summary.round(2))
            else: write_section(outfile, "2. MISSING VALUE SUMMARY", "No missing values found.")

            # --- Section 3: Overall Descriptive Statistics ---
            # Include ALL detected prediction columns
            stats_cols = ([ACTUAL_COL] if ACTUAL_COL in df.columns else []) + \
                         list(pred_cols_config.values()) + \
                         list(abs_error_cols_available.values()) + \
                         list(uncertainty_cols_config.values()) + \
                         [NORM_RESID_COL, REL_ACC_COL, TEMP_COL, SIZE_COL]
            stats_cols_final = sorted([col for col in stats_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])])
            if stats_cols_final:
                try:
                    overall_stats = df[stats_cols_final].describe()
                    write_section(outfile, "3. OVERALL DESCRIPTIVE STATISTICS", overall_stats)
                    # Store overall std dev for later use if calculated
                    overall_actual_std = overall_stats.loc['std', ACTUAL_COL] if ACTUAL_COL in overall_stats.columns else 0
                    overall_pred_std = {key: overall_stats.loc['std', pred_col] if key in pred_cols_config and (pred_col := pred_cols_config[key]) in overall_stats.columns else 0 for key in model_keys_ordered} # Use ordered keys
                except Exception as e:
                    logger.error(f"Error calculating overall descriptive stats: {e}")
                    write_section(outfile, "3. OVERALL DESCRIPTIVE STATISTICS", f"Error calculating stats: {e}")
                    overall_actual_std = 0 # Ensure it's defined
                    overall_pred_std = {} # Ensure it's defined
            else:
                logger.warning("Could not calculate overall descriptive stats (no valid numeric columns found).")
                write_section(outfile, "3. OVERALL DESCRIPTIVE STATISTICS", "No valid numeric columns found for stats.")
                overall_actual_std = 0 # Ensure it's defined
                overall_pred_std = {} # Ensure it's defined


            # --- Section 4: Data Distributions (Added Counts) ---
            write_section(outfile, "4. DATA DISTRIBUTIONS", "Counts based on a single representative temperature.")
            # Select one temp (e.g., lowest available) for counts
            representative_temp = df[TEMP_COL].min() if TEMP_COL in df.columns and df[TEMP_COL].nunique() > 0 else None
            df_single_temp = df[df[TEMP_COL] == representative_temp] if representative_temp is not None else pd.DataFrame()
            note_counts = f"(Counts from Temp={representative_temp:.0f}K, N={len(df_single_temp)})" if representative_temp is not None else "(Counts not available)"

            if TEMP_COL in df.columns:
                 dist_df = pd.concat([df[TEMP_COL].value_counts(normalize=True).mul(100), df[TEMP_COL].value_counts()], axis=1, keys=['Percent', 'Count']).round(2)
                 write_section(outfile, f"4.1 {TEMP_COL.upper()} DISTRIBUTION", dist_df)
            if RESNAME_COL in df.columns:
                 dist_df = pd.concat([df[RESNAME_COL].value_counts(normalize=True).mul(100), df_single_temp[RESNAME_COL].value_counts() if not df_single_temp.empty else None], axis=1, keys=['Percent', 'Count (Sample Temp)']).round(2)
                 write_section(outfile, f"4.2 {RESNAME_COL.upper()} DISTRIBUTION", dist_df, note=note_counts)
            if CORE_EXT_COL in df.columns:
                 dist_df = pd.concat([df[CORE_EXT_COL].value_counts(normalize=True).mul(100), df_single_temp[CORE_EXT_COL].value_counts() if not df_single_temp.empty else None], axis=1, keys=['Percent', 'Count (Sample Temp)']).round(2)
                 write_section(outfile, f"4.3 {CORE_EXT_COL.upper()} DISTRIBUTION", dist_df, note=note_counts)
            if 'ss_group' in df.columns:
                 dist_df = pd.concat([df['ss_group'].value_counts(normalize=True).mul(100), df_single_temp['ss_group'].value_counts() if not df_single_temp.empty else None], axis=1, keys=['Percent', 'Count (Sample Temp)']).round(2)
                 write_section(outfile, "4.4 GROUPED SECONDARY STRUCTURE (H/E/L) DISTRIBUTION", dist_df, note=note_counts)


            # --- Section 5: Comprehensive Model Comparison ---
            write_section(outfile, "5. COMPREHENSIVE MODEL COMPARISON", "Comparing performance across ALL detected models.")
            if performance_possible:
                # 5.1 Overall Performance Metrics Table
                logger.info("Calculating overall performance metrics for all models...")
                overall_metrics_list = []
                # Iterate using the ordered list of keys
                for model_key in model_keys_ordered:
                    if model_key not in pred_cols_config: continue # Should not happen if keys are derived correctly
                    pred_col = pred_cols_config[model_key]
                    df_pair = df[[ACTUAL_COL, pred_col]].dropna()
                    if len(df_pair) < MIN_POINTS_FOR_METRICS: logger.warning(f"Skipping overall metrics for {model_key}: Not enough valid data points ({len(df_pair)})."); continue
                    y_true, y_pred = df_pair[ACTUAL_COL].values, df_pair[pred_col].values
                    metrics = {"Model": model_key}; # Store metrics for this model
                    try: metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
                    except Exception: pass
                    try: metrics["MAE"] = mean_absolute_error(y_true, y_pred)
                    except Exception: pass
                    try: metrics["MedAE"] = median_absolute_error(y_true, y_pred)
                    except Exception: pass
                    if np.var(y_true) > 1e-9 and np.var(y_pred) > 1e-9:
                         try: pcc, _ = pearsonr(y_true, y_pred); metrics["PCC"] = pcc if not np.isnan(pcc) else 0.0
                         except Exception: pass
                         try: metrics["R2"] = r2_score(y_true, y_pred)
                         except Exception: pass
                    overall_metrics_list.append(metrics)

                if overall_metrics_list:
                    overall_metrics_df = pd.DataFrame(overall_metrics_list).set_index("Model")
                    # Reindex to ensure consistent order in the output table
                    overall_metrics_df = overall_metrics_df.reindex(model_keys_ordered).dropna(how='all')
                    write_section(outfile, "5.1 OVERALL PERFORMANCE METRICS", overall_metrics_df)
                else: write_section(outfile, "5.1 OVERALL PERFORMANCE METRICS", "Could not calculate overall metrics for any model.")

                # 5.2 Performance by Temperature
                if TEMP_COL in df.columns:
                     logger.info("Calculating performance by temperature for all models...");
                     # Pass the full configs to the function
                     temp_perf = df.groupby(TEMP_COL).apply(
                         lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                         include_groups=False
                     )
                     if not temp_perf.empty:
                         # Dynamically generate columns to show based on available metrics for ordered keys
                         cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc', 'r2'] if f"{key}_{m}" in temp_perf.columns]
                         write_section(outfile, "5.2 PERFORMANCE METRICS BY TEMPERATURE", temp_perf[cols_to_show])
                         temp_report = "Model Comparison Highlights by Temperature:\n";
                         for temp, row in temp_perf.iterrows():
                             temp_report += f"  T={temp:.0f}K (N={row.get('count', 0)}): ";
                             # Get metrics for all available models at this temp
                             maes = {key: row.get(f"{key}_mae", np.inf) for key in model_keys_ordered if f"{key}_mae" in row}
                             pccs = {key: row.get(f"{key}_pcc", -np.inf) for key in model_keys_ordered if f"{key}_pcc" in row}
                             best_mae_model = min(maes, key=maes.get) if maes else "N/A"
                             best_pcc_model = max(pccs, key=pccs.get) if pccs else "N/A"
                             mae_val = maes.get(best_mae_model, np.inf)
                             pcc_val = pccs.get(best_pcc_model, -np.inf)
                             temp_report += f"Best MAE={best_mae_model}({mae_val:.3f}), Best PCC={best_pcc_model}({pcc_val:.3f})\n"
                         write_section(outfile, "5.2.1 TEMP PERFORMANCE SUMMARY", temp_report)
                     else: write_section(outfile, "5.2 PERFORMANCE METRICS BY TEMPERATURE", "No results after grouping by temperature.")

                # 5.3 Prediction Correlations (MODIFIED -> R-squared)
                logger.info("Calculating R-squared between model predictions (including Actual RMSF)...");
                # Get prediction columns based on ordered keys that actually exist
                pred_corr_cols = ([ACTUAL_COL] if ACTUAL_COL in df.columns else []) + [pred_cols_config[key] for key in model_keys_ordered if key in pred_cols_config]
                pred_corr_keys = (['Actual'] if ACTUAL_COL in df.columns else []) + [key for key in model_keys_ordered if key in pred_cols_config] # Keys corresponding to the cols

                if len(pred_corr_cols) > 1: # Check > 1 because Actual counts as one
                    try:
                        # Check if all columns are numeric BEFORE correlation
                        numeric_pred_corr_cols = [col for col in pred_corr_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                        if len(numeric_pred_corr_cols) > 1:
                             # Calculate Pearson correlation first
                             pred_pcc_matrix = df[numeric_pred_corr_cols].corr(method='pearson', min_periods=max(100, int(0.05*len(df))))
                             # MODIFIED: Square the PCC matrix to get R-squared
                             pred_r2_matrix = pred_pcc_matrix.pow(2)
                             # Update keys to match the numeric columns used
                             numeric_pred_corr_keys = ([k for k, c in zip(pred_corr_keys, pred_corr_cols) if c in numeric_pred_corr_cols])
                             pred_r2_matrix.columns = numeric_pred_corr_keys
                             pred_r2_matrix.index = numeric_pred_corr_keys
                             # MODIFIED: Update title and content
                             write_section(outfile, "5.3 PREDICTION R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", pred_r2_matrix)
                        else:
                            write_section(outfile, "5.3 PREDICTION R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", "Not enough numeric prediction columns (and Actual) available for calculation.")
                    except Exception as e: logger.error(f"Prediction R-squared calculation error: {e}"); write_section(outfile, "5.3 PREDICTION R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", f"Error: {e}")
                else: write_section(outfile, "5.3 PREDICTION R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", "Requires at least Actual RMSF and 1 model prediction.")


                # 5.4 Absolute Error Correlations (MODIFIED -> R-squared)
                logger.info("Calculating R-squared between model absolute errors (including Actual RMSF)...");
                # Get abs error columns based on ordered keys that actually exist and were calculated
                abs_error_corr_cols = ([ACTUAL_COL] if ACTUAL_COL in df.columns else []) + [abs_error_cols_available[key] for key in model_keys_ordered if key in abs_error_cols_available]
                abs_error_corr_keys = (['Actual'] if ACTUAL_COL in df.columns else []) + [key for key in model_keys_ordered if key in abs_error_cols_available] # Keys for labels

                if len(abs_error_corr_cols) > 1: # Check > 1 (Actual + 1 error column)
                    try:
                        # Ensure columns are numeric BEFORE correlation
                        numeric_abs_error_corr_cols = [col for col in abs_error_corr_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                        if len(numeric_abs_error_corr_cols) > 1:
                            # Calculate Pearson correlation first
                            abs_error_pcc_matrix = df[numeric_abs_error_corr_cols].corr(method='pearson', min_periods=max(100, int(0.05*len(df))))
                            # MODIFIED: Square the PCC matrix to get R-squared
                            abs_error_r2_matrix = abs_error_pcc_matrix.pow(2)
                            # Update keys to match numeric columns
                            numeric_abs_error_corr_keys = ([k for k, c in zip(abs_error_corr_keys, abs_error_corr_cols) if c in numeric_abs_error_corr_cols])
                            abs_error_r2_matrix.columns = numeric_abs_error_corr_keys
                            abs_error_r2_matrix.index = numeric_abs_error_corr_keys
                            # MODIFIED: Update title and note
                            write_section(outfile, "5.4 ABSOLUTE ERROR R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", abs_error_r2_matrix, note="Shows squared correlation (R^2) between errors. High value means models tend to make errors on the same samples. R^2 between errors and Actual RMSF is also shown.")
                        else:
                             write_section(outfile, "5.4 ABSOLUTE ERROR R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", "Not enough numeric error columns (and Actual) available for calculation.")
                    except Exception as e: logger.error(f"Absolute error R-squared calculation error: {e}"); write_section(outfile, "5.4 ABSOLUTE ERROR R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", f"Error: {e}")
                else: write_section(outfile, "5.4 ABSOLUTE ERROR R-SQUARED MATRIX (Coefficient of Determination, Incl. Actual)", "Requires at least Actual RMSF and 1 model with calculated absolute errors.")
            # --- Section 6: Uncertainty Analysis ---
            # [Keep Section 6 as it was in the previous version]
            write_section(outfile, "6. UNCERTAINTY ANALYSIS", "Comparing uncertainty estimates for models where available.")
            # Use the ordered list of models with uncertainty
            if models_with_uncertainty:
                logger.info(f"Analyzing uncertainty for models: {models_with_uncertainty}")
                # 6.1 Uncertainty Statistics
                unc_stats_list = []
                for model_key in models_with_uncertainty: # Iterate through ordered keys
                    unc_col = uncertainty_cols_config[model_key]
                    if unc_col in df.columns:
                        try: # Add try-except for robustness
                            stats = df[unc_col].describe().to_dict()
                            stats['Model'] = model_key
                            unc_stats_list.append(stats)
                        except Exception as e:
                            logger.warning(f"Could not get describe stats for uncertainty column {unc_col} of model {model_key}: {e}")
                if unc_stats_list:
                    unc_stats_df = pd.DataFrame(unc_stats_list).set_index('Model')[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                    # Reindex to ensure consistent order
                    unc_stats_df = unc_stats_df.reindex(models_with_uncertainty).dropna(how='all')
                    write_section(outfile, "6.1 UNCERTAINTY DISTRIBUTION STATISTICS", unc_stats_df)
                else: write_section(outfile, "6.1 UNCERTAINTY DISTRIBUTION STATISTICS", "No uncertainty columns found or no data.")

                # 6.2 Uncertainty vs. Absolute Error Correlation
                unc_err_corr_list = []
                for model_key in models_with_uncertainty: # Iterate through ordered keys
                    unc_col = uncertainty_cols_config[model_key]; abs_err_col = abs_error_cols_available.get(model_key)
                    if unc_col in df.columns and abs_err_col in df.columns:
                        df_pair = df[[unc_col, abs_err_col]].dropna()
                        if len(df_pair) >= MIN_POINTS_FOR_METRICS and df_pair[unc_col].nunique() > 1 and df_pair[abs_err_col].nunique() > 1:
                            try: corr, _ = pearsonr(df_pair[unc_col], df_pair[abs_err_col]); unc_err_corr_list.append({"Model": model_key, "Uncertainty-Error PCC": corr})
                            except Exception as e: logger.warning(f"Could not calculate uncertainty-error correlation for {model_key}: {e}"); unc_err_corr_list.append({"Model": model_key, "Uncertainty-Error PCC": np.nan})
                        else: unc_err_corr_list.append({"Model": model_key, "Uncertainty-Error PCC": np.nan})
                if unc_err_corr_list:
                    unc_err_corr_df = pd.DataFrame(unc_err_corr_list).set_index("Model")
                    # Reindex for order
                    unc_err_corr_df = unc_err_corr_df.reindex(models_with_uncertainty).dropna(how='all')
                    write_section(outfile, "6.2 OVERALL UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION", unc_err_corr_df, note="Positive correlation is desired.")

                # 6.3 Simple Calibration Check (% within 1 uncertainty interval)
                calibration_list = []
                for model_key in models_with_uncertainty: # Iterate through ordered keys
                    unc_col = uncertainty_cols_config[model_key]; err_col = error_cols_available.get(model_key) # Signed error
                    if unc_col in df.columns and err_col in df.columns:
                        df_pair = df[[unc_col, err_col]].dropna()
                        if not df_pair.empty:
                            errors = df_pair[err_col].values; stds = df_pair[unc_col].values; stds_safe = np.maximum(stds, 1e-9)
                            within_1std = np.mean(np.abs(errors) <= stds_safe) * 100 # Percentage
                            calibration_list.append({"Model": model_key, "% within 1 Uncertainty": within_1std})
                if calibration_list:
                    calib_df = pd.DataFrame(calibration_list).set_index("Model")
                    # Reindex for order
                    calib_df = calib_df.reindex(models_with_uncertainty).dropna(how='all')
                    write_section(outfile, "6.3 SIMPLE CALIBRATION CHECK", calib_df.round(2), note="Expected ~68.2% for well-calibrated Gaussian uncertainty.")

                # 6.4 Error Binned by Uncertainty Quantiles
                for model_key in models_with_uncertainty: # Iterate through ordered keys
                    unc_col = uncertainty_cols_config[model_key]; abs_err_col = abs_error_cols_available.get(model_key)
                    if unc_col in df.columns and abs_err_col in df.columns:
                        df_subset = df[[unc_col, abs_err_col]].dropna()
                        if len(df_subset) > QUANTILE_BINS * 2 and df_subset[unc_col].nunique() > QUANTILE_BINS:
                            try:
                                df_subset['unc_quantile'] = pd.qcut(df_subset[unc_col], q=QUANTILE_BINS, labels=False, duplicates='drop')
                                binned_error = df_subset.groupby('unc_quantile')[abs_err_col].agg(['mean', 'median', 'count'])
                                quantile_bounds = pd.qcut(df_subset[unc_col], q=QUANTILE_BINS, duplicates='drop').cat.categories
                                bin_labels = [f"{i.left:.3f}-{i.right:.3f}" for i in quantile_bounds]
                                # Handle case where fewer bins are created due to duplicates
                                num_bins_created = len(binned_error)
                                binned_error.index = pd.Index(bin_labels[:num_bins_created], name=f'{model_key}_Unc_Quantile')
                                write_section(outfile, f"6.4 MEAN ABSOLUTE ERROR BINNED BY {model_key} UNCERTAINTY QUANTILE", binned_error)
                            except Exception as e: logger.warning(f"Failed to bin error by uncertainty for {model_key}: {e}")
            else: write_section(outfile, "6. UNCERTAINTY ANALYSIS", "Skipped: No uncertainty columns found or configured.")


            # --- Detailed Breakdowns (Focused on PRIMARY_MODEL_KEY_FOR_RANKING, but show all models) ---
            # Check if the primary model key is valid and exists
            if PRIMARY_MODEL_KEY_FOR_RANKING not in model_keys_ordered:
                 primary_model_key = model_keys_ordered[0] if model_keys_ordered else None
                 if primary_model_key: logger.warning(f"Configured primary model '{PRIMARY_MODEL_KEY_FOR_RANKING}' not found in detected models, using '{primary_model_key}' for ranking instead.")
                 else: logger.warning("No models found, cannot set a primary model for ranking.")
            else:
                 primary_model_key = PRIMARY_MODEL_KEY_FOR_RANKING # Use the configured one (now ATTENTION-ESM)
                 logger.info(f"Using '{primary_model_key}' as the primary model for ranking and case studies.")

            if performance_possible and primary_model_key:
                logger.info(f"Starting detailed breakdowns, focusing on '{primary_model_key}' for ranking/selection.")
                primary_pred_col = pred_cols_config.get(primary_model_key) # Use .get for safety
                primary_abs_error_col = abs_error_cols_available.get(primary_model_key) # Use .get

                # 7. Domain Performance
                if DOMAIN_COL in df.columns:
                    logger.info("Analyzing performance per domain (all models)...");
                    # Calculate domain performance metrics if not already done or if needed again
                    # This check avoids recalculating if it was done globally before, but ensures it exists for this section
                    if 'domain_perf' not in locals() or domain_perf.empty:
                         domain_perf = df.groupby(DOMAIN_COL).apply(
                             lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                             include_groups=False
                         )

                    if not domain_perf.empty:
                        # Calculate mean performance across domains (show all models)
                        mean_domain_perf = domain_perf.mean().to_frame(name='Mean Across Domains')
                        write_section(outfile, f"7. MEAN DOMAIN PERFORMANCE METRICS (Avg. across {len(domain_perf)} domains)", mean_domain_perf)

                        # 7.1 Primary Model Domain Scoreboard (MODIFIED to use primary_model_key dynamically)
                        mae_metric_primary = f"{primary_model_key}_mae" # Metric for scoreboard based on the now primary model
                        if mae_metric_primary in domain_perf.columns:
                            all_best_domains_multi = defaultdict(list); all_worst_domains_multi = defaultdict(list)
                            # Define metrics to rank based on the primary model
                            metrics_to_rank_info = { f"{primary_model_key}_{m}": ("pcc" in m or "r2" in m or "expl" in m) for m in ['mae', 'rmse', 'pcc', 'r2', 'explained_variance'] if f"{primary_model_key}_{m}" in domain_perf.columns}

                            if not metrics_to_rank_info:
                                logger.warning(f"No performance metrics found for primary model '{primary_model_key}' in domain results. Cannot create scoreboard.")
                                write_section(outfile, f"7.1 {primary_model_key} DOMAIN PERFORMANCE SCOREBOARD", f"Skipped: No metrics found for {primary_model_key}.")
                            else:
                                for metric, higher_is_better_flag in metrics_to_rank_info.items():
                                    top_indices, bottom_indices = get_top_bottom_indices(domain_perf, metric, n=N_TOP_BOTTOM)
                                    all_best_domains_multi[metric].extend(top_indices); all_worst_domains_multi[metric].extend(bottom_indices)

                                flat_best_domains = [d for sublist in all_best_domains_multi.values() for d in sublist]; flat_worst_domains = [d for sublist in all_worst_domains_multi.values() for d in sublist]
                                best_domain_counts = Counter(flat_best_domains); worst_domain_counts = Counter(flat_worst_domains)
                                scoreboard = f"Domains frequently in Top {N_TOP_BOTTOM} (by {primary_model_key} metrics): "; scoreboard += ", ".join([f"{d}({c})" for d, c in best_domain_counts.most_common(N_TOP_BOTTOM + 5)])
                                scoreboard += f"\nDomains frequently in Bottom {N_TOP_BOTTOM}: "; scoreboard += ", ".join([f"{d}({c})" for d, c in worst_domain_counts.most_common(N_TOP_BOTTOM + 5)])
                                # MODIFIED: Title reflects the primary model used
                                write_section(outfile, f"7.1 {primary_model_key} DOMAIN PERFORMANCE SCOREBOARD", scoreboard)

                            # 7.2/7.3 Best/Worst Domain Analysis (Selected by primary MAE, show all models)
                            top_primary_mae_domains, bottom_primary_mae_domains = get_top_bottom_indices(domain_perf, mae_metric_primary, n=N_TOP_BOTTOM)
                            top_domain_chars = analyze_domain_characteristics(df, top_primary_mae_domains, pred_cols_config, uncertainty_cols_config)
                            bottom_domain_chars = analyze_domain_characteristics(df, bottom_primary_mae_domains, pred_cols_config, uncertainty_cols_config)
                            # MODIFIED: Titles reflect the primary model used for selection
                            if not top_domain_chars.empty: write_section(outfile, f"7.2 CHARACTERISTICS OF TOP {len(top_primary_mae_domains)} DOMAINS (Lowest {mae_metric_primary.upper()})", top_domain_chars)
                            if not bottom_domain_chars.empty: write_section(outfile, f"7.3 CHARACTERISTICS OF BOTTOM {len(bottom_primary_mae_domains)} DOMAINS (Highest {mae_metric_primary.upper()})", bottom_domain_chars)
                        else:
                            write_section(outfile, f"7.1-7.3 DOMAIN RANKING ({primary_model_key})", f"Skipped: Required metric '{mae_metric_primary}' not found in domain performance results.")
                    else: write_section(outfile, "7. DOMAIN PERFORMANCE", "No domain metrics calculated.")

                # 8. "Nailed It" Case Study (Based on primary_model_key)
                logger.info(f"Finding 'Nailed It' domain candidate (based on {primary_model_key})...")
                nailed_it_found = False
                # Ensure domain_perf exists and is not empty
                if 'domain_perf' in locals() and not domain_perf.empty and 'actual_stddev' in domain_perf.columns:
                    pcc_key, r2_key, mae_key = f'{primary_model_key}_pcc', f'{primary_model_key}_r2', f'{primary_model_key}_mae'
                    pred_std_key = f'{primary_model_key}_pred_stddev'
                    if all(k in domain_perf.columns for k in [pcc_key, r2_key, mae_key, 'actual_stddev', pred_std_key]):
                        # Use calculated overall standard deviations if available
                        actual_std_thresh = overall_actual_std * NAILED_IT_MIN_ACTUAL_STD_FACTOR if overall_actual_std > 0 else NAILED_IT_MIN_ACTUAL_STD_FACTOR # Fallback
                        # Use .get for primary model's std dev with a default
                        primary_overall_pred_std = overall_pred_std.get(primary_model_key, 0)
                        pred_std_thresh = primary_overall_pred_std * NAILED_IT_MIN_PRED_STD_FACTOR if primary_overall_pred_std > 0 else NAILED_IT_MIN_PRED_STD_FACTOR # Fallback

                        # Filter candidates based on criteria
                        candidates = domain_perf[
                            (domain_perf[pcc_key].astype(float) >= NAILED_IT_MIN_PCC) &
                            (domain_perf[r2_key].astype(float) >= NAILED_IT_MIN_R2) &
                            (domain_perf[mae_key].astype(float) <= NAILED_IT_MAX_MAE) &
                            (domain_perf['actual_stddev'].astype(float) >= actual_std_thresh) &
                            (domain_perf[pred_std_key].astype(float) >= pred_std_thresh)
                        ]
                        if not candidates.empty:
                            candidates_sorted = candidates.sort_values(by=pcc_key, ascending=False)
                            best_nailed_it = candidates_sorted.index[0]; logger.info(f"Selected '{best_nailed_it}' as 'Nailed It' candidate.")
                            best_nailed_it_chars = analyze_domain_characteristics(df, [best_nailed_it], pred_cols_config, uncertainty_cols_config) # Pass unc config
                            criteria_note = (f"Criteria ({primary_model_key}): PCC>={NAILED_IT_MIN_PCC}, R2>={NAILED_IT_MIN_R2}, MAE<={NAILED_IT_MAX_MAE}, ActualStd>={actual_std_thresh:.3f}, PredStd>={pred_std_thresh:.3f}")
                            write_section(outfile, f"8. 'NAILED IT' DOMAIN CASE STUDY ({primary_model_key} Selection): {best_nailed_it}", best_nailed_it_chars, note=criteria_note)
                            nailed_it_found = True
                    else: logger.warning(f"Missing required metrics ({pcc_key}, {r2_key}, {mae_key}, actual_stddev, {pred_std_key}) in domain_perf for 'Nailed It'.")
                if not nailed_it_found: write_section(outfile, f"8. 'NAILED IT' DOMAIN CASE STUDY ({primary_model_key} Selection)", "No domains found matching criteria.")


                # 9. Amino Acid Performance (Show all models)
                # [Keep Section 9 as it was - it shows all models, sorted by primary]
                if RESNAME_COL in df.columns:
                    logger.info("Analyzing performance per amino acid (all models)...");
                    aa_perf = df.groupby(RESNAME_COL).apply(
                        lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                        include_groups=False
                    )
                    if not aa_perf.empty:
                        cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in aa_perf.columns]
                        sort_metric_aa = f"{primary_model_key}_mae";
                        aa_perf_sorted = aa_perf.sort_values(sort_metric_aa, ascending=True, na_position='last') if sort_metric_aa in aa_perf.columns else aa_perf
                        write_section(outfile, f"9. AMINO ACID PERFORMANCE (Sorted by {sort_metric_aa.upper()})", aa_perf_sorted[cols_to_show])


                # 10. Positional Performance (Show all models)
                # [Keep Section 10 as it was - it shows all models]
                if NORM_RESID_COL in df.columns:
                    logger.info("Analyzing performance by normalized residue position (all models)...")
                    try:
                        # Ensure the binning column exists or create it
                        if 'pos_bin_labeled' not in df.columns or df['pos_bin_labeled'].isnull().any():
                            df['pos_bin_labeled'] = pd.cut(df[NORM_RESID_COL], bins=POS_BINS, labels=POS_BIN_LABELS, include_lowest=True, right=True)

                        if 'pos_bin_labeled' in df.columns and not df['pos_bin_labeled'].isnull().all():
                             pos_perf = df.groupby('pos_bin_labeled', observed=False).apply(
                                 lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                                 include_groups=False
                             )
                             cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in pos_perf.columns]
                             pos_perf.index.name = f"{NORM_RESID_COL}_bin";
                             write_section(outfile, f"10. PERFORMANCE BY {NORM_RESID_COL.upper()} BIN", pos_perf[cols_to_show])
                        else:
                            write_section(outfile, f"10. PERFORMANCE BY {NORM_RESID_COL.upper()} BIN", "Failed to create or use position bins.")
                    except Exception as e: write_section(outfile, f"10. PERFORMANCE BY {NORM_RESID_COL.upper()} BIN", f"Error: {e}")


                # 11. Core/Exterior Performance (Show all models)
                # [Keep Section 11 as it was - it shows all models]
                if CORE_EXT_COL in df.columns:
                    logger.info("Analyzing performance by core/exterior (all models)...");
                    core_ext_perf = df.groupby(CORE_EXT_COL).apply(
                        lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                        include_groups=False
                    )
                    cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in core_ext_perf.columns]
                    write_section(outfile, "11. PERFORMANCE BY CORE/EXTERIOR", core_ext_perf[cols_to_show])


                # 12. Secondary Structure Performance (Show all models)
                # [Keep Section 12 as it was - it shows all models]
                if 'ss_group' in df.columns:
                    logger.info("Analyzing performance by secondary structure group (all models)...");
                    ss_perf = df.groupby('ss_group').apply(
                        lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                        include_groups=False
                    )
                    cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in ss_perf.columns]
                    write_section(outfile, "12. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)", ss_perf[cols_to_show])


                # 13. Temperature Sensitivity/Resistance (Based on primary_model_key)
                temp_sensitivity_df = pd.DataFrame() # Initialize
                mae_col_primary = f"{primary_model_key}_mae" # Define here for use in this section
                if TEMP_COL in df.columns:
                     logger.info(f"Analyzing domain temperature sensitivity (based on {primary_model_key} MAE)...");
                     available_temps = df[TEMP_COL].dropna().unique();
                     min_temp = available_temps.min() if len(available_temps)>0 else np.nan;
                     max_temp = available_temps.max() if len(available_temps)>0 else np.nan
                     if not pd.isna(min_temp) and not pd.isna(max_temp) and not np.isclose(min_temp, max_temp):
                         domains_at_min = set(df[np.isclose(df[TEMP_COL], min_temp)][DOMAIN_COL].unique());
                         domains_at_max = set(df[np.isclose(df[TEMP_COL], max_temp)][DOMAIN_COL].unique())
                         spanning_domains = sorted(list(domains_at_min.intersection(domains_at_max)));
                         logger.info(f"Found {len(spanning_domains)} domains spanning T={min_temp}K to T={max_temp}K.")

                         if spanning_domains:
                             spanning_df = df[df[DOMAIN_COL].isin(spanning_domains)].copy()
                             # Calculate performance at min/max temps for ALL models
                             perf_at_min = spanning_df[np.isclose(spanning_df[TEMP_COL], min_temp)].groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                             perf_at_max = spanning_df[np.isclose(spanning_df[TEMP_COL], max_temp)].groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                             avg_rmsf_at_min = spanning_df[np.isclose(spanning_df[TEMP_COL], min_temp)].groupby(DOMAIN_COL)[ACTUAL_COL].mean()
                             avg_rmsf_at_max = spanning_df[np.isclose(spanning_df[TEMP_COL], max_temp)].groupby(DOMAIN_COL)[ACTUAL_COL].mean()

                             # Check if primary MAE exists in both min/max perf tables
                             if mae_col_primary in perf_at_min.columns and mae_col_primary in perf_at_max.columns:
                                 delta_mae_primary = (perf_at_max[mae_col_primary] - perf_at_min[mae_col_primary]).rename(f'delta_mae_{primary_model_key}')
                                 delta_actual_rmsf = (avg_rmsf_at_max - avg_rmsf_at_min).rename('delta_actual_rmsf')

                                 # Get overall primary MAE for these domains from the domain_perf table (calculated earlier)
                                 # Ensure domain_perf exists and primary MAE column is present
                                 if 'domain_perf' in locals() and not domain_perf.empty and mae_col_primary in domain_perf.columns:
                                     # Reindex domain_perf to match spanning_domains before accessing .loc
                                     try:
                                         domain_perf_aligned = domain_perf.reindex(spanning_domains)
                                         overall_mae_primary = domain_perf_aligned.loc[spanning_domains, mae_col_primary].rename(f'overall_{mae_col_primary}')
                                         # Combine the series, ensuring index alignment
                                         temp_sensitivity_df = pd.concat([delta_mae_primary, delta_actual_rmsf, overall_mae_primary], axis=1).dropna()
                                     except KeyError as e:
                                         logger.warning(f"KeyError accessing overall domain performance for sensitivity analysis (likely index mismatch): {e}")
                                         temp_sensitivity_df = pd.concat([delta_mae_primary, delta_actual_rmsf], axis=1).dropna() # Proceed without overall MAE
                                 else:
                                     logger.warning(f"Overall domain performance for {primary_model_key} not available for sensitivity analysis.")
                                     temp_sensitivity_df = pd.concat([delta_mae_primary, delta_actual_rmsf], axis=1).dropna() # Proceed without overall MAE


                                 if not temp_sensitivity_df.empty:
                                     corr_note = ""; change_corr = np.nan
                                     delta_mae_col_name = f'delta_mae_{primary_model_key}' # Use the dynamic name
                                     if delta_mae_col_name in temp_sensitivity_df.columns and 'delta_actual_rmsf' in temp_sensitivity_df.columns:
                                         corr_df_subset = temp_sensitivity_df[[delta_mae_col_name, 'delta_actual_rmsf']].dropna()
                                         if corr_df_subset.nunique().min() > 1 and len(corr_df_subset) >= MIN_POINTS_FOR_METRICS:
                                             try:
                                                 change_corr, _ = pearsonr(corr_df_subset[delta_mae_col_name], corr_df_subset['delta_actual_rmsf']);
                                                 corr_note = f"Corr(Delta {primary_model_key} MAE vs Delta Actual RMSF): {change_corr:.3f}" if not pd.isna(change_corr) else "Corr: NaN" ;
                                             except Exception as e: corr_note = f"Error calc sensitivity corr: {e}"
                                         else: corr_note = "Not enough variance/data for sensitivity correlation."
                                     else: corr_note = f"Required columns ('{delta_mae_col_name}', 'delta_actual_rmsf') not found for correlation."

                                     # Sort by the delta MAE column for the primary model
                                     temp_sensitivity_df_sorted = temp_sensitivity_df.sort_values(delta_mae_col_name)
                                     # MODIFIED: Titles reflect the primary model
                                     write_section(outfile, f"13.1 {primary_model_key} TEMPERATURE SENSITIVITY (T={min_temp:.0f}K to T={max_temp:.0f}K, {len(spanning_domains)} Domains)", temp_sensitivity_df_sorted, note=corr_note)

                                     # Identify and analyze resistant/sensitive domains (based on primary delta_mae)
                                     resistant_indices, sensitive_indices = get_top_bottom_indices(temp_sensitivity_df, delta_mae_col_name, n=N_TOP_BOTTOM)
                                     if resistant_indices:
                                         resistant_chars = analyze_domain_characteristics(df, resistant_indices, pred_cols_config, uncertainty_cols_config)
                                         # MODIFIED: Titles reflect the primary model
                                         write_section(outfile, f"13.2 CHARACTERISTICS OF {len(resistant_indices)} MOST ERROR-RESISTANT DOMAINS (Smallest {primary_model_key} MAE Change)", resistant_chars)
                                     if sensitive_indices:
                                         sensitive_chars = analyze_domain_characteristics(df, sensitive_indices, pred_cols_config, uncertainty_cols_config)
                                         # MODIFIED: Titles reflect the primary model
                                         write_section(outfile, f"13.3 CHARACTERISTICS OF {len(sensitive_indices)} MOST ERROR-SENSITIVE DOMAINS (Largest {primary_model_key} MAE Change)", sensitive_chars)
                                 else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "No spanning domains with valid MAE change.")
                             else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", f"Required metric '{mae_col_primary}' not available in min/max temp performance.")
                         else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "No spanning domains found.")
                     else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "Cannot analyze: Temp range insufficient or only one temp.")

                # 14. "Temperature Mastered" Case Study (Based on primary_model_key)
                logger.info(f"Finding 'Temperature Mastered' domain candidate (based on {primary_model_key})...")
                temp_mastered_found = False
                # Check if necessary columns exist in temp_sensitivity_df
                overall_mae_col_name = f'overall_{mae_col_primary}' # From section 13
                delta_mae_col_name = f'delta_mae_{primary_model_key}' # From section 13
                if not temp_sensitivity_df.empty and \
                   overall_mae_col_name in temp_sensitivity_df.columns and \
                   delta_mae_col_name in temp_sensitivity_df.columns and \
                   'delta_actual_rmsf' in temp_sensitivity_df.columns:
                    candidates_temp = temp_sensitivity_df[
                        (temp_sensitivity_df['delta_actual_rmsf'].abs() >= TEMP_MASTERED_MIN_ACTUAL_DELTA) &
                        (temp_sensitivity_df[overall_mae_col_name] <= TEMP_MASTERED_MAX_MAE) &
                        (temp_sensitivity_df[delta_mae_col_name].abs() <= TEMP_MASTERED_MAX_DELTA_MAE_ABS)
                    ]
                    if not candidates_temp.empty:
                         candidates_temp_sorted = candidates_temp.sort_values(by=['delta_actual_rmsf', delta_mae_col_name], ascending=[False, True]) # Sort by largest actual change, then smallest error change
                         best_temp_mastered = candidates_temp_sorted.index[0]; logger.info(f"Selected '{best_temp_mastered}' as 'Temperature Mastered' candidate.")
                         best_temp_mastered_chars = analyze_domain_characteristics(df, [best_temp_mastered], pred_cols_config, uncertainty_cols_config) # Pass unc config
                         criteria_note = (f"Criteria ({primary_model_key}): |ActualDelta| >= {TEMP_MASTERED_MIN_ACTUAL_DELTA}, OverallMAE <= {TEMP_MASTERED_MAX_MAE}, |DeltaMAE| <= {TEMP_MASTERED_MAX_DELTA_MAE_ABS}")
                         # MODIFIED: Title reflects the primary model
                         write_section(outfile, f"14. 'TEMPERATURE MASTERED' DOMAIN CASE STUDY ({primary_model_key} Selection): {best_temp_mastered}", best_temp_mastered_chars, note=criteria_note)
                         temp_mastered_found = True
                elif temp_sensitivity_df.empty:
                    logger.info("Skipping 'Temperature Mastered' - temp_sensitivity_df is empty.")
                else:
                     logger.info(f"Skipping 'Temperature Mastered' - required columns missing in temp_sensitivity_df (needs: {overall_mae_col_name}, {delta_mae_col_name}, delta_actual_rmsf).")

                if not temp_mastered_found: write_section(outfile, f"14. 'TEMPERATURE MASTERED' DOMAIN CASE STUDY ({primary_model_key} Selection)", "No domains found matching criteria or prerequisite data missing.")


                # 15. Flexibility Magnitude vs. Performance (Show all models)
                # [Keep Section 15 as it was - it shows all models]
                logger.info("Analyzing performance by actual flexibility magnitude (all models)...");
                try:
                    if ACTUAL_COL in df.columns and df[ACTUAL_COL].nunique() > QUANTILE_BINS:
                        # Ensure quantile label column exists or create it
                        if 'rmsf_quantile_label' not in df.columns or df['rmsf_quantile_label'].isnull().any():
                             df['rmsf_quantile_label'] = pd.qcut(df[ACTUAL_COL], q=QUANTILE_BINS, labels=False, duplicates='drop')

                        if 'rmsf_quantile_label' in df.columns and not df['rmsf_quantile_label'].isnull().all():
                            flex_perf = df.groupby('rmsf_quantile_label').apply(
                                lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                                include_groups=False
                            )
                            cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in flex_perf.columns]
                            try:
                                quantile_bounds = pd.qcut(df[ACTUAL_COL], q=QUANTILE_BINS, duplicates='drop').cat.categories
                                bin_labels = [f"{i.left:.2f}-{i.right:.2f}" for i in quantile_bounds];
                                num_bins_created = len(flex_perf)
                                flex_perf.index = pd.Index(bin_labels[:num_bins_created], name='Actual_RMSF_Quantile')
                            except Exception: flex_perf.index.name = 'Actual_RMSF_Quantile_Idx' # Fallback index name
                            write_section(outfile, f"15. PERFORMANCE BY ACTUAL RMSF QUANTILE ({len(flex_perf)} bins)", flex_perf[cols_to_show])
                        else:
                            write_section(outfile, f"15. PERFORMANCE BY ACTUAL RMSF QUANTILE", "Failed to create or use RMSF quantile bins.")
                    else: write_section(outfile, f"15. PERFORMANCE BY ACTUAL RMSF QUANTILE", "Not enough unique actual RMSF values for quantile binning.")
                except Exception as e: write_section(outfile, "15. PERFORMANCE BY ACTUAL RMSF QUANTILE", f"Error: {e}")


                # 16. Relative Accessibility vs. Performance (Show all models)
                # [Keep Section 16 as it was - it shows all models]
                if REL_ACC_COL in df.columns:
                     logger.info("Analyzing performance by relative accessibility (all models)...");
                     try:
                        acc_col_no_na = df[REL_ACC_COL].fillna(df[REL_ACC_COL].median()) # Impute missing with median for binning
                        if acc_col_no_na.nunique() > QUANTILE_BINS:
                            # Ensure quantile label column exists or create it
                            if 'access_quantile_label' not in df.columns or df['access_quantile_label'].isnull().any():
                                df['access_quantile_label'] = pd.qcut(acc_col_no_na, q=QUANTILE_BINS, labels=False, duplicates='drop')

                            if 'access_quantile_label' in df.columns and not df['access_quantile_label'].isnull().all():
                                access_perf = df.groupby('access_quantile_label').apply(
                                    lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                                    include_groups=False
                                )
                                cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in access_perf.columns]
                                try:
                                    quantile_bounds = pd.qcut(acc_col_no_na, q=QUANTILE_BINS, duplicates='drop').cat.categories
                                    bin_labels = [f"{i.left:.2f}-{i.right:.2f}" for i in quantile_bounds];
                                    num_bins_created = len(access_perf)
                                    access_perf.index = pd.Index(bin_labels[:num_bins_created], name='RelAccess_Quantile')
                                except Exception: access_perf.index.name = 'RelAccess_Quantile_Idx' # Fallback
                                write_section(outfile, f"16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE ({len(access_perf)} bins)", access_perf[cols_to_show])
                            else:
                                write_section(outfile, f"16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", "Failed to create or use accessibility bins.")
                        else: write_section(outfile, f"16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", "Not enough unique relative accessibility values for quantile binning.")
                     except Exception as e: write_section(outfile, "16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", f"Error: {e}")


                # 17. Domain Size vs. Performance (Show all models)
                # [Keep Section 17 as it was - it shows all models]
                if SIZE_COL in df.columns and DOMAIN_COL in df.columns:
                    logger.info("Analyzing performance by domain size (all models)...");
                    try:
                        # Create a mapping from domain_id to its size (taking the first value per domain)
                        domain_size_map = df.drop_duplicates(subset=[DOMAIN_COL])[[DOMAIN_COL, SIZE_COL]].set_index(DOMAIN_COL)[SIZE_COL].dropna()
                        if len(domain_size_map) >= QUANTILE_BINS * 2: # Need enough domains for binning
                             # Bin the domain sizes themselves
                             size_quantile_labels, size_bins = pd.qcut(domain_size_map, q=QUANTILE_BINS, labels=False, duplicates='drop', retbins=True)
                             # Map these bin labels back to the original dataframe via domain_id
                             df['size_quantile_group'] = df[DOMAIN_COL].map(size_quantile_labels)

                             if not df['size_quantile_group'].isna().all(): # Check if mapping worked
                                 size_perf = df.dropna(subset=['size_quantile_group']).groupby('size_quantile_group').apply(
                                     lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config),
                                     include_groups=False
                                 )
                                 cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in size_perf.columns]
                                 # Create labels from the bins returned by qcut
                                 bin_labels = [f"{int(size_bins[i])}-{int(size_bins[i+1])}" for i in range(len(size_bins)-1)];
                                 num_bins_created = len(size_perf)
                                 # Map the integer index (0, 1, 2...) to the bin label strings
                                 label_map = {i: bin_labels[i] for i in range(num_bins_created)}
                                 size_perf.index = size_perf.index.map(label_map); size_perf.index.name = 'DomainSize_Quantile'
                                 write_section(outfile, f"17. PERFORMANCE BY DOMAIN SIZE QUANTILE ({len(size_perf)} bins)", size_perf[cols_to_show])
                             else: write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", "Mapping size bins failed or resulted in all NaNs.")
                        else: write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", f"Not enough unique domains ({len(domain_size_map)}) with size info for quantile binning.")
                    except Exception as e: logger.error(f"Error during domain size analysis: {e}"); write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", f"Error: {e}")


                # 18. Model Disagreement Analysis
                # [Keep Section 18 as it was - it uses all models]
                logger.info("Analyzing model disagreement...");
                pred_cols_for_std = list(pred_cols_config.values()) # Get all prediction columns
                if len(pred_cols_for_std) > 1:
                    try:
                        # Calculate std dev across ALL model predictions row-wise
                        pred_df_subset = df[pred_cols_for_std].copy() # Operate on a copy
                        # Use skipna=True for robustness if some models have NaNs for a residue
                        df['prediction_stddev'] = pred_df_subset.std(axis=1, skipna=True)
                        write_section(outfile, "18. MODEL PREDICTION STANDARD DEVIATION STATS", df['prediction_stddev'].describe())

                        # Correlate disagreement with the primary model's absolute error
                        if primary_abs_error_col and primary_abs_error_col in df.columns:
                             disagreement_error_corr_df = df[[primary_abs_error_col, 'prediction_stddev']].dropna()
                             if disagreement_error_corr_df.shape[0] > MIN_POINTS_FOR_METRICS and disagreement_error_corr_df.nunique().min() > 1:
                                  disagree_corr, _ = pearsonr(disagreement_error_corr_df[primary_abs_error_col], disagreement_error_corr_df['prediction_stddev'])
                                  write_section(outfile, "18.1 MODEL DISAGREEMENT VS. ERROR (Primary Model)", f"Correlation between prediction_stddev and {primary_abs_error_col}: {disagree_corr:.3f} (N={len(disagreement_error_corr_df)})")
                             else:
                                 write_section(outfile, "18.1 MODEL DISAGREEMENT VS. ERROR (Primary Model)", "Not enough data or variance to calculate correlation.")
                        else:
                             write_section(outfile, "18.1 MODEL DISAGREEMENT VS. ERROR (Primary Model)", f"Skipped: Primary model absolute error column '{primary_abs_error_col}' not found.")
                    except Exception as e: write_section(outfile, "18. MODEL DISAGREEMENT ANALYSIS", f"Error: {e}")
                else: write_section(outfile, "18. MODEL DISAGREEMENT ANALYSIS", "Skipped: Requires >= 2 models.")


                # 19. Outlier Residue Analysis (Based on primary_model_key error, show all models) (MODIFIED)
                logger.info(f"Analyzing outlier residues (based on {primary_model_key} error)...");
                if primary_abs_error_col and primary_abs_error_col in df.columns:
                     # Define columns to show for outliers, including ALL predictions and uncertainties
                     outlier_cols_to_show = [DOMAIN_COL, RESID_COL, RESNAME_COL, TEMP_COL, ACTUAL_COL] + \
                                            list(pred_cols_config.values()) + \
                                            [primary_abs_error_col] + \
                                            list(uncertainty_cols_config.values()) + \
                                            [CORE_EXT_COL, 'ss_group', 'prediction_stddev'] # Add disagreement std dev
                     # Filter list to only include columns that actually exist in the dataframe
                     outlier_cols_to_show = [c for c in outlier_cols_to_show if c in df.columns]

                     # MODIFIED: Change number of outliers
                     n_outliers = 5 # Show top 5 outliers

                     # Get the rows with the largest absolute error for the primary model
                     outliers_df = df.nlargest(n_outliers, primary_abs_error_col)[outlier_cols_to_show]

                     # MODIFIED: Update title dynamically
                     write_section(outfile, f"19. TOP {n_outliers} OUTLIER RESIDUES (Highest {primary_abs_error_col.upper()})", outliers_df)

                     # Summarize characteristics of these outlier residues
                     outlier_summary = {};
                     if not outliers_df.empty:
                          if RESNAME_COL in outliers_df: outlier_summary['AA Types (%)'] = outliers_df[RESNAME_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if 'ss_group' in outliers_df: outlier_summary['SS Group (%)'] = outliers_df['ss_group'].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if CORE_EXT_COL in outliers_df: outlier_summary['Core/Ext (%)'] = outliers_df[CORE_EXT_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if TEMP_COL in outliers_df: outlier_summary['Temp Dist (%)'] = outliers_df[TEMP_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if 'prediction_stddev' in outliers_df: outlier_summary['Avg Model Disagreement (StdDev)'] = outliers_df['prediction_stddev'].mean()
                          # Add uncertainty summary if available for the primary model
                          primary_unc_col = uncertainty_cols_config.get(primary_model_key)
                          if primary_unc_col and primary_unc_col in outliers_df.columns:
                              outlier_summary[f'Avg {primary_model_key} Uncertainty'] = outliers_df[primary_unc_col].mean()

                          write_section(outfile, "19.1 SUMMARY CHARACTERISTICS OF OUTLIER RESIDUES", outlier_summary)
                else: write_section(outfile, "19. OUTLIER RESIDUE ANALYSIS", f"Skipped: Primary model absolute error column '{primary_abs_error_col}' not found.")
            else:
                write_section(outfile, "DETAILED BREAKDOWNS (SECTIONS 7-19)", "Skipped: Performance analysis not possible or no primary model identified.")

    except FileNotFoundError:
        logger.error(f"Output directory '{OUTPUT_DIR}' not found or cannot be created.")
    except Exception as e:
        logger.error(f"An critical error occurred during the main analysis: {e}", exc_info=True) # Log traceback

    logger.info(f"--- Analysis Report Generation Finished ({OUTPUT_TXT_PATH}) ---")

if __name__ == "__main__":
    main()