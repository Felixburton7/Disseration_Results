# Example filename: scripts/general_analysis3.py
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
# Input dataset path (contains base data + RF, NN, LGBM, ESM, Voxel predictions/uncertainty)
# Assumes script is run from ./home/s_felix/drDataScience/scripts OR paths are absolute
# INPUT_DIR = "/home/s_felix/packages/DeepFlex/data" # Or where analysis2 is saved
INPUT_DIR = "/home/s_felix/drDataScience/data" # Using the output dir as requested base location in prev steps
INPUT_BASE_FILENAME = "analysis3_holdout_dataset" # Input file name without extension

# Directory to save the output ANALYSIS REPORT ONLY
OUTPUT_DIR = "/home/s_felix/drDataScience/analysis_outputs" # Where the report will be saved

# Output analysis text file name
OUTPUT_TXT_FILENAME = "general_analysis4_report.txt"

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
PREDICTION_PATTERN = re.compile(r"^(deepflex_(?:NN|RF|LGBM)_rmsf|esm_rmsf|voxel_rmsf)$", re.IGNORECASE)
UNCERTAINTY_PATTERN = re.compile(r"^(deepflex_(?:NN|RF)_rmsf_uncertainty)$", re.IGNORECASE) # Add LGBM if it has uncertainty

# --- Analysis Constants ---
N_TOP_BOTTOM = 10; POS_BINS = 5
POS_BIN_LABELS = ['N-term (0-0.2)', 'Mid-N (0.2-0.4)', 'Middle (0.4-0.6)', 'Mid-C (0.6-0.8)', 'C-term (0.8-1.0)']
QUANTILE_BINS = 10; MIN_POINTS_FOR_METRICS = 10
PRIMARY_MODEL_KEY_FOR_RANKING = 'RF' # Model to use for sorting/selecting best/worst/outliers

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
def write_section(outfile, title, content, note=None):
    """Writes a formatted section to the output text file."""
    outfile.write(f"\n{'=' * 80}\n")
    outfile.write(f"## {title.upper()} ##\n")
    if note: outfile.write(f"   ({note})\n")
    outfile.write(f"{'-' * 80}\n")
    if isinstance(content, (pd.DataFrame, pd.Series)):
        with pd.option_context('display.max_rows', 200, 'display.max_columns', 50, 'display.width', 250, 'display.float_format', '{:,.4f}'.format):
             outfile.write(content.to_string())
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

    # Initialize results with NaN
    for model_key in pred_cols_config:
        for metric in metric_keys: results[f"{model_key}_{metric}"] = np.nan
        results[f"{model_key}_pred_stddev"] = np.nan
        if uncertainty_cols_config and model_key in uncertainty_cols_config:
            for unc_metric in uncertainty_metrics: results[f"{model_key}_{unc_metric}"] = np.nan
    results['actual_stddev'] = np.nan

    if group_count < MIN_POINTS_FOR_METRICS: return pd.Series(results)
    if actual_col not in group: return pd.Series(results)
    actual_vals = group[actual_col].dropna()
    # Require minimum points *after* dropping NaNs in actual
    if actual_vals.nunique() <= 1 or len(actual_vals) < MIN_POINTS_FOR_METRICS: return pd.Series(results)
    results['actual_stddev'] = actual_vals.std()

    for model_key, pred_col in pred_cols_config.items():
        if pred_col not in group: continue
        df_pair = group[[actual_col, pred_col]].dropna()
        # Require min points after dropping NaNs in pred too
        if len(df_pair) < MIN_POINTS_FOR_METRICS: continue

        y_true, y_pred = df_pair[actual_col].values, df_pair[pred_col].values
        if np.var(y_pred) > 1e-9: results[f'{model_key}_pred_stddev'] = np.std(y_pred)
        else: results[f'{model_key}_pred_stddev'] = 0.0

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

        # Uncertainty Metrics
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
    subset_df = df[df[DOMAIN_COL].isin(domain_list)].copy()
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

    domain_perf = subset_df.groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
    all_data_to_concat = [s for s in features.values() if isinstance(s, (pd.Series, pd.DataFrame))]
    if not domain_perf.empty: all_data_to_concat.append(domain_perf)
    if not all_data_to_concat: return pd.DataFrame()
    try: analysis = pd.concat(all_data_to_concat, axis=1, join='outer')
    except Exception as e: logger.error(f"Concat failed in analyze_domain_characteristics: {e}"); return pd.DataFrame()
    return analysis.round(3)


# --- Main Analysis Function ---
def main():
    logger.info("--- Starting Comprehensive General Analysis Report (Analysis 3) ---")

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
            if col.lower().startswith('deepflex_'): model_key = col.split('_')[1].upper()
            elif col.lower() == 'esm_rmsf': model_key = 'ESM'
            elif col.lower() == 'voxel_rmsf': model_key = 'Voxel'
            else: model_key = col
            pred_cols_config[model_key] = col
            if model_key not in model_keys_found: model_keys_found.append(model_key)
            logger.info(f"  Found Prediction: '{model_key}' -> {col}")
        elif unc_match:
            model_key = col.split('_')[1].upper()
            uncertainty_cols_config[model_key] = col
            if model_key not in model_keys_found: model_keys_found.append(model_key)
            logger.info(f"  Found Uncertainty: '{model_key}' -> {col}")
    model_keys_ordered = sorted([mk for mk in model_keys_found if mk in pred_cols_config])
    models_with_uncertainty = sorted(uncertainty_cols_config.keys())
    logger.info(f"Detected models for analysis: {model_keys_ordered}")
    logger.info(f"Models with uncertainty: {models_with_uncertainty}")
    performance_possible = ACTUAL_COL in df.columns and bool(pred_cols_config)

    # --- Calculate Error Columns (if needed) ---
    error_cols_available = {}; abs_error_cols_available = {}
    if performance_possible:
        for model_key, pred_col in pred_cols_config.items():
             error_col = f"{model_key}_error"; abs_error_col = f"{model_key}_abs_error"
             if error_col not in df.columns:
                 logger.debug(f"Calculating error columns for {model_key}...")
                 # Important: Use .loc to avoid SettingWithCopyWarning if df is a slice
                 df.loc[:, error_col] = df[pred_col] - df[ACTUAL_COL]
                 df.loc[:, abs_error_col] = df[error_col].abs()
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
            basic_info = { "Source Data File": input_file_used, "Total Rows": df.shape[0], "Total Columns": df.shape[1], "Unique Domains": num_unique_domains, "Memory Usage": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB", "Models Detected": model_keys_ordered, "Uncertainty Models": models_with_uncertainty, "Prediction Cols": list(pred_cols_config.values()) }
            write_section(outfile, "1. BASIC INFORMATION", basic_info)

            # --- Section 2: Missing Values ---
            missing_values = df.isnull().sum(); missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            if not missing_values.empty: missing_summary = pd.DataFrame({'count': missing_values, 'percentage': (missing_values / df.shape[0]) * 100}); write_section(outfile, "2. MISSING VALUE SUMMARY", missing_summary.round(2))
            else: write_section(outfile, "2. MISSING VALUE SUMMARY", "No missing values found.")

            # --- Section 3: Overall Descriptive Statistics ---
            stats_cols = ([ACTUAL_COL] if ACTUAL_COL in df.columns else []) + list(pred_cols_config.values()) + list(abs_error_cols_available.values()) + list(uncertainty_cols_config.values()) + [NORM_RESID_COL, REL_ACC_COL, TEMP_COL, SIZE_COL]
            stats_cols_final = sorted([col for col in stats_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])])
            if stats_cols_final: overall_stats = df[stats_cols_final].describe(); write_section(outfile, "3. OVERALL DESCRIPTIVE STATISTICS", overall_stats)
            else: logger.warning("Could not calculate overall descriptive stats."); write_section(outfile, "3. OVERALL DESCRIPTIVE STATISTICS", "Error calculating stats.")
            # Store overall std dev for later use if calculated
            overall_actual_std = overall_stats.loc['std', ACTUAL_COL] if 'overall_stats' in locals() and ACTUAL_COL in overall_stats.columns else 0
            overall_pred_std = {key: overall_stats.loc['std', pred_col] if 'overall_stats' in locals() and key in pred_cols_config and (pred_col := pred_cols_config[key]) in overall_stats.columns else 0 for key in pred_cols_config}


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
            write_section(outfile, "5. COMPREHENSIVE MODEL COMPARISON", "Comparing performance across detected models.")
            if performance_possible:
                # 5.1 Overall Performance Metrics Table
                logger.info("Calculating overall performance metrics for all models...")
                overall_metrics_list = []
                for model_key in model_keys_ordered:
                    if model_key not in pred_cols_config: continue
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

                if overall_metrics_list: overall_metrics_df = pd.DataFrame(overall_metrics_list).set_index("Model"); write_section(outfile, "5.1 OVERALL PERFORMANCE METRICS", overall_metrics_df)
                else: write_section(outfile, "5.1 OVERALL PERFORMANCE METRICS", "Could not calculate overall metrics for any model.")

                # 5.2 Performance by Temperature
                if TEMP_COL in df.columns:
                     logger.info("Calculating performance by temperature for all models..."); temp_perf = df.groupby(TEMP_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                     if not temp_perf.empty:
                         cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc', 'r2'] if f"{key}_{m}" in temp_perf.columns]
                         write_section(outfile, "5.2 PERFORMANCE METRICS BY TEMPERATURE", temp_perf[cols_to_show])
                         temp_report = "Model Comparison Highlights by Temperature:\n";
                         for temp, row in temp_perf.iterrows():
                             temp_report += f"  T={temp:.0f}K (N={row.get('count', 0)}): "; maes = {key: row.get(f"{key}_mae", np.inf) for key in model_keys_ordered}; pccs = {key: row.get(f"{key}_pcc", -np.inf) for key in model_keys_ordered}; best_mae_model = min(maes, key=maes.get); best_pcc_model = max(pccs, key=pccs.get)
                             temp_report += f"Best MAE={best_mae_model}({maes[best_mae_model]:.3f}), Best PCC={best_pcc_model}({pccs[best_pcc_model]:.3f})\n"
                         write_section(outfile, "5.2.1 TEMP PERFORMANCE SUMMARY", temp_report)
                     else: write_section(outfile, "5.2 PERFORMANCE METRICS BY TEMPERATURE", "No results after grouping by temperature.")

                # 5.3 Prediction Correlations
                logger.info("Calculating correlations between model predictions..."); pred_corr_cols = [pred_cols_config[key] for key in model_keys_ordered if key in pred_cols_config]
                if len(pred_corr_cols) > 1:
                     try: pred_corr_matrix = df[pred_corr_cols].corr(method='pearson', min_periods=max(100, int(0.05*len(df)))); pred_corr_matrix.columns = model_keys_ordered; pred_corr_matrix.index = model_keys_ordered; write_section(outfile, "5.3 PREDICTION CORRELATION MATRIX (PEARSON)", pred_corr_matrix)
                     except Exception as e: logger.error(f"Prediction correlation error: {e}"); write_section(outfile, "5.3 PREDICTION CORRELATION MATRIX", "Error.")
                else: write_section(outfile, "5.3 PREDICTION CORRELATION MATRIX", "Requires at least 2 models.")

                # 5.4 Absolute Error Correlations
                logger.info("Calculating correlations between model absolute errors..."); abs_error_corr_cols = [abs_error_cols_available[key] for key in model_keys_ordered if key in abs_error_cols_available]
                if len(abs_error_corr_cols) > 1:
                     try: abs_error_corr_matrix = df[abs_error_corr_cols].corr(method='pearson', min_periods=max(100, int(0.05*len(df)))); abs_error_corr_matrix.columns = model_keys_ordered; abs_error_corr_matrix.index = model_keys_ordered; write_section(outfile, "5.4 ABSOLUTE ERROR CORRELATION MATRIX (PEARSON)", abs_error_corr_matrix, note="High correlation means models tend to make errors on the same samples.")
                     except Exception as e: logger.error(f"Absolute error correlation error: {e}"); write_section(outfile, "5.4 ABSOLUTE ERROR CORRELATION MATRIX", "Error.")
                else: write_section(outfile, "5.4 ABSOLUTE ERROR CORRELATION MATRIX", "Requires at least 2 models with calculated absolute errors.")
            else: write_section(outfile, "5. COMPREHENSIVE MODEL COMPARISON", "Skipped: Performance analysis not possible.")

            # --- Section 6: Uncertainty Analysis ---
            write_section(outfile, "6. UNCERTAINTY ANALYSIS (RF & NN)", "Comparing uncertainty estimates.")
            if models_with_uncertainty:
                logger.info(f"Analyzing uncertainty for models: {models_with_uncertainty}")
                # 6.1 Uncertainty Statistics
                unc_stats_list = []
                for model_key in models_with_uncertainty:
                    unc_col = uncertainty_cols_config[model_key]
                    if unc_col in df.columns: stats = df[unc_col].describe().to_dict(); stats['Model'] = model_key; unc_stats_list.append(stats)
                if unc_stats_list: unc_stats_df = pd.DataFrame(unc_stats_list).set_index('Model')[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]; write_section(outfile, "6.1 UNCERTAINTY DISTRIBUTION STATISTICS", unc_stats_df)
                else: write_section(outfile, "6.1 UNCERTAINTY DISTRIBUTION STATISTICS", "No uncertainty columns found or no data.")

                # 6.2 Uncertainty vs. Absolute Error Correlation
                unc_err_corr_list = []
                for model_key in models_with_uncertainty:
                    unc_col = uncertainty_cols_config[model_key]; abs_err_col = abs_error_cols_available.get(model_key)
                    if unc_col in df.columns and abs_err_col in df.columns:
                        df_pair = df[[unc_col, abs_err_col]].dropna()
                        if len(df_pair) >= MIN_POINTS_FOR_METRICS and df_pair[unc_col].nunique() > 1 and df_pair[abs_err_col].nunique() > 1:
                            try: corr, _ = pearsonr(df_pair[unc_col], df_pair[abs_err_col]); unc_err_corr_list.append({"Model": model_key, "Uncertainty-Error PCC": corr})
                            except Exception as e: logger.warning(f"Could not calculate uncertainty-error correlation for {model_key}: {e}")
                        else: unc_err_corr_list.append({"Model": model_key, "Uncertainty-Error PCC": np.nan})
                if unc_err_corr_list: unc_err_corr_df = pd.DataFrame(unc_err_corr_list).set_index("Model"); write_section(outfile, "6.2 OVERALL UNCERTAINTY VS. ABSOLUTE ERROR CORRELATION", unc_err_corr_df, note="Positive correlation is desired.")

                # 6.3 Simple Calibration Check (% within 1 uncertainty interval)
                calibration_list = []
                for model_key in models_with_uncertainty:
                    unc_col = uncertainty_cols_config[model_key]; err_col = error_cols_available.get(model_key) # Signed error
                    if unc_col in df.columns and err_col in df.columns:
                        df_pair = df[[unc_col, err_col]].dropna()
                        if not df_pair.empty:
                            errors = df_pair[err_col].values; stds = df_pair[unc_col].values; stds_safe = np.maximum(stds, 1e-9)
                            within_1std = np.mean(np.abs(errors) <= stds_safe) * 100 # Percentage
                            calibration_list.append({"Model": model_key, "% within 1 Uncertainty": within_1std})
                if calibration_list: calib_df = pd.DataFrame(calibration_list).set_index("Model"); write_section(outfile, "6.3 SIMPLE CALIBRATION CHECK", calib_df.round(2), note="Expected ~68.2% for well-calibrated Gaussian uncertainty.")

                # 6.4 Error Binned by Uncertainty Quantiles
                for model_key in models_with_uncertainty:
                    unc_col = uncertainty_cols_config[model_key]; abs_err_col = abs_error_cols_available.get(model_key)
                    if unc_col in df.columns and abs_err_col in df.columns:
                        df_subset = df[[unc_col, abs_err_col]].dropna()
                        if len(df_subset) > QUANTILE_BINS * 2 and df_subset[unc_col].nunique() > QUANTILE_BINS:
                            try:
                                df_subset['unc_quantile'] = pd.qcut(df_subset[unc_col], q=QUANTILE_BINS, labels=False, duplicates='drop')
                                binned_error = df_subset.groupby('unc_quantile')[abs_err_col].agg(['mean', 'median', 'count'])
                                quantile_bounds = pd.qcut(df_subset[unc_col], q=QUANTILE_BINS, duplicates='drop').cat.categories; bin_labels = [f"{i.left:.3f}-{i.right:.3f}" for i in quantile_bounds]
                                binned_error.index = pd.Index(bin_labels[:len(binned_error)], name=f'{model_key}_Unc_Quantile')
                                write_section(outfile, f"6.4 MEAN ABSOLUTE ERROR BINNED BY {model_key} UNCERTAINTY QUANTILE", binned_error)
                            except Exception as e: logger.warning(f"Failed to bin error by uncertainty for {model_key}: {e}")
            else: write_section(outfile, "6. UNCERTAINTY ANALYSIS", "Skipped: No uncertainty columns found.")

            # --- Detailed Breakdowns (Focused on RF, but show all models) ---
            primary_model_key = PRIMARY_MODEL_KEY_FOR_RANKING if PRIMARY_MODEL_KEY_FOR_RANKING in model_keys_ordered else (model_keys_ordered[0] if model_keys_ordered else None)
            if performance_possible and primary_model_key:
                logger.info(f"Starting detailed breakdowns, focusing on '{primary_model_key}' for ranking/selection.")
                primary_pred_col = pred_cols_config[primary_model_key]
                primary_abs_error_col = abs_error_cols_available.get(primary_model_key)

                # 7. Domain Performance
                if DOMAIN_COL in df.columns:
                    logger.info("Analyzing performance per domain (all models)..."); domain_perf = df.groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                    if not domain_perf.empty:
                        write_section(outfile, f"7. OVERALL DOMAIN PERFORMANCE METRICS (Avg. across {len(domain_perf)} domains)", domain_perf.mean())
                        # 7.1 RF Domain Scoreboard
                        mae_metric_rf = f"{primary_model_key}_mae" # Use RF MAE for scoreboard
                        all_best_domains_multi = defaultdict(list); all_worst_domains_multi = defaultdict(list)
                        metrics_to_rank_info = { f"{primary_model_key}_{m}": ("pcc" in m or "r2" in m or "expl" in m) for m in ['mae', 'rmse', 'pcc', 'r2', 'explained_variance'] if f"{primary_model_key}_{m}" in domain_perf.columns}
                        for metric, higher_is_better_flag in metrics_to_rank_info.items(): top_indices, bottom_indices = get_top_bottom_indices(domain_perf, metric, n=N_TOP_BOTTOM); all_best_domains_multi[metric].extend(top_indices); all_worst_domains_multi[metric].extend(bottom_indices)
                        flat_best_domains = [d for sublist in all_best_domains_multi.values() for d in sublist]; flat_worst_domains = [d for sublist in all_worst_domains_multi.values() for d in sublist]
                        best_domain_counts = Counter(flat_best_domains); worst_domain_counts = Counter(flat_worst_domains)
                        scoreboard = f"Domains frequently in Top {N_TOP_BOTTOM} (by {primary_model_key} metrics): "; scoreboard += ", ".join([f"{d}({c})" for d, c in best_domain_counts.most_common(N_TOP_BOTTOM + 5)])
                        scoreboard += f"\nDomains frequently in Bottom {N_TOP_BOTTOM}: "; scoreboard += ", ".join([f"{d}({c})" for d, c in worst_domain_counts.most_common(N_TOP_BOTTOM + 5)])
                        write_section(outfile, f"7.1 {primary_model_key} DOMAIN PERFORMANCE SCOREBOARD", scoreboard) # Explicitly label RF
                        # 7.2/7.3 Best/Worst Domain Analysis (Selected by RF MAE, show all models)
                        top_rf_mae_domains, bottom_rf_mae_domains = get_top_bottom_indices(domain_perf, mae_metric_rf, n=N_TOP_BOTTOM)
                        top_domain_chars = analyze_domain_characteristics(df, top_rf_mae_domains, pred_cols_config, uncertainty_cols_config)
                        bottom_domain_chars = analyze_domain_characteristics(df, bottom_rf_mae_domains, pred_cols_config, uncertainty_cols_config)
                        if not top_domain_chars.empty: write_section(outfile, f"7.2 CHARACTERISTICS OF TOP {len(top_rf_mae_domains)} DOMAINS (Lowest {mae_metric_rf.upper()})", top_domain_chars)
                        if not bottom_domain_chars.empty: write_section(outfile, f"7.3 CHARACTERISTICS OF BOTTOM {len(bottom_rf_mae_domains)} DOMAINS (Highest {mae_metric_rf.upper()})", bottom_domain_chars)
                    else: write_section(outfile, "7. DOMAIN PERFORMANCE", "No domain metrics calculated.")

                # 8. "Nailed It" Case Study (Based on RF)
                logger.info(f"Finding 'Nailed It' domain candidate (based on {primary_model_key})...")
                nailed_it_found = False
                if 'domain_perf' in locals() and not domain_perf.empty and 'actual_stddev' in domain_perf.columns:
                    pcc_key, r2_key, mae_key = f'{primary_model_key}_pcc', f'{primary_model_key}_r2', f'{primary_model_key}_mae'
                    pred_std_key = f'{primary_model_key}_pred_stddev'
                    if all(k in domain_perf.columns for k in [pcc_key, r2_key, mae_key, 'actual_stddev', pred_std_key]):
                        actual_std_thresh = overall_actual_std * NAILED_IT_MIN_ACTUAL_STD_FACTOR
                        pred_std_thresh = overall_pred_std.get(primary_model_key, 0) * NAILED_IT_MIN_PRED_STD_FACTOR
                        candidates = domain_perf[ (domain_perf[pcc_key] >= NAILED_IT_MIN_PCC) & (domain_perf[r2_key] >= NAILED_IT_MIN_R2) & (domain_perf[mae_key] <= NAILED_IT_MAX_MAE) & (domain_perf['actual_stddev'] >= actual_std_thresh) & (domain_perf[pred_std_key] >= pred_std_thresh) ]
                        if not candidates.empty:
                            candidates_sorted = candidates.sort_values(by=pcc_key, ascending=False)
                            best_nailed_it = candidates_sorted.index[0]; logger.info(f"Selected '{best_nailed_it}' as 'Nailed It' candidate.")
                            best_nailed_it_chars = analyze_domain_characteristics(df, [best_nailed_it], pred_cols_config, uncertainty_cols_config) # Pass unc config
                            criteria_note = (f"Criteria ({primary_model_key}): PCC>={NAILED_IT_MIN_PCC}, R2>={NAILED_IT_MIN_R2}, MAE<={NAILED_IT_MAX_MAE}, ActualStd>={actual_std_thresh:.3f}, PredStd>={pred_std_thresh:.3f}")
                            write_section(outfile, f"8. 'NAILED IT' DOMAIN CASE STUDY ({primary_model_key} Selection): {best_nailed_it}", best_nailed_it_chars, note=criteria_note)
                            nailed_it_found = True
                    else: logger.warning(f"Missing required RF metrics in domain_perf for 'Nailed It'.")
                if not nailed_it_found: write_section(outfile, f"8. 'NAILED IT' DOMAIN CASE STUDY ({primary_model_key} Selection)", "No domains found matching criteria.")

                # 9. Amino Acid Performance (Show all models)
                if RESNAME_COL in df.columns:
                    logger.info("Analyzing performance per amino acid (all models)..."); aa_perf = df.groupby(RESNAME_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                    if not aa_perf.empty: cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in aa_perf.columns]; sort_metric_aa = f"{primary_model_key}_mae"; aa_perf_sorted = aa_perf.sort_values(sort_metric_aa, ascending=True, na_position='last') if sort_metric_aa in aa_perf.columns else aa_perf; write_section(outfile, f"9. AMINO ACID PERFORMANCE (Sorted by {sort_metric_aa.upper()})", aa_perf_sorted[cols_to_show])

                # 10. Positional Performance (Show all models)
                if NORM_RESID_COL in df.columns:
                    logger.info("Analyzing performance by normalized residue position (all models)...")
                    try:
                        if 'pos_bin_labeled' not in df.columns: df['pos_bin_labeled'] = pd.cut(df[NORM_RESID_COL], bins=POS_BINS, labels=POS_BIN_LABELS, include_lowest=True, right=True)
                        if 'pos_bin_labeled' in df.columns:
                             pos_perf = df.groupby('pos_bin_labeled', observed=False).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                             cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in pos_perf.columns]
                             pos_perf.index.name = f"{NORM_RESID_COL}_bin"; write_section(outfile, f"10. PERFORMANCE BY {NORM_RESID_COL.upper()} BIN", pos_perf[cols_to_show])
                    except Exception as e: write_section(outfile, f"10. PERFORMANCE BY {NORM_RESID_COL.upper()} BIN", f"Error: {e}")

                # 11. Core/Exterior Performance (Show all models)
                if CORE_EXT_COL in df.columns: logger.info("Analyzing performance by core/exterior (all models)..."); core_ext_perf = df.groupby(CORE_EXT_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False); cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in core_ext_perf.columns]; write_section(outfile, "11. PERFORMANCE BY CORE/EXTERIOR", core_ext_perf[cols_to_show])

                # 12. Secondary Structure Performance (Show all models)
                if 'ss_group' in df.columns: logger.info("Analyzing performance by secondary structure group (all models)..."); ss_perf = df.groupby('ss_group').apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False); cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in ss_perf.columns]; write_section(outfile, "12. PERFORMANCE BY SECONDARY STRUCTURE (H/E/L)", ss_perf[cols_to_show])

                # 13. Temperature Sensitivity/Resistance (Based on RF)
                temp_sensitivity_df = pd.DataFrame() # Initialize
                if TEMP_COL in df.columns:
                     logger.info(f"Analyzing domain temperature sensitivity (based on {primary_model_key} MAE)..."); available_temps = df[TEMP_COL].dropna().unique(); min_temp = available_temps.min() if len(available_temps)>0 else np.nan; max_temp = available_temps.max() if len(available_temps)>0 else np.nan
                     if not pd.isna(min_temp) and not pd.isna(max_temp) and not np.isclose(min_temp, max_temp):
                         domains_at_min = set(df[np.isclose(df[TEMP_COL], min_temp)][DOMAIN_COL].unique()); domains_at_max = set(df[np.isclose(df[TEMP_COL], max_temp)][DOMAIN_COL].unique())
                         spanning_domains = sorted(list(domains_at_min.intersection(domains_at_max))); logger.info(f"Found {len(spanning_domains)} domains spanning T={min_temp}K to T={max_temp}K.")
                         if spanning_domains:
                             spanning_df = df[df[DOMAIN_COL].isin(spanning_domains)].copy()
                             # Calculate performance at min/max temps for ALL models (needed for characteristics table)
                             perf_at_min = spanning_df[np.isclose(spanning_df[TEMP_COL], min_temp)].groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                             perf_at_max = spanning_df[np.isclose(spanning_df[TEMP_COL], max_temp)].groupby(DOMAIN_COL).apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                             avg_rmsf_at_min = spanning_df[np.isclose(spanning_df[TEMP_COL], min_temp)].groupby(DOMAIN_COL)[ACTUAL_COL].mean()
                             avg_rmsf_at_max = spanning_df[np.isclose(spanning_df[TEMP_COL], max_temp)].groupby(DOMAIN_COL)[ACTUAL_COL].mean()
                             mae_col_rf = f"{primary_model_key}_mae" # RF MAE column name
                             # Check if RF MAE exists in both min/max perf tables
                             if mae_col_rf in perf_at_min.columns and mae_col_rf in perf_at_max.columns:
                                 delta_mae_rf = (perf_at_max[mae_col_rf] - perf_at_min[mae_col_rf]).rename('delta_mae_RF')
                                 delta_actual_rmsf = (avg_rmsf_at_max - avg_rmsf_at_min).rename('delta_actual_rmsf')
                                 # Get overall RF MAE for these domains from the domain_perf table
                                 if 'domain_perf' in locals() and mae_col_rf in domain_perf.columns:
                                     overall_mae_rf = domain_perf.loc[spanning_domains, mae_col_rf].rename(f'overall_{mae_col_rf}')
                                     temp_sensitivity_df = pd.concat([delta_mae_rf, delta_actual_rmsf, overall_mae_rf], axis=1).dropna()
                                 if not temp_sensitivity_df.empty:
                                     corr_note = ""; change_corr = np.nan # ... (keep correlation calculation) ...
                                     if temp_sensitivity_df[['delta_mae_RF', 'delta_actual_rmsf']].nunique().min() > 1 and len(temp_sensitivity_df) >= MIN_POINTS_FOR_METRICS: 
                                         try: 
                                             change_corr, _ = pearsonr(temp_sensitivity_df['delta_mae_RF'], temp_sensitivity_df['delta_actual_rmsf']); corr_note = f"Corr(Delta {primary_model_key} MAE vs Delta Actual RMSF): {change_corr:.3f}" if not pd.isna(change_corr) else "Corr: NaN" ; 
                                         except Exception as e: corr_note = f"Error calc sensitivity corr: {e}"
                                     else: corr_note = "Not enough variance/data for sensitivity correlation."
                                     write_section(outfile, f"13.1 {primary_model_key} TEMPERATURE SENSITIVITY (T={min_temp:.0f}K to T={max_temp:.0f}K, {len(spanning_domains)} Domains)", temp_sensitivity_df.sort_values('delta_mae_RF'), note=corr_note)
                                     # Identify and analyze resistant/sensitive domains (based on RF delta_mae)
                                     resistant_indices, sensitive_indices = get_top_bottom_indices(temp_sensitivity_df, 'delta_mae_RF', n=N_TOP_BOTTOM)
                                     if resistant_indices: resistant_chars = analyze_domain_characteristics(df, resistant_indices, pred_cols_config, uncertainty_cols_config); write_section(outfile, f"13.2 CHARACTERISTICS OF {len(resistant_indices)} MOST ERROR-RESISTANT DOMAINS (Smallest {primary_model_key} MAE Change)", resistant_chars)
                                     if sensitive_indices: sensitive_chars = analyze_domain_characteristics(df, sensitive_indices, pred_cols_config, uncertainty_cols_config); write_section(outfile, f"13.3 CHARACTERISTICS OF {len(sensitive_indices)} MOST ERROR-SENSITIVE DOMAINS (Largest {primary_model_key} MAE Change)", sensitive_chars)
                                 else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "No spanning domains with valid MAE change.")
                             else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", f"Required metric '{mae_col_rf}' not available in min/max temp performance.")
                         else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "No spanning domains found.")
                     else: write_section(outfile, f"13. {primary_model_key} TEMPERATURE SENSITIVITY/RESISTANCE", "Cannot analyze: Temp range insufficient.")

                # 14. "Temperature Mastered" Case Study (Based on RF)
                logger.info(f"Finding 'Temperature Mastered' domain candidate (based on {primary_model_key})...")
                temp_mastered_found = False
                if not temp_sensitivity_df.empty and f'overall_{mae_col_rf}' in temp_sensitivity_df.columns: # Use mae_col_rf defined above
                    candidates_temp = temp_sensitivity_df[ (temp_sensitivity_df['delta_actual_rmsf'].abs() >= TEMP_MASTERED_MIN_ACTUAL_DELTA) & (temp_sensitivity_df[f'overall_{mae_col_rf}'] <= TEMP_MASTERED_MAX_MAE) & (temp_sensitivity_df['delta_mae_RF'].abs() <= TEMP_MASTERED_MAX_DELTA_MAE_ABS) ]
                    if not candidates_temp.empty:
                         candidates_temp_sorted = candidates_temp.sort_values(by=['delta_actual_rmsf', 'delta_mae_RF'], ascending=[False, True])
                         best_temp_mastered = candidates_temp_sorted.index[0]; logger.info(f"Selected '{best_temp_mastered}' as 'Temperature Mastered' candidate.")
                         best_temp_mastered_chars = analyze_domain_characteristics(df, [best_temp_mastered], pred_cols_config, uncertainty_cols_config) # Pass unc config
                         criteria_note = (f"Criteria ({primary_model_key}): |ActualDelta| >= {TEMP_MASTERED_MIN_ACTUAL_DELTA}, OverallMAE <= {TEMP_MASTERED_MAX_MAE}, |DeltaMAE| <= {TEMP_MASTERED_MAX_DELTA_MAE_ABS}")
                         write_section(outfile, f"14. 'TEMPERATURE MASTERED' DOMAIN CASE STUDY ({primary_model_key} Selection): {best_temp_mastered}", best_temp_mastered_chars, note=criteria_note)
                         temp_mastered_found = True
                if not temp_mastered_found: write_section(outfile, f"14. 'TEMPERATURE MASTERED' DOMAIN CASE STUDY ({primary_model_key} Selection)", "No domains found matching criteria.")

                # 15. Flexibility Magnitude vs. Performance (Show all models)
                logger.info("Analyzing performance by actual flexibility magnitude (all models)...");
                try:
                    if ACTUAL_COL in df.columns and df[ACTUAL_COL].nunique() > QUANTILE_BINS:
                        df['rmsf_quantile_label'] = pd.qcut(df[ACTUAL_COL], q=QUANTILE_BINS, labels=False, duplicates='drop')
                        flex_perf = df.groupby('rmsf_quantile_label').apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                        cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in flex_perf.columns]
                        try: quantile_bounds = pd.qcut(df[ACTUAL_COL], q=QUANTILE_BINS, duplicates='drop').cat.categories; bin_labels = [f"{i.left:.2f}-{i.right:.2f}" for i in quantile_bounds]; flex_perf.index = pd.Index(bin_labels[:len(flex_perf)], name='Actual_RMSF_Quantile')
                        except Exception: flex_perf.index.name = 'Actual_RMSF_Quantile_Idx'
                        write_section(outfile, f"15. PERFORMANCE BY ACTUAL RMSF QUANTILE ({len(flex_perf)} bins)", flex_perf[cols_to_show])
                    else: write_section(outfile, f"15. PERFORMANCE BY ACTUAL RMSF QUANTILE", "Not enough unique values.")
                except Exception as e: write_section(outfile, "15. PERFORMANCE BY ACTUAL RMSF QUANTILE", f"Error: {e}")

                # 16. Relative Accessibility vs. Performance (Show all models)
                if REL_ACC_COL in df.columns:
                     logger.info("Analyzing performance by relative accessibility (all models)...");
                     try:
                        acc_col_no_na = df[REL_ACC_COL].fillna(df[REL_ACC_COL].median())
                        if acc_col_no_na.nunique() > QUANTILE_BINS:
                            df['access_quantile_label'] = pd.qcut(acc_col_no_na, q=QUANTILE_BINS, labels=False, duplicates='drop')
                            access_perf = df.groupby('access_quantile_label').apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                            cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in access_perf.columns]
                            try: quantile_bounds = pd.qcut(acc_col_no_na, q=QUANTILE_BINS, duplicates='drop').cat.categories; bin_labels = [f"{i.left:.2f}-{i.right:.2f}" for i in quantile_bounds]; access_perf.index = pd.Index(bin_labels[:len(access_perf)], name='RelAccess_Quantile')
                            except Exception: access_perf.index.name = 'RelAccess_Quantile_Idx'
                            write_section(outfile, f"16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE ({len(access_perf)} bins)", access_perf[cols_to_show])
                        else: write_section(outfile, f"16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", "Not enough unique values.")
                     except Exception as e: write_section(outfile, "16. PERFORMANCE BY RELATIVE ACCESSIBILITY QUANTILE", f"Error: {e}")

                # 17. Domain Size vs. Performance (Show all models)
                if SIZE_COL in df.columns and DOMAIN_COL in df.columns:
                    logger.info("Analyzing performance by domain size (all models)...");
                    try:
                        domain_size_map = df.drop_duplicates(subset=[DOMAIN_COL])[[DOMAIN_COL, SIZE_COL]].set_index(DOMAIN_COL)[SIZE_COL].dropna()
                        if len(domain_size_map) >= QUANTILE_BINS * 2:
                             domain_size_map['size_quantile_label'], size_bins = pd.qcut(domain_size_map, q=QUANTILE_BINS, labels=False, duplicates='drop', retbins=True)
                             size_quantile_mapping = domain_size_map['size_quantile_label'].to_dict()
                             # Use .map() which is often safer than merge for adding a column based on index/key
                             df['size_quantile_group'] = df[DOMAIN_COL].map(size_quantile_mapping)
                             if not df['size_quantile_group'].isna().all():
                                 size_perf = df.dropna(subset=['size_quantile_group']).groupby('size_quantile_group').apply(lambda x: calculate_group_metrics(x, ACTUAL_COL, pred_cols_config, uncertainty_cols_config), include_groups=False)
                                 cols_to_show = ['count'] + [f"{key}_{m}" for key in model_keys_ordered for m in ['mae', 'pcc'] if f"{key}_{m}" in size_perf.columns]
                                 bin_labels = [f"{int(size_bins[i])}-{int(size_bins[i+1])}" for i in range(len(size_bins)-1)]; label_map = dict(enumerate(bin_labels))
                                 size_perf.index = size_perf.index.map(label_map); size_perf.index.name = 'DomainSize_Quantile'
                                 write_section(outfile, f"17. PERFORMANCE BY DOMAIN SIZE QUANTILE ({len(size_perf)} bins)", size_perf[cols_to_show])
                             else: write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", "Mapping size bins failed.")
                        else: write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", f"Not enough unique domains ({len(domain_size_map)}).")
                    except Exception as e: logger.error(f"Error during domain size analysis: {e}"); write_section(outfile, "17. PERFORMANCE BY DOMAIN SIZE QUANTILE", f"Error: {e}")

                # 18. Model Disagreement Analysis
                logger.info("Analyzing model disagreement...");
                if len(pred_cols_config) > 1:
                    try:
                        pred_df_subset = df[list(pred_cols_config.values())]; df['prediction_stddev'] = pred_df_subset.std(axis=1)
                        write_section(outfile, "18. MODEL PREDICTION STANDARD DEVIATION STATS", df['prediction_stddev'].describe())
                        if primary_abs_error_col and primary_abs_error_col in df.columns:
                             disagreement_error_corr_df = df[[primary_abs_error_col, 'prediction_stddev']].dropna()
                             if disagreement_error_corr_df.shape[0] > MIN_POINTS_FOR_METRICS and disagreement_error_corr_df.nunique().min() > 1:
                                  disagree_corr, _ = pearsonr(disagreement_error_corr_df[primary_abs_error_col], disagreement_error_corr_df['prediction_stddev'])
                                  write_section(outfile, "18.1 MODEL DISAGREEMENT VS. ERROR (Primary Model)", f"Correlation between prediction_stddev and {primary_abs_error_col}: {disagree_corr:.3f}")
                    except Exception as e: write_section(outfile, "18. MODEL DISAGREEMENT ANALYSIS", f"Error: {e}")
                else: write_section(outfile, "18. MODEL DISAGREEMENT ANALYSIS", "Skipped: Requires >= 2 models.")

                # 19. Outlier Residue Analysis (Based on RF error, show all models)
                logger.info("Analyzing outlier residues (based on RF error)...");
                if primary_abs_error_col and primary_abs_error_col in df.columns:
                     outlier_cols_to_show = [DOMAIN_COL, RESID_COL, RESNAME_COL, TEMP_COL, ACTUAL_COL] + list(pred_cols_config.values()) + [primary_abs_error_col] + list(uncertainty_cols_config.values()) + [CORE_EXT_COL, 'ss_group']
                     outlier_cols_to_show = [c for c in outlier_cols_to_show if c in df.columns] # Filter available
                     n_outliers = N_TOP_BOTTOM * 3; outliers_df = df.nlargest(n_outliers, primary_abs_error_col)[outlier_cols_to_show]
                     write_section(outfile, f"19. TOP {len(outliers_df)} OUTLIER RESIDUES (Highest {primary_abs_error_col.upper()})", outliers_df)
                     outlier_summary = {};
                     if not outliers_df.empty:
                          if RESNAME_COL in outliers_df: outlier_summary['AA Types (%)'] = outliers_df[RESNAME_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if 'ss_group' in outliers_df: outlier_summary['SS Group (%)'] = outliers_df['ss_group'].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if CORE_EXT_COL in outliers_df: outlier_summary['Core/Ext (%)'] = outliers_df[CORE_EXT_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          if TEMP_COL in outliers_df: outlier_summary['Temp Dist (%)'] = outliers_df[TEMP_COL].value_counts(normalize=True).mul(100).round(1).to_dict()
                          write_section(outfile, "19.1 SUMMARY CHARACTERISTICS OF OUTLIER RESIDUES", outlier_summary)
                else: write_section(outfile, "19. OUTLIER RESIDUE ANALYSIS", f"Skipped: Abs error column '{primary_abs_error_col}' not found.")
            else: write_section(outfile, "DETAILED BREAKDOWNS (SECTIONS 7-19)", "Skipped: Performance analysis not possible or no primary model identified.")

    except Exception as e:
        logger.error(f"An critical error occurred during the main analysis: {e}", exc_info=True)

    logger.info(f"--- Analysis Report Generation Finished ({OUTPUT_TXT_PATH}) ---")

if __name__ == "__main__":
    main()