# scripts/main_pipeline.py
import pandas as pd
import os
import argparse

from load_data import load_lob_data, load_hedge_data
from preprocess_lob import filter_and_prepare_lob_data, add_public_lob_features
from preprocess_hedge import filter_and_prepare_hedge_data
from matching_engine import match_hedges_to_lob

def get_tick_size_for_ric(ric_code):
    """ Returns the tick size for a given RIC. Expand as needed. """

    ric_prefix = ric_code[:2]
    if ric_prefix == 'FF': # FTSE Futures
        return 0.5
    elif ric_prefix == 'ES': # E-mini S&P
        return 0.25
    elif ric_prefix == 'NQ': # E-mini NASDAQ
        return 0.25
    print(f"Warning: Tick size not defined for RIC starting with {ric_prefix}. Defaulting to 0.5.")
    return 0.5


def run_pipeline_for_single_ric(lob_df_for_ric, hedge_df_for_ric, specific_ric, 
                                time_threshold_seconds, instrument_tick_size, 
                                 verbose, output_path_base):
    """
    Runs the preprocessing and matching for a single RIC's pre-loaded data.
    """
    print(f"\n--- Processing RIC: {specific_ric} ---")

    # 2. Preprocess LOB Data for this RIC
    print("Step 2.1: Filtering and Preparing LOB data...")
    df_lob_filtered = filter_and_prepare_lob_data(lob_df_for_ric, specific_ric=specific_ric, verbose=verbose)
    if df_lob_filtered.empty:
        print(f"LOB data for {specific_ric} became empty after filtering. Skipping this RIC.")
        return None
    
    print("Step 2.2: Adding public features to LOB data...")
    df_lob_enriched = add_public_lob_features(df_lob_filtered, verbose=verbose)
    if df_lob_enriched.empty:
        print(f"LOB data for {specific_ric} became empty after adding features. Skipping this RIC.")
        return None
    print(f"Enriched LOB data shape for {specific_ric}: {df_lob_enriched.shape}")

    # 4. Preprocess Hedge Data for this RIC
    print("\nStep 3.1: Filtering and Preparing hedge data...")
    df_hedge_cleaned = filter_and_prepare_hedge_data(hedge_df_for_ric, specific_ric=specific_ric, verbose=verbose)
    if df_hedge_cleaned.empty:
        print(f"Hedge data for {specific_ric} became empty after filtering or no hedge data loaded. Skipping matching for this RIC.")
        return None
    print(f"Cleaned hedge data shape for {specific_ric}: {df_hedge_cleaned.shape}")

    # 5. Match Hedges to LOB
    print("\nStep 4.1: Matching hedges to LOB states...")
    df_matched_data = match_hedges_to_lob(
        df_hedge_cleaned, 
        df_lob_enriched, 
        time_threshold_seconds,
        tick_size=instrument_tick_size,
    )
    
    if df_matched_data.empty:
        print(f"No matches found for {specific_ric}.")
    else:
        print(f"Successfully matched data shape for {specific_ric}: {df_matched_data.shape}")
        
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        
        output_filename = os.path.join(output_path_base, f"{specific_ric}_matched_lob_hedge.parquet")
        try:
            df_matched_data.to_parquet(output_filename, index=False)
            print(f"Saved matched data for {specific_ric} to: {output_filename}")
        except Exception as e:
            print(f"Error saving matched data to {output_filename} for {specific_ric}: {e}")
            output_csv_filename = os.path.join(output_path_base, f"{specific_ric}_matched_lob_hedge.csv")
            try:
                df_matched_data.to_csv(output_csv_filename, index=False)
                print(f"Saved matched data as CSV for {specific_ric} to: {output_csv_filename}")
            except Exception as e_csv:
                 print(f"Error saving matched data as CSV for {specific_ric}: {e_csv}")
    
    return df_matched_data


def main_orchestrator(index_name, specific_ric_to_process, index_family_for_hedge,
                      base_data_path='../data', 
                      time_threshold_seconds=2,
                      verbose=False,
                      output_path_base='../data/processed_matched_data'):
    """
    Main orchestrator for the pipeline.
    If specific_ric_to_process is None, it processes all found RICs for the index_name.
    """
    print(f"--- Starting Pipeline Orchestration for Index: {index_name} ---")
    if specific_ric_to_process:
        print(f"Targeting specific RIC: {specific_ric_to_process}")
    else:
        print("Targeting all available RICs for the index.")

    # 1. Load Initial LOB Data (either all for index, or just for specific_ric)
    print("\nStep 1: Loading initial LOB data...")
    df_lob_initial = load_lob_data(base_data_path, index_name, specific_ric=specific_ric_to_process)
    if df_lob_initial.empty:
        print(f"Failed to load any LOB data. Exiting pipeline.")
        return

    if 'Alias Underlying RIC' not in df_lob_initial.columns:
        print("Error: 'Alias Underlying RIC' column missing in loaded LOB data. Cannot determine RICs.")
        return

    # Determine which RICs to process
    if specific_ric_to_process:
        rics_to_process = [specific_ric_to_process]
    else:
        rics_to_process = df_lob_initial['Alias Underlying RIC'].unique()
        if not rics_to_process.size: # handle case where unique returns empty array
            print(f"No unique RICs found in LOB data for index {index_name}. Exiting.")
            return
        print(f"Found RICs to process: {list(rics_to_process)}")

    all_matched_data = {} # To store results if needed

    for ric_code in rics_to_process:
        # Select LOB data for the current RIC
        lob_df_for_this_ric = df_lob_initial[df_lob_initial['Alias Underlying RIC'] == ric_code].copy()
        if lob_df_for_this_ric.empty:
            print(f"No LOB data for RIC {ric_code} after filtering initial load. Skipping.")
            continue

        # Load Hedge Data for this specific RIC
        # ric_month_year often same as ric_code for these futures
        print(f"\nLoading hedge data for RIC: {ric_code} using index_family: {index_family_for_hedge}...")
        base_futures_path = os.path.join(base_data_path, 'futures_data_local')
        hedge_df_for_this_ric = load_hedge_data(
            base_futures_path, 
            index_family_for_hedge, # This needs to be correct for the RIC
            ric_month_year=ric_code,  # Assuming subdir is same as RIC
            full_ric=ric_code
        )
        
        # If no hedge data, we can still process LOB and save enriched LOB, or just skip matching
        if hedge_df_for_this_ric.empty:
            print(f"No hedge data found for RIC {ric_code}. Matching will be skipped for this RIC.")
            # You might still want to process and save just the enriched LOB data
            # For now, the pipeline for single RIC handles empty hedge data by skipping matching.

        instrument_tick_size = get_tick_size_for_ric(ric_code)

        matched_result_for_ric = run_pipeline_for_single_ric(
            lob_df_for_this_ric, hedge_df_for_this_ric, ric_code,
            time_threshold_seconds, instrument_tick_size,
            verbose, output_path_base
        )
        if matched_result_for_ric is not None and not matched_result_for_ric.empty:
            all_matched_data[ric_code] = matched_result_for_ric
    
    print(f"\n--- Pipeline Orchestration Finished for Index: {index_name} ---")
    return all_matched_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LOB and Hedge data matching pipeline.")
    parser.add_argument("--index_name", type=str, required=True, 
                        help="Name of the index (e.g., FTSE, NASDAQ) to find LOB dir.")
    parser.add_argument("--ric", type=str, default=None, 
                        help="Optional: Specific RIC to process (e.g., FFIH4). If None, processes all RICs for the index.")
    parser.add_argument("--index_family", type=str, required=True, 
                        help="Index family for hedge data path (e.g., FF for FTSE futures, ES for S&P E-mini). This should match the prefix of RICs in the index.")
    
    parser.add_argument("--base_path", type=str, default="../data", help="Base path to the 'data' directory.")
    parser.add_argument("--threshold_sec", type=float, default=5, help="Time threshold in seconds for matching.") # default 5 second
    parser.add_argument("--size_tolerance", type=int, default=1, help="Absolute size tolerance for matching.") # default 1
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    parser.add_argument("--output_dir", type=str, default="../data/processed_matched_data", 
                        help="Directory to save matched data.")

    args = parser.parse_args()

    main_orchestrator(
        index_name=args.index_name,
        specific_ric_to_process=args.ric,
        index_family_for_hedge=args.index_family,
        base_data_path=args.base_path,
        time_threshold_seconds=args.threshold_sec,
        verbose=args.verbose,
        output_path_base=args.output_dir
    )

    # Example commands from 'scripts' directory:
    # Process a specific RIC:
    # python main_pipeline.py --index_name FTSE --ric FFIH4 --index_family FF --verbose
    # Process all RICs found under FTSE LOB data (assuming hedge data is available for them under FF family):
    # python main_pipeline.py --index_name FTSE --index_family FF --verbose 