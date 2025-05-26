
import pandas as pd
import numpy as np

def cumulative_sum_by_key(df, lob_state_depth=10):
    """
    Calculates the cumulative sum across a sequence of columns in a Pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        #print("Error or empty DataFrame: Input must be a non-empty Pandas DataFrame for cumulative_sum.")
        return pd.DataFrame() # Return empty if df is bad

    if not isinstance(lob_state_depth, int) or lob_state_depth <= 0:
        print("Error: lob_state_depth must be a positive integer.")
        return pd.DataFrame()

    cumulative_sum_data = {}
    # Check if any required L1 key exists before proceeding
    if f'L1-AskSize' not in df.columns or f'L1-BidSize' not in df.columns:
        print(f"Warning: L1-AskSize or L1-BidSize not found. Cannot calculate cumulative volumes.")
        return pd.DataFrame()
        
    for i in range(1, lob_state_depth + 1):
        akey = f'L{i}-AskSize'
        bkey = f'L{i}-BidSize'
        cumulative_akey = f'L{i}-AskVolume'
        cumulative_bkey = f'L{i}-BidVolume'

        # Only proceed if current level's size columns exist
        if akey not in df.columns or bkey not in df.columns:
            print(f"Warning: Key '{akey}' or '{bkey}' not found. Stopping cumulative sum at L{i-1}.")
            break # Stop if a level is missing

        if i == 1:
            cumulative_sum_data[cumulative_akey] = df[akey].fillna(0)
            cumulative_sum_data[cumulative_bkey] = df[bkey].fillna(0)
        else:
            # Ensure previous level's cumulative volume exists
            prev_ask_vol_key = f'L{i-1}-AskVolume'
            prev_bid_vol_key = f'L{i-1}-BidVolume'
            if prev_ask_vol_key not in cumulative_sum_data or prev_bid_vol_key not in cumulative_sum_data:
                # This case should ideally be caught by the break above
                print(f"Warning: Previous cumulative volume for L{i-1} not found. Stopping.")
                break
            cumulative_sum_data[cumulative_akey] = df[akey].fillna(0) + cumulative_sum_data[prev_ask_vol_key]
            cumulative_sum_data[cumulative_bkey] = df[bkey].fillna(0) + cumulative_sum_data[prev_bid_vol_key]
            
    return pd.DataFrame(cumulative_sum_data)


def filter_and_prepare_lob_data(df_o, specific_ric, verbose=False):
    """
    Filters LOB data by RIC, standardizes time, and ensures it's sorted.
    """
    if df_o.empty:
        print("LOB DataFrame is empty. Skipping filtering.")
        return df_o
        
    if 'Date-Time' not in df_o.columns:
        print('LOB Date-Time column not found. Cannot run on already filtered data or malformed data.')
        return df_o

    if specific_ric and 'Alias Underlying RIC' in df_o.columns:
        df_o = df_o[df_o['Alias Underlying RIC'] == specific_ric].copy()
        if df_o.empty:
            print(f"No LOB data found for RIC: {specific_ric}")
            return df_o
    elif specific_ric:
        print(f"Warning: 'Alias Underlying RIC' column not in LOB data, cannot filter by RIC {specific_ric}.")
    
    df_o.rename(columns={'Date-Time': 'lob_time'}, inplace=True)
    
    # Convert to pd.timestamp if not already (load_data should do this, but good to be safe)
    if not pd.api.types.is_datetime64_any_dtype(df_o['lob_time']):
        df_o['lob_time'] = pd.to_datetime(df_o['lob_time'])

    # Remove timezone if present
    if df_o['lob_time'].dt.tz is not None:
        df_o['lob_time'] = df_o['lob_time'].dt.tz_localize(None)
    
    df_o.sort_values(by=['lob_time'], inplace=True)
    
    if not df_o['lob_time'].is_monotonic_increasing:
        # Attempt to resolve by dropping duplicates, keeping the first occurrence
        df_o.drop_duplicates(subset=['lob_time'], keep='first', inplace=True)
        if not df_o['lob_time'].is_monotonic_increasing: # Check again
             # If still not monotonic, it's a more serious issue.
             # For now, we might raise an error or just print a strong warning.
             # Let's try sorting one more time very carefully if it helps in weird cases.
             df_o.reset_index(drop=True, inplace=True) # Reset index before sort
             df_o.sort_values(by=['lob_time'], inplace=True, kind='mergesort') # Stable sort

             if not df_o['lob_time'].is_monotonic_increasing:
                 duplicate_times = df_o[df_o['lob_time'].duplicated(keep=False)]
                 print(f"WARNING: df_o['lob_time'] is STILL not sorted correctly after de-duplication attempts! This will cause issues with searchsorted. Problematic times (first few):\n{duplicate_times['lob_time'].head()}")
                 # raise ValueError("df_o['lob_time'] is not sorted correctly after de-duplication! Critical for matching.")

    df_o.reset_index(drop=True, inplace=True)
    return df_o


def add_public_lob_features(df_o, verbose=False):
    """
    Calculates and adds public LOB features.
    """
    if df_o.empty:
        print("LOB DataFrame is empty. Skipping feature addition.")
        return df_o

    required_l1_cols = ['L1-AskPrice', 'L1-BidPrice', 'L1-AskSize', 'L1-BidSize']
    if not all(col in df_o.columns for col in required_l1_cols):
        print(f"Warning: Missing one or more L1 columns ({required_l1_cols}). Cannot calculate some public features.")
    
    # Calculate basic features safely
    market_spread = pd.Series(np.nan, index=df_o.index)
    mid_price = pd.Series(np.nan, index=df_o.index)
    bbo_imbalance_calc = pd.Series(np.nan, index=df_o.index)

    if 'L1-AskPrice' in df_o.columns and 'L1-BidPrice' in df_o.columns:
        market_spread = df_o['L1-AskPrice'] - df_o['L1-BidPrice']
        mid_price = (df_o['L1-AskPrice'] + df_o['L1-BidPrice']) / 2
        # Ensure mid_price is not zero for division
        non_zero_mid_price_mask = (mid_price != 0) & (~mid_price.isna())
        # BBO Size & Imbalance: ‘BidSize_L1’, ‘AskSize_L1’, and a ratio like ‘(BidSz-AskSz)/(BidSz+AskSz)’ or ‘log(BidSz/AskSz)’.
        bbo_imbalance_calc[non_zero_mid_price_mask] = (df_o['L1-BidSize'][non_zero_mid_price_mask] - df_o['L1-AskSize'][non_zero_mid_price_mask]) / (df_o['L1-BidSize'][non_zero_mid_price_mask] + df_o['L1-AskSize'][non_zero_mid_price_mask])
    df_public_features_basic = pd.DataFrame({
        'MarketSpread': market_spread,
        'MidPrice': mid_price,
        'BBOImbalance': bbo_imbalance_calc,
    })

    # Cumulative volume and volume imbalance
    cum_volume_df = cumulative_sum_by_key(df_o, 10) # Assuming depth 10
    volume_imbalance_dfs = []
    if not cum_volume_df.empty:
        for i in range(1, 11):
            ask_vol_col = f'L{i}-AskVolume'
            bid_vol_col = f'L{i}-BidVolume'
            imb_col_name = f'L{i}-VolumeImbalance'
            
            temp_imb_series = pd.Series(np.nan, index=df_o.index)
            if ask_vol_col in cum_volume_df.columns and bid_vol_col in cum_volume_df.columns:
                # Avoid division by zero or NaN
                mask = (cum_volume_df[bid_vol_col] != 0) & (~cum_volume_df[bid_vol_col].isna()) & \
                       (~cum_volume_df[ask_vol_col].isna())
                temp_imb_series[mask] = cum_volume_df[ask_vol_col][mask] / cum_volume_df[bid_vol_col][mask]
            volume_imbalance_dfs.append(pd.DataFrame({imb_col_name: temp_imb_series}))
    
    volume_imbalance_final_df = pd.concat(volume_imbalance_dfs, axis=1) if volume_imbalance_dfs else pd.DataFrame(index=df_o.index)

    # Diff features
    # Ensure some ask/bid columns exist before trying to filter
    ask_bid_cols = [col for col in df_o.columns if '-Ask' in col or '-Bid' in col]
    df_o_diff = pd.DataFrame(index=df_o.index) # Initialize empty
    if ask_bid_cols:
        df_o_diff_temp = df_o.filter(regex='-AskPrice|-BidPrice|-AskSize|-BidSize').diff().fillna(0) # focus on Price/Size diffs
        df_o_diff = df_o_diff_temp.rename(columns={col: col + '-diff' for col in df_o_diff_temp.columns})

    # Combine all results into single dataframe
    # Start with a copy of df_o to avoid modifying original if passed around
    df_enriched = df_o.copy() 
    for new_df in [df_public_features_basic, cum_volume_df, volume_imbalance_final_df, df_o_diff]:
        if not new_df.empty:
            # Ensure indices align before concat, especially if df_o was modified (e.g. dropped rows)
            new_df = new_df.reindex(df_enriched.index)
            df_enriched = pd.concat([df_enriched, new_df], axis=1)
  
    if verbose and not df_enriched.empty:
        print("Sample of enriched LOB data:")
        print(df_enriched.head())
    elif df_enriched.empty and not df_o.empty:
        print("Warning: Enriched LOB DataFrame is empty, possibly due to missing columns in input.")
        return df_o # Return original if enrichment failed badly

    return df_enriched
