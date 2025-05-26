
import pandas as pd
import numpy as np
def filter_and_prepare_hedge_data(df_h, specific_ric, verbose=False):
    """
    Cleans and preprocesses the hedge order DataFrame.
    """
    if df_h.empty:
        print("Hedge DataFrame is empty. Skipping filtering.")
        return df_h

    if 'TRANSACTTIME' not in df_h.columns:
        print('Hedge TRANSACTTIME column not found. Cannot run on already filtered data or malformed data.')
        return df_h

    # Select relevant columns
    relevant_cols = ['CLORDID', 'SIDE', 'ORDERQTY', 'PRICE', 'CURRENCY', 
                     'TIMEINFORCE', 'MKT_PRICE', 'BID', 'OFFER', 'VWAP', 
                     'STATUS', 'TRANSACTTIME', 'EXECTYPE', 'CUMQTY', 
                     'LEAVESQTY', 'EXEC_PRICE', 'RIC']
    missing_cols = [col for col in relevant_cols if col not in df_h.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns in hedge data: {missing_cols}")
        relevant_cols = [col for col in relevant_cols if col in df_h.columns]
        if not relevant_cols or 'TRANSACTTIME' not in relevant_cols: 
             print("Error: Critical columns missing for hedge data processing.")
             return pd.DataFrame()


    df_h_processed = df_h[relevant_cols].copy()

    if specific_ric and 'RIC' in df_h_processed.columns:
        df_h_processed = df_h_processed[df_h_processed['RIC'] == specific_ric]
        if df_h_processed.empty:
            print(f"No hedge data found for RIC: {specific_ric}")
            return df_h_processed
    elif specific_ric:
        print(f"Warning: 'RIC' column not in hedge data, cannot filter by RIC {specific_ric}.")


    # Filter 1: Remove cancelled orders
    if 'ORDERQTY' in df_h_processed.columns and 'EXECTYPE' in df_h_processed.columns:
        df_h_processed = df_h_processed[~((df_h_processed['ORDERQTY'] == 1) & (df_h_processed['EXECTYPE'] == 4))]
    
    # Filter 2: Remove orders with no volume impact indicated by CUMQTY and LEAVESQTY
    if 'CUMQTY' in df_h_processed.columns and 'LEAVESQTY' in df_h_processed.columns:
        df_h_processed = df_h_processed[~((df_h_processed['CUMQTY'] == 0) & (df_h_processed['LEAVESQTY'] == 0))]

    subset_cols_for_duplicates = [col for col in ['CLORDID', 'SIDE', 'ORDERQTY', 'PRICE', 'CURRENCY', 
                                                 'TIMEINFORCE', 'MKT_PRICE', 'BID', 'OFFER', 'VWAP', 
                                                 'STATUS', 'EXECTYPE', 'CUMQTY', 'LEAVESQTY', 
                                                 'EXEC_PRICE', 'RIC'] if col in df_h_processed.columns]
    if subset_cols_for_duplicates:
        df_h_processed.drop_duplicates(subset=subset_cols_for_duplicates, keep='first', inplace=True)

    df_h_processed.rename(columns={'TRANSACTTIME': 'hedge_time'}, inplace=True)
    
    if not pd.api.types.is_datetime64_any_dtype(df_h_processed['hedge_time']):
        df_h_processed['hedge_time'] = pd.to_datetime(df_h_processed['hedge_time'])
    
    if df_h_processed['hedge_time'].dt.tz is not None:
        df_h_processed['hedge_time'] = df_h_processed['hedge_time'].dt.tz_localize(None)
    
    # Sort by time, then by CUMQTY to handle partial fills chronologically for a given order
    # Ensure 'CUMQTY' exists if used in sorting
    sort_by_cols = ['hedge_time']
    if 'CUMQTY' in df_h_processed.columns:
        sort_by_cols.append('CUMQTY')
    df_h_processed.sort_values(by=sort_by_cols, inplace=True)
    
    df_h_processed.reset_index(drop=True, inplace=True)
    
    if verbose and not df_h_processed.empty : print(f"Processed {len(df_h_processed)} hedge records.")
    return df_h_processed
