
import pandas as pd
import numpy as np

def find_closest_lob_indexes(df_h_sorted_for_lookup, df_o_prepared_sorted, hedge_irow, t_threshold_seconds):
    """
    Finds LOB snapshot indices within a time window around a hedge order's execution.
    Assumes df_o_prepared_sorted['lob_time'] is sorted and timezone-naive.
    Assumes df_h_sorted_for_lookup['hedge_time'] is timezone-naive.
    """
    if hedge_irow >= len(df_h_sorted_for_lookup):
        print(f"Error: hedge_irow {hedge_irow} is out of bounds for df_h.")
        return -1, -1 
        
    hedge_time = df_h_sorted_for_lookup['hedge_time'].iloc[hedge_irow]
    if pd.isna(hedge_time):
        print(f"Warning: NaT hedge_time at irow {hedge_irow}. Cannot find LOB window.")
        return -1, -1

    lower_bound = hedge_time - pd.Timedelta(seconds=1) # pp - using assymetric time interval
    upper_bound = hedge_time + pd.Timedelta(seconds=t_threshold_seconds)
    
    if df_o_prepared_sorted.empty or 'lob_time' not in df_o_prepared_sorted.columns:
        print("Warning: LOB DataFrame is empty or missing 'lob_time' for searchsorted.")
        return -1, -1

    try:
        start_idx = df_o_prepared_sorted['lob_time'].searchsorted(lower_bound, side='left')
        end_idx = df_o_prepared_sorted['lob_time'].searchsorted(upper_bound, side='right') # searchsorted returns insertion point
    except Exception as e:
        print(f"Error during searchsorted: {e}. Ensure lob_time is sorted and comparable.")
        return -1, -1
        
    
    return start_idx, end_idx # These are for iloc[start_idx:end_idx]


def match_hedges_to_lob(df_h_cleaned_sorted, df_o_enriched_prepared, 
                          t_threshold_seconds, tick_size): # Added tick_size for fuzzy matching
    """
    Matches hedge order executions to LOB states, with exact and fuzzy price matching.
    """

    print("THRESHOLD TIME:", t_threshold_seconds)
    print("TICK SIZE:", tick_size)
    if df_h_cleaned_sorted.empty:
        print("Hedge DataFrame is empty. No matching to perform.")
        return pd.DataFrame()
    if df_o_enriched_prepared.empty:
        print("LOB DataFrame is empty. No matching to perform.")
        return pd.DataFrame()

    matched_data_list = []
    num_input_hedge_fill_events = 0
    num_exact_price_matches = 0
    num_fuzzy_price_matches = 0
    
    for irow in range(len(df_h_cleaned_sorted)):
        current_hedge_event = df_h_cleaned_sorted.iloc[irow]

        # Consider only fills for matching
        if 'EXECTYPE' in current_hedge_event and current_hedge_event['EXECTYPE'] not in [1, 2]: # 1=Partial, 2=Last Fill
            continue 

        num_input_hedge_fill_events += 1 # Count actual fill events we attempt to match

        lob_start_idx, lob_end_idx = find_closest_lob_indexes(df_h_cleaned_sorted, df_o_enriched_prepared, irow, t_threshold_seconds)

        if lob_start_idx == -1 or lob_start_idx >= lob_end_idx:
            continue

        lob_window_df = df_o_enriched_prepared.iloc[lob_start_idx:lob_end_idx]

        if current_hedge_event['SIDE'] == 1: # Buy
            lob_price_suffix = '-AskPrice'
            lob_size_suffix = '-AskSize'
            matched_level_sign = 1
        elif current_hedge_event['SIDE'] == 2: # Sell
            lob_price_suffix = '-BidPrice'
            lob_size_suffix = '-BidSize'
            matched_level_sign = -1
        else:
            continue
        
        ord_size_this_fill = 0
        if irow > 0 and current_hedge_event['CLORDID'] == df_h_cleaned_sorted.iloc[irow-1]['CLORDID'] \
           and 'CUMQTY' in current_hedge_event and 'CUMQTY' in df_h_cleaned_sorted.iloc[irow-1]:
            ord_size_this_fill = current_hedge_event['CUMQTY'] - df_h_cleaned_sorted.iloc[irow-1]['CUMQTY']
        elif 'CUMQTY' in current_hedge_event and current_hedge_event['CUMQTY'] > 0:
            ord_size_this_fill = current_hedge_event['CUMQTY']
        
        if ord_size_this_fill <= 0 or pd.isna(ord_size_this_fill):
            continue

        exec_price = current_hedge_event['EXEC_PRICE']
        if pd.isna(exec_price):
            continue

        match_found_details = None
        match_type = None # To track 'exact' or 'fuzzy'

        # --- Pass 1: Try Exact Price Match ---
        for lob_level_num in range(1, 11):
            lob_price_col = f'L{lob_level_num}{lob_price_suffix}'
            lob_size_col = f'L{lob_level_num}{lob_size_suffix}'

            if lob_price_col not in lob_window_df.columns or lob_size_col not in lob_window_df.columns:
                continue

            for lob_original_idx, matched_lob_row in lob_window_df.iterrows():
                if pd.notna(matched_lob_row[lob_price_col]) and \
                   abs(matched_lob_row[lob_price_col] - exec_price) < 1e-9: # Exact match
                    
                    if pd.notna(matched_lob_row[lob_size_col]) and \
                       ord_size_this_fill <= matched_lob_row[lob_size_col]:
                        
                        match_details_dict = matched_lob_row.to_dict()
                        match_details_dict['original_lob_index'] = lob_original_idx
                        match_details_dict['matched_hedge_clordid'] = current_hedge_event['CLORDID']
                        match_details_dict['matched_hedge_exec_price'] = exec_price
                        match_details_dict['matched_hedge_side'] = current_hedge_event['SIDE']
                        match_details_dict['matched_hedge_exectype'] = current_hedge_event.get('EXECTYPE', np.nan)
                        match_details_dict['matched_hedge_time'] = current_hedge_event['hedge_time']
                        match_details_dict['matched_hedge_this_fill_size'] = ord_size_this_fill
                        match_details_dict['matched_lob_level_interacted'] = matched_level_sign * lob_level_num
                        match_details_dict['match_type'] = 'exact'
                        
                        match_found_details = match_details_dict
                        match_type = 'exact'
                        break 
            if match_found_details:
                break
        
        # --- Pass 2: Try Fuzzy Price Match (if no exact match found) ---
        if not match_found_details:
            for lob_level_num in range(1, 11):
                lob_price_col = f'L{lob_level_num}{lob_price_suffix}'
                lob_size_col = f'L{lob_level_num}{lob_size_suffix}'

                if lob_price_col not in lob_window_df.columns or lob_size_col not in lob_window_df.columns:
                    continue

                for lob_original_idx, matched_lob_row in lob_window_df.iterrows():
                    lob_level_price = matched_lob_row[lob_price_col]
                    if pd.notna(lob_level_price) and \
                       abs(lob_level_price - exec_price) <= tick_size + 1e-9: # Fuzzy match (within tick size)
                        
                        if pd.notna(matched_lob_row[lob_size_col]) and \
                           ord_size_this_fill <= matched_lob_row[lob_size_col]:
                            
                            match_details_dict = matched_lob_row.to_dict()
                            match_details_dict['original_lob_index'] = lob_original_idx
                            match_details_dict['matched_hedge_clordid'] = current_hedge_event['CLORDID']
                            # Store both actual exec price and the LOB price it matched to if fuzzy
                            match_details_dict['matched_hedge_exec_price'] = exec_price 
                            match_details_dict['matched_lob_price_at_level'] = lob_level_price 
                            match_details_dict['matched_hedge_side'] = current_hedge_event['SIDE']
                            match_details_dict['matched_hedge_exectype'] = current_hedge_event.get('EXECTYPE', np.nan)
                            match_details_dict['matched_hedge_time'] = current_hedge_event['hedge_time']
                            match_details_dict['matched_hedge_this_fill_size'] = ord_size_this_fill
                            match_details_dict['matched_lob_level_interacted'] = matched_level_sign * lob_level_num
                            match_details_dict['match_type'] = 'fuzzy'

                            match_found_details = match_details_dict
                            match_type = 'fuzzy'
                            break
                if match_found_details:
                    break
        
        # Add to list if a match was found (either exact or fuzzy)
        if match_found_details:
            matched_data_list.append(match_found_details)
            if match_type == 'exact':
                num_exact_price_matches +=1
            elif match_type == 'fuzzy':
                num_fuzzy_price_matches +=1

    if not matched_data_list:
        print("No matches found between hedge events and LOB snapshots.")
        return pd.DataFrame()

    df_matched_result = pd.DataFrame(matched_data_list)
    
    num_matched_fill_events = len(df_matched_result) # Now each row is a distinct fill event matched
    
    print(f"\n--- Matching Engine Report ---")
    print(f"Attempted to match {num_input_hedge_fill_events} hedge fill execution events.")
    print(f"Total matched fill events: {num_matched_fill_events}")
    print(f"  - Exact price matches: {num_exact_price_matches}")
    print(f"  - Fuzzy price matches (within +/- {tick_size:.4f}): {num_fuzzy_price_matches}")

    if num_input_hedge_fill_events > 0:
        perc_fills_matched = (num_matched_fill_events / num_input_hedge_fill_events) * 100
        print(f"Percentage of hedge fill events matched: {perc_fills_matched:.2f}%")
    else:
        print("No hedge fill events to process for percentage calculation.")
    print("--- End Report ---")

    return df_matched_result
