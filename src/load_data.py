# scripts/load_data.py
import pandas as pd
import os

def load_lob_data(base_lob_path, index_name, specific_ric=None, start_date=None, end_date=None):
    """
    Loads LOB data for a given index, optionally filtered by a specific RIC and date range.

    Args:
        base_lob_path (str): Base path to the LOB data (e.g., '../data')
        index_name (str): Name of the index (e.g., 'FTSE', 'NASDAQ'). This will be used 
                          to form the directory name like 'FTSE_2024_data_parquet'.
        specific_ric (str, optional): Specific RIC to filter for (e.g., 'FFIH4'). 
                                      If None, loads all RICs found.
        start_date (str or pd.Timestamp, optional): Start date for filtering.
        end_date (str or pd.Timestamp, optional): End date for filtering.

    Returns:
        pd.DataFrame: Concatenated LOB data.
    """
    # Construct the directory path based on index_name, assuming a pattern like INDEX_2024_data_parquet
    lob_dir = os.path.join(base_lob_path, f"{index_name}_2024_data_parquet") 
    if not os.path.isdir(lob_dir):
        print(f"Error: LOB directory not found: {lob_dir}")
        return pd.DataFrame()

    print(f"Loading LOB data from: {lob_dir}")
    date_filtered_files = []
    if start_date and end_date:
        for file in os.listdir(lob_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(lob_dir, file)
                date_filtered_files.append(file_path)
        print(f"Date-filtered LOB files: {date_filtered_files}")
    else:
        for file in os.listdir(lob_dir):
            if file.endswith('.parquet'):
                date_filtered_files.append(os.path.join(lob_dir, file))

    df_list = []
    for file_path in date_filtered_files:
        try:
            df_temp = pd.read_parquet(file_path, columns=None if not specific_ric else ['Alias Underlying RIC'])
            if specific_ric:
                if 'Alias Underlying RIC' in df_temp.columns:
                    if specific_ric in df_temp['Alias Underlying RIC'].values:
                        df_temp = pd.read_parquet(file_path)
                        df_temp = df_temp[df_temp['Alias Underlying RIC'] == specific_ric]
                        if not df_temp.empty:
                            df_list.append(df_temp)
                # else: skip file
            else:
                df_temp = pd.read_parquet(file_path)
                if not df_temp.empty:
                    df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading or processing LOB file {file_path}: {e}")

    if not df_list:
        print(f"No LOB data loaded for {index_name}" + (f" and RIC {specific_ric}" if specific_ric else ""))
        return pd.DataFrame()

    df_o = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_o)} LOB records.")
    return df_o


def load_hedge_data(base_futures_path, index_family, ric_month_year, full_ric):
    """
    Loads hedge data for a specific futures contract.

    Args:
        base_futures_path (str): Base path to futures data (e.g., '../data/futures_data_local')
        index_family (str): Index family directory (e.g., 'FF' for FTSE Futures)
        ric_month_year (str): Subdirectory for the contract (e.g., 'FFIH4')
        full_ric (str): The full RIC name, also typically the filename (e.g., 'FFIH4.parquet')

    Returns:
        pd.DataFrame: Hedge data.
    """
    # Construct the file path based on the provided components
    # e.g. ../data/futures_data_local/FF/FFIH4/FFIH4.parquet
    file_path = os.path.join(base_futures_path, index_family, ric_month_year, f"{full_ric}.parquet")
    
    if not os.path.exists(file_path):
        print(f"Error: Hedge data file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        print(f"Loading hedge data from: {file_path}")
        df_h = pd.read_parquet(file_path)
        print(f"Loaded {len(df_h)} hedge records.")
        return df_h
    except Exception as e:
        print(f"Error reading hedge file {file_path}: {e}")
        return pd.DataFrame()

