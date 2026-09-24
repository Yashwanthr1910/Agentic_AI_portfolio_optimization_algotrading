import os
import time
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

CONSTITUENTS_FILE = "data/raw/nifty500_constituents.csv"

OUTPUT_DIR = "data/raw/daily"

MASTER_OUTPUT = "data/processed/nifty500_daily.parquet"

START_DATE = "2015-01-01"

# yfinance end date is exclusive
END_DATE = "2026-01-01"

INTERVAL = "1d"

BATCH_SIZE = 25


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ============================================================
# LOAD NIFTY 500 SYMBOLS
# ============================================================

def load_nifty500_symbols():

    df = pd.read_csv(
    CONSTITUENTS_FILE,
    sep="\t"
    )

    print("\nAvailable columns:")
    print(df.columns.tolist())

    # Find symbol column
    symbol_column = None

    possible_columns = [
        "Symbol",
        "SYMBOL",
        "symbol"
    ]

    for column in possible_columns:

        if column in df.columns:
            symbol_column = column
            break

    if symbol_column is None:

        raise ValueError(
            "Symbol column not found in nifty500_constituents.csv"
        )

    symbols = (
        df[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(f"\nTotal symbols found: {len(symbols)}")

    return symbols


# ============================================================
# CONVERT NSE SYMBOLS TO YAHOO FINANCE SYMBOLS
# ============================================================

def convert_to_yahoo_symbols(symbols):

    yahoo_symbols = []

    for symbol in symbols:

        yahoo_symbol = f"{symbol}.NS"

        yahoo_symbols.append(yahoo_symbol)

    return yahoo_symbols


# ============================================================
# SPLIT LIST INTO BATCHES
# ============================================================

def create_batches(symbols, batch_size):

    for i in range(0, len(symbols), batch_size):

        yield symbols[i:i + batch_size]


# ============================================================
# DOWNLOAD BATCH
# ============================================================

def download_batch(batch):

    print("\nDownloading batch:")

    print(batch)

    data = yf.download(
        tickers=batch,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False
    )

    return data


# ============================================================
# CONVERT MULTI-TICKER DATA TO LONG FORMAT
# ============================================================

def convert_to_long_format(data, batch):

    all_stock_data = []

    for ticker in batch:

        try:

            if ticker not in data.columns.get_level_values(0):

                print(f"No data found for {ticker}")

                continue

            stock_data = data[ticker].copy()

            stock_data = stock_data.dropna(how="all")

            if stock_data.empty:

                print(f"Empty data for {ticker}")

                continue

            stock_data = stock_data.reset_index()

            stock_data["Symbol"] = ticker.replace(".NS", "")

            all_stock_data.append(stock_data)

        except Exception as e:

            print(f"Error processing {ticker}: {e}")

    if not all_stock_data:

        return pd.DataFrame()

    combined_data = pd.concat(
        all_stock_data,
        ignore_index=True
    )

    return combined_data


# ============================================================
# SAVE INDIVIDUAL STOCK FILES
# ============================================================

def save_individual_files(data):

    for symbol, group in data.groupby("Symbol"):

        file_path = os.path.join(
            OUTPUT_DIR,
            f"{symbol}.csv"
        )

        group.to_csv(
            file_path,
            index=False
        )


# ============================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================

def download_nifty500_data():

    # Load NSE symbols
    symbols = load_nifty500_symbols()

    # Convert to Yahoo Finance symbols
    yahoo_symbols = convert_to_yahoo_symbols(symbols)

    print(f"\nDownloading data for {len(yahoo_symbols)} stocks")

    all_data = []

    batches = list(
        create_batches(
            yahoo_symbols,
            BATCH_SIZE
        )
    )

    print(f"Total batches: {len(batches)}")

    # Download each batch
    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"Downloading batch "
            f"{batch_number}/{len(batches)}"
        )

        print("=" * 60)

        try:

            data = download_batch(batch)

            if data.empty:

                print("Batch returned no data")

                continue

            long_data = convert_to_long_format(
                data,
                batch
            )

            if not long_data.empty:

                # Save individual stock CSVs
                save_individual_files(
                    long_data
                )

                all_data.append(
                    long_data
                )

            time.sleep(2)

        except Exception as e:

            print(
                f"Batch {batch_number} failed: {e}"
            )

            continue

    # ========================================================
    # COMBINE ALL DATA
    # ========================================================

    if not all_data:

        print("\nNo data downloaded.")

        return

    master_data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Sort
    master_data = master_data.sort_values(
        by=["Symbol", "Date"]
    )

    # Remove duplicates
    master_data = master_data.drop_duplicates(
        subset=["Symbol", "Date"]
    )

    # Save Parquet
    master_data.to_parquet(
        MASTER_OUTPUT,
        index=False
    )

    # Optional CSV
    master_csv = (
        "data/processed/"
        "nifty500_daily.csv"
    )

    master_data.to_csv(
        master_csv,
        index=False
    )

    print("\n" + "=" * 60)

    print("DOWNLOAD COMPLETE")

    print("=" * 60)

    print(
        f"Total rows: {len(master_data)}"
    )

    print(
        f"Unique stocks: "
        f"{master_data['Symbol'].nunique()}"
    )

    print(
        f"Date range: "
        f"{master_data['Date'].min()} "
        f"to "
        f"{master_data['Date'].max()}"
    )

    print(
        f"\nParquet saved to:"
        f"\n{MASTER_OUTPUT}"
    )

    print(
        f"\nCSV saved to:"
        f"\n{master_csv}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    download_nifty500_data()