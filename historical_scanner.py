
"""
Historical Spike Scanner (Full Market)

Scans for stock spikes from Jan 1, 2015 to present.
Criteria:
- RVOL (20-day) >= 2.5
- 7.5% <= 1-day Return <= 80%
- Price >= $5
- Market Cap >= $1B (Current Market Cap used as proxy)
- Dollar Volume >= $5M
- Volume >= 1M shares

Output: Appends to 'historical_spikes.xlsx'
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import os
import time
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = "2015-01-01"
RVOL_LOOKBACK = 20
RVOL_THRESHOLD = 2.5
MIN_RETURN = 0.075      # 7.5%
MAX_RETURN = 0.80       # 80%
MIN_PRICE = 5
MIN_MARKET_CAP = 1_000_000_000       # $1B
MIN_DOLLAR_VOL = 5_000_000           # $5M
MIN_EVENT_VOLUME = 1_000_000         # 1M shares

OUTPUT_FILE = "historical_spikes.xlsx"
PARTIAL_FILE = "historical_spikes_partial.csv"

# ============================================================================
# TICKER SOURCES
# ============================================================================
def get_all_tickers():
    """Fetch all US tickers from rreichel3 GitHub repo (approx 7000 tickers)"""
    print("Fetching full ticker list from GitHub...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        resp = requests.get(url)
        if resp.status_code == 200:
            tickers = resp.text.splitlines()
            # Clean up tickers (remove empty strings)
            tickers = [t.strip() for t in tickers if t.strip()]
            print(f"Successfully loaded {len(tickers)} tickers.")
            return tickers
        else:
            print(f"Failed to fetch tickers. Status: {resp.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching ticker list: {e}")
        return []

# ============================================================================
# SCANNER LOGIC
# ============================================================================
def process_ticker(ticker, start_date):
    """
    Downloads data for a single ticker and identifies spike events.
    Returns a DataFrame of valid spike events.
    """
    try:
        # Download data
        # We start a bit earlier to have enough data for the first RVOL calculation
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        
        if df.empty or len(df) < RVOL_LOOKBACK + 1:
            return None
            
        # Ensure we have single-level columns (yfinance sometimes returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Calculate Metrics
        
        # 1. Previous Close
        df['Prev_Close'] = df['Close'].shift(1)
        
        # 2. Daily Return
        df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
        
        # 3. Dollar Volume
        df['Dollar_Vol'] = df['Close'] * df['Volume']
        
        # 4. Rolling Average Volume (for RVOL)
        # Shifted by 1 so we don't include today's volume in the average
        df['Avg_Vol_20'] = df['Volume'].rolling(window=RVOL_LOOKBACK).mean().shift(1)
        
        # 5. RVOL
        df['RVOL'] = df['Volume'] / df['Avg_Vol_20']
        
        # ---------------------------------------------------------
        # Apply Filters
        # ---------------------------------------------------------
        # We need to drop the initial rows where rolling calc is NaN
        df = df.dropna()
        
        # Create boolean mask
        mask = (
            (df['RVOL'] >= RVOL_THRESHOLD) &
            (df['Return'] >= MIN_RETURN) &
            (df['Return'] <= MAX_RETURN) &
            (df['Close'] >= MIN_PRICE) &
            (df['Volume'] >= MIN_EVENT_VOLUME) &
            (df['Dollar_Vol'] >= MIN_DOLLAR_VOL)
        )
        
        spikes = df[mask].copy()
        
        if spikes.empty:
            return None
            
        # Add metadata
        spikes['Ticker'] = ticker
        
        # Market Cap Check (Current)
        # This is the most expensive part time-wise if we do it for every valid ticker.
        # But we only check market cap IF we found spike events.
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get('marketCap', 0)
        except:
            mcap = 0
            
        if mcap < MIN_MARKET_CAP:
            return None
            
        spikes['Market_Cap_Current'] = mcap
        
        # Format for output
        spikes = spikes.reset_index()
        result_cols = ['Date', 'Ticker', 'Close', 'Return', 'Volume', 'RVOL', 'Dollar_Vol', 'Market_Cap_Current']
        spikes = spikes[result_cols]
        
        return spikes
        
    except Exception as e:
        # print(f"Error processing {ticker}: {e}")
        return None

def main():
    print("="*60)
    print("HISTORICAL SPIKE SCANNER - FULL MARKET (~7000 Tickers)")
    print("="*60)
    
    tickers = get_all_tickers()
    
    if not tickers:
        print("No tickers found. Exiting.")
        return

    print(f"Starting scan for {len(tickers)} tickers...")
    print("This will take a significant amount of time (1-2+ hours).")
    print("Progress is saved incrementally to 'historical_spikes_partial.csv'.")
    
    # Initialize or append mode logic could be complex, sticking to fresh run for simplicity
    # defined by user request "find every single one".
    
    # Clean partial file if it exists (fresh start)
    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)
        
    total_found = 0
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        # Progress indication
        if (i+1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)
            remaining_sec = avg_time * (len(tickers) - (i+1))
            remaining_min = remaining_sec / 60
            print(f"Processed {i+1}/{len(tickers)} ({((i+1)/len(tickers))*100:.1f}%) - Found {total_found} events. Est. remaining: {remaining_min:.0f} min")
            
        # Small delay to keep API happy
        # time.sleep(0.05) 
        
        result = process_ticker(ticker, START_DATE)
        
        if result is not None:
            total_found += len(result)
            
            # Append to partial CSV immediately
            # If file doesn't exist, write header. If it does, skip header.
            header = not os.path.exists(PARTIAL_FILE)
            result.to_csv(PARTIAL_FILE, mode='a', header=header, index=False)
            
    # Finalize
    print("\nScan Complete!")
    
    if os.path.exists(PARTIAL_FILE):
        print("Converting partial CSV to final Excel...")
        full_df = pd.read_csv(PARTIAL_FILE)
        # Sort by date
        if 'Date' in full_df.columns:
            full_df = full_df.sort_values(by='Date')
            
        full_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Saved {len(full_df)} events to {OUTPUT_FILE}")
        
        # Cleanup partial
        # os.remove(PARTIAL_FILE) # Keep it just in case until user confirms
    else:
        print("No events found.")

if __name__ == "__main__":
    main()
