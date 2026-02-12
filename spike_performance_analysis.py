"""
Post-Spike Performance Analysis
Analyzes how spike events perform over 1, 5, 10, 20, and 100 trading days after the spike.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Time horizons to analyze (in trading days)
TIME_HORIZONS = [1, 5, 10, 20, 100]

def get_post_spike_returns(ticker, spike_date, horizons=TIME_HORIZONS, max_days_to_fetch=150):
    """
    Calculate returns for various time horizons after a spike event.
    
    Args:
        ticker: Stock ticker symbol
        spike_date: Date of the spike event
        horizons: List of trading days to calculate returns for
        max_days_to_fetch: Maximum calendar days to fetch (to ensure we get enough trading days)
    
    Returns:
        dict: Returns for each time horizon
    """
    try:
        spike_date = pd.to_datetime(spike_date)
        
        # Fetch data from spike date to max_days_to_fetch days after
        start_date = spike_date
        end_date = spike_date + timedelta(days=max_days_to_fetch)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty or len(hist) < 2:
            return {h: None for h in horizons}
        
        # Find the spike date in the data (or closest trading day)
        spike_idx = None
        spike_date_normalized = spike_date.normalize()
        
        if spike_date_normalized in hist.index:
            spike_idx = hist.index.get_loc(spike_date_normalized)
        else:
            # Find closest trading day
            closest_idx = hist.index.get_indexer([spike_date_normalized], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(hist):
                spike_idx = closest_idx
        
        if spike_idx is None or spike_idx >= len(hist) - 1:
            return {h: None for h in horizons}
        
        # Get the closing price on the spike day
        spike_close = hist.iloc[spike_idx]['Close']
        
        # Calculate returns for each horizon
        returns = {}
        for horizon in horizons:
            target_idx = spike_idx + horizon
            
            if target_idx < len(hist):
                future_close = hist.iloc[target_idx]['Close']
                returns[horizon] = (future_close - spike_close) / spike_close
            else:
                returns[horizon] = None
        
        return returns
    
    except Exception as e:
        print(f"Error processing {ticker} on {spike_date}: {str(e)}")
        return {h: None for h in horizons}

def analyze_spike_performance(csv_path, output_path=None, sample_size=None):
    """
    Analyze post-spike performance for all events in the CSV.
    
    Args:
        csv_path: Path to the historical spikes CSV
        output_path: Path to save the results (optional)
        sample_size: Number of events to analyze (None = all)
    
    Returns:
        DataFrame: Results with post-spike returns for each time horizon
    """
    print("Loading spike events...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Total spike events: {len(df):,}")
    
    if sample_size and sample_size < len(df):
        print(f"Analyzing random sample of {sample_size:,} events...")
        df = df.sample(n=sample_size, random_state=42).copy()
    else:
        print(f"Analyzing all {len(df):,} events...")
    
    # Sort by date for better progress tracking
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize columns for each time horizon
    for horizon in TIME_HORIZONS:
        df[f'Return_{horizon}D'] = None
    
    # Process each spike event
    print("\nFetching post-spike performance data...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing spikes"):
        returns = get_post_spike_returns(row['Ticker'], row['Date'])
        
        for horizon in TIME_HORIZONS:
            df.at[idx, f'Return_{horizon}D'] = returns[horizon]
    
    # Convert return columns to numeric
    for horizon in TIME_HORIZONS:
        df[f'Return_{horizon}D'] = pd.to_numeric(df[f'Return_{horizon}D'], errors='coerce')
    
    # Save results if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return df

def generate_summary_report(df):
    """
    Generate a comprehensive summary report of post-spike performance.
    
    Args:
        df: DataFrame with post-spike returns
    """
    print("\n" + "="*80)
    print("POST-SPIKE PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nTotal spike events analyzed: {len(df):,}")
    
    # Summary statistics for each time horizon
    print("\n" + "="*80)
    print("PERFORMANCE BY TIME HORIZON")
    print("="*80)
    
    summary_data = []
    
    for horizon in TIME_HORIZONS:
        col = f'Return_{horizon}D'
        valid_returns = df[col].dropna()
        
        if len(valid_returns) == 0:
            continue
        
        # Calculate statistics
        mean_return = valid_returns.mean()
        median_return = valid_returns.median()
        std_return = valid_returns.std()
        min_return = valid_returns.min()
        max_return = valid_returns.max()
        
        # Win rate (positive returns)
        win_rate = (valid_returns > 0).sum() / len(valid_returns)
        
        # Percentage of data available
        data_availability = len(valid_returns) / len(df)
        
        summary_data.append({
            'Horizon': f'{horizon}D',
            'Count': len(valid_returns),
            'Availability': f'{data_availability:.1%}',
            'Mean': mean_return,
            'Median': median_return,
            'Std Dev': std_return,
            'Min': min_return,
            'Max': max_return,
            'Win Rate': win_rate
        })
        
        print(f"\n{horizon}-Day Performance:")
        print(f"  Data points:     {len(valid_returns):,} ({data_availability:.1%} of total)")
        print(f"  Mean return:     {mean_return:>8.2%}")
        print(f"  Median return:   {median_return:>8.2%}")
        print(f"  Std deviation:   {std_return:>8.2%}")
        print(f"  Min return:      {min_return:>8.2%}")
        print(f"  Max return:      {max_return:>8.2%}")
        print(f"  Win rate:        {win_rate:>8.1%} (% positive returns)")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Identify optimal holding period
    print("\n" + "="*80)
    print("OPTIMAL HOLDING PERIOD ANALYSIS")
    print("="*80)
    
    best_mean = summary_df.loc[summary_df['Mean'].idxmax()]
    best_median = summary_df.loc[summary_df['Median'].idxmax()]
    best_winrate = summary_df.loc[summary_df['Win Rate'].idxmax()]
    
    print(f"\nBest by Mean Return:      {best_mean['Horizon']} ({best_mean['Mean']:.2%})")
    print(f"Best by Median Return:    {best_median['Horizon']} ({best_median['Median']:.2%})")
    print(f"Best by Win Rate:         {best_winrate['Horizon']} ({best_winrate['Win Rate']:.1%})")
    
    # Risk-adjusted returns (Sharpe-like ratio)
    summary_df['Risk_Adj_Return'] = summary_df['Mean'] / summary_df['Std Dev']
    best_risk_adj = summary_df.loc[summary_df['Risk_Adj_Return'].idxmax()]
    print(f"Best Risk-Adjusted:       {best_risk_adj['Horizon']} (ratio: {best_risk_adj['Risk_Adj_Return']:.3f})")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("RETURN DISTRIBUTION BY HORIZON")
    print("="*80)
    
    for horizon in TIME_HORIZONS:
        col = f'Return_{horizon}D'
        valid_returns = df[col].dropna()
        
        if len(valid_returns) == 0:
            continue
        
        print(f"\n{horizon}-Day Returns Distribution:")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        print("  Percentiles:")
        for p in percentiles:
            val = valid_returns.quantile(p/100)
            print(f"    {p}th: {val:>8.2%}")
        
        # Return buckets
        print("  Return Buckets:")
        buckets = [
            (float('-inf'), -0.20, "< -20%"),
            (-0.20, -0.10, "-20% to -10%"),
            (-0.10, -0.05, "-10% to -5%"),
            (-0.05, 0.00, "-5% to 0%"),
            (0.00, 0.05, "0% to 5%"),
            (0.05, 0.10, "5% to 10%"),
            (0.10, 0.20, "10% to 20%"),
            (0.20, float('inf'), "> 20%")
        ]
        
        for low, high, label in buckets:
            count = ((valid_returns > low) & (valid_returns <= high)).sum()
            pct = count / len(valid_returns)
            print(f"    {label:>15}: {count:>5} ({pct:>6.1%})")
    
    # Year-over-year analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY YEAR")
    print("="*80)
    
    df['Year'] = df['Date'].dt.year
    
    for horizon in TIME_HORIZONS:
        col = f'Return_{horizon}D'
        print(f"\n{horizon}-Day Returns by Year:")
        
        yearly_stats = df.groupby('Year')[col].agg(['count', 'mean', 'median'])
        yearly_stats = yearly_stats[yearly_stats['count'] > 0]
        
        for year, row in yearly_stats.iterrows():
            if row['count'] > 0:
                print(f"  {year}: Mean={row['mean']:>7.2%}, Median={row['median']:>7.2%}, N={int(row['count']):>4}")
    
    # Top and bottom performers
    print("\n" + "="*80)
    print("TOP 10 BEST PERFORMING SPIKES (20-Day Returns)")
    print("="*80)
    
    top_20d = df.nlargest(10, 'Return_20D')[['Date', 'Ticker', 'Return', 'Return_20D']]
    for idx, row in top_20d.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')} {row['Ticker']:>6}: Spike={row['Return']:>7.2%}, 20D={row['Return_20D']:>7.2%}")
    
    print("\n" + "="*80)
    print("TOP 10 WORST PERFORMING SPIKES (20-Day Returns)")
    print("="*80)
    
    bottom_20d = df.nsmallest(10, 'Return_20D')[['Date', 'Ticker', 'Return', 'Return_20D']]
    for idx, row in bottom_20d.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')} {row['Ticker']:>6}: Spike={row['Return']:>7.2%}, 20D={row['Return_20D']:>7.2%}")
    
    return summary_df

def main():
    """
    Main execution function.
    """
    csv_path = r"historical_spikes_partial.csv"
    output_path = r"spike_performance_results.csv"
    
    print("="*80)
    print("SPIKE PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing post-spike returns for time horizons: {TIME_HORIZONS}")
    print("\nThis will fetch historical data from Yahoo Finance for each spike event.")
    print("This may take several minutes depending on the number of events...\n")
    
    # Ask user if they want to analyze all or a sample
    response = input("Analyze all events or a sample? (all/sample): ").strip().lower()
    
    if response == 'sample':
        sample_size = int(input("Enter sample size: ").strip())
    else:
        sample_size = None
    
    # Run the analysis
    results_df = analyze_spike_performance(csv_path, output_path, sample_size)
    
    # Generate summary report
    summary_df = generate_summary_report(results_df)
    
    # Save summary
    summary_path = r"spike_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. {output_path} - Full results with all returns")
    print(f"  2. {summary_path} - Summary statistics by time horizon")

if __name__ == "__main__":
    main()
