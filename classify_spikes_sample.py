"""
Spike Event Classification - Sample Test (100 events)
Phase 1: Fetch news and classify using OpenAI

Requirements:
- pip install openai finnhub-python pandas openpyxl
- OPENAI_API_KEY environment variable
- FINNHUB_API_KEY environment variable (free from finnhub.io)
"""

import pandas as pd
import openai
import finnhub
import os
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_SIZE = 100
OPENAI_MODEL = "gpt-4o-mini"  # Cheap and effective

# Categories for classification
CATEGORIES = [
    "Earnings Beat",
    "Product Launch/Approval", 
    "M&A Activity",
    "Analyst Upgrade",
    "Contract Win",
    "Technical Breakout",
    "Sector Momentum",
    "Short Squeeze",
    "Regulatory News",
    "Guidance Raise",
    "Insider Buying",
    "Other/Unknown"
]

def setup_apis():
    """
    Setup API clients. Requires environment variables:
    - OPENAI_API_KEY
    - FINNHUB_API_KEY
    """
    openai_key = os.getenv('OPENAI_API_KEY')
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    
    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: $env:OPENAI_API_KEY='your-key-here'")
        return None, None
    
    if not finnhub_key:
        print("ERROR: FINNHUB_API_KEY environment variable not set")
        print("Get free key at: https://finnhub.io/register")
        print("Set it with: $env:FINNHUB_API_KEY='your-key-here'")
        return None, None
    
    # Setup OpenAI
    openai.api_key = openai_key
    
    # Setup Finnhub
    finnhub_client = finnhub.Client(api_key=finnhub_key)
    
    return openai, finnhub_client

def fetch_news_for_spike(finnhub_client, ticker, date_str):
    """
    Fetch news headlines for a spike event.
    Returns list of headline strings.
    """
    try:
        # Parse date
        spike_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get news from 3 days before (to catch earnings/catalysts) to 1 day after
        from_date = (spike_date - timedelta(days=3)).strftime('%Y-%m-%d')
        to_date = (spike_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch company news
        news = finnhub_client.company_news(ticker, _from=from_date, to=to_date)
        
        # Extract headlines (top 7)
        headlines = [item['headline'] for item in news[:7] if 'headline' in item]
        
        return headlines
        
    except Exception as e:
        print(f"  Error fetching news for {ticker}: {str(e)}")
        return []

def classify_spike_with_openai(openai_client, ticker, date, return_pct, rvol, headlines):
    """
    Use OpenAI to classify the spike event.
    Returns dict with category, confidence, and reasoning.
    """
    try:
        # Build prompt
        headlines_text = "\n".join([f"- {h}" for h in headlines]) if headlines else "- No news headers found in API"
        
        prompt = f"""Classify this stock spike event into ONE category based on the available information.

Ticker: {ticker}
Date: {date}
Return: {return_pct:.1f}%
Relative Volume (RVOL): {rvol:.1f}x

News Headlines from API:
{headlines_text}

Categories:
{chr(10).join([f'- {cat}' for cat in CATEGORIES])}

Instructions:
1. If news headlines are provided, use them to identify the catalyst.
2. If NO news headlines are provided or they are irrelevant:
   - Search your INTERNAL KNOWLEDGE for significant events for {ticker} around {date}.
   - Common catalysts include Earnings, FDA approvals, Buyouts/Mergers, or Short Squeezes.
   - If you recall a specific event, classify based on that.
   - If you cannot find a Reason/Catalyst, classify as "Technical Breakout".
3. Provide confidence score (0.0 to 1.0). If relying on internal knowledge for a well-known event (like GME in 2021), confidence can be high. If unknown, keep it low.
4. In "reasoning", explicitly state if you used provided news or internal knowledge.

Respond ONLY with valid JSON in this exact format:
{{
  "category": "Category Name",
  "confidence": 0.85,
  "reasoning": "Brief explanation (e.g. 'Internal knowledge: Apple released iPhone 15 on this date')"
}}"""

        # Call OpenAI
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at identifying stock market catalysts and recalling historical market events."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(result_text)
            return {
                'category': result.get('category', 'Other/Unknown'),
                'confidence': result.get('confidence', 0.5),
                'reasoning': result.get('reasoning', 'No reasoning provided')
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                'category': 'Other/Unknown',
                'confidence': 0.3,
                'reasoning': 'Failed to parse AI response'
            }
            
    except Exception as e:
        print(f"  Error classifying {ticker}: {str(e)}")
        return {
            'category': 'Other/Unknown',
            'confidence': 0.0,
            'reasoning': f'Error: {str(e)}'
        }

def main():
    """
    Main execution function.
    """
    print("="*80)
    print("SPIKE EVENT CLASSIFICATION - SAMPLE TEST")
    print("="*80)
    
    # Setup APIs
    print("\nSetting up API clients...")
    openai_client, finnhub_client = setup_apis()
    
    if not openai_client or not finnhub_client:
        print("\nERROR: Could not setup APIs. Please set environment variables.")
        print("\nTo set environment variables in PowerShell:")
        print("  $env:OPENAI_API_KEY='your-openai-key'")
        print("  $env:FINNHUB_API_KEY='your-finnhub-key'")
        return
    
    print("✓ APIs configured successfully")
    
    # Load spike data
    csv_path = r"historical_spikes_partial.csv"
    print(f"\nLoading spike data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total spikes: {len(df):,}")
    
    # Take random sample
    print(f"\nSelecting random sample of {SAMPLE_SIZE} events...")
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    df_sample = df_sample.reset_index(drop=True)
    
    print(f"Sample date range: {df_sample['Date'].min()} to {df_sample['Date'].max()}")
    
    # Initialize columns
    df_sample['News_Headlines'] = ''
    df_sample['Category'] = ''
    df_sample['Confidence'] = 0.0
    df_sample['Reasoning'] = ''
    
    # Process each spike
    print("\n" + "="*80)
    print("PROCESSING SAMPLE")
    print("="*80)
    
    total_cost = 0.0
    
    for idx, row in df_sample.iterrows():
        print(f"\n[{idx+1}/{SAMPLE_SIZE}] {row['Ticker']} on {row['Date']}")
        
        # Fetch news
        print("  Fetching news...")
        headlines = fetch_news_for_spike(finnhub_client, row['Ticker'], row['Date'])
        
        if headlines:
            print(f"  Found {len(headlines)} headlines")
            df_sample.at[idx, 'News_Headlines'] = ' | '.join(headlines)
        else:
            print("  No news found")
            df_sample.at[idx, 'News_Headlines'] = 'No news available'
        
        # Classify with OpenAI
        print("  Classifying with OpenAI...")
        classification = classify_spike_with_openai(
            openai_client,
            row['Ticker'],
            row['Date'],
            row['Return'] * 100,
            row['RVOL'],
            headlines
        )
        
        df_sample.at[idx, 'Category'] = classification['category']
        df_sample.at[idx, 'Confidence'] = classification['confidence']
        df_sample.at[idx, 'Reasoning'] = classification['reasoning']
        
        print(f"  → Category: {classification['category']}")
        print(f"  → Confidence: {classification['confidence']:.2f}")
        
        # Estimate cost (rough)
        # ~300 tokens per request, $0.15 per 1M input + $0.60 per 1M output
        estimated_cost = (300 / 1_000_000) * 0.75  # Average of input/output
        total_cost += estimated_cost
        
        # Rate limiting (Finnhub free tier: 60 req/min, OpenAI: generous)
        time.sleep(1.5)  # ~40 requests/min to be safe
    
    # Save results
    output_path = r"spike_classification_sample.xlsx"
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Full sample with classifications
        df_sample.to_excel(writer, sheet_name='Classified Spikes', index=False)
        
        # Category summary
        category_summary = df_sample.groupby('Category').agg({
            'Ticker': 'count',
            'Confidence': 'mean',
            'Return': 'mean'
        }).rename(columns={'Ticker': 'Count', 'Confidence': 'Avg_Confidence', 'Return': 'Avg_Spike_Return'})
        category_summary = category_summary.sort_values('Count', ascending=False)
        category_summary.to_excel(writer, sheet_name='Category Summary')
        
        # High confidence classifications
        high_conf = df_sample[df_sample['Confidence'] >= 0.7].copy()
        high_conf = high_conf.sort_values('Confidence', ascending=False)
        high_conf.to_excel(writer, sheet_name='High Confidence', index=False)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal events processed: {SAMPLE_SIZE}")
    print(f"Events with news: {(df_sample['News_Headlines'] != 'No news available').sum()}")
    print(f"Average confidence: {df_sample['Confidence'].mean():.2f}")
    print(f"High confidence (≥0.7): {(df_sample['Confidence'] >= 0.7).sum()}")
    
    print("\nCategory Distribution:")
    print("-"*80)
    for category, count in df_sample['Category'].value_counts().items():
        pct = count / SAMPLE_SIZE * 100
        avg_conf = df_sample[df_sample['Category'] == category]['Confidence'].mean()
        print(f"  {category:25s}: {count:3d} ({pct:5.1f}%) - Avg Conf: {avg_conf:.2f}")
    
    print(f"\nEstimated API cost: ${total_cost:.4f}")
    print(f"Projected cost for all {len(df):,} events: ${total_cost * len(df) / SAMPLE_SIZE:.2f}")
    
    print("\n" + "="*80)
    print("SAMPLE TEST COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the Excel file to validate classification quality")
    print("2. Check if categories make sense")
    print("3. If satisfied, run full classification on all events")

if __name__ == "__main__":
    main()
