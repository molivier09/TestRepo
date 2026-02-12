# Spike Classification Setup Guide

## Prerequisites

You'll need:
1. **OpenAI API Key** - Get from https://platform.openai.com/api-keys
2. **Finnhub API Key** - Get FREE from https://finnhub.io/register

## Step 1: Install Required Packages

```powershell
pip install openai finnhub-python pandas openpyxl
```

Or use the requirements file:
```powershell
pip install -r requirements_classification.txt
```

## Step 2: Set Up API Keys

### Get Your API Keys:

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)

**Finnhub (FREE):**
1. Go to https://finnhub.io/register
2. Sign up for free account
3. Copy your API key from the dashboard

### Set Environment Variables:

In PowerShell:
```powershell
$env:OPENAI_API_KEY='sk-your-openai-key-here'
$env:FINNHUB_API_KEY='your-finnhub-key-here'
```

**Note:** These environment variables only last for the current PowerShell session. You'll need to set them again if you close the terminal.

To make them permanent (optional):
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-key', 'User')
[System.Environment]::SetEnvironmentVariable('FINNHUB_API_KEY', 'your-key', 'User')
```

## Step 3: Run the Sample Classification

```powershell
cd "c:\Users\maxim\OneDrive\Desktop\EP New"
python classify_spikes_sample.py
```

## What the Script Does

1. **Loads 100 random spike events** from your CSV
2. **Fetches news headlines** for each event (±1 day from spike date)
3. **Uses OpenAI GPT-4o-mini** to classify the catalyst type
4. **Saves results to Excel** with 3 sheets:
   - Classified Spikes (all 100 events with categories)
   - Category Summary (count and stats by category)
   - High Confidence (classifications with confidence ≥0.7)

## Expected Output

```
[1/100] AAPL on 2023-11-02
  Fetching news...
  Found 3 headlines
  Classifying with OpenAI...
  → Category: Earnings Beat
  → Confidence: 0.95

[2/100] TSLA on 2024-01-15
  Fetching news...
  Found 5 headlines
  Classifying with OpenAI...
  → Category: Product Launch/Approval
  → Confidence: 0.88
...
```

## Cost Estimate

- **Sample (100 events):** ~$0.03
- **Full dataset (11,000 events):** ~$3.00

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
- Make sure you set the environment variable in the same PowerShell window
- Check with: `echo $env:OPENAI_API_KEY`

### "FINNHUB_API_KEY environment variable not set"
- Same as above
- Check with: `echo $env:FINNHUB_API_KEY`

### "Rate limit exceeded"
- Finnhub free tier: 60 requests/minute
- Script has built-in delays (1.5 sec between requests)
- Should not hit limits with current settings

### "No news found" for many events
- Normal for older events (pre-2020)
- Script will classify as "Technical Breakout" when no news available

## Next Steps After Sample

1. **Review the Excel output** (`spike_classification_sample.xlsx`)
2. **Check classification quality** - Do the categories make sense?
3. **Verify confidence scores** - Are high-confidence ones accurate?
4. **If satisfied**, run full classification on all 11,000 events

## Full Classification Script

After validating the sample, I can create a script to:
- Process all 11,000 events
- Include progress checkpoints (resume if interrupted)
- Merge with your existing performance data
- Generate comprehensive analysis by category
