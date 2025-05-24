# Bitcoin Trading Agent

This project implements a Bitcoin trading agent that uses historical data to make trading decisions.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To download historical Bitcoin data:
```bash
python download_data.py
```

This will download 5 years of Bitcoin price data and save it to `bitcoin_data.csv`.

## Data Format

The downloaded data includes the following columns:
- Open: Opening price
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Volume: Trading volume
- Dividends: Dividend payments (if any)
- Stock Splits: Stock splits (if any) 