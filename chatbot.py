# chatbot_optimized.py
# Financial Chatbot - Optimized with SIMPLE Risk Score, Multi-Source News Sentiment + ETS Forecasting
# (Removed pandas-ta dependency and Technical Strategy Scanning)
# (Now uses Polygon.io data for manual momentum indicator calculations and trading signals)
# (Simplified Risk Score calculation - removed dependency on Riskfolio-Lib, reduced factors)


# WARNING: Hardcoding API keys directly in the script is a SIGNIFICANT SECURITY RISK.
#          It is STRONGLY RECOMMENDED to use Environment Variables or Streamlit Secrets (`secrets.toml`)
#          to manage your API keys securely.
#          This code includes the keys as requested, but this is NOT best practice.

import os
import re
import time
import io
import traceback
import logging
from datetime import datetime, timedelta, date, timezone
from itertools import product
import requests
import json
from collections import Counter
import sys
import warnings
# --- Added imports for Rate Limiting ---
from functools import wraps
import requests.exceptions


import openai  # ✅ MAKE SURE THIS LINE IS PRESENT!

# --- Data Science & Math ---
import pandas as pd
import numpy as np
# Removed scipy.special.binom as it's not used in the simplified score
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing # For ETS
# Removed scipy.stats as it's not used in the simplified score
# Removed sklearn.covariance as it's not used in the simplified score
from sklearn.metrics import mean_squared_error # For ETS evaluation
# Removed numpy.linalg.inv as it's not used in the simplified score

# --- Financial Data APIs ---
import yfinance as yf

# --- Technical Analysis (Manual Implementation) ---
# Removed: import pandas_ta as ta - Manual implementation below

# --- AI & Streamlit ---
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError # Explicitly import AuthenticationError

# --- NEWS SENTIMENT LIBS ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser # For RSS feeds
from dateutil import parser as date_parser # For flexible date parsing

# --- Configure Logging and Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # For ETS function logging

warnings.filterwarnings("ignore", module="statsmodels") # General statsmodels warnings
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="Non-stationary starting parameters found")
warnings.filterwarnings("ignore", message="invalid value encountered in divide") # Handle potential div by zero in manual TA
warnings.filterwarnings("ignore", message="invalid value encountered in subtract") # Handle potential invalid op in manual TA
warnings.filterwarnings("ignore", message="Could not find frequency for") # yfinance history index sometimes lacks freq
# Suppress UserWarning from pandas when doing EWMA on Series (often happens in MACD)
warnings.filterwarnings("ignore", message="The behavior of Series.ewm now depends on the Series dtype")


# --- Constants, Weights, Ranges (UPDATED for Simplified Risk Score) ---
# Simplified Weights for the new factors
DEFAULT_WEIGHTS = {
    'volatility': 0.25,       # Higher volatility = Higher Risk
    'market_cap': 0.25,       # Lower market cap = Higher Risk (inverted score)
    'liquidity': 0.20,        # Lower volume = Higher Risk (inverted score)
    'beta': 0.15,             # Higher beta = Higher Risk
    'price_vs_sma': 0.15,     # Larger absolute deviation = Higher Risk
    # Removed: semi_deviation, piotroski, vix, cvar, cdar, gmd
}
_weight_sum = sum(DEFAULT_WEIGHTS.values())
# Ensure weights sum to 1 even with simplification
if abs(_weight_sum - 1.0) > 1e-6:
    DEFAULT_WEIGHTS = {k: v / _weight_sum for k, v in DEFAULT_WEIGHTS.items()}
    logging.info(f"Normalized simplified weights: {DEFAULT_WEIGHTS}")


# Simplified Ranges for the new factors
VOLATILITY_RANGE = (0.05, 0.80) # Adjusted range, annualized std dev (e.g. 5% to 80%)
# Market Cap & Volume ranges remain, used with log scale and inverted score
MARKET_CAP_RANGE_LOG = (np.log10(50e6), np.log10(500e9)) # $50M to $500B
VOLUME_RANGE_LOG = (np.log10(10000), np.log10(5e6)) # 10k shares to 5M shares
BETA_RANGE = (0.5, 2.5) # Remains the same
PRICE_VS_SMA_RANGE = (-0.30, 0.30) # Absolute deviation, e.g., -30% to +30%. We use abs(deviation) for scoring range (0, 0.30)
VIX_RANGE = (10.0, 40.0) # Market VIX range (e.g. 10 to 40). Factor included separately now.

# Cache durations remain the same
VIX_CACHE_DURATION_SECONDS = 3600
HISTORY_CACHE_DURATION_SECONDS = 300 # Cache unified history for 5 mins
NEWS_SENTIMENT_CACHE_DURATION_SECONDS = 1800 # Cache combined news sentiment for 30 mins
ETS_FORECAST_CACHE_DURATION_SECONDS = 1800 # Cache ETS forecast results for 30 mins
POLYGON_HISTORY_CACHE_DURATION_SECONDS = 300 # Cache Polygon history for 5 mins

# --- Load API Keys from Streamlit Secrets ---
# Ensure .streamlit/secrets.toml exists and contains these keys:
# OPENAI_API_KEY="sk-..."
# NEWS_API_KEY="YOUR_NEWSAPI_KEY"
# FMP_API_KEY="YOUR_FMP_KEY"
# POLYGON_API_KEY="YOUR_POLYGON_KEY"

# Attempt to load keys from secrets.toml. If running locally and not using `streamlit run`,
# you might need to set these as environment variables or load differently.
# For Streamlit Cloud deployment, secrets.toml is the standard.
try:
    logging.info("Attempting to load API keys from secrets.toml...")
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    FMP_API_KEY = st.secrets["FMP_API_KEY"]
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
    logging.info("Successfully attempted loading API keys from secrets.toml.")
except FileNotFoundError:
    logging.error("secrets.toml not found. Attempting to load from environment variables.")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
except Exception as e:
    logging.error(f"Error loading API keys from secrets.toml: {e}. Attempting to load from environment variables.")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
    FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")


# --- NEWS SENTIMENT Configuration ---
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'
FMP_ENDPOINT = 'https://financialmodelingprep.com/api'
RSS_FEEDS = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US',
    'https://www.reuters.com/pf/reuters/us/rss/technologySector',
    'http://feeds.marketwatch.com/marketwatch/topstories/',
    'https://www.cnbc.com/id/19854910/device/rss/rss.html',
    'https://seekingalpha.com/feed.xml'
]
NEWS_DAYS_BACK = 7
NEWS_PAGE_SIZE = 30
NEWS_FMP_LIMIT = 30
NEWS_MAX_ARTICLES_DISPLAY = 50

# --- Sentiment Analyzer Global Instance ---
vader_analyzer = SentimentIntensityAnalyzer()

# --- Streamlit Page Config ---
st.set_page_config(page_title="📈 Financial Chat + Risk + News + Forecasting", layout="wide", initial_sidebar_state="expanded")

# --- API Key Validation ---
missing_or_invalid_keys = []
# Basic check: is the key non-empty and *looks* roughly like the expected format?
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    missing_or_invalid_keys.append("OpenAI API Key")
if not NEWS_API_KEY or len(NEWS_API_KEY) < 20: # NewsAPI keys are typically 32 chars
    missing_or_invalid_keys.append("NewsAPI Key")
if not FMP_API_KEY or len(FMP_API_KEY) < 20: # FMP keys vary, but 20+ is a rough heuristic
    missing_or_invalid_keys.append("Financial Modeling Prep (FMP) API Key")
if not POLYGON_API_KEY or len(POLYGON_API_KEY) < 20: # Polygon keys vary, but 20+ is a rough heuristic
    missing_or_invalid_keys.append("Polygon.io API Key")


if missing_or_invalid_keys:
    keys_str = ', '.join(missing_or_invalid_keys)
    st.error(f"❌ Invalid or missing API keys: {keys_str}. Please check the `.streamlit/secrets.toml` file or environment variables.")
    logging.error(f"API keys invalid or missing: {keys_str}")
    st.stop()
else:
    logging.info("✅ All required API Keys seem to be loaded successfully.")


# --- Riskfolio-Lib Import/Handling ---
# Remove all Riskfolio imports and handling
# try:
#     import riskfolio.src.AuxFunctions as af
#     import riskfolio.src.DBHT as db
#     import riskfolio.src.GerberStatistic as gs
#     import riskfolio.src.RiskFunctions as rk
#     import riskfolio.src.OwaWeights as owa
#     RISKFOLIO_AVAILABLE = True
#     logging.info("Riskfolio-Lib found and imported successfully.")
#     # No specific Riskfolio functions are required for the simplified score
#     # The fallback functions below will be used regardless.
# except ImportError:
logging.info("Riskfolio-Lib is not used in this simplified risk score version.")
RISKFOLIO_AVAILABLE = False # Explicitly set to False

# --- Define Fallback Functions (Only needed ones kept) ---
# Keep necessary helper functions that were previously fallbacks or used internally
def _calculate_semi_deviation(X):
    """Calculates Semi-Deviation (downside standard deviation)."""
    a = np.array(X, ndmin=1).flatten();
    if len(a) < 2: return np.nan
    mu = np.mean(a); diff = a - mu; downside_diff = diff[diff < 0];
    if len(downside_diff) < 1: return 0.0 # Or NaN? Riskfolio returns 0 if no downside
    variance = np.sum(downside_diff**2) / max(1, len(a) - 1); return np.sqrt(variance)

def _calculate_mdd(prices: pd.Series):
    """Calculates Maximum Drawdown."""
    if prices is None or prices.empty or len(prices) < 2:
        logging.debug("MDD calculation: Insufficient data.")
        return np.nan
    try:
        # Ensure prices are positive to avoid division by zero/negative NAV calculation
        if (prices <= 0).any():
             logging.warning("MDD calculation: Prices contain zero or negative values, cannot calculate.")
             return np.nan
        # Convert prices to numpy array for robust calculation
        prices_arr = prices.values.astype(float)
        # Compute cumulative maximum prices
        cumulative_max = np.maximum.accumulate(prices_arr)
        # Handle case where cumulative_max can be zero (e.g., if all prices were 0 or negative, though checked above)
        # Replace zero or near-zero cumulative_max values with NaN to avoid division issues
        cumulative_max_safe = cumulative_max.copy()
        cumulative_max_safe[cumulative_max_safe < 1e-9] = np.nan # Use a small threshold
        # Calculate drawdowns
        drawdown = (prices_arr - cumulative_max) / cumulative_max_safe
        # Maximum Drawdown is the minimum (most negative) value in the drawdown series
        max_drawdown = np.nanmin(drawdown) # Use nanmin to ignore NaNs
        # MDD is typically reported as a positive percentage or a negative value.
        # Let's return the negative value.
        return min(0.0, max_drawdown) if np.isfinite(max_drawdown) else np.nan
    except Exception as e:
        logging.warning(f"MDD calculation error: {e}")
        return np.nan

def _calculate_cvar(returns: pd.Series, alpha: float = 0.01):
    """
    Calculates Conditional Value at Risk (CVaR) or Expected Shortfall for a Series of Returns.
    Returns the value as a positive number representing the potential loss percentage.
    """
    if returns is None or returns.empty or len(returns) < int(1/alpha) + 1: # Need enough points to estimate alpha quantile
        logging.debug(f"CVaR calculation: Insufficient data ({len(returns)} returns) for alpha={alpha}.")
        return np.nan

    try:
        # Convert returns to numpy array and remove NaNs
        returns_arr = returns.dropna().values.astype(float)
        if len(returns_arr) < int(1/alpha) + 1:
            logging.debug(f"CVaR calculation: Insufficient valid data ({len(returns_arr)} returns) after dropping NaNs for alpha={alpha}.")
            return np.nan

        # Sort returns in ascending order
        sorted_returns = np.sort(returns_arr)

        # Calculate VaR (Value at Risk) at the alpha quantile
        if alpha <= 0 or alpha >= 1:
             logging.warning(f"CVaR calculation: Invalid alpha value {alpha}. Must be between 0 and 1 (exclusive).")
             return np.nan

        var_level = np.quantile(sorted_returns, alpha)

        # CVaR (Expected Shortfall) is the average of returns less than or equal to VaR level
        cvar_returns = sorted_returns[sorted_returns <= var_level]

        if len(cvar_returns) == 0:
             logging.debug(f"CVaR calculation: No returns found below VaR level {var_level:.4f}.")
             return 0.0 # No returns in the worst alpha% tail

        # CVaR is the average of these worst-case returns. It's reported as a positive value (loss).
        average_loss_in_tail = np.mean(cvar_returns)

        # Return as a positive value representing the loss percentage
        return -average_loss_in_tail # Negate the average return to get a positive loss value

    except Exception as e:
        logging.warning(f"CVaR calculation error: {e}")
        return np.nan


# --- Initialize Clients ---
try: client = OpenAI(api_key=OPENAI_API_KEY); logging.info("Initialized OpenAI client.")
except AuthenticationError as e: st.error(f"OpenAI API Authentication Error: {e}. Please check your API key in secrets.toml or environment variables."); logging.error(f"OpenAI Authentication Error: {e}", exc_info=True); st.stop()
except Exception as e: st.error(f"Error initializing OpenAI client: {e}"); logging.error(f"Initialization Error: {e}", exc_info=True); st.stop()

# --- Global Settings ---
MODEL_NAME = "gpt-4o-mini"; # Using a potentially faster/cheaper model
MAX_TOKENS = 1200; TEMPERATURE = 0.5

# --- Helper Functions (Including Date Parsing) ---
def parse_date(date_string):
    if not date_string: return None
    try:
        dt = date_parser.parse(date_string)
        # Ensure timezone awareness and convert to UTC, then format
        if dt.tzinfo is None: # Assume local timezone if naive
             dt = dt.replace(tzinfo=datetime.now(timezone.utc).astimezone().tzinfo)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat().replace('+00:00', 'Z') # Use ISO 8601 format with Z
    except (ValueError, TypeError, OverflowError):
        logging.debug(f"Could not parse date: {date_string}")
        return None

# --- Rate Limiting Decorator ---
# Apply this decorator to Yahoo Finance functions that might hit rate limits
def rate_limit_retry(max_attempts=3, initial_delay=5): # Increased initial delay as suggested
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = initial_delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                    # Catch specific requests exceptions for HTTP errors
                    status_code = getattr(e.response, 'status_code', None)
                    error_message = str(e)

                    is_rate_limit = (status_code == 429 or "Too Many Requests" in error_message)

                    if is_rate_limit:
                        attempts += 1
                        if attempts == max_attempts:
                            logging.error(f"Rate limit hit for {func.__name__}, exhausting retries ({attempts}/{max_attempts}). Last error: {error_message}")
                            raise # Re-raise after max attempts
                        logging.warning(f"Rate limit hit for {func.__name__}, retrying after {delay}s (attempt {attempts}/{max_attempts}). Status: {status_code} Msg: {error_message[:100]}...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        # Add jitter (optional but good practice)
                        delay += np.random.uniform(0, initial_delay * 0.5)
                    else:
                        # Re-raise if it's not a rate limit error
                        logging.error(f"Non-rate limit error in {func.__name__} (attempt {attempts+1}/{max_attempts}): {e}")
                        raise
                except Exception as e:
                    # Catch any other unexpected exceptions
                    logging.error(f"Unexpected error in {func.__name__} (attempt {attempts+1}/{max_attempts}): {e}", exc_info=True)
                    # Decide if these should be retried. For YF, maybe not unless it's clearly transient.
                    # For now, we'll just re-raise after logging. If you want to retry *any* exception, move
                    # the retry logic outside the specific requests.exceptions catch.
                    raise
            # This part should theoretically not be reached if max_attempts is reached,
            # as the exception is re-raised inside the loop.
            logging.error(f"Reached end of retry loop for {func.__name__} unexpectedly.")
            raise Exception(f"Failed after {max_attempts} attempts due to unexpected logic.")
        return wrapper
    return decorator

# --- Decorated Yahoo Finance functions ---
# Decorate the core functions that might hit rate limits
@st.cache_data(ttl=300) # Keep caching, apply retry decorator *around* the fetch
@rate_limit_retry(max_attempts=3, initial_delay=5)
def get_stock_data_yf_retry(ticker):
    """Fetches yfinance info with retries."""
    logging.info(f"Attempting yfinance info fetch for {ticker}")
    yf_ticker = ticker.replace('$', '').upper()
    ticker_obj = yf.Ticker(yf_ticker)
    info = ticker_obj.info
    # Basic check if info is valid - yfinance can return empty dict sometimes
    if not info or not info.get('symbol'):
         # Try a history fetch as a final check if info is sparse
         logging.warning(f"Info sparse/invalid for {ticker}, attempting history check inside retry.")
         hist = ticker_obj.history(period="1d")
         if hist.empty:
              logging.warning(f"History also empty for {ticker}. Likely invalid ticker.")
              raise ValueError(f"No data found for ticker {ticker}") # Raise to indicate fetch failed
         else:
              logging.info(f"History found for {ticker}, info sparse.")
              if not info: info = {}
              if 'symbol' not in info: info['symbol'] = yf_ticker
              if 'quoteType' not in info: info['quoteType'] = 'EQUITY' # Assume EQUITY if history exists
              if info.get('currentPrice') is None and not hist.empty: info['currentPrice'] = hist['Close'].iloc[-1] # Patch price

    logging.info(f"Successful yfinance info fetch for {ticker}.")
    return info, ticker_obj # Return info and ticker object


@st.cache_data(ttl=HISTORY_CACHE_DURATION_SECONDS)
@rate_limit_retry(max_attempts=3, initial_delay=5)
def get_unified_yfinance_history_retry(ticker: str, period="3y"):
    """
    Fetches Yahoo Finance history with retries for Risk Score & ETS Forecast.
    Returns Close series for ETS and full OHLCV df for Risk Score factors.
    """
    logging.info(f"Attempting unified yfinance history fetch for {ticker}, period: {period}")
    yf_ticker_obj = yf.Ticker(ticker)
    df = yf_ticker_obj.history(period=period, interval="1d", auto_adjust=False)

    if df is None or df.empty:
         logging.warning(f"No data returned from yfinance history fetch for {ticker}.")
         # Instead of returning None, raise an error so the retry decorator can handle it
         # If retries are exhausted, the final error will be caught by the caller
         raise ValueError(f"yfinance history data is empty for ticker {ticker}")

    # Process the dataframe if fetch was successful
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
         missing_cols = [col for col in required_cols if col not in df.columns]
         logging.warning(f"Missing required columns ({missing_cols}) in unified yfinance data for {ticker}. Found: {df.columns.tolist()}.")
         if 'Close' not in df.columns:
              logging.error(f"Unified yfinance history for {ticker} missing 'Close' column.")
              raise ValueError(f"Missing 'Close' column in history for {ticker}") # Indicate fatal data issue
         available_cols = [col for col in required_cols if col in df.columns]
         df_processed = df[available_cols].copy() # Keep only available required columns
    else:
         df_processed = df[required_cols].copy()

    if isinstance(df_processed.index, pd.DatetimeIndex) and df_processed.index.tz is not None:
        try: df_processed.index = df_processed.index.tz_convert('America/New_York').tz_localize(None) # Convert and remove timezone
        except Exception as tz_err: logging.warning(f"Unified yfinance index tz conversion failed: {tz_err}."); pass
    elif not isinstance(df_processed.index, pd.DatetimeIndex): logging.warning(f"Unified yfinance index for {ticker} not DatetimeIndex.")

    logging.info(f"Successful unified yfinance history fetch for {ticker} ({len(df_processed)} rows).")
    return df_processed['Close'].copy(), df_processed # Return Close series and the potentially subsetted df


# --- Apply decorator to VIX fetch as well ---
@st.cache_data(ttl=VIX_CACHE_DURATION_SECONDS)
@rate_limit_retry(max_attempts=3, initial_delay=3) # VIX is quick, maybe fewer retries/delay
def _get_vix_close_retry():
    """Fetches VIX close price with retries."""
    logging.info("Attempting VIX (^VIX) history fetch...")
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period="5d", interval="1d")
    if vix_data is None or vix_data.empty or 'Close' not in vix_data.columns:
        logging.warning("No VIX data or Close column found.")
        raise ValueError("VIX data is empty or missing Close column") # Raise error for retry
    latest_close = vix_data['Close'].dropna().iloc[-1] if not vix_data['Close'].dropna().empty else None
    if latest_close is None:
         logging.warning("Latest VIX close price is NaN after dropna.")
         raise ValueError("Latest VIX close price is NaN") # Raise error for retry
    logging.info(f"Successful VIX fetch: {latest_close:.2f}")
    return latest_close


@st.cache_data(ttl=300)
def get_stock_data(ticker):
    """
    Fetches Yahoo Finance summary data using the retry-decorated function.
    Returns processed data dictionary or None.
    """
    try:
        info, ticker_obj = get_stock_data_yf_retry(ticker)
        if not info or not info.get('symbol'):
             logging.warning(f"get_stock_data_yf_retry returned empty/invalid info for {ticker}")
             return None # Return None if fetch ultimately failed

        quote_type = info.get('quoteType', 'N/A')
        # Warn for non-supported types
        if quote_type not in ['EQUITY', 'ETF', 'N/A', 'Undefined']:
             logging.warning(f"Ticker {ticker} is not typical EQUITY/ETF (Type: {quote_type}). Some features may be unreliable.");
             if quote_type not in ['N/A', 'Undefined']:
                 st.toast(f"⚠️ '{ticker}' not a supported stock type (Type: {quote_type}). Features may be limited.", icon="⚠️")

        num_opinions = info.get("numberOfAnalystOpinions")
        data = {
            "ticker": info.get("symbol", ticker).upper(), "companyName": info.get("shortName", info.get("longName", "N/A")), "sector": info.get("sector", "N/A"), "industry": info.get("industry", "N/A"), "quoteType": quote_type,
            "priceForDisplay": info.get("currentPrice", info.get("regularMarketPrice", info.get("regularMarketOpen", info.get("previousClose")))), "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"), "forwardPE": info.get("forwardPE"), "dividendYield": info.get("dividendYield"), "sma50": info.get("fiftyDayAverage"), "sma200": info.get("twoHundredDayAverage"),
            "beta": get_stock_beta(info), # Use helper for beta
            "dayLow": info.get("dayLow"), "dayHigh": info.get("dayHigh"), "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"), "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("volume", info.get("regularMarketVolume")), "recommendationKey": info.get("recommendationKey"), "targetMeanPrice": info.get("targetMeanPrice"),
            "targetLowPrice": info.get("targetLowPrice"), "targetHighPrice": info.get("targetHighPrice"), "numberOfAnalystOpinions": int(num_opinions) if num_opinions is not None and isinstance(num_opinions, (int, float)) and num_opinions > 0 else None,
            "website": info.get("website"), "longBusinessSummary": info.get("longBusinessSummary")
        }
        if data["priceForDisplay"] is None: logging.warning(f"Could not determine price for {ticker} from fetched info.")
        logging.info(f"Successfully processed Yahoo summary data for {data['ticker']} (Type: {data['quoteType']})"); return data
    except ValueError as ve:
         # Catch the specific ValueError raised by get_stock_data_yf_retry on final failure
         logging.warning(f"get_stock_data: Fetch ultimately failed for {ticker}: {ve}")
         st.toast(f"⚠️ Couldn't find Yahoo equity data for '{ticker}'.", icon="⚠️")
         return None
    except Exception as e:
        # Catch any other unexpected errors during processing after fetch
        logging.error(f"Error processing yfinance info for {ticker}: {e}", exc_info=True); st.toast(f"⚠️ Error processing data from Yahoo for {ticker}.", icon="❌")
        return None


@st.cache_data(ttl=HISTORY_CACHE_DURATION_SECONDS)
def get_unified_yfinance_history(ticker: str, period="3y"):
    """
    Fetches Yahoo Finance history using the retry-decorated function.
    Returns Close series for ETS and full OHLCV df for Risk Score factors.
    Handles potential errors from the decorated function.
    """
    try:
        # Call the retry-decorated function
        close_series, full_df = get_unified_yfinance_history_retry(ticker, period=period)
        # The retry function raises ValueError on ultimate failure, which will be caught here
        if full_df is None or full_df.empty:
             logging.warning(f"get_unified_yfinance_history_retry returned empty/invalid data for {ticker}. History unavailable.")
             return None, None
        return close_series, full_df
    except ValueError as ve:
        # Catch the specific ValueError raised by the retry function on final failure
        logging.warning(f"get_unified_yfinance_history: Fetch ultimately failed for {ticker}: {ve}")
        st.toast(f"⚠️ yfinance history unavailable for {ticker}.", icon="⚠️")
        return None, None
    except Exception as e:
        # Catch any other unexpected errors during processing after fetch
        logging.error(f"Error processing yfinance history for {ticker}: {e}", exc_info=True); st.toast(f"⚠️ Error processing history data for {ticker}.", icon="❌")
        return None, None


# Manual Momentum Indicator Calculations
# @st.cache_data(ttl=POLYGON_HISTORY_CACHE_DURATION_SECONDS) # Cache is handled at the fetch level
def calculate_momentum_indicators(df: pd.DataFrame):
    """
    Calculates standard momentum indicators (RSI, MACD, CCI, Stoch, Williams %R)
    manually using pandas operations on OHLCV data.
    Requires 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    """
    logging.info(f"Calculating momentum indicators for {len(df)} rows.")
    if df is None or df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logging.warning("Input DataFrame for momentum indicators is empty or missing required OHLCV columns.")
        return pd.DataFrame() # Return empty DataFrame if input is invalid

    df_ta = df.copy() # Work on a copy

    # --- RSI (14) ---
    logging.debug("Calculating RSI...")
    delta = df_ta['Close'].diff(); gain = (delta.where(delta > 0, 0)).fillna(0); loss = (-delta.where(delta < 0, 0)).fillna(0)
    # Use Wilder's smoothing for EMA-like calculation (standard for RSI)
    alpha_rsi = 1/14; avg_gain = gain.ewm(alpha=alpha_rsi, adjust=False).mean(); avg_loss = loss.ewm(alpha=alpha_rsi, adjust=False).mean()
    # Handle case where avg_loss is zero by replacing 0 with NaN before division, then handle Inf/NaN
    # Add a small epsilon to avg_loss to prevent division by zero if it's exactly 0
    rs = avg_gain / (avg_loss + 1e-9) # Add epsilon here for division
    df_ta['RSI'] = 100 - (100 / (1 + rs))
    # Handle cases where the original avg_loss was truly NaN or result is Inf/NaN
    df_ta['RSI'] = df_ta['RSI'].replace([np.inf, -np.inf], np.nan) # Replace inf/neg-inf with NaN first
    df_ta['RSI'] = df_ta['RSI'].fillna(0) # Fill initial NaNs (first 14 periods) and results of division by zero/nan

    # --- MACD (12, 26, 9) ---
    logging.debug("Calculating MACD...")
    ema_fast = df_ta['Close'].ewm(span=12, adjust=False).mean(); ema_slow = df_ta['Close'].ewm(span=26, adjust=False).mean()
    df_ta['MACD'] = ema_fast - ema_slow; df_ta['MACD_Signal'] = df_ta['MACD'].ewm(span=9, adjust=False).mean()
     # Fill initial NaNs from EWM calculations with 0
    df_ta['MACD'] = df_ta['MACD'].fillna(0)
    df_ta['MACD_Signal'] = df_ta['MACD_Signal'].fillna(0)


    # --- CCI (20) ---
    logging.debug("Calculating CCI...")
    tp = (df_ta['High'] + df_ta['Low'] + df_ta['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()

    # Mean Deviation Calculation Function - FIXED FOR NUMPY ARRAY INPUT
    def mean_deviation(series_array):
        # Ensure input is a numpy array and filter out NaNs
        valid_series_array = series_array[~np.isnan(series_array)]
        # Check if the resulting array is empty after removing NaNs
        if valid_series_array.size == 0:
            return np.nan # Cannot calculate mean deviation for empty array
        # Calculate mean deviation using the valid (non-NaN) data
        # Handle case where all values are the same (mean_deviation is 0)
        if np.allclose(valid_series_array, valid_series_array[0]): return 0.0
        return np.mean(np.abs(valid_series_array - np.mean(valid_series_array)))

    # Apply mean deviation rolling window
    # Use raw=True to pass numpy array to the aggregation function for performance
    md_tp = tp.rolling(window=20).apply(mean_deviation, raw=True)
    # Handle division by zero or zero mean deviation (add epsilon or replace with NaN)
    denominator = 0.015 * md_tp;
    # Add a small epsilon to the denominator if it's zero or near-zero, or replace with NaN for robustness
    # Using mask + isclose is robust
    denominator = denominator.mask(np.isclose(denominator, 0, atol=1e-9), np.nan) # Use np.isclose for robustness with tolerance
    df_ta['CCI'] = (tp - ma_tp) / denominator
    # Replace Inf/NaN results from division by zero/NaN inputs with 0.
    df_ta['CCI'] = df_ta['CCI'].replace([np.inf, -np.inf], np.nan).fillna(0)


    # --- Stochastic Oscillator (14, 3) ---
    logging.debug("Calculating Stochastic...")
    lowest_low = df_ta['Low'].rolling(window=14).min(); highest_high = df_ta['High'].rolling(window=14).max()
    # Handle case where highest_high == lowest_low by replacing the range with NaN or adding epsilon
    range_hl = highest_high - lowest_low;
    # Add a small epsilon to the denominator if it's zero or near-zero
    range_hl_safe = range_hl.copy(); range_hl_safe[range_hl_safe < 1e-9] = np.nan # Use NaN for robustness
    df_ta['Stoch_%K'] = 100 * ((df_ta['Close'] - lowest_low) / range_hl_safe) # Use safe denominator
    # Replace Inf/NaN results with NaN first, then fill NaNs with 0.
    df_ta['Stoch_%K'] = df_ta['Stoch_%K'].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Calculate %D (3-day SMA of %K), fill initial NaNs with 0.
    df_ta['Stoch_%D'] = df_ta['Stoch_%K'].rolling(window=3).mean().fillna(0)


    # --- Williams %R (14) ---
    logging.debug("Calculating Williams %R...")
    highest_high_w = df_ta['High'].rolling(window=14).max(); lowest_low_w = df_ta['Low'].rolling(window=14).min()
    # Handle case where highest_high_w == lowest_low_w by replacing the range with NaN or adding epsilon
    range_hw = highest_high_w - lowest_low_w;
    # Add a small epsilon to the denominator if it's zero or near-zero
    range_hw_safe = range_hw.copy(); range_hw_safe[range_hw_safe < 1e-9] = np.nan # Use NaN for robustness
    df_ta['Williams_%R'] = ((highest_high_w - df_ta['Close']) / range_hw_safe) * -100 # Use safe denominator
    # Replace Inf/NaN results with NaN first, then fill NaNs with 0.
    df_ta['Williams_%R'] = df_ta['Williams_%R'].replace([np.inf, -np.inf], np.nan).fillna(0)

    logging.info(f"Finished calculating momentum indicators. Total rows: {len(df_ta)}. Latest row indicator NaNs: {df_ta.iloc[-1][['RSI', 'MACD', 'MACD_Signal', 'CCI', 'Stoch_%K', 'Williams_%R']].isna().sum()}")

    # Only return the calculated columns, ensuring they are numeric
    indicator_cols = ['RSI', 'MACD', 'MACD_Signal', 'CCI', 'Stoch_%K', 'Williams_%R']
    # Drop rows where any of the final indicator values are NaN (usually just the initial period needed for calculation)
    df_ta_cleaned = df_ta[indicator_cols].dropna()
    logging.info(f"Indicator data after dropping initial NaNs: {len(df_ta_cleaned)} rows.")

    return df_ta_cleaned.astype(float) # Ensure float type


@st.cache_data(ttl=ETS_FORECAST_CACHE_DURATION_SECONDS)
def forecast_stock_ets_advanced( ticker: str, close_prices: pd.Series, forecast_days: int = 7, volatility_window_recent: int = 21, volatility_window_long: int = 252, volatility_threshold: float = 1.5, seasonal_period: int = 21, eval_test_size: int = 21 ):
    logger.info(f"[Forecast] Starting ETS forecast for {ticker} ({forecast_days} days)...")
    if close_prices is None or close_prices.empty:
        logger.error(f"[Forecast] No close price data provided for {ticker}.")
        return None, None, "Data Error - Missing Prices"
    # Ensure sufficient data for calculation and evaluation
    # Need enough data for long_vol_window+1 for log returns, seasonal_period+1 for seasonality, and eval_test_size + seasonal_period + 1 for evaluation train set
    min_data_required_calc = max(volatility_window_long + 1, seasonal_period + 1)
    min_data_required_eval = eval_test_size + seasonal_period + 1 if eval_test_size > 0 else 0 # Only needed if eval_test_size > 0
    min_data_required = max(min_data_required_calc, min_data_required_eval, 30) # Minimum 30 points often needed for ETS

    if len(close_prices) < min_data_required:
        logger.error(f"[Forecast] Not enough historical data ({len(close_prices)} days) for robust ETS model or evaluation. Need at least {min_data_required}.")
        return None, None, f"Data Error - Insufficient Data ({len(close_prices)} days)"

    logger.info("[Forecast] Analyzing volatility for seasonality decision...")
    use_seasonal = False
    try:
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_returns) >= volatility_window_long:
            recent_std = log_returns[-volatility_window_recent:].std()
            long_std = log_returns[-volatility_window_long:].std()
            # Avoid division by zero/very small numbers for long_std
            if long_std > 1e-9:
                 if (recent_std / long_std) > volatility_threshold:
                     use_seasonal = True
                     logger.info("[Forecast] High recent volatility detected relative to long term, enabling seasonality.")
                 else: logger.info("[Forecast] Volatility within normal range, using non-seasonal model.")
            else:
                logger.warning("[Forecast] Long-term volatility is zero or near zero. Cannot compare recent volatility. Using non-seasonal model.")
        else:
            logger.warning("[Forecast] Not enough data for long-term volatility comparison, using non-seasonal model.")
    except Exception as e: logger.error(f"[Forecast] Error calculating volatility: {e}"); logger.warning("[Forecast] Proceeding with non-seasonal model due to volatility calculation error."); use_seasonal = False

    model_params = { 'trend': 'add', 'initialization_method': 'estimated', 'use_boxcox': False, 'damped_trend': True }
    if use_seasonal: model_params.update({ 'seasonal': 'add', 'seasonal_periods': seasonal_period })

    rmse = None; evaluation_summary = "Evaluation: Not performed (insufficient data or error)."
    # Perform evaluation only if sufficient data is available for training *and* testing sets
    if eval_test_size > 0 and len(close_prices) > (eval_test_size + (seasonal_period if use_seasonal else 1) + 1): # Ensure enough data for train + seasonal period + 1 for model init
        train_eval = close_prices[:-(eval_test_size)]; test_eval = close_prices[-(eval_test_size):]
        if len(train_eval) > (seasonal_period if use_seasonal else 1) + 1: # Double check train set size is sufficient for model
            logger.info(f"[Forecast] Evaluating model on last {eval_test_size} days...")
            try:
                # Explicitly setting freq=None as date index might not have it from yfinance/polygon
                eval_model = ExponentialSmoothing(train_eval, freq=None, **model_params) # Pass freq=None
                with warnings.catch_warnings(): warnings.simplefilter("ignore"); fitted_eval = eval_model.fit(optimized=True)
                eval_forecast = fitted_eval.forecast(steps=eval_test_size)

                # Ensure eval_forecast index aligns with test_eval for error calculation
                if isinstance(test_eval.index, pd.DatetimeIndex) and isinstance(eval_forecast.index, pd.DatetimeIndex) and len(test_eval) == len(eval_forecast):
                     # Attempt to reindex eval_forecast to match test_eval if dates align
                     if test_eval.index.equals(eval_forecast.index):
                          pass # Indexes already match, great.
                     elif len(test_eval) == len(eval_forecast):
                          # Assume they correspond index-wise if lengths match but dates might differ slightly or freq is off
                          eval_forecast.index = test_eval.index
                     else: # Should not happen if lengths match, but as a safeguard
                          logger.warning("[Forecast] Evaluation forecast and test indices/lengths don't match unexpected case. Cannot calculate RMSE/MAPE reliably.")
                          rmse = None; evaluation_summary = "Evaluation Error: Index/Length mismatch (unexpected)"
                elif len(test_eval) == len(eval_forecast):
                     # Fallback: if not DatetimeIndex but lengths match, proceed assuming correspondence
                     pass # Indexes match by position
                else: # Length mismatch if not DatetimeIndex
                     logger.warning("[Forecast] Evaluation forecast and test lengths don't match. Cannot calculate RMSE/MAPE reliably.")
                     rmse = None; evaluation_summary = "Evaluation Error: Length mismatch"


                if rmse is None: # If index/length matching failed or any other step failed, rmse would be None
                     evaluation_summary = evaluation_summary # Keep the specific error message
                else:
                    rmse = np.sqrt(mean_squared_error(test_eval, eval_forecast))
                    avg_price_eval = test_eval.mean()
                    # Handle case where test_eval mean is zero or near-zero
                    mape = np.mean(np.abs((test_eval - eval_forecast) / test_eval)) * 100 if avg_price_eval > 1e-6 else np.inf
                    evaluation_summary = f"Evaluation (last {eval_test_size} days): RMSE={rmse:.2f} (MAPE={mape:.2f}%)"; logger.info(f"[Forecast] {evaluation_summary}")
            except Exception as e: logger.error(f"[Forecast] Error during evaluation fitting/forecasting: {e}"); evaluation_summary = f"Evaluation Error: {e}"; rmse = None
        else:
             logger.warning(f"[Forecast] Not enough data in evaluation training set ({len(train_eval)} days) for model fit. Need > {(seasonal_period if use_seasonal else 1) + 1}.")
             evaluation_summary = f"Evaluation skipped: Train data insufficient ({len(train_eval)})"

    else: logger.warning(f"[Forecast] Not enough data to perform separate evaluation (need > {eval_test_size + (seasonal_period if use_seasonal else 1) + 1} for test+train). Evaluation skipped.")


    logger.info("[Forecast] Fitting final model on full dataset...")
    try:
        # Explicitly setting freq=None again
        final_model = ExponentialSmoothing(close_prices, freq=None, **model_params) # Pass freq=None
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); fitted_final_model = final_model.fit(optimized=True)
        forecast_result = fitted_final_model.forecast(steps=forecast_days)

        # Attempt to create business day index for forecast
        if not close_prices.empty and isinstance(close_prices.index, pd.DatetimeIndex):
            last_date = close_prices.index[-1]
            try:
                 # Start one calendar day after the last date to ensure next trading day
                 start_forecast_date = last_date + pd.Timedelta(days=1)
                 future_dates = pd.bdate_range(start=start_forecast_date, periods=forecast_days, freq='B')
                 if len(future_dates) == forecast_days: forecast_result.index = future_dates
                 else: logger.warning(f"[Forecast] Could not generate expected number of business dates ({forecast_days}) for forecast index. Generated: {len(future_dates)}. Using default numerical index.")
            except Exception as date_err:
                 logger.warning(f"[Forecast] Error generating forecast date index: {date_err}. Using default numerical index.")
        else:
            logger.warning("[Forecast] Cannot generate date index for forecast (input index issue or empty data). Using default numerical index.")

        # Construct model description dynamically
        trend_part = model_params.get('trend', 'N')
        seasonal_part = model_params.get('seasonal', 'N') if use_seasonal else 'N'
        damped_part = 'damped' if model_params.get('damped_trend') else ''
        model_used_desc = f"ETS({trend_part},{seasonal_part},{'A' if use_seasonal else 'N'}{f' {damped_part}' if damped_part else ''})".strip()
        model_used_desc = model_used_desc.replace("ETS(A,N,N)", "ETS(A)").replace("ETS(A,A,A damped)", "ETS(A,A) Damped") # Simplify common cases

        logger.info(f"[Forecast] Forecast generation successful using {model_used_desc}.")

        return forecast_result, evaluation_summary, model_used_desc
    except Exception as e: logger.error(f"[Forecast] Error during final model fitting/forecasting: {e}"); return None, evaluation_summary, "Model Error"

# --- Main ---
# Removed: main function stub that was left from previous copy/paste

# [Multi-Source News Sentiment functions remain the same]
# --- START: MULTI-SOURCE NEWS SENTIMENT FUNCTIONS ---
def analyze_sentiment_vader(text):
    if not text: return "Neutral", 0.0
    vs = vader_analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05: sentiment = "Positive"
    elif compound_score <= -0.05: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, compound_score
def get_news_newsapi(ticker, api_key, days_back=NEWS_DAYS_BACK, page_size=NEWS_PAGE_SIZE):
    logging.info(f"[News] Fetching news from NewsAPI for '{ticker}'...")
    if not api_key or len(api_key) < 20: logging.warning("[News] NewsAPI key not set or seems invalid. Skipping NewsAPI."); return []
    to_date = datetime.now(); from_date = to_date - timedelta(days=days_back); from_date_str = from_date.strftime('%Y-%m-%d')
    query = f'"{ticker}" AND (stock OR shares OR earnings OR market OR business OR company OR analyst OR investment)'; # Expanded query slightly
    params = { 'q': query, 'apiKey': api_key, 'from': from_date_str, 'sortBy': 'relevancy', 'language': 'en', 'pageSize': min(page_size, 100), }
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=15); response.raise_for_status(); data = response.json()
        if data.get('status') == 'ok':
            articles = data.get('articles', []); formatted_articles = []
            for article in articles:
                title = article.get('title');
                if not title or title == '[Removed]': continue
                formatted_articles.append({ 'title': title, 'description': article.get('description'), 'url': article.get('url'), 'publishedAt': parse_date(article.get('publishedAt')), 'source_api': 'NewsAPI', 'source_name': article.get('source', {}).get('name', 'NewsAPI') })
            logging.info(f"[News] NewsAPI: Found {len(formatted_articles)} articles (total results: {data.get('totalResults')})."); return formatted_articles
        else:
            if data.get('code') == 'rateLimited': logging.error("[News] Error from NewsAPI: Rate limit exceeded.")
            elif data.get('code') == 'maximumResultsReached': logging.warning("[News] Error from NewsAPI: Maximum results reached for plan.")
            elif data.get('code') == 'apiKeyInvalid': logging.error("[News] Error from NewsAPI: Invalid API key.")
            else: logging.error(f"[News] Error from NewsAPI: {data.get('code')} - {data.get('message')}")
            return []
    except requests.exceptions.Timeout: logging.error("[News] NewsAPI Network/Request Error: Request timed out."); return []
    except requests.exceptions.RequestException as e: logging.error(f"[News] NewsAPI Network/Request Error: {e}"); return []
    except json.JSONDecodeError: logging.error("[News] Error: Could not decode JSON response from NewsAPI."); return []
    except Exception as e: logging.error(f"[News] An unexpected error occurred with NewsAPI: {e}", exc_info=True); return []
def get_news_fmp(ticker, api_key, limit=NEWS_FMP_LIMIT):
    logging.info(f"[News] Fetching news from Financial Modeling Prep for '{ticker}'...")
    if not api_key or len(api_key) < 20: logging.warning("[News] FMP API key not set or seems invalid. Skipping FMP."); return []
    fmp_news_url = f"{FMP_ENDPOINT}/v3/stock_news"; params = {'tickers': ticker, 'limit': limit, 'apikey': api_key}
    try:
        response = requests.get(fmp_news_url, params=params, timeout=15)
        if response.status_code != 200:
             if response.status_code == 401: logging.error(f"[News] FMP Request Error: Unauthorized (401). Check API key.")
             elif response.status_code == 403: logging.error(f"[News] FMP Request Error: Forbidden (403). Check API key or plan limits.")
             elif response.status_code == 404: logging.warning(f"[News] FMP Request Warning: Not Found (404) - Ticker {ticker} may not be supported or no news.")
             else: response.raise_for_status(); # Raise for other 4xx/5xx errors
             return []
        data = response.json()
        if isinstance(data, dict) and 'Error Message' in data:
            error_message = data['Error Message'];
            if "Limit Reach" in error_message: logging.error(f"[News] Error from FMP: API limit reached. {error_message}")
            else: logging.error(f"[News] Error from FMP: {error_message}")
            return []
        if not isinstance(data, list):
            if data is None: logging.warning(f"[News] FMP returned None for ticker {ticker}.")
            else: logging.error(f"[News] Error from FMP: Unexpected response format. Expected list, got {type(data)}");
            return []

        formatted_articles = []
        for article in data:
             title = article.get('title');
             if not title: continue
             # Prioritize article text if available, otherwise use summary/text
             text_content = article.get('article', article.get('text', article.get('summary')))
             formatted_articles.append({ 'title': title, 'description': text_content, 'url': article.get('url'), 'publishedAt': parse_date(article.get('publishedDate')), 'source_api': 'FMP', 'source_name': article.get('site', 'Financial Modeling Prep') })
        logging.info(f"[News] FMP: Found {len(formatted_articles)} articles for {ticker}."); return formatted_articles
    except requests.exceptions.Timeout: logging.error("[News] FMP Network/Request Error: Request timed out."); return []
    except requests.exceptions.RequestException as e: logging.error(f"[News] FMP Network/Request Error: {e}"); return []
    except json.JSONDecodeError: logging.error(f"[News] Error: Could not decode JSON response from FMP. Raw: {response.text[:200]}..."); return []
    except Exception as e: logging.error(f"[News] An unexpected error occurred with FMP: {e}", exc_info=True); return []
def get_news_rss(ticker, feed_urls, days_back=NEWS_DAYS_BACK):
    logging.info("[News] Fetching news from RSS Feeds...")
    all_rss_articles = []; cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back);
    for url_template in feed_urls:
        url = url_template.replace('{ticker}', ticker) if '{ticker}' in url_template else url_template
        logging.debug(f"[News]   Parsing RSS feed: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; FinancialBot/1.0; +http://example.com/bot)'}; feed_data = feedparser.parse(url, request_headers=headers, timeout=10)
            if feed_data.bozo: logging.warning(f"[News]     Warning: Malformed feed or error parsing {url}. Exception: {feed_data.bozo_exception}")
            for entry in feed_data.entries:
                title = entry.get('title');
                if not title: continue
                # Combine potential fields for description
                summary = entry.get('summary') or entry.get('description')
                if entry.get('content'):
                     for content_item in entry['content']:
                          if content_item.get('type') == 'text/html' or content_item.get('type') == 'text/plain':
                              if summary: summary = f"{summary} {content_item['value']}"
                              else: summary = content_item['value']
                summary = summary or '' # Ensure summary is a string

                link = entry.get('link'); published_time_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                published_dt_aware = None; parsed_date_iso = None
                if published_time_struct:
                    try: ts = time.mktime(published_time_struct); published_dt_aware = datetime.fromtimestamp(ts, timezone.utc); parsed_date_iso = published_dt_aware.isoformat().replace('+00:00', 'Z')
                    except (TypeError, ValueError, OverflowError): pass
                elif entry.get('published'):
                    parsed_date_iso = parse_date(entry.get('published')) # Use the robust parse_date function
                    if parsed_date_iso:
                         try: published_dt_aware = date_parser.isoparse(parsed_date_iso.replace('Z', '+00:00'))
                         except ValueError: pass

                # Filter by date if published_dt_aware was successfully parsed
                if published_dt_aware and published_dt_aware < cutoff_date:
                    logging.debug(f"[News]     RSS ipping (Old Article): {title[:50]}...")
                    continue

                # Ticker relevance check
                is_ticker_specific_feed = '{ticker}' in url_template; ticker_lower = ticker.lower(); title_lower = title.lower(); summary_lower = summary.lower()
                # Use word boundaries or '$' prefix for better matching
                pattern = r'\b' + re.escape(ticker_lower) + r'\b|\(' + re.escape(ticker_lower) + r'\)|' + re.escape(f'${ticker_lower}') + r'\b'
                mentions_ticker = bool(re.search(pattern, title_lower) or re.search(pattern, summary_lower))

                if is_ticker_specific_feed or (not is_ticker_specific_feed and mentions_ticker):
                     all_rss_articles.append({ 'title': title, 'description': summary.strip(), 'url': link, 'publishedAt': parsed_date_iso or 'N/A', 'source_api': 'RSS', 'source_name': feed_data.feed.get('title', url) })
                elif not is_ticker_specific_feed and not mentions_ticker: logging.debug(f"[News]     RSS ipping (General Feed, No Ticker Match): {title[:50]}...")
        except requests.exceptions.Timeout: logging.warning(f"[News]     RSS Feed Timeout: {url}")
        except Exception as e: logging.error(f"[News]     Error processing RSS feed {url}: {e}")
        time.sleep(0.1) # Be polite to servers
    logging.info(f"[News] RSS Feeds: Found {len(all_rss_articles)} potentially relevant articles within date range."); return all_rss_articles

@st.cache_data(ttl=NEWS_SENTIMENT_CACHE_DURATION_SECONDS)
def get_multi_source_news_sentiment(ticker: str, news_api_key: str, fmp_api_key: str):
    logging.info(f"--- Starting Multi-Source News Sentiment Analysis for {ticker} ---"); start_time = time.time()
    # Fetch from sources
    newsapi_articles = get_news_newsapi(ticker, news_api_key);
    fmp_articles = get_news_fmp(ticker, fmp_api_key);
    rss_articles = get_news_rss(ticker, RSS_FEEDS);

    all_articles = newsapi_articles + fmp_articles + rss_articles; logging.info(f"[News] Total articles fetched across sources: {len(all_articles)}")

    # Deduplicate - prefer URL, fallback to title
    seen_urls = set(); seen_titles_lower = set(); unique_articles = []
    for article in all_articles:
         url = article.get('url'); title = article.get('title'); title_lower = title.lower().strip() if title else None
         # Use a tuple (url, title_lower) for uniqueness check to handle cases where URL is missing
         unique_key = (url, title_lower)
         if unique_key not in seen_urls:
              unique_articles.append(article);
              seen_urls.add(unique_key)

    logging.info(f"[News] Articles after deduplication: {len(unique_articles)}")

    # Sort by date (newest first)
    def get_sort_key(article):
        date_str = article.get('publishedAt');
        if date_str and date_str != 'N/A':
            try: return date_parser.isoparse(date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError): return datetime.min.replace(tzinfo=timezone.utc) # Return min datetime if parsing fails
        return datetime.min.replace(tzinfo=timezone.utc) # Return min datetime if no date

    unique_articles.sort(key=get_sort_key, reverse=True);
    articles_to_analyze = unique_articles[:NEWS_MAX_ARTICLES_DISPLAY] # Limit number of articles for analysis/display

    logging.info(f"[News] Analyzing sentiment for latest {len(articles_to_analyze)} unique articles...")
    positive_count, negative_count, neutral_count = 0, 0, 0; compound_scores = []
    analyzed_articles_details = [] # To store results for display/logging

    if not articles_to_analyze:
        logging.warning(f"[News] No unique articles found to analyze for {ticker}.")
        summary_str = f"Multi-Source News Sentiment ({NEWS_DAYS_BACK}d, VADER):\n  - No relevant news articles found for {ticker}.";
        counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0, 'avg_score': 0.0};
        return summary_str, counts, [] # Return empty details list

    for article in articles_to_analyze:
        title = article.get('title', ''); description = article.get('description', '');
        # Use title and description/text for analysis, prioritize text if available
        text_to_analyze = f"{title}. {description.strip()}" if description and len(description.strip()) > 10 else title
        if not text_to_analyze: continue # Skip if no text to analyze

        sentiment_label, compound_score = analyze_sentiment_vader(text_to_analyze);
        compound_scores.append(compound_score)

        if sentiment_label == "Positive": positive_count += 1
        elif sentiment_label == "Negative": negative_count += 1
        else: neutral_count += 1

        analyzed_articles_details.append({
            'title': title,
            'source': article.get('source_name', article.get('source_api', 'N/A')),
            'published': article.get('publishedAt', 'N/A'),
            'sentiment': sentiment_label,
            'compound_score': compound_score,
            'url': article.get('url')
        })

    total_analyzed = len(analyzed_articles_details); # Use count of articles that actually got analyzed
    avg_score = np.mean(compound_scores) if compound_scores else 0.0

    # Determine overall bias based on counts (simple majority/heuristic)
    if total_analyzed > 0:
        # A more robust heuristic: Positive needs to outweigh Negative by some margin, considering Neutrals.
        # Simple: Positive count vs Negative count
        if positive_count > negative_count: overall_bias = "Positive Bias ✅"
        elif negative_count > positive_count: overall_bias = "Negative Bias ❌"
        else: overall_bias = "Neutral/Mixed Bias ⚖️"
        # Alternative heuristic considering Neutral:
        # if positive_count > negative_count + neutral_count * 0.2: overall_bias = "Positive Bias ✅" # Needs to outweigh negative + small portion of neutral
        # elif negative_count > positive_count + neutral_count * 0.2: overall_bias = "Negative Bias ❌"
        # else: overall_bias = "Neutral/Mixed Bias ⚖️"
    else:
        overall_bias = "N/A"

    summary_str = f"Multi-Source News Sentiment ({NEWS_DAYS_BACK}d, VADER Analysis on ~{total_analyzed} unique articles):\n";
    summary_str += f"  - ✅ Positive Articles: {positive_count}\n";
    summary_str += f"  - ❌ Negative Articles: {negative_count}\n";
    summary_str += f"  - ➖ Neutral Articles: {neutral_count}\n";
    summary_str += f"  - Overall Sentiment (heuristic): {overall_bias}"

    counts = { 'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count, 'total': total_analyzed, 'avg_score': avg_score };
    end_time = time.time()
    logging.info(f"[News] Multi-source sentiment analysis for {ticker} completed in {end_time - start_time:.2f} sec.");
    logging.info(f"[News] Result: P={positive_count}, N={negative_count}, Neut={neutral_count}, AvgScore={avg_score:.3f}, Bias: {overall_bias}");
    return summary_str, counts, analyzed_articles_details # Return details list


# --- END: MULTI-SOURCE NEWS SENTIMENT FUNCTIONS ---

# --- Other Helper Functions (Piotroi, Normalization, etc.) ---
def get_financial_data(ticker_obj, statement_type: str, periods: int = 2):
    """Fetches financial statements from yfinance."""
    try:
        if statement_type == 'balance_sheet': data = ticker_obj.balance_sheet
        elif statement_type == 'income_stmt': data = ticker_obj.income_stmt
        elif statement_type == 'cashflow': data = ticker_obj.cashflow
        else: logging.warning(f"Invalid statement type: {statement_type}"); return None
        if data is not None and not data.empty:
            actual_periods = min(periods, data.shape[1])
            if actual_periods >= 1: return data.iloc[:, :actual_periods]
            else: logging.warning(f"[{ticker_obj.ticker}] Report '{statement_type}' has no columns."); return None
        else: logging.warning(f"[{ticker_obj.ticker}] Report '{statement_type}' is empty/unavailable."); return None
    except AttributeError: logging.warning(f"[{ticker_obj.ticker}] Statement '{statement_type}' not found."); return None
    except Exception as e: logging.warning(f"[{ticker_obj.ticker}] Error getting report '{statement_type}': {e}"); return None
def safe_get(series: pd.Series, key: str, default=None):
    """Safely get a value from a Pandas Series, handling None/NaN/string conversions."""
    if series is None: return default
    value = series.get(key, default)
    if pd.isna(value): return default
    if isinstance(value, str):
        try: return pd.to_numeric(value)
        except ValueError: return default
    return value
def get_stock_beta(ticker_info: dict):
    """Extracts beta from yfinance info, handling different keys and types."""
    if not ticker_info: return None
    beta = ticker_info.get('beta', ticker_info.get('beta3Year'))
    if beta is not None and pd.isna(beta): return None # Explicitly check for NaN
    if beta is not None and not isinstance(beta, (int, float)):
        try: beta = float(beta)
        except (ValueError, TypeError): beta = None
    # yfinance often returns beta of 0 for indices/ETFs. Check quoteType.
    if beta is not None and beta == 0.0 and ticker_info.get('quoteType', '').upper() in ['INDEX', 'ETF']:
         logging.debug(f"Beta is 0.0 for {ticker_info.get('symbol')}, likely an index/ETF, returning None.")
         return None
    return beta
def normalize_score(value, range_min, range_max, higher_is_riskier=True, is_log_range=False):
    """Normalizes a raw value to a 0-100 risk score based on a defined range."""
    if value is None or pd.isna(value):
        logging.debug(f"Normalize: Input value is None/NaN, returning default 50.0")
        return 50.0 # Return neutral score if data is missing

    current_min, current_max = range_min, range_max
    value_to_norm = float(value) # Ensure it's a float

    if is_log_range:
        # Ensure value and range bounds are positive before taking log
        if value_to_norm > 1e-9 and current_min is not None and current_max is not None and current_min > 1e-9 and current_max > 1e-9:
            try:
                value_to_norm = np.log10(value_to_norm)
                current_min = np.log10(current_min)
                current_max = np.log10(current_max)
                logging.debug(f"Normalize Log: {value:.2f} -> log10({value:.2f})={value_to_norm:.2f} (Range: log10({range_min:.2f})={current_min:.2f} to log10({range_max:.2f})={current_max:.2f})")
            except (ValueError, TypeError, OverflowError) as e:
                logging.warning(f"Log10 failed for value {value} or range ({range_min}, {range_max}): {e}. Using original value linearly.")
                value_to_norm = float(value) # Revert to original float value
                current_min, current_max = range_min, range_max # Revert range
        else:
             # Value or range bounds are zero or negative, can't take log. Assign edge score.
             # If value is <=0, it's likely a small/zero market cap/volume, which is riskier.
             # If the range bounds are <=0 or None, it's likely a configuration error, return neutral.
             if value_to_norm <= 1e-9 and current_min is not None and current_max is not None and current_min > 1e-9 and current_max > 1e-9:
                 edge_score = 0.0 if higher_is_riskier else 100.0 # Low value means higher risk (mcap, vol)
                 logging.debug(f"Normalize Log: Value {value:.2f} <= {1e-9}, assigning edge score {edge_score}")
                 return edge_score
             else:
                 logging.warning(f"Normalize Log: Range bounds invalid or value non-positive for log: Val={value}, Range=({range_min}, {range_max}). Returning 50.0")
                 return 50.0


    # Check for invalid range after potential log transformation
    if current_min is None or current_max is None or abs(current_max - current_min) < 1e-9:
        logging.warning(f"Normalize: Range min {current_min} and max {current_max} are too close or None. Returning 50.0")
        return 50.0 # Avoid division by zero if range is negligible or invalid

    # Clip the value to be within the (potentially log-transformed) range
    clipped_value = np.clip(value_to_norm, current_min, current_max)

    # Perform normalization (linear interpolation between 0 and 1)
    # Ensure denominator is not zero
    denominator = current_max - current_min
    if abs(denominator) < 1e-9:
         logging.warning(f"Normalize: Denominator near zero during normalization ({denominator:.4f}). Returning 50.0")
         return 50.0

    normalized = (clipped_value - current_min) / denominator

    # Apply risk direction (higher is riskier or lower is riskier)
    # Calculate ri_score based on higher_is_riskier flag
    # Ensure score is strictly between 0 and 100
    ri_score = np.clip(normalized * 100 if higher_is_riskier else (1 - normalized) * 100, 0, 100)

    logging.debug(f"Normalize: Val={value:.4f} (NormVal={value_to_norm:.4f}), Range=({range_min:.4f},{range_max:.4f}), Log={is_log_range}, HigherRisk={higher_is_riskier} => Clipped={clipped_value:.4f}, Normalized={normalized:.4f}, Score={ri_score:.2f}")
    return ri_score

def calculate_sma(data: pd.Series, window: int):
    """Calculates the last value of a Simple Moving Average."""
    if data is None or data.empty or len(data) < window:
         logging.debug(f"SMA{window}: Insufficient data ({len(data)}).")
         return np.nan # Use np.nan instead of None for consistency with pandas calculations
    try:
        # Explicitly handle potential NaN/Inf in input data before rolling
        valid_data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) < window:
             logging.debug(f"SMA{window}: Insufficient *valid* data ({len(valid_data)}).")
             return np.nan
        return valid_data.rolling(window=window).mean().iloc[-1]
    except Exception as e: logging.warning(f"SMA{window} error: {e}"); return np.nan


@st.cache_data(ttl=3600*6)
def calculate_piotroski_f_score(ticker_symbol: str):
    """Calculates the Piotroski F-Score. (Not used in simplified risk score)."""
    logging.info(f"[{ticker_symbol}] Calculating Piotroski F-Score (or cache)..."); score = 0; details = {}
    try:
        # Note: These fetches are NOT using the retry decorator directly, relying on yfinance's internal retries or hoping these endpoints are less rate-limited
        # If these prove flaky, they would also need a similar retry wrapper.
        ticker = yf.Ticker(ticker_symbol); income = get_financial_data(ticker, 'income_stmt', 2); balance = get_financial_data(ticker, 'balance_sheet', 2); cashflow = get_financial_data(ticker, 'cashflow', 2)
        if income is None or balance is None or cashflow is None or \
           income.shape[1] < 2 or balance.shape[1] < 2 or cashflow.shape[1] < 2: logging.warning(f"[{ticker_symbol}] Insufficient financials for F-Score (need 2 periods)."); return None, details
        inc_t, inc_tm1 = income.iloc[:, 0], income.iloc[:, 1]; bal_t, bal_tm1 = balance.iloc[:, 0], balance.iloc[:, 1]; cf_t, cf_tm1 = cashflow.iloc[:, 0], cashflow.iloc[:, 1]

        # Profitability
        net_income_t = safe_get(inc_t, 'Net Income', 0); details['NI > 0']=(net_income_t is not None and net_income_t > 0); score += details.get('NI > 0', 0)
        op_cashflow_t=safe_get(cf_t, 'Operating Cash Flow', safe_get(cf_t, 'Cash Flow From Continuing Operating Activities', 0)); details['OCF > 0']=(op_cashflow_t is not None and op_cashflow_t > 0); score += details.get('OCF > 0', 0)

        assets_t=safe_get(bal_t, 'Total Assets', 0); assets_tm1=safe_get(bal_tm1, 'Total Assets', 0);
        # Calculate ROA only if assets are positive in both periods
        roa_check = False; roa_t, roa_tm1 = 0, 0
        if assets_t is not None and assets_tm1 is not None and assets_t > 0 and assets_tm1 > 0:
             net_income_tm1 = safe_get(inc_tm1, 'Net Income', 0) or 0 # Ensure net_income_tm1 is numeric
             roa_t = (net_income_t / assets_t) if assets_t != 0 and net_income_t is not None else 0;
             roa_tm1 = (net_income_tm1 / assets_tm1) if assets_tm1 != 0 and net_income_tm1 is not None else 0;
             roa_check=(roa_t > roa_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate ROA change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta ROA > 0']=roa_check; score += details.get('Delta ROA > 0', 0)

        details['OCF > NI']=(op_cashflow_t > net_income_t) if op_cashflow_t is not None and net_income_t is not None else False; score += details.get('OCF > NI', 0)

        # Leverage, Liquidity & Source of Funds
        debt_lt_t=safe_get(bal_t, 'Long Term Debt And Capital Lease Obligation', safe_get(bal_t, 'Long Term Debt', 0)); debt_lt_tm1=safe_get(bal_tm1, 'Long Term Debt And Capital Lease Obligation', safe_get(bal_tm1, 'Long Term Debt', 0)); leverage_check=False
        # Calculate leverage ratio and check change only if assets are positive
        if assets_t is not None and assets_tm1 is not None and assets_t > 0 and assets_tm1 > 0:
             # Handle potential None debt values by treating them as 0 for ratio calculation
             leverage_t = (debt_lt_t or 0) / assets_t if assets_t != 0 else np.inf;
             leverage_tm1 = (debt_lt_tm1 or 0) / assets_tm1 if assets_tm1 != 0 else np.inf;
             leverage_check=(leverage_t < leverage_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate Leverage change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta Leverage < 0']=leverage_check; score += details.get('Delta Leverage < 0', 0)

        current_assets_t=safe_get(bal_t, 'Current Assets', 0); current_liab_t=safe_get(bal_t, 'Current Liabilities', 0);
        current_assets_tm1=safe_get(bal_tm1, 'Current Assets', 0); current_liab_tm1=safe_get(bal_tm1, 'Current Liabilities', 0);
        # Calculate current ratio only if current liabilities are positive
        current_ratio_t = (current_assets_t / current_liab_t) if current_liab_t is not None and current_liab_t > 0 and current_assets_t is not None else (np.inf if (current_assets_t is not None and current_assets_t > 0) else 0);
        current_ratio_tm1 = (current_assets_tm1 / current_liab_tm1) if current_liab_tm1 is not None and current_liab_tm1 > 0 and current_assets_tm1 is not None else (np.inf if (current_assets_tm1 is not None and current_assets_tm1 > 0) else 0);
        details['Delta Current Ratio > 0']=(current_ratio_t > current_ratio_tm1); score += details.get('Delta Current Ratio > 0', 0)

        shares_t=safe_get(bal_t, 'Share Issued', safe_get(inc_t,'Diluted Average Shares',None));
        shares_tm1=safe_get(bal_tm1, 'Share Issued', safe_get(inc_tm1,'Diluted Average Shares',None));
        shares_check=None
        # Check shares issued or use equity proxy if shares data unreliable
        if shares_t is not None and shares_tm1 is not None and shares_tm1 > 0:
             # Allow a small increase (e.g., 1%) to account for minor options/RSUs
             shares_check = (shares_t <= shares_tm1 * 1.01)
             details['Shares Issued Not Increased']=shares_check
        else:
            logging.warning(f"[{ticker_symbol}] Shares Issued not found or zero ({shares_t}, {shares_tm1}). Using equity proxy for F-Score.")
            equity_t=safe_get(bal_t, 'Stockholders Equity', 0); equity_tm1=safe_get(bal_tm1, 'Stockholders Equity', 0);
            ni_for_calc=net_income_t if net_income_t is not None and pd.notna(net_income_t) else 0;
            # Growth in equity not explained by net income (proxy for share issuance)
            equity_growth_non_re = (equity_t - equity_tm1) - ni_for_calc if equity_t is not None and equity_tm1 is not None else None

            if equity_growth_non_re is not None:
                 # Check if equity growth not explained by NI is relatively small compared to previous equity or a small absolute value
                 if equity_tm1 is not None and equity_tm1 > 0: shares_check = (equity_growth_non_re < equity_tm1 * 0.02) # Allow 2% non-re growth
                 else: shares_check = (equity_growth_non_re < 1e6) # Small absolute increase threshold if previous equity is zero/negative

                 details['Shares Issued Not Increased (Equity Proxy)'] = shares_check
            else:
                logging.warning(f"[{ticker_symbol}] Cannot calculate equity growth for F-Score shares check. Equity: t={equity_t}, tm1={equity_tm1}, NI: {ni_for_calc}");
                details['Shares Issued Not Increased (Equity Proxy)'] = False # Assume issuance increased if calculation fails

        score += details.get('Shares Issued Not Increased', 0) if 'Shares Issued Not Increased' in details else details.get('Shares Issued Not Increased (Equity Proxy)', 0)

        # Operating Efficiency
        gross_profit_t=safe_get(inc_t, 'Gross Profit', 0); revenue_t=safe_get(inc_t, 'Total Revenue', safe_get(inc_t,'Operating Revenue', 0));
        gross_profit_tm1=safe_get(inc_tm1, 'Gross Profit', 0); revenue_tm1=safe_get(inc_tm1, 'Total Revenue', safe_get(inc_tm1,'Operating Revenue', 0));
        # Calculate gross margin only if revenue is positive
        gross_margin_t=(gross_profit_t/revenue_t) if revenue_t is not None and revenue_t > 0 and gross_profit_t is not None else 0;
        gross_margin_tm1=(gross_profit_tm1/revenue_tm1) if revenue_tm1 is not None and revenue_tm1 > 0 and revenue_tm1 is not None else 0;
        details['Delta Gross Margin > 0']=(gross_margin_t > gross_margin_tm1); score += details.get('Delta Gross Margin > 0', 0)

        turnover_check=False
        # Calculate asset turnover only if assets are positive
        if assets_t is not None and assets_tm1 is not None and assets_t > 0 and assets_tm1 > 0:
            asset_turnover_t=(revenue_t/assets_t) if revenue_t is not None else 0;
            asset_turnover_tm1=(revenue_tm1/assets_tm1) if revenue_tm1 is not None else 0;
            turnover_check=(asset_turnover_t > asset_turnover_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate Asset Turnover change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta Asset Turnover > 0']=turnover_check; score += details.get('Delta Asset Turnover > 0', 0)

        final_score = score # Total score out of 9
        logging.info(f"[{ticker_symbol}] Piotroski F-Score calculated: {final_score}/9"); logging.debug(f"[{ticker_symbol}] F-Score Details: {details}");
        return final_score, details
    except Exception as e: logging.error(f"[{ticker_symbol}] General F-Score error: {e}", exc_info=True); logging.debug(traceback.format_exc()); return None, details


@st.cache_data(ttl=600)
def calculate_dynamic_risk_score(ticker: str, df_history: pd.DataFrame, info: dict, weights: dict):
    """
    Calculate a Simplified Dynamic Risk Score (0-100) for a stock based on core factors.

    Args:
        ticker (str): Stock ticker symbol.
        df_history (pd.DataFrame): OHLCV DataFrame from Yahoo Finance (full history).
        info (dict): Yahoo Finance ticker info dictionary.
        weights (dict): Weights for each risk factor (only relevant ones used).

    Returns:
        tuple: (final_score, intermediate_scores, weight_sum)
            - final_score (float): Final risk score (0-100).
            - intermediate_scores (dict): Scores for each factor.
            - weight_sum (float): Sum of weights used.
    """
    logging.info(f"Calculating simplified risk score for {ticker}")
    intermediate_scores = {}
    # Use weights copy filtered for only the factors included in the simplified score
    simplified_factors = ['volatility', 'market_cap', 'liquidity', 'beta', 'price_vs_sma', 'vix']
    effective_weights = {k: weights.get(k, 0) for k in simplified_factors}

    # Validate input data presence before proceeding
    if df_history is None or df_history.empty or 'Close' not in df_history.columns:
        logging.error(f"Risk Calculation for {ticker}: No valid OHLCV data (df_history) for calculation.")
        # Set all factor scores to NaN and effective weights to 0
        for factor in simplified_factors: intermediate_scores[factor] = np.nan; effective_weights[factor] = 0.0
        return None, intermediate_scores, 0.0

    if not info:
         logging.warning(f"Risk Calculation for {ticker}: No Yahoo Ticker info. Some risk factors will be unavailable.")
         # For factors dependent on info, set score to NaN and weight to 0
         for factor in ['market_cap', 'beta']: intermediate_scores[factor] = np.nan; effective_weights[factor] = 0.0


    # Ensure Close prices series is available for return-based calcs
    close_prices = df_history['Close']
    if close_prices.empty:
         logging.error(f"Risk Calculation for {ticker}: Close price series is empty. Cannot calculate return-based risk factors.")
         # Set return/price-based factors to NaN and weight to 0
         for factor in ['volatility', 'price_vs_sma']: intermediate_scores[factor] = np.nan; effective_weights[factor] = 0.0
         # Also check Volume presence for liquidity
         if 'Volume' not in df_history.columns: intermediate_scores['liquidity'] = np.nan; effective_weights['liquidity'] = 0.0


    # Use a recent period for volatility and returns (e.g., 1 year)
    # Ensure recent_period_days is not more than available data points minus 1 (for pct_change)
    recent_period_days = min(252, len(df_history)) # Use up to ~1 year or whatever is available
    if recent_period_days > 1:
        recent_close_prices = close_prices.iloc[-recent_period_days:]
        recent_returns = recent_close_prices.pct_change().dropna()
        min_return_points = 20 # Minimum points for volatility
        if len(recent_returns) >= min_return_points:
            volatility = recent_returns.std() * np.sqrt(252) # Annualized
            intermediate_scores['volatility'] = normalize_score(volatility, *VOLATILITY_RANGE)
            logging.debug(f"[{ticker}] Volatility ({len(recent_returns)} days): {volatility:.4f}, Score: {intermediate_scores['volatility']:.2f}")
        else:
            intermediate_scores['volatility'] = np.nan
            effective_weights['volatility'] = 0.0
            logging.debug(f"[{ticker}] Volatility: Insufficient return data ({len(recent_returns)} returns) for window {min_return_points}.")
    else:
        intermediate_scores['volatility'] = np.nan
        effective_weights['volatility'] = 0.0
        logging.debug(f"[{ticker}] Volatility: Insufficient history data ({len(df_history)} days) for window {recent_period_days}.")


    # 2. Market Cap (smaller = riskier) - Depends on info dict
    if 'market_cap' not in intermediate_scores: # Check if already marked unavailable
         market_cap = info.get('marketCap')
         if market_cap is not None and pd.notna(market_cap) and market_cap > 0:
             # Invert score: smaller market cap = higher risk
             score = normalize_score(market_cap, *MARKET_CAP_RANGE_LOG, is_log_range=True)
             intermediate_scores['market_cap'] = 100 - score
             logging.debug(f"[{ticker}] Market Cap: {format_val(market_cap, '$', prec=0)}, Score: {intermediate_scores['market_cap']:.2f}")
         else:
             intermediate_scores['market_cap'] = np.nan
             effective_weights['market_cap'] = 0.0
             logging.debug(f"[{ticker}] Market Cap: Not available, zero, or invalid.")


    # 3. Liquidity (average daily volume over recent period, lower = riskier)
    # Use the same recent period as volatility for volume average, depends on Volume column
    if 'liquidity' not in intermediate_scores: # Check if already marked unavailable
         recent_volume_series = df_history['Volume'].iloc[-recent_period_days:] if 'Volume' in df_history.columns and recent_period_days > 0 else pd.Series(dtype=float)
         avg_volume = recent_volume_series.mean() if not recent_volume_series.empty and recent_volume_series.dropna().shape[0] > 0 else None

         if avg_volume is not None and pd.notna(avg_volume) and avg_volume > 0:
             # Invert score: lower volume = higher risk
             score = normalize_score(avg_volume, *VOLUME_RANGE_LOG, is_log_range=True)
             intermediate_scores['liquidity'] = 100 - score
             logging.debug(f"[{ticker}] Liquidity (Avg Volume {len(recent_volume_series)} days): {avg_volume:.0f}, Score: {intermediate_scores['liquidity']:.2f}")
         else:
             intermediate_scores['liquidity'] = np.nan
             effective_weights['liquidity'] = 0.0
             logging.debug(f"[{ticker}] Liquidity: Not available, zero, invalid, or insufficient history.")


    # 4. Beta (higher = riskier) - Depends on info dict
    if 'beta' not in intermediate_scores: # Check if already marked unavailable
         beta = get_stock_beta(info) # Use helper function
         if beta is not None and pd.notna(beta):
             intermediate_scores['beta'] = normalize_score(beta, *BETA_RANGE)
             logging.debug(f"[{ticker}] Beta: {beta:.2f}, Score: {intermediate_scores['beta']:.2f}")
         else:
             intermediate_scores['beta'] = np.nan
             effective_weights['beta'] = 0.0
             logging.debug(f"[{ticker}] Beta: Not available or invalid.")

    # 5. Price vs. SMA (deviation from 50-day SMA, larger abs deviation = riskier)
    # Needs enough data for SMA50, depends on Close prices
    if 'price_vs_sma' not in intermediate_scores: # Check if already marked unavailable
        if len(close_prices) >= 50:
            sma50 = calculate_sma(close_prices, 50) # Use our SMA helper
            current_price = close_prices.iloc[-1] if not close_prices.empty else None
            if sma50 is not None and pd.notna(sma50) and current_price is not None and pd.notna(current_price) and sma50 > 0:
                deviation = (current_price - sma50) / sma50
                # Normalize the absolute value against the positive range of the deviation magnitude
                # The range for deviation magnitude is [0, max(abs(min_dev), abs(max_dev))]
                deviation_magnitude_range_max = max(abs(PRICE_VS_SMA_RANGE[0] or 0), abs(PRICE_VS_SMA_RANGE[1] or 0))
                if deviation_magnitude_range_max > 1e-9: # Avoid division by zero in normalize_score
                     intermediate_scores['price_vs_sma'] = normalize_score(abs(deviation), 0, deviation_magnitude_range_max, higher_is_riskier=True)
                     logging.debug(f"[{ticker}] Price vs SMA50 Deviation: {deviation:.4f} (Abs: {abs(deviation):.4f}), Score: {intermediate_scores['price_vs_sma']:.2f}")
                else:
                     intermediate_scores['price_vs_sma'] = np.nan
                     effective_weights['price_vs_sma'] = 0.0
                     logging.warning(f"[{ticker}] Price vs SMA: Deviation magnitude range is zero or near zero ({deviation_magnitude_range_max}).")
            else:
                intermediate_scores['price_vs_sma'] = np.nan
                effective_weights['price_vs_sma'] = 0.0
                logging.debug(f"[{ticker}] Price vs SMA: SMA50 or Current Price calculation failed or SMA50 is zero/negative.")
        else:
            intermediate_scores['price_vs_sma'] = np.nan
            effective_weights['price_vs_sma'] = 0.0
            logging.debug(f"[{ticker}] Price vs SMA: Insufficient data ({len(close_prices)} prices). Need >= 50 for SMA50.")


    # 6. VIX (market-wide volatility, higher = riskier) - Fetch separately using retry decorator
    # This factor adds market context and is less dependent on the individual stock's history/info
    # VIX score is added regardless of whether stock data was fully available, as long as VIX data is available
    vix = _get_vix_close_retry() # Call the inner cached, retry-decorated function
    if vix is not None and pd.notna(vix):
        intermediate_scores['vix'] = normalize_score(vix, *VIX_RANGE)
        logging.debug(f"[{ticker}] VIX: {vix:.2f}, Score: {intermediate_scores['vix']:.2f}")
    else:
        intermediate_scores['vix'] = np.nan
        effective_weights['vix'] = 0.0
        logging.debug(f"[{ticker}] VIX: Not available.")


    # Calculate final score
    logging.debug(f"[{ticker}] Intermediate Scores (before filtering NaN): {intermediate_scores}")
    valid_scores = {k: v for k, v in intermediate_scores.items() if not np.isnan(v)}
    logging.debug(f"[{ticker}] Valid Scores: {valid_scores}")

    # Only use weights for factors that were successfully calculated
    # Ensure weights for simplified factors are used
    valid_weights_mapping = {k: effective_weights.get(k, 0) for k in valid_scores.keys()}
    weight_sum = sum(valid_weights_mapping.values())
    logging.debug(f"[{ticker}] Valid Weights (mapped): {valid_weights_mapping}")
    logging.debug(f"[{ticker}] Calculated Weight Sum for final score: {weight_sum:.4f}")


    final_score = None
    if weight_sum > 1e-9:  # Check against a small epsilon to avoid near-zero division
        # Normalize weights to sum to 1 using only weights of valid factors
        normalized_weights = {k: v / weight_sum for k, v in valid_weights_mapping.items()}
        final_score = sum(score * normalized_weights[factor] for factor, score in valid_scores.items())
        # Apply a base score offset and clip
        final_score = min(final_score + 35.0, 100.0)  # Add offset, cap at 100. Offset might need tuning.
        final_score = max(final_score, 0.0) # Ensure not below 0
        logging.info(f"Simplified Risk score for {ticker} calculated: {final_score:.2f} (using {len(valid_scores)}/{len(simplified_factors)} factors)")
    else:
        logging.warning(f"[{ticker}] No valid factors calculated or total effective weight is zero. Cannot calculate risk score.")
        final_score = None  # Return None if no valid factors
    return final_score, intermediate_scores, weight_sum


def get_risk_category(score):
    """Categorizes a risk score into Low, Medium, or High."""
    # Also check for pandas NaN explicitly
    if score is None or pd.isna(score): return "N/A"
    try:
        score_float = float(score);
        if score_float >= 65: return "⚠️ High Risk"
        elif score_float >= 50: return "⚖️ Medium Risk"
        else: return "✅ Low Risk"
    except (ValueError, TypeError): logging.error(f"Could not convert risk score '{score}' to float for categorization."); return "N/A"


# @st.cache_data(ttl=POLYGON_HISTORY_CACHE_DURATION_SECONDS) # Cache is handled at the fetch/calc_indicators level
def generate_trading_signals(ticker: str, indicators_df: pd.DataFrame):
    """
    Generates simple BUY/SELL/HOLD signals based on the latest momentum indicator values.
    Requires a DataFrame with indicator columns (RSI, MACD, MACD_Signal, CCI, Stoch_%K, Williams_%R).
    """
    logging.info(f"Generating trading signals for {ticker}")
    try:
        if indicators_df is None or indicators_df.empty:
            logging.warning(f"No indicator data provided for trading signals for {ticker}.")
            return "Trading Signals: Unavailable (No indicator data)", []

        # Ensure required indicator columns exist AND have valid data in the latest row
        required_indicators = ['RSI', 'MACD', 'MACD_Signal', 'CCI', 'Stoch_%K', 'Williams_%R']
        if not all(col in indicators_df.columns for col in required_indicators):
            missing = [col for col in required_indicators if col not in indicators_df.columns]
            logging.error(f"Missing required indicator columns for {ticker}: {missing}")
            return f"Trading Signals: Error - Missing indicator data ({', '.join(missing)})", []

        # Get the latest row AFTER dropping initial NaNs in calculate_momentum_indicators
        if indicators_df.empty:
             logging.warning(f"Indicator data became empty after dropping initial NaNs for {ticker}.")
             return "Trading Signals: Unavailable (Insufficient valid indicator data)", []

        # Get the latest row, ensuring it's not all NaN for the required columns
        latest = indicators_df.iloc[-1][required_indicators] # Get the latest values

        # Check if the latest row has *any* valid indicator data (at least one non-NaN value)
        if latest.isna().all():
             logging.warning(f"Latest row of indicator data is all NaN for {ticker}. Cannot generate signals.")
             return "Trading Signals: Unavailable (Latest indicator data incomplete)", []


        signals = []

        # RSI
        # Ensure indicator value is not NaN before using
        if latest['RSI'] is not None and pd.notna(latest['RSI']):
            if latest['RSI'] < 30: signals.append("BUY (RSI Oversold)")
            elif latest['RSI'] > 70: signals.append("SELL (RSI Overbought)")
            logging.debug(f"RSI: {latest['RSI']:.2f}") # Log indicator value

        # MACD
        if latest['MACD'] is not None and pd.notna(latest['MACD']) and latest['MACD_Signal'] is not None and pd.notna(latest['MACD_Signal']):
            logging.debug(f"MACD: {latest['MACD']:.4f}, Signal: {latest['MACD_Signal']:.4f}") # Log indicator values
            # MACD crosses Signal line (Bullish cross)
            # Check current MACD > Signal AND previous MACD <= Signal (requires at least 2 rows)
            if len(indicators_df) >= 2:
                prev = indicators_df.iloc[-2][required_indicators]
                # Ensure previous values are also valid before checking crossover
                if prev['MACD'] is not None and pd.notna(prev['MACD']) and prev['MACD_Signal'] is not None and pd.notna(prev['MACD_Signal']):
                    logging.debug(f"MACD Previous: {prev['MACD']:.4f}, Signal Previous: {prev['MACD_Signal']:.4f}")
                    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                         signals.append("BUY (MACD Bullish Crossover)")
                    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                         signals.append("SELL (MACD Bearish Crossover)")
                else:
                    logging.debug("Previous MACD/Signal data invalid, checking current position only.")
                    # Fallback to simple MACD vs Signal position if not enough data for crossover or prev data is invalid
                    if latest['MACD'] > latest['MACD_Signal']: signals.append("BUY (MACD Above Signal)")
                    elif latest['MACD'] < latest['MACD_Signal']: signals.append("SELL (MACD Below Signal)")
            else:
                 logging.debug("Insufficient data for MACD crossover (need >= 2 rows), checking current position only.")
                 # Fallback to simple MACD vs Signal position if not enough data for crossover
                 if latest['MACD'] > latest['MACD_Signal']: signals.append("BUY (MACD Above Signal)")
                 elif latest['MACD'] < latest['MACD_Signal']: signals.append("SELL (MACD Below Signal)")
        else:
            logging.debug("MACD or Signal data not available.")


        # CCI
        if latest['CCI'] is not None and pd.notna(latest['CCI']):
            logging.debug(f"CCI: {latest['CCI']:.2f}")
            if latest['CCI'] < -100: signals.append("BUY (CCI Oversold)")
            elif latest['CCI'] > 100: signals.append("SELL (CCI Overbought)")

        # Stochastic (%K crossing 20/80, or %K crossing %D 20/80)
        # Let's use %K crossing thresholds for simplicity, as %D requires more logic
        if latest['Stoch_%K'] is not None and pd.notna(latest['Stoch_%K']):
             logging.debug(f"Stoch %K: {latest['Stoch_%K']:.2f}")
             if latest['Stoch_%K'] < 20: signals.append("BUY (Stoch %K Oversold)")
             elif latest['Stoch_%K'] > 80: signals.append("SELL (Stoch %K Overbought)")
             # Could add %K/%D crossover signals here if desired

        # Williams %R
        if latest['Williams_%R'] is not None and pd.notna(latest['Williams_%R']):
            logging.debug(f"Williams %R: {latest['Williams_%R']:.2f}")
            # Williams %R is inverted compared to Stochastic (%R < -80 is oversold/BUY, %R > -20 is overbought/SELL)
            if latest['Williams_%R'] <= -80: signals.append("BUY (Williams %R Oversold)")
            elif latest['Williams_%R'] >= -20: signals.append("SELL (Williams %R Overbought)")


        # --- Determine Final Decision based on signal count ---
        if not signals:
            final_decision = "HOLD ✋"
        else:
            buy_signals_count = sum(1 for s in signals if "BUY" in s)
            sell_signals_count = sum(1 for s in signals if "SELL" in s)

            if buy_signals_count > sell_signals_count:
                final_decision = "BUY ✅"
            elif sell_signals_count > buy_signals_count:
                final_decision = "SELL ❌"
            else:
                final_decision = "HOLD ✋" # Equal buy/sell signals or only neutral signals (though no neutral signals defined here)

        # Refine the signals string
        signals_str = ', '.join(signals) if signals else 'None detected (all indicators neutral)' # Added this line

        summary = f"Trading Signals (Polygon.io, Momentum Indicators):\n- Final Decision: {final_decision}\n- Signals: {signals_str}" # Used signals_str here
        logging.info(f"Trading signals generated for {ticker}: {summary}")
        return summary, signals
    except Exception as e:
        logging.error(f"Error generating trading signals for {ticker}: {e}", exc_info=True)
        return f"Trading Signals: Error during calculation for {ticker}", []


# --- Technical Strategy Functions REMOVED ---
# --- Strategy Scanning Function REMOVED ---

# [LLM Prompt Formatting functions remain the same]
def format_stock_data_for_prompt(data):
    """Formats key Yahoo Finance data into a human-readable string for the LLM prompt."""
    if not data or not data.get('ticker'): return "No current Yahoo Finance summary data found for this ticker."
    ticker = data['ticker']; lines = [ f"Context - Summary data for {ticker} ({data.get('companyName', 'N/A')}) from Yahoo Finance:", f"- Current/Last Price: {format_val(data.get('priceForDisplay'), '$', prec=2)}", f"- Market Cap: {format_val(data.get('marketCap'), '$', prec=2)}", f"- P/E (Trailing): {format_val(data.get('trailingPE'), prec=2)}", f"- P/E (Forward): {format_val(data.get('forwardPE'), prec=2)}", f"- Dividend Yield: {format_val(data.get('dividendYield', 0) * 100 if data.get('dividendYield') is not None else 0, suffix='%', prec=2)}", f"- 50d SMA: {format_val(data.get('sma50'), '$', prec=2)}", f"- 200d SMA: {format_val(data.get('sma200'), '$', prec=2)}", f"- Beta: {format_val(data.get('beta'), prec=2)}", f"- Day Range: {format_val(data.get('dayLow'), '$', prec=2)} - {format_val(data.get('dayHigh'), '$', prec=2)}", f"- 52 Week Range: {format_val(data.get('fiftyTwoWeekLow'), '$', prec=2)} - {format_val(data.get('fiftyTwoWeekHigh'), '$', prec=2)}", f"- Volume: {format_val(data.get('volume'), prec=0)}", ]
    rec_key = data.get('recommendationKey'); num_opinions = data.get('numberOfAnalystOpinions'); mean_target = data.get('targetMeanPrice'); low_target = data.get('targetLowPrice'); high_target = data.get('targetHighPrice'); mapped_rec = map_recommendation_key_to_english(rec_key); analyst_line = "- Analyst Consensus (Aggregated Yahoo): Not Available"
    # Construct analyst line only if relevant data exists
    if mapped_rec not in ["Not Available", "N/A"] and (mean_target is not None or (num_opinions is not None and isinstance(num_opinions, (int, float)) and num_opinions > 0)):
        analyst_count_str = f"{num_opinions} analysts" if num_opinions is not None and isinstance(num_opinions, (int, float)) and num_opinions > 0 else "count unavailable";
        target_mean_str = format_val(mean_target, '$', prec=2);
        target_low_str = format_val(low_target, '$', prec=2);
        target_high_str = format_val(high_target, '$', prec=2);
        # Only add range if mean target is also available and low/high are not 'Not Available'
        range_str = f", ranging from {target_low_str} to {target_high_str}" if target_mean_str != "Not Available" and target_low_str != "Not Available" and target_high_str != "Not Available" else "";
        analyst_line = f"- Analyst Consensus (Aggregated Yahoo): Based on {analyst_count_str}, recommendation: {mapped_rec}"
        if target_mean_str != "Not Available": analyst_line += f", avg target: {target_mean_str}{range_str}."
        else: analyst_line += " (target price N/A)."
    lines.append(analyst_line)
    if data.get('sector') and data['sector'] != 'N/A': lines.append(f"- Sector: {data['sector']}")
    if data.get('industry') and data['industry'] != 'N/A': lines.append(f"- Industry: {data['industry']}")

    # Filter out lines with "Not Available" or "N/A" unless it's the analyst line
    filtered_lines = [lines[0]] + [ln for ln in lines[1:] if not (ln.strip().endswith(": Not Available") or ln.strip().endswith(": N/A") ) or ln.startswith("- Analyst Consensus")];

    # Check if core data (like price) is present
    if not any("Price" in ln for ln in filtered_lines): return f"Could not retrieve key data (like price) from Yahoo for {ticker}."; # Simplified message


    return "\n".join(filtered_lines)

def format_val(v, prefix="", suffix="", prec=2):
    """Formats a numeric value nicely with currency/percentage/scale suffixes."""
    if v is None or pd.isna(v) or str(v).lower() == 'n/a' or str(v).strip() == '': return "Not Available"
    try:
        v_float = float(v);
        if prec > 0 or v_float != int(v_float): # Use float formatting if precision > 0 or value has decimal part
            # Scale formatting for large numbers (only for non-percentage values)
            if suffix != '%':
                 if abs(v_float) >= 1e12: formatted_num = f"{v_float / 1e12:,.{prec}f}T"
                 elif abs(v_float) >= 1e9: formatted_num = f"{v_float / 1e9:,.{prec}f}B"
                 elif abs(v_float) >= 1e6: formatted_num = f"{v_float / 1e6:,.{prec}f}M"
                 else: formatted_num = f"{v_float:,.{prec}f}" # Standard float format for smaller numbers
            else: # Percentage formatting - no T/B/M scaling
                 formatted_num = f"{v_float:,.{prec}f}"
        else: # Integer formatting if precision is 0 and value is whole number
            formatted_num = f"{int(v_float):,}"
        return f"{prefix}{formatted_num}{suffix}"
    except (ValueError, TypeError): return str(v).strip() if str(v).strip() else "Not Available"

def map_recommendation_key_to_english(key):
    """Maps Yahoo Finance recommendation keys to human-readable strings."""
    mapping = { 'strong_buy': 'Strong Buy', 'buy': 'Buy', 'hold': 'Hold', 'sell': 'Sell', 'strong_sell': 'Strong Sell', 'underperform': 'Underperform', 'outperform': 'Outperform', 'none': 'N/A' }
    if key is None: return "Not Available"
    return mapping.get(str(key).lower(), str(key).capitalize() if key else "Not Available")


# [Wikipedia Index Lookup functions]
# --- START: WIKIPEDIA LOOKUP FUNCTIONS ---
def build_sp500_ticker_map(cache_duration_hours=24, force_refresh=False):
    """Builds or loads a mapping of S&P 500 company names to tickers from Wikipedia."""
    cache_file = "sp500_data.pkl"
    ticker_map = None
    loaded_from_cache = False
    logging.info(f"Checking S&P 500 cache (file: {cache_file}, force_refresh={force_refresh}).")
    if not force_refresh and os.path.exists(cache_file):
        try:
            cache_data = pd.read_pickle(cache_file)
            last_fetch_time = cache_data.get('timestamp', 0)
        except Exception as e:
            logging.warning(f"S&P 500 cache read error: {e}.")
            last_fetch_time = 0
        if (time.time() - last_fetch_time) / 3600 < cache_duration_hours:
            logging.info("Using cached S&P 500 data.")
            ticker_map = cache_data.get('ticker_map')
            loaded_from_cache = bool(ticker_map)
        else:
            logging.info("S&P 500 cache expired.")
    if ticker_map is None:
        logging.info("Fetching fresh S&P 500 data.")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; FinancialBot/1.0; +http://example.com/bot)'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = io.StringIO(response.text)
            tables = pd.read_html(html_content, flavor='lxml')
            sp500_table = tables[0] # S&P 500 table is typically the first one
            logging.info(f"S&P 500 table columns (raw): {sp500_table.columns.tolist()}") # Log columns
            logging.info(f"S&P 500 table rows (raw): {len(sp500_table)}") # Log row count

            ticker_col = 'Symbol'
            name_col = 'Security'

            if ticker_col not in sp500_table.columns or name_col not in sp500_table.columns:
                logging.error(f"Required columns '{ticker_col}' or '{name_col}' not found in S&P 500 table. Found: {sp500_table.columns.tolist()}")
                return None

            scraped_ticker_map = {}
            for _, row in sp500_table.iterrows():
                ticker_val, name_val = row.get(ticker_col), row.get(name_col)
                if isinstance(ticker_val, str) and isinstance(name_val, str) and ticker_val.strip() and name_val.strip():
                    ticker_clean = ticker_val.strip().replace('.', '-')
                    name_lower = name_val.strip().lower()
                    name_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc|co)\.?\b|\.$|,', '', name_lower, flags=re.IGNORECASE).strip()
                    scraped_ticker_map[name_lower] = ticker_clean
                    if name_cleaned and name_cleaned != name_lower and name_cleaned not in scraped_ticker_map:
                        scraped_ticker_map[name_cleaned] = ticker_clean
            ticker_map = scraped_ticker_map
            logging.info(f"Scraped {len(ticker_map)} S&P 500 entries before overrides.")
        except Exception as e:
            logging.error(f"Error fetching S&P 500 data: {e}")
            return None
    if ticker_map is not None:
        # Expanded overrides for top 50 stocks and Apple + TSLA (as it's now in S&P 500 too)
        overrides = {
            # Apple-specific mappings
            "apple": "AAPL", "apple inc": "AAPL", "apple inc.": "AAPL",
            # Top 50 stocks by market cap (approximated based on recent data)
            "microsoft": "MSFT", "microsoft corporation": "MSFT",
            "nvidia": "NVDA", "nvidia corporation": "NVDA",
            "amazon": "AMZN", "amazon.com": "AMZN",
            "meta": "META", "meta platforms": "META", "facebook": "META",
            "alphabet": "GOOGL", "google": "GOOGL", "alphabet class c": "GOOG",
            "tesla": "TSLA", "tesla inc": "TSLA", "tesla, inc.": "TSLA", # Added TSLA overrides here too
            "berkshire hathaway": "BRK-B", "berkshire hathaway inc": "BRK-B",
            "jpmorgan chase": "JPM", "jpmorgan chase & co": "JPM",
            "visa": "V", "visa inc": "V",
            "walmart": "WMT", "walmart inc": "WMT",
            "exxon mobil": "XOM", "exxon mobil corporation": "XOM",
            "unitedhealth group": "UNH", "unitedhealth group incorporated": "UNH",
            "mastercard": "MA", "mastercard incorporated": "MA",
            "procter & gamble": "PG", "procter & gamble company": "PG",
            "johnson & johnson": "JNJ",
            "home depot": "HD", "home depot inc": "HD",
            "costco wholesale": "COST", "costco wholesale corporation": "COST",
            "abbvie": "ABBV", "abbvie inc": "ABBV",
            "chevron": "CVX", "chevron corporation": "CVX",
            "merck": "MRK", "merck & co inc": "MRK",
            "coca-cola": "KO", "coca-cola company": "KO",
            "pepsico": "PEP", "pepsico inc": "PEP",
            "broadcom": "AVGO", "broadcom inc": "AVGO",
            "thermo fisher scientific": "TMO", "thermo fisher scientific inc": "TMO",
            "cisco systems": "CSCO", "cisco": "CSCO",
            "accenture": "ACN", "accenture plc": "ACN",
            "mcdonald's": "MCD", "mcdonald's corporation": "MCD",
            "pfizer": "PFE", "pfizer inc": "PFE",
            "salesforce": "CRM", "salesforce inc": "CRM",
            "bank of america": "BAC", "bank of america corporation": "BAC",
            "netflix": "NFLX", "netflix inc": "NFLX",
            "adobe": "ADBE", "adobe inc": "ADBE",
            "advanced micro devices": "AMD", "amd": "AMD",
            "linde": "LIN", "linde plc": "LIN",
            "qualcomm": "QCOM", "qualcomm incorporated": "QCOM",
            "intel": "INTC", "intel corporation": "INTC",
            "wells fargo": "WFC", "wells fargo & company": "WFC",
            "oracle": "ORCL", "oracle corporation": "ORCL",
            "applied materials": "AMAT", "applied materials inc": "AMAT",
            "union pacific": "UNP", "union pacific corporation": "UNP",
            "texas instruments": "TXN", "texas instruments incorporated": "TXN",
            "at&t": "T", "at&t inc": "T",
            "verizon communications": "VZ", "verizon": "VZ",
            "morgan stanley": "MS", "morgan stanley": "MS",
            "goldman sachs": "GS", "goldman sachs group inc": "GS",
            "comcast": "CMCSA", "comcast corporation": "CMCSA",
            "charles schwab": "SCHW", "charles schwab corporation": "SCHW",
            "intuit": "INTU", "intuit inc": "INTU",
            "amgen": "AMGN", "amgen inc": "AMGN",
            "paypal": "PYPL", "paypal holdings": "PYPL"
        }
        ticker_map.update(overrides)
        logging.info(f"S&P 500 map updated with {len(overrides)} overrides, total size: {len(ticker_map)}.")
        logging.info(f"Sample S&P 500 mappings: {dict(list(ticker_map.items())[:5])}") # Log sample
        if not loaded_from_cache or force_refresh:
            try:
                pd.to_pickle({'timestamp': time.time(), 'ticker_map': ticker_map}, cache_file)
                logging.info(f"Saved S&P 500 map to cache.")
            except Exception as e:
                logging.warning(f"Warning: Could not write S&P 500 cache: {e}")
    else:
        logging.error("ERROR: S&P 500 Ticker map is None.")
        return None
    return ticker_map


def build_nasdaq100_ticker_map(cache_duration_hours=24, force_refresh=False):
    """Builds or loads a mapping of Nasdaq 100 company names to tickers from Wikipedia."""
    cache_file = "nasdaq100_data.pkl"; ticker_map = None; loaded_from_cache = False
    logging.info(f"Checking Nasdaq 100 cache (file: {cache_file}, force_refresh={force_refresh}).")
    if not force_refresh and os.path.exists(cache_file):
        try: cache_data = pd.read_pickle(cache_file); last_fetch_time = cache_data.get('timestamp', 0)
        except Exception as e: logging.warning(f"Nasdaq 100 cache read error: {e}."); last_fetch_time = 0
        if (time.time() - last_fetch_time) / 3600 < cache_duration_hours: logging.info("Using cached Nasdaq 100 data."); ticker_map = cache_data.get('ticker_map'); loaded_from_cache = bool(ticker_map)
        else: logging.info("Nasdaq 100 cache expired.")
    if ticker_map is None:
        logging.info("Fetching fresh Nasdaq 100 data."); url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; FinancialBot/1.0; +http://example.com/bot)'}; response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); nasdaq_table = None
            try:
                html_content = io.StringIO(response.text); tables = pd.read_html(html_content, flavor='lxml')
                # Auto-find table: Look for a table with "Ticker" or "Symbol" and "Company" or "Security"
                logging.warning(f"Expected Nasdaq 100 table index might vary. Auto-finding...")
                found_table = False
                for i, df in enumerate(tables):
                    cols_lower = {str(col).lower() for col in df.columns}
                    has_ticker = any(t in cols_lower for t in ['ticker symbol', 'ticker', 'symbol']) # Include "Ticker symbol" as seen
                    has_name = any(n in cols_lower for n in ['company', 'security'])
                    # Nasdaq 100 has exactly 101 rows (including index row). Check for this count +/- a few.
                    if has_ticker and has_name and len(df) > 95 and len(df) < 110:
                        nasdaq_table = df
                        logging.info(f"Found Nasdaq 100 table at index {i}.")
                        found_table = True
                        break
                if not found_table: raise IndexError("Could not find Nasdaq 100 table with expected columns and row count.")
            except Exception as e: logging.error(f"Error reading Nasdaq 100 HTML: {e}."); return None

            # --- Add logging for table columns and row count ---
            if nasdaq_table is not None:
                 logging.info(f"Nasdaq 100 table columns: {nasdaq_table.columns.tolist()}")
                 logging.info(f"Nasdaq 100 table rows: {len(nasdaq_table)}")
            # --- End logging ---

            ticker_col, name_col = None, None; possible_ticker_cols = ['Ticker Symbol', 'Ticker', 'Symbol']; possible_name_cols = ['Company', 'Security']
            for col in nasdaq_table.columns:
                col_str = str(col)
                if col_str in possible_ticker_cols and ticker_col is None: ticker_col = col_str
                if col_str in possible_name_cols and name_col is None: name_col = col_str

            if not ticker_col or not name_col: logging.error(f"Could not find Nasdaq 100 columns (looked for {possible_ticker_cols} and {possible_name_cols}). Found: {nasdaq_table.columns.tolist()}"); return None

            scraped_ticker_map = {}
            for _, row in nasdaq_table.iterrows():
                 ticker_val, name_val = row.get(ticker_col), row.get(name_col)
                 if isinstance(ticker_val, str) and isinstance(name_val, str) and ticker_val.strip() and name_val.strip():
                    ticker_clean = ticker_val.strip().replace('.', '-');
                    name_lower = name_val.strip().lower();
                    name_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc|co)\.?\b|\.$|,', '', name_lower, flags=re.IGNORECASE).strip()
                    scraped_ticker_map[name_lower] = ticker_clean
                    if name_cleaned and name_cleaned != name_lower and name_cleaned not in scraped_ticker_map: scraped_ticker_map[name_cleaned] = ticker_clean
            ticker_map = scraped_ticker_map; logging.info(f"Scraped {len(ticker_map)} Nasdaq 100 entries before overrides.")
        except requests.exceptions.RequestException as e: logging.error(f"FATAL: Error fetching Nasdaq 100 URL '{url}': {e}"); return None
        except Exception as e: logging.error(f"FATAL: Unexpected error during Nasdaq 100 fetch: {e}", exc_info=True); return None

    if ticker_map is not None:
        # Add specific overrides (some overlap with S&P 500, ok) - Added TSLA overrides
        overrides = { "google": "GOOGL", "alphabet": "GOOGL", "alphabet class c": "GOOG", "alphabet inc.": "GOOGL",
                      "meta": "META", "facebook": "META", "meta platforms": "META", "fb": "META",
                      "amazon": "AMZN", "amazon.com": "AMZN",
                      "paypal": "PYPL", "paypal holdings": "PYPL",
                      "netflix": "NFLX",
                      "nvidia": "NVDA", "nvidia corporation": "NVDA",
                      "moderna": "MRNA",
                      "intel": "INTC", "intel corporation": "INTC",
                      "cisco": "CSCO", "cisco systems": "CSCO",
                      "adobe": "ADBE", "adobe inc.": "ADBE",
                      "tesla": "TSLA", "tesla, inc.": "TSLA", "tesla inc": "TSLA", # Added these overrides
                      "microsoft": "MSFT", "microsoft corporation": "MSFT",
                      "apple": "AAPL", "apple inc.": "AAPL",
                    }
        ticker_map.update(overrides); logging.info(f"Nasdaq 100 map updated with overrides, size: {len(ticker_map)}.")
        logging.info(f"Sample Nasdaq 100 mappings: {dict(list(ticker_map.items())[:5])}") # Log sample
        if not loaded_from_cache or force_refresh:
            try: pd.to_pickle({'timestamp': time.time(), 'ticker_map': ticker_map}, cache_file); logging.info(f"Saved Nasdaq 100 map to cache.")
            except Exception as e: logging.warning(f"Warning: Could not write Nasdaq 100 cache: {e}")
    else: logging.error("ERROR: Nasdaq 100 Ticker map is None."); return None
    return ticker_map

def get_ticker_from_combined_map(query, combined_map):
    """Looks up a ticker in the combined S&P 500 and Nasdaq 100 map by company name."""
    if not combined_map: logging.warning("Combined ticker map unavailable."); return None
    query_lower = query.lower().strip();
    if not query_lower: return None

    # Attempt direct match
    ticker = combined_map.get(query_lower)
    if ticker: logging.debug(f"Combined map direct hit for '{query_lower}': {ticker}"); return ticker

    # Attempt cleaned name match
    query_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc|co)\.?\b|\.$|,', '', query_lower, flags=re.IGNORECASE).strip()
    if query_cleaned != query_lower:
         ticker = combined_map.get(query_cleaned)
         if ticker: logging.debug(f"Combined map cleaned hit for '{query_lower}' -> '{query_cleaned}': {ticker}"); return ticker

    # Attempt some specific tricky cases not easily regex'd
    specific_tricks = {
        "coca cola": "coca-cola",
        "johnson and johnson": "johnson & johnson",
        "google": "alphabet", # Maps google to alphabet, which should then match GOOGL/GOOG
    }
    for tricky_in, tricky_out in specific_tricks.items():
        if query_lower == tricky_in:
            ticker = combined_map.get(tricky_out)
            if ticker: logging.debug(f"Combined map specific trick hit for '{query_lower}' -> '{tricky_out}': {ticker}"); return ticker
            # Also try cleaned version of the tricky output name if the direct map hit didn't work
            tricky_out_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc|co)\.?\b|\.$|,', '', tricky_out, flags=re.IGNORECASE).strip()
            if tricky_out_cleaned != tricky_out:
                 ticker = combined_map.get(tricky_out_cleaned)
                 if ticker: logging.debug(f"Combined map specific trick + cleaned hit for '{query_lower}' -> '{tricky_out_cleaned}': {ticker}"); return ticker


    # Attempt " Inc" removal
    if query_lower.endswith(" inc"):
        ticker = combined_map.get(query_lower[:-4].strip())
        if ticker: logging.debug(f"Combined map 'inc' variation hit for '{query_lower}': {ticker}"); return ticker

    logging.debug(f"Query '{query}' not found in combined map."); return None

@st.cache_resource(ttl=3600 * 12) # Cache for 12 hours
def load_combined_ticker_map():
    """Loads or builds the combined S&P 500 and Nasdaq 100 ticker map, with caching."""
    logging.info("\n" + "="*30 + " Building/Loading Combined Ticker Map " + "="*30);
    logging.info("--- Processing S&P 500 ---");
    sp500_map = build_sp500_ticker_map();
    if sp500_map is None: logging.warning("Failed to build S&P 500 map."); sp500_map = {}

    logging.info("\n--- Processing Nasdaq 100 ---");
    nasdaq100_map = build_nasdaq100_ticker_map()
    if nasdaq100_map is None: logging.warning("Failed to build Nasdaq 100 map."); nasdaq100_map = {}

    logging.info("\n--- Merging Maps ---");
    combined_tickers = sp500_map.copy();
    combined_tickers.update(nasdaq100_map); # Nasdaq 100 entries will overwrite S&P 500 if there's overlap (e.g., Apple)
    # --- Add logging for combined map size ---
    logging.info(f"Total entries in combined map: {len(combined_tickers)}");
    # --- End logging ---
    logging.info(" Combined Ticker Map Ready " + "="*30 + "\n");
    return combined_tickers

# --- New helper function for the direct yf history check with retry ---
@rate_limit_retry(max_attempts=3, initial_delay=5)
def check_ticker_history_with_retry(ticker):
    """Performs a yfinance history check for validation with rate limit retries."""
    logging.info(f"Attempting yf history check for '{ticker}' (inside retry)")
    ticker_obj = yf.Ticker(ticker)
    # Check history for the last day - if it exists, it's likely a valid symbol
    hist = ticker_obj.history(period="1d", interval="1d")
    if hist is None or hist.empty:
        # Raise an error if history is empty, so the retry decorator handles it
        logging.warning(f"yf history check for '{ticker}' returned empty history.")
        raise ValueError(f"No history found for ticker {ticker}")
    logging.info(f"yf history check for '{ticker}' successful.")
    return hist # Return hist if successful


# --- Rewritten fallback_yahoo_search function ---
# @st.cache_data(ttl=3600) # Caching applied in lookup_ticker_by_company_name caller
def fallback_yahoo_search(search_term):
    logging.info(f"Step 3: Falling back to Yahoo Finance Search API for '{search_term}'...")
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        # Increased quotesCount for more potential matches, filtered later
        params = {"q": search_term, "quotesCount": 20, "newsCount": 0}
        # Use a descriptive User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; FinancialBot/1.0; +https://yourwebsite.com/financialbot)'} # Replace with your bot info/website
        res = requests.get(url, params=params, headers=headers, timeout=10)
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = res.json()

        quotes = data.get("quotes", [])
        allowed_quote_types = ["EQUITY", "ETF"]
        # Add common major exchanges suffix or exchDisp
        # Expanded list of allowed exchanges/suffixes
        allowed_exchanges_suffix = ['.TA', '.TL', '.AS', '.BR', '.DE', '.PA', '.L', '.TO', '.V', '.HE', '.SW', '.OL', '.VI', '.IC', '.IR', '.MI', '.LS', '.MC', '.VX', '.ST', '.CO', '.TR'] # Added OSL, VIE, ICE, IRL, MIL, LIS, MAD, SWX, STO, CPH, IST
        allowed_exchanges_disp = ["TLV", "TASE", "NMS", "NYQ", "ASE", "NASDAQ", "NYSE", "AMEX", "BATS", "AMS", "BRU", "GER", "PAR", "LSE", "TOR", "VAN", "HEL", "EBS", "OSL", "VIE", "ICE", "IRL", "MIL", "LIS", "MAD", "SWX", "STO", "CPH", "IST"]

        highest_score = -1 # Start below 0 so any valid score is higher
        best_match = None
        search_term_lower = search_term.lower()

        logging.debug(f"Step 3: Found {len(quotes)} raw search results.")

        for item in quotes:
            symbol = item.get("symbol")
            quote_type = item.get("quoteType")
            score = item.get("score", 0) or 0 # Use 0 if score is None or missing
            short_name = item.get("shortname", "").lower()
            long_name = item.get("longname", "").lower()
            exch_disp = item.get("exchDisp", "")
            is_yahoo_finance = item.get("isYahooFinance", False) # Prioritize results flagged as primary

            # Basic filters: Valid symbol, allowed type, not an index/future/option/currency/mutual fund
            if not symbol or quote_type not in allowed_quote_types or '^' in symbol or any(ft in quote_type.upper() for ft in ['FUTURE', 'INDEX', 'CURRENCY', 'OPTION', 'MUTUALFUND']):
                 logging.debug(f"Step 3 Filtered (Type/Symbol): {symbol} ({quote_type})"); continue

            # Exchange filter: must be a US exchange or one of the explicitly allowed international ones
            is_allowed_exchange = False
            if '.' in symbol:
                 suffix = '.' + symbol.split('.')[-1].upper()
                 if suffix in allowed_exchanges_suffix: is_allowed_exchange = True
            # Also check exchDisp for both US and international listings
            if exch_disp in allowed_exchanges_disp or exch_disp in ["NMS", "NYQ", "ASE", "NASDAQ", "NYSE", "AMEX", "BATS"]:
                 is_allowed_exchange = True

            if not is_allowed_exchange:
                 logging.debug(f"Step 3 Filtered (Exchange Suffix/Disp: {symbol.split('.')[-1].upper() if '.' in symbol else 'N/A'}/{exch_disp}): {symbol}"); continue


            # Calculate a relevance score - refined scoring logic
            current_score = score # Start with Yahoo's score

            # Bonus for exact name matches (case-insensitive)
            if search_term_lower == short_name: current_score += 2000 # Strongest match
            elif search_term_lower == long_name: current_score += 1000
            # Bonus for exact ticker match (case-insensitive)
            if search_term_lower.upper() == symbol.upper(): current_score += 1500 # Very strong match

            # Bonus for starts with name matches
            if short_name.startswith(search_term_lower): current_score += 500
            elif long_name.startswith(search_term_lower): current_score += 250

            # Bonus for contains name matches, weighted by length ratio
            if short_name and search_term_lower in short_name: current_score += (len(search_term_lower) / len(short_name)) * 100
            elif long_name and search_term_lower in long_name: current_score += (len(search_term_lower) / len(long_name)) * 50

            # Bonus for primary Yahoo Finance listing
            if is_yahoo_finance: current_score += 50

            # Penalize indices/futures/etc. even if they somehow slipped filters (belt-and-suspenders)
            if any(ft in quote_type.upper() for ft in ['FUTURE', 'INDEX', 'CURRENCY', 'OPTION', 'MUTUALFUND']):
                 current_score -= 500 # Large penalty

            # Small bonus for US exchanges if relevant (can be tuned)
            if exch_disp in ["NMS", "NYQ", "ASE", "NASDAQ", "NYSE", "AMEX", "BATS"]:
                 current_score += 5

            logging.debug(f"Step 3 Candidate: {symbol} ({quote_type}, {exch_disp}), Score: {current_score:.2f}, Names: '{short_name}'/'{long_name}'")

            if current_score > highest_score:
                highest_score = current_score; best_match = symbol

        if best_match and highest_score > 0: # Only return a match if the score is positive (i.e. better than initial -1)
            logging.info(f"Step 3 SUCCESS: Best match from Fallback Search (Score: {highest_score:.2f}): {best_match}"); return best_match.upper()
        else:
            logging.info(f"Step 3 FAILED: No suitable EQUITY/ETF found via Fallback Search (best score {highest_score:.2f})."); return None

    except requests.exceptions.RequestException as e:
        logging.warning(f"Step 3 EXCEPTION (Request): Fallback search failed: {e}")
        return None
    except json.JSONDecodeError:
        logging.warning(f"Step 3 EXCEPTION (JSON): Fallback search received invalid JSON.")
        return None
    except Exception as e:
        logging.warning(f"Step 3 EXCEPTION (Other): Fallback search failed: {e}", exc_info=False) # Don't log stack trace for common search errors
        return None


# --- Load Combined Ticker Map ---
COMBINED_TICKERS = load_combined_ticker_map()

# --- Chat Management ---
if "messages" not in st.session_state: st.session_state.messages = []
if "predefined_question" not in st.session_state: st.session_state.predefined_question = None

# --- UI Elements ---
# Updated Title
st.markdown('<h1 style="text-align: left;">📈 Financial Chat, Risk Score, News & Forecasting</h1>', unsafe_allow_html=True)
# Updated description
st.markdown(f'<p style="text-align: left; font-size: small;"><br>Ask about stocks ($AAPL, Microsoft), compare, or discuss finance. Includes Dynamic Risk Score, Recent News Sentiment (Multi-Source/{NEWS_DAYS_BACK}d/VADER), and ETS Price Forecasting (Calculated). No charts or technical scans.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.markdown("## Examples")
    # Updated MENU_OPTIONS - removed Scan Signals
    MENU_OPTIONS = {
        "🔍 Stock Info": ["What's up with $TSLA?", "$KO", "$TEVA", "MSFT data?", "3M Company info?"],
        "📊 TA Concepts": ["What is SMA?", "Explain Moving Averages?", "What is Support/Resistance?", "Candlesticks?", "What are technical indicators?"], # Kept TA concepts
        "⚖️ Risk": [" $AMD?", "Explain the risk score model", " $TQQQ?", " GOOG?"],
        "📰 News Sentiment": ["News for $MSFT?", " News for $NVDA?", "News for for META?"],
        "📈 ETS Forecast": ["Forecast $AAPL price", "What's the ETS forecast for $MSFT?", "Price projection for $GOOG?"],
        "💼 Portfolio": ["How to diversify?", "Risks of single stocks?"],
        "📰 Market/General": ["Impact of interest rates?", "Inflation effect?", "What are ETFs?"],
    }
    for category, questions in MENU_OPTIONS.items():
        # Expanded default categories adjusted
        is_expanded = (category in ["🔍 Stock Info", "⚖️ Risk", "📰 News Sentiment", "📈 ETS Forecast"])
        with st.expander(f"**{category}**", expanded=is_expanded):
            for i, q in enumerate(questions):
                safe_category = re.sub(r'\W+', '', category); button_key = f"menu_{safe_category}_{i}"
                if st.button(q, key=button_key, use_container_width=True): st.session_state.predefined_question = q; st.rerun() # Use st.rerun()

    st.caption("Click a question to ask."); st.divider(); st.info("Enter a ticker symbol ($GOOGL) or company name (Microsoft, 3M) for specific data."); st.divider()

    # Display API Key warnings in sidebar
    # Riskfolio is no longer used, so no need to warn about it
    # if not RISKFOLIO_AVAILABLE: st.warning("Riskfolio-Lib not found. Some advanced risk factors/methods disabled.", icon="⚠️")
    # API key checks are done at the start, assuming they stop the app if critical keys are missing/invalid.
    # Optional: Display warnings here if keys *were* provided but were invalid, if the app didn't stop.
    # For now, relying on the initial st.error and st.stop() is sufficient.


# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(str(msg["content"]), unsafe_allow_html=True)

# --- Ticker Extraction ---
# Added more common non-ticker words + some TA/finance concepts
FORBIDDEN_TICKERS = {"TELL", "LOVE", "LIFE", "SOLO", "PLAY", "YOU", "REAL", "CASH", "WORK", "HOPE", "GOOD", "SAFE", "FAST", "COOK", "HUGE", "YOLO", "BOOM", "DUDE", "WISH", "ME", "ARE", "IS", "THE", "FOR", "AND", "NOW", "SEE", "CAN", "HAS", "WAS", "BUY", "SELL", "ALL", "ONE", "TWO", "BIG", "NEW", "OLD", "TOP", "LOW", "HIGH", "DATA", "FREE", "NEWS", "RISK", "CHART", "ETF", "FUND", "INDEX", "STOCK", "SHARES", "PRICE", "TRADE", "HOLD", "EXIT", "ENTRY","WHAT","ABOUT","ME","SENTIMENT","RISK","surge", "plunge", "spike", "dip", "correction", "crash", "rally", "breakout", "reversal", "pullback", "capital", "liquidity", "float", "burn rate", "runway", "dry powder",
 "trade", "order", "fill", "execution", "volume", "spread", "slippage", "scalping",
 "bullish", "bearish", "FOMO", "HODL", "panic sell", "greed", "fear",
 "investor", "trader", "market maker", "broker", "institutional investor",
 "drawdown", "stop-loss", "risk/reward", "volatility", "leverage",
 "earnings", "PE ratio", "support", "resistance", "RSI", "MACD", "candlestick", "volume profile",
 "forecast", "prediction", "projection", "technical", "fundamental", "analysis",
 "average", "moving", "simple", "exponential", "deviation", "standard", "beta",
 "dividend", "yield", "cap", "market", "sector", "industry", "economic", "inflation", "interest", "rates",
 "report", "statement", "balance", "income", "cashflow", "profit", "revenue", "asset", "liability", "equity",
 "bond", "treasury", "future", "option", "currency", "commodity", "gold", "oil",
 "vix", "cpi", "fed", "federal", "reserve", "ecb"}
def extract_tickers(text):
    """Extracts potential ticker symbols (prefixed with $) or standalone words that might be tickers."""
    # Find words that look like tickers, potentially prefixed with $
    # Relaxed regex slightly to allow more potential symbols, relying more on subsequent lookup validation
    potential_candidates = re.findall(r"(?<![a-zA-Z0-9_])(?:[$])?([A-Za-z0-9\-\.]{1,10})\b", text, re.IGNORECASE)
    # Remove duplicates, clean up, filter by length and the forbidden list
    # Ensure ticker length is >= 1 and <= 10 after cleaning '$'
    clean_tickers = [t.upper().replace('$', '') for t in potential_candidates if t and 1 <= len(t.replace('$','')) <= 10 and not t.replace('$','').isdigit()] # Filter out pure numbers
    # Filter forbidden words - apply uppercase to comparison
    unique_clean = [t for t in clean_tickers if t not in FORBIDDEN_TICKERS]
    # Final deduplication preserving order (optional but good practice)
    seen = set(); unique_clean_deduped = [t for t in unique_clean if not (t in seen or seen.add(t))]

    logging.info(f"Extracted potential ticker candidates: {potential_candidates}, filtered: {unique_clean_deduped} from text: '{text}'");
    return unique_clean_deduped


@st.cache_data(ttl=3600) # Cache Yahoo search results for 1 hour
def lookup_ticker_by_company_name(query):
    """Attempts to find a stock ticker for a given query (ticker or company name)."""
    if not query or len(query.strip()) < 1: return None
    search_term = query.strip(); logging.info(f"=== Starting Ticker Lookup for: '{search_term}' ===");

    # Step 1: Check if the query itself is a plausible ticker
    direct_ticker_attempt = search_term.upper().replace('$', '').strip();
    # A plausible ticker is 1-10 alphanumeric chars, allows hyphens/dots (for international/OTC),
    # but filter out long strings that are only digits (unlikely tickers).
    # Also filter against common forbidden words immediately
    is_potential_ticker = bool(re.fullmatch(r'[A-Z0-9\-\.]{1,10}', direct_ticker_attempt)) and not (direct_ticker_attempt.isdigit() and len(direct_ticker_attempt) > 4) and direct_ticker_attempt not in FORBIDDEN_TICKERS

    if is_potential_ticker:
        logging.info(f"Step 1: Query '{search_term}' looks like ticker '{direct_ticker_attempt}'. Direct yf check with retry...")
        try:
            # Use the decorated helper function for the history check
            hist = check_ticker_history_with_retry(direct_ticker_attempt)
            # If check_ticker_history_with_retry returns without raising, history was found
            # Note: check_ticker_history_with_retry already logs success/warning internally

            # Optional: Check info to filter out non-EQUITY/ETF if stricter filtering needed,
            # but history is a strong validation for existence. Let's add a quick info check here
            # WITHOUT retry, just one attempt, primarily for quoteType filtering.
            try:
                quick_info = yf.Ticker(direct_ticker_attempt).info
                if quick_info and quick_info.get('quoteType', '').upper() not in ['EQUITY', 'ETF']:
                     logging.warning(f"Step 1 FAILED: Ticker '{direct_ticker_attempt}' exists but not supported type (Type: {quick_info.get('quoteType')}).")
                     # Fall through to next steps
                else:
                     logging.info(f"Step 1 SUCCESS: Direct yf '{direct_ticker_attempt}' confirmed via history (and optional type check).")
                     return direct_ticker_attempt.upper()
            except Exception as info_e:
                 # Log info fetch error but proceed if history check passed
                 logging.warning(f"Step 1 INFO check failed for '{direct_ticker_attempt}': {info_e}. History check passed, assuming valid.")
                 return direct_ticker_attempt.upper()


        except ValueError as ve: # Catch the specific error raised by check_ticker_history_with_retry on final failure
             logging.warning(f"Step 1 FAILED: Direct yf check '{direct_ticker_attempt}' - {ve} (after retries).")
             # Fall through to next steps
        except Exception as e: # Catch any other unexpected errors from the decorated function
             logging.warning(f"Step 1 EXCEPTION: Direct yf check failed for '{direct_ticker_attempt}': {e}.")
             # Fall through to next steps
    else: logging.info(f"Step 1: Query '{search_term}' not formatted like plausible ticker or is forbidden word.")

    # Step 2: Check Combined S&P/Nasdaq Map
    # Only check map if query is longer than a typical ticker (avoid ambiguous 1-3 letter lookups)
    if len(search_term) > 4 or '-' in search_term or ' ' in search_term:
        logging.info(f"Step 2: Checking Combined S&P/Nasdaq Map for '{search_term}'...")
        map_ticker = get_ticker_from_combined_map(search_term, COMBINED_TICKERS)
        if map_ticker:
            # Verify the map result with yfinance history as a safety check (using retry helper)
            try:
                hist_map = check_ticker_history_with_retry(map_ticker)
                # If history check passes, return the map ticker
                # Optional: Add quick info check here as well if stricter filtering is desired
                logging.info(f"Step 2 SUCCESS: Found '{search_term}' in Combined Map: {map_ticker}, yf history confirmed.");
                return map_ticker.upper()
            except ValueError as ve: # Catch the specific error raised by history check on final failure
                 logging.warning(f"Step 2 FAILED: Map result '{map_ticker}' from '{search_term}' not confirmed by yf history (after retries). {ve}")
                 # Fall through to step 3
            except Exception as e: # Catch any other unexpected errors
                 logging.warning(f"Step 2 EXCEPTION: yf check on map result '{map_ticker}' failed: {e}.")
                 # Fall through to step 3
        else:
             logging.info(f"Step 2 FAILED: Query '{search_term}' not in combined map.")
    else:
        logging.info(f"Step 2: Skipping Combined Map check for short query '{search_term}'.")


    # Step 3: Fallback to Yahoo Finance Search API
    # Pass the task to the rewritten function
    logging.info(f"Step 3: Attempting Fallback Yahoo Finance Search API for '{search_term}'...")
    fallback_match = fallback_yahoo_search(search_term)

    if fallback_match:
        # Optional: Verify fallback match with history check?
        # This adds latency but increases confidence. Let's skip for speed after a search API hit,
        # assuming the search API is somewhat reliable for EQUITY/ETF types.
        logging.info(f"Step 3 SUCCESS: Found match '{fallback_match}' via Fallback Search.")
        return fallback_match.upper()
    else:
        logging.info(f"Step 3 FAILED: No suitable match found via Fallback Search.")
        return None

    finally: logging.info(f"=== Finished Ticker Lookup for: '{search_term}' ===")


# --- Load Combined Ticker Map ---
COMBINED_TICKERS = load_combined_ticker_map()

# --- Chat Management ---
if "messages" not in st.session_state: st.session_state.messages = []
if "predefined_question" not in st.session_state: st.session_state.predefined_question = None

# --- UI Elements ---
# Updated Title
st.markdown('<h1 style="text-align: left;">📈 Financial Chat, Risk Score, News & Forecasting</h1>', unsafe_allow_html=True)
# Updated description
st.markdown(f'<p style="text-align: left; font-size: small;"><br>Ask about stocks ($AAPL, Microsoft), compare, or discuss finance. Includes Dynamic Risk Score, Recent News Sentiment (Multi-Source/{NEWS_DAYS_BACK}d/VADER), and ETS Price Forecasting (Calculated). No charts or technical scans.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.markdown("## Examples")
    # Updated MENU_OPTIONS - removed Scan Signals
    MENU_OPTIONS = {
        "🔍 Stock Info": ["What's up with $TSLA?", "$KO", "$TEVA", "MSFT data?", "3M Company info?"],
        "📊 TA Concepts": ["What is SMA?", "Explain Moving Averages?", "What is Support/Resistance?", "Candlesticks?", "What are technical indicators?"], # Kept TA concepts
        "⚖️ Risk": [" $AMD?", "Explain the risk score model", " $TQQQ?", " GOOG?"],
        "📰 News Sentiment": ["News for $MSFT?", " News for $NVDA?", "News for for META?"],
        "📈 ETS Forecast": ["Forecast $AAPL price", "What's the ETS forecast for $MSFT?", "Price projection for $GOOG?"],
        "💼 Portfolio": ["How to diversify?", "Risks of single stocks?"],
        "📰 Market/General": ["Impact of interest rates?", "Inflation effect?", "What are ETFs?"],
    }
    for category, questions in MENU_OPTIONS.items():
        # Expanded default categories adjusted
        is_expanded = (category in ["🔍 Stock Info", "⚖️ Risk", "📰 News Sentiment", "📈 ETS Forecast"])
        with st.expander(f"**{category}**", expanded=is_expanded):
            for i, q in enumerate(questions):
                safe_category = re.sub(r'\W+', '', category); button_key = f"menu_{safe_category}_{i}"
                if st.button(q, key=button_key, use_container_width=True): st.session_state.predefined_question = q; st.rerun() # Use st.rerun()

    st.caption("Click a question to ask."); st.divider(); st.info("Enter a ticker symbol ($GOOGL) or company name (Microsoft, 3M) for specific data."); st.divider()

    # Display API Key warnings in sidebar
    # Riskfolio is no longer used, so no need to warn about it
    # if not RISKFOLIO_AVAILABLE: st.warning("Riskfolio-Lib not found. Some advanced risk factors/methods disabled.", icon="⚠️")
    # API key checks are done at the start, assuming they stop the app if critical keys are missing/invalid.
    # Optional: Display warnings here if keys *were* provided but were invalid, if the app didn't stop.
    # For now, relying on the initial st.error and st.stop() is sufficient.


# --- User Input Handling ---
user_input_triggered = None
# Check for predefined question first
if st.session_state.predefined_question:
    user_input_triggered = st.session_state.predefined_question
    # Clear the session state flag immediately
    del st.session_state.predefined_question
    logging.info(f"Processing predefined: '{user_input_triggered}'")
# Then check chat input
else:
    # Updated input hint
    chat_input_value = st.chat_input(f"Ask about stocks ($AAPL, Microsoft), risk, news ({NEWS_DAYS_BACK}d), forecast, or finance...")
    if chat_input_value:
        user_input_triggered = chat_input_value.strip()
        logging.info(f"Processing user input: '{user_input_triggered}'")


# --- Main Processing Logic ---
if user_input_triggered:
    user_input = user_input_triggered
    # Add user message to history *before* starting processing
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare for assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty() # Use a placeholder to show intermediate status
        placeholder.markdown("⏳ Thinking...")

        # --- Ticker Identification & Lookup ---
        # Start with extracted tickers
        extracted_tickers = extract_tickers(user_input);
        lookup_query = None
        if extracted_tickers:
             # Take the first extracted ticker as the primary lookup query
             lookup_query = extracted_tickers[0]; logging.info(f"Prioritizing extracted ticker '{lookup_query}'.")
        else:
            # If no tickers extracted, try to identify a potential company name from the query structure
            # Command prefixes updated - removed 'scan'
            command_prefixes = ("compare ", "risk ", "risky ", "risk score for ", "what is the risk score for ", "news sentiment for ", "recent news sentiment for ", "sentiment analysis for ", "forecast ", "price forecast for ", "ets forecast for ", "price projection for for ", "tell me about ") # Added "tell me about"
            question_starters = ("what", "how", "explain", "who", "why", "list", "define", "is ", "are ")
            input_lower = user_input.lower()
            is_command_prefix = any(input_lower.startswith(p) for p in command_prefixes)
            is_short_question = any(input_lower.startswith(p) for p in question_starters) and len(input_lower.split()) < 5 # Short questions might be about a specific entity

            potential_name_part = user_input.strip();
            found_prefix = None

            if is_command_prefix:
                 for prefix in command_prefixes:
                     if input_lower.startswith(prefix):
                         potential_name_part = user_input[len(prefix):].strip();
                         found_prefix = prefix
                         break
            elif is_short_question:
                 # For short questions like "Is Apple?", try the main subject
                 parts = input_lower.split()
                 if len(parts) > 1: potential_name_part = " ".join(parts[1:])
                 else: potential_name_part = None # Query is just "What" or "Is", no name part


            # Only use the potential name part if it seems like a company name (not just a financial term)
            # Refined check: is it just one word that is a forbidden term? Or a very short phrase of forbidden terms?
            finance_terms_general_set = set(FORBIDDEN_TICKERS) # Use the set for faster lookups
            potential_name_lower = potential_name_part.lower() if potential_name_part else None
            if potential_name_lower:
                 # Check if the entire potential name is a forbidden term
                 is_just_forbidden_word = potential_name_lower in finance_terms_general_set
                 # Check if the potential name is a short phrase where all words are forbidden terms
                 is_short_phrase_of_forbidden = False
                 potential_words = potential_name_lower.split()
                 if 1 < len(potential_words) <= 3 and all(word in finance_terms_general_set for word in potential_words):
                     is_short_phrase_of_forbidden = True

                 if not is_just_forbidden_word and not is_short_phrase_of_forbidden:
                      lookup_query = potential_name_part; logging.info(f"Trying potential name part '{lookup_query}' for name lookup after '{found_prefix or ''}' prefix.")
                 else:
                      logging.info(f"Potential name part '{potential_name_part}' looks like a financial term or forbidden phrase. Skipping ticker lookup based on name part.")

            else:
                 logging.info("No obvious ticker or plausible name part found. Skipping ticker lookup.")


        # --- Validate the lookup query ---
        validated_ticker = None; primary_ticker = None
        if lookup_query:
            placeholder.markdown(f"⏳ Verifying '{lookup_query}'...")
            try: validated_ticker = lookup_ticker_by_company_name(lookup_query)
            except Exception as lookup_err:
                logging.error(f"Ticker lookup function failed: {lookup_err}", exc_info=True)
                validated_ticker = None # Ensure validation fails on unexpected errors

            if validated_ticker:
                primary_ticker = validated_ticker
                logging.info(f"Lookup success. Using primary ticker: {primary_ticker}")
                # Inform the user if the name lookup found a ticker they didn't specify directly
                is_query_like_ticker = bool(re.fullmatch(r'[\$\.A-Z0-9\-]{1,10}', lookup_query.upper().replace('$','')))
                # Only show toast if the query wasn't already a perfect or near-perfect match for the found ticker
                if not is_query_like_ticker and lookup_query.upper().replace('$','') != primary_ticker.upper():
                     st.toast(f"Found data for **{primary_ticker}** (based on '{lookup_query}')", icon="💡")
            else:
                logging.info(f"Could not validate/find equity/ETF ticker for '{lookup_query}'.")
                # Provide feedback to the user if a lookup was attempted but failed
                if lookup_query and len(lookup_query) > 1: # Avoid toast for single letters or empty queries
                     st.toast(f"Couldn't find stock/ETF matching '{lookup_query}'.", icon="⚠️")
                primary_ticker = None # Ensure primary_ticker is None if lookup failed
        else:
            logging.info("No lookup query generated from user input.")
            primary_ticker = None


        # --- Initialize Context Variables ---
        stock_data = None
        close_prices_history_yf = None # Store close prices from Yahoo history for ETS
        full_history_df_yf = None # Store full df from Yahoo history for Risk
        polygon_ohlcv_df = None # Store full df from Polygon for TA/Signals
        indicators_df_polygon = None # Store calculated indicators from Polygon
        trading_signals_summary = "Trading Signals: Not calculated."
        trading_signals_list = [] # List of individual signals
        news_sentiment_summary = "News Sentiment: Not calculated."
        news_sentiment_counts_dict = {'positive': None, 'negative': None, 'neutral': None, 'total': None, 'avg_score': None}
        news_articles_details = [] # Detailed list of news articles
        ets_forecast_summary = "ETS Price Forecast: Not calculated."
        stock_data_for_prompt = "No specific ticker identified or data failed."
        risk_score_final = None; risk_score_description = "Risk Score (Model): Not Calculated"
        risk_category = "N/A" # Initialize risk category

        # --- Fetch Data, Calculate Risk, News & Forecast if ticker identified ---
        if primary_ticker:
            with st.spinner(f"Gathering data & insights for **{primary_ticker}**..."):
                logging.info(f"--- Processing Data for Ticker: {primary_ticker} ---")

                # 1. Fetch Yahoo Summary Data (Uses retry-decorated function internally)
                placeholder.markdown(f"⏳ Fetching Yahoo Finance summary for **{primary_ticker}**...")
                # get_stock_data now calls get_stock_data_yf_retry internally and handles its errors
                stock_data = get_stock_data(primary_ticker)
                if stock_data:
                     stock_data_for_prompt = format_stock_data_for_prompt(stock_data)
                     logging.info(f"Yahoo summary fetch OK for {primary_ticker}.")
                else:
                     stock_data_for_prompt = f"Could not retrieve summary data from Yahoo for {primary_ticker}.";
                     logging.warning(f"Yahoo summary fetch failed for {primary_ticker}.")


                # 2. Fetch Unified Yahoo History (Needed for Risk & ETS) (Uses retry-decorated function internally)
                placeholder.markdown(f"⏳ Fetching Yahoo Finance history (for Risk/ETS) for **{primary_ticker}**...")
                # get_unified_yfinance_history now calls get_unified_yfinance_history_retry internally and handles its errors
                close_prices_history_yf, full_history_df_yf = get_unified_yfinance_history(primary_ticker, period="3y")
                if full_history_df_yf is None or full_history_df_yf.empty:
                     logging.warning(f"Unified Yahoo history fetch failed or empty for {primary_ticker}. Risk & ETS might be unavailable.")
                else:
                     logging.info(f"Unified Yahoo history fetch OK for {primary_ticker} ({len(full_history_df_yf)} rows).")


                # 3. Calculate Risk Score (Uses full Yahoo history df and info)
                placeholder.markdown(f"⏳ Calculating Dynamic Risk Score for **{primary_ticker}**...")
                intermediate_risk_scores = {}; factors_weight_sum = 0.0
                # Pass both history df and info dict to the simplified risk score function
                # Risk score calculation now has internal checks for input data presence
                risk_score_final, intermediate_risk_scores, factors_weight_sum = calculate_dynamic_risk_score(primary_ticker, full_history_df_yf, stock_data, weights=DEFAULT_WEIGHTS)


                if risk_score_final is not None and pd.notna(risk_score_final): # Check for pd.notna here too
                    risk_category = get_risk_category(risk_score_final)
                    risk_score_description = f"Risk Score (Model): {risk_score_final:.2f}/100 ({risk_category})"
                    logging.info(f"Risk score OK: {risk_score_description}.")
                else:
                    risk_score_description = f"Risk Score (Model): Could not calculate for {primary_ticker} (Insufficient history, data errors, or factors unavailable).";
                    risk_category = "N/A"
                    logging.warning(f"Simplified risk score calc returned None/NaN for {primary_ticker} or input data invalid.")
                    # Show a toast if risk score failed specifically for this ticker
                    # Only show toast if the ticker was actually found/validated
                    if primary_ticker: st.toast(f"⚠️ Risk score unavailable for {primary_ticker}. Requires historical price data and basic info.", icon="⚠️")


                # 4. Fetch Polygon.io data and Calculate Trading Signals
                placeholder.markdown(f"⏳ Fetching Polygon.io data & calculating trading signals for **{primary_ticker}**...")
                # Fetch enough history for indicators (e.g., 3 years)
                polygon_ohlcv_df = fetch_polygon_price_data(primary_ticker, days_back=365*3)

                if polygon_ohlcv_df is not None and not polygon_ohlcv_df.empty:
                    # Calculate indicators using the Polygon data
                    indicators_df_polygon = calculate_momentum_indicators(polygon_ohlcv_df)
                    # Check if indicators were calculated successfully and the latest row is not all NaN
                    if indicators_df_polygon is not None and not indicators_df_polygon.empty and not indicators_df_polygon.iloc[-1].isna().all():
                        # Generate signals from the latest indicator values
                        trading_signals_summary, trading_signals_list = generate_trading_signals(primary_ticker, indicators_df_polygon)
                        logging.info(f"Trading signals calculated for {primary_ticker}.")
                    else:
                        trading_signals_summary = f"Trading Signals: Calculation failed for {primary_ticker} (Indicator data incomplete or insufficient)."
                        trading_signals_list = []
                        logging.warning(f"Indicator calculation failed or returned empty/NaN for {primary_ticker}.")
                else:
                    trading_signals_summary = f"Trading Signals: Unavailable due to missing Polygon.io data for {primary_ticker}."
                    trading_signals_list = []
                    logging.warning(f"Polygon.io data fetch failed or empty for {primary_ticker}.")

                # Check if signals are unavailable and show a toast
                if "Unavailable" in trading_signals_summary or "Error" in trading_signals_summary or "failed" in trading_signals_summary.lower() or trading_signals_list == []:
                     if primary_ticker: st.toast(f"⚠️ Trading signals unavailable for {primary_ticker}.", icon="⚠️")


                # 5. Fetch Multi-Source News Sentiment
                placeholder.markdown(f"⏳ Fetching Recent News Sentiment (Multi-Source/VADER) for **{primary_ticker}**...")
                try:
                    news_sentiment_summary, news_sentiment_counts_dict, news_articles_details = get_multi_source_news_sentiment(primary_ticker, NEWS_API_KEY, FMP_API_KEY)
                    logging.info(f"Multi-source news sentiment fetch completed for {primary_ticker}.")
                except Exception as news_err:
                    logging.error(f"Error calling multi-source news sentiment: {news_err}", exc_info=True)
                    news_sentiment_summary = f"News Sentiment: Error during analysis for {primary_ticker}."
                    news_sentiment_counts_dict = {'positive': None, 'negative': None, 'neutral': None, 'total': None, 'avg_score': None}
                    news_articles_details = []
                # Show toast if no news found
                if news_sentiment_counts_dict.get('total', 0) == 0 and primary_ticker:
                     st.toast(f"⚠️ No recent news found for {primary_ticker}.", icon="⚠️")


                # 6. Generate ETS Forecast (Uses close_prices series from Yahoo history)
                placeholder.markdown(f"⏳ Generating ETS Price Forecast for **{primary_ticker}**...")
                forecast_days = 7 # Define forecast horizon
                # Check if Yahoo close price history is available for forecasting
                # Need enough data for ETS (min_data_required is checked inside the function)
                if close_prices_history_yf is not None and not close_prices_history_yf.empty:
                    try:
                        # Pass the Close price series from Yahoo history to the forecast function
                        forecast_values, eval_metric_str, model_desc = forecast_stock_ets_advanced(primary_ticker, close_prices_history_yf, forecast_days=forecast_days)
                        if forecast_values is not None and not forecast_values.empty:
                            forecast_lines = [f"- Forecast Period: Next {forecast_days} business days"]
                            forecast_lines.append(f"- Model Used: {model_desc}")
                            if eval_metric_str and "Error" not in eval_metric_str and "skipped" not in eval_metric_str.lower():
                                forecast_lines.append(f"- {eval_metric_str}")
                            else:
                                forecast_lines.append(f"- Evaluation: {eval_metric_str}") # Report skip/error too
                            forecast_lines.append("- Forecasted Prices:")
                            if not forecast_values.empty:
                                for date, value in forecast_values.items():
                                     # Format date nicely for prompt
                                     date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date);
                                     forecast_lines.append(f"  - {date_str}: {value:.2f}")
                            else:
                                forecast_lines.append("  - No forecast values generated.")
                                logging.warning(f"ETS forecast values empty for {primary_ticker}.")

                            ets_forecast_summary = "\n".join(forecast_lines); logging.info(f"ETS Forecast generated successfully for {primary_ticker}.")
                        else:
                            # If forecast_values is None, model_desc and eval_metric_str should indicate why
                            ets_forecast_summary = f"ETS Price Forecast ({model_desc}): Could not generate forecast for {primary_ticker}. Reason: {eval_metric_str or 'Model fitting error'}";
                            logging.warning(f"ETS forecast generation failed for {primary_ticker}. Reason: {eval_metric_str or 'Model fitting error'}")
                    except Exception as forecast_err:
                        logging.error(f"Error calling ETS forecast function for {primary_ticker}: {forecast_err}", exc_info=True);
                        ets_forecast_summary = f"ETS Price Forecast: Error during calculation for {primary_ticker}."
                else:
                    ets_forecast_summary = f"ETS Price Forecast: Unavailable due to missing historical price data from Yahoo for {primary_ticker}.";
                    logging.warning(f"ETS forecast skipped for {primary_ticker} due to missing history.")

                # Show toast if forecast failed
                if "Unavailable" in ets_forecast_summary or "Error" in ets_forecast_summary or "Could not generate" in ets_forecast_summary:
                     if primary_ticker: st.toast(f"⚠️ Price forecast unavailable for {primary_ticker}.", icon="⚠️")


            placeholder.markdown(f"⏳ Compiling info & generating response for **{primary_ticker}**...")

            # --- Prepare messages for OpenAI ---
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Include recent chat history (last N turns)
            history_limit = 6 # Include last 6 pairs of messages + current
            # Find the starting index to include from st.session_state.messages
            # Each turn is 2 messages (user + assistant), so history_limit * 2
            # Account for the current user message already added
            start_index = max(0, len(st.session_state.messages) - (history_limit * 2) -1 ) # -1 because current user message is already added below
            relevant_history = st.session_state.messages[start_index:]
            # Append relevant history, excluding the current user message which will be added last
            full_messages.extend([msg for msg in relevant_history if msg["role"] != "user" or msg is not relevant_history[-1] ])


            # --- Add the Specific Context Block ---
            context_message_content = ""
            context_message_content += f"\n--- Start of Context for {primary_ticker} ---\n"
            context_message_content += f"Yahoo Finance Summary:\n{stock_data_for_prompt}\n\n" # Includes core data, SMAs, Beta, Analyst Rec

            context_message_content += f"Dynamic Risk Score Calculation Result:\n{risk_score_description}\n\n" # Includes score and category

            context_message_content += f"Recent News Sentiment Summary (Multi-Source / VADER Analysis):\n{news_sentiment_summary}\n\n" # Includes positive/negative/neutral counts and bias

            context_message_content += f"ETS Price Forecast (Calculated from Yahoo History):\n{ets_forecast_summary}\n\n" # Includes forecast period, model, evaluation, and predicted prices

            context_message_content += f"Trading Signals (Polygon.io, Momentum Indicators):\n{trading_signals_summary}\n" # Includes final decision and list of signals

            context_message_content += f"--- End of Context for {primary_ticker} ---\n"

            # Add final instructions about prioritization and sources
            context_message_content += f"Please prioritize information from the provided context. Follow guidelines (missing data, analyst format, state sources: Yahoo Finance, Multi-Source News/VADER, ETS Model, Polygon.io/Momentum Indicators etc.). Note news covers last {NEWS_DAYS_BACK} days. ETS Forecast is short-term. Technical strategy signals are NOT available."

            # Insert the context message just before the last user message
            # The current user message is already added to session_state, and we've built the full_messages list
            # by appending history up to the point *before* adding the current user message.
            # The current user message is the last one in session_state.messages.
            # We need to add the context before the user message.
            # The full_messages list currently contains [SYSTEM_PROMPT, ...history...]
            # We need [SYSTEM_PROMPT, ...history..., CONTEXT, USER_MESSAGE]
            # Let's reconstruct full_messages carefully:
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Add only the history *before* the current interaction
            if len(st.session_state.messages) > 1: # If there's more than just the current user message
                 # Add messages from the start up to the one *before* the latest user message
                 full_messages.extend(st.session_state.messages[:-1])

            full_messages.append({"role": "system", "content": context_message_content})
            full_messages.append(st.session_state.messages[-1]) # Add the current user message back as the last item


            logging.info(f"Prepared full message list for OpenAI. Total messages: {len(full_messages)}. Last message role: {full_messages[-1]['role']}")
            logging.debug(f"Context (start): {context_message_content[:800]}...")

        else:
            # No ticker identified - general query or lookup failed
            logging.info("No ticker identified. Skipping data fetching steps.");
            placeholder.markdown("⏳ Generating general response...")
            # Build messages with only the system prompt and recent history
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            history_limit = 6; # Include last 6 pairs
            start_index = max(0, len(st.session_state.messages) - (history_limit * 2) -1) # -1 for current user message
            relevant_history = st.session_state.messages[start_index:]
            # Add history except the current user message
            full_messages.extend([msg for msg in relevant_history if msg["role"] != "user" or msg is not relevant_history[-1] ])

            # Add a context block indicating no ticker was found
            context_message_content = "\n--- Start of Context ---\nNo specific stock ticker could be identified from the user's request, or the ticker lookup failed. Please provide a general financial response if possible, or ask the user to clarify the ticker/company.\n--- End of Context---\n"
            full_messages.append({"role": "system", "content": context_message_content})
            full_messages.append(st.session_state.messages[-1]) # Add the current user message


        # --- Call OpenAI API ---
        try:
            logging.info(f"Sending request to {MODEL_NAME}.")
            if not client: raise ValueError("OpenAI client is not initialized.") # Should not happen due to early check

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=full_messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=True
            )
            # Stream the response to the placeholder
            response_content = placeholder.write_stream(stream);
            gpt_reply = response_content

            # Append the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": gpt_reply})
            logging.info(f"Streamed response OK.")

            # Display Data Dashboard AFTER response generation if a ticker was processed
            if primary_ticker and stock_data:
                 # Ensure risk_category is correct for display even if risk_score_final was None or NaN
                 risk_category_display = get_risk_category(risk_score_final)
                 display_stock_data_dashboard(
                     stock_data,
                     risk_score_final,
                     risk_category_display,
                     news_sentiment_counts_dict,
                     trading_signals_summary # Pass the summary string
                 )
                 # Optional: Display news article details if available
                 if news_articles_details:
                     with st.expander(f"Recent News Articles ({len(news_articles_details)} shown)"):
                          for article in news_articles_details:
                               title = article.get('title', 'N/A'); source = article.get('source', 'N/A'); published = article.get('published', 'N/A'); sentiment = article.get('sentiment', 'N/A'); score = article.get('compound_score', 'N/A'); url = article.get('url', '#')
                               sentiment_emoji = "✅" if sentiment == "Positive" else "❌" if sentiment == "Negative" else "➖"
                               # Safely format URL to prevent Markdown issues if URL is malformed/missing
                               article_link = f"[{title}]({url})" if url and url != '#' else title
                               st.markdown(f"- **{article_link}**")
                               st.caption(f"Source: {source} | Published: {published} | Sentiment: {sentiment_emoji} {sentiment} (Score: {score:.3f})")

            elif primary_ticker:
                 st.warning(f"Could not display data snapshot for {primary_ticker} (Yahoo summary data missing).")

        except AuthenticationError as e:
            error_message = f"😥 OpenAI API Authentication Error: Invalid API Key or configuration issue. Please check your key in secrets.toml or environment variables. Error: {e}";
            placeholder.error(error_message); logging.error(f"OpenAI API Authentication Error: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": error_message});
            # Stop the script execution as API key is fundamental
            st.stop()
        except OpenAIError as e:
            error_message = f"😥 OpenAI API Error: {e}. Try again later."; placeholder.error(error_message); logging.error(f"OpenAI API Error: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": error_message})
        except Exception as e:
            error_message = f"😥 An unexpected error occurred: {e}. Please check logs for details."; placeholder.error(error_message); logging.error(f"Unexpected Error during OpenAI call or processing: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": f"Internal Error: {e}"})


# ... (end of the main processing logic and API calls) ...

# --- Footer ---
st.divider() # Divider before the donation link

# --- Start of Donation Section ---
st.markdown('<p style="font-size: 16px; font-weight: bold; text-align: center;">Like this tool? Consider supporting its development: <a href="https://paypal.me/niveyal">☕ Buy me a coffee (PayPal.me)</a> (Optional, but appreciated!)</p>', unsafe_allow_html=True)
# --- End of Donation Section ---

st.divider() # Divider between donation and disclaimer

# Updated disclaimer in dashboard footer
st.caption(f"*Data Sources: Yahoo Finance (via yfinance) for summary/history/analyst data/risk score components. Polygon.io for daily price data used in momentum indicators/trading signals. Multi-source News (NewsAPI, FMP, RSS - {NEWS_DAYS_BACK}d) with VADER sentiment analysis. Wikipedia for index lookups. ETS Forecasts are model calculations based on historical data. All data may be delayed. This tool provides informational analysis only and is NOT financial advice.*")
