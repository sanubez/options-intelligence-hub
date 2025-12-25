"""
═══════════════════════════════════════════════════════════════════
    DELTA EXCHANGE OPTIONS INTELLIGENCE HUB - STREAMLIT VERSION
    By: Santanu Bez
    Tagline: "Decision-Support & Risk-Context Engine"
    
    VERSION: 4.0 - PROFESSIONAL EDITION
    - Percentile-based OI Analysis
    - IV-RV Volatility Framework
    - Controlled Auto-Refresh
    - Institutional UI/UX
═══════════════════════════════════════════════════════════════════
"""

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import math
import os
import pickle
import warnings
import json
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Options Intelligence Hub v4.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS - INSTITUTIONAL THEME
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Base Theme */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #1a4d5c;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d9ff;
        font-size: 26px;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00d9ff !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #0a0a0a;
        border-bottom: 1px solid #1a1a1a;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0f0f0f;
        border: 1px solid #1a1a1a;
        color: #a0a0a0;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1a1a1a;
        border-color: #00d9ff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0a0a0a;
        border-bottom: 2px solid #00d9ff;
        color: #00d9ff;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .regime-card {
        padding: 32px;
        border-radius: 16px;
        border: 2px solid;
        text-align: center;
        margin: 16px 0;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    .regime-optimal {
        border-color: #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    
    .regime-favorable {
        border-color: #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
    }
    
    .regime-neutral {
        border-color: #ffaa00;
        box-shadow: 0 0 20px rgba(255, 170, 0, 0.2);
    }
    
    .regime-unfavorable {
        border-color: #ff6b6b;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.2);
    }
    
    .regime-dangerous {
        border-color: #ff0066;
        box-shadow: 0 0 20px rgba(255, 0, 102, 0.2);
    }
    
    /* Percentile Bar */
    .percentile-container {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    
    .percentile-bar {
        height: 32px;
        background: linear-gradient(90deg, #00ff88 0%, #ffaa00 50%, #ff6b6b 100%);
        border-radius: 16px;
        position: relative;
        margin: 8px 0;
    }
    
    .percentile-marker {
        position: absolute;
        width: 4px;
        height: 40px;
        background-color: #ffffff;
        top: -4px;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
    }
    
    /* Status Indicators */
    .status-healthy {
        color: #00ff88;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ffaa00;
        font-weight: 600;
    }
    
    .status-critical {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Debug Box */
    .debug-box {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        margin: 12px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #a0a0a0;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #0f1f2a;
        border-left: 4px solid #00d9ff;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    
    .warning-box {
        background-color: #2a1f0f;
        border-left: 4px solid #ffaa00;
        padding: 16px;
        margin: 12px 0;
        border-radius: 4px;
    }
    
    /* Volatility Matrix */
    .vol-matrix {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    
    .vol-cell {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid;
        font-weight: 600;
    }
    
    .vol-premium-rich {
        border-color: #00ff88;
        background: linear-gradient(135deg, #001a0f 0%, #002a1a 100%);
    }
    
    .vol-trap {
        border-color: #ff6b6b;
        background: linear-gradient(135deg, #1a0000 0%, #2a0000 100%);
    }
    
    .vol-dangerous {
        border-color: #ff0066;
        background: linear-gradient(135deg, #1a0010 0%, #2a0020 100%);
    }
    
    .vol-dead {
        border-color: #666666;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00ccff 100%);
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Data Age Indicator */
    .data-age-fresh {
        color: #00ff88;
        font-size: 12px;
        font-weight: 600;
    }
    
    .data-age-stale {
        color: #ffaa00;
        font-size: 12px;
        font-weight: 600;
    }
    
    .data-age-old {
        color: #ff6b6b;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration management"""
    
    # Timezone
    IST = pytz.timezone('Asia/Kolkata')
    
    # API Configuration
    BASE_URL = 'https://api.india.delta.exchange'
    REQUEST_TIMEOUT = 15
    
    # Analysis Parameters
    SPOT_RANGE_PERCENT = 2.5  # Increased for better OI coverage
    CANDLE_RESOLUTION = "15m"
    HISTORICAL_CANDLES = 100  # Increased for better RV calculation
    
    # EMA Periods
    EMA_FAST = 8
    EMA_MID = 13
    EMA_SLOW = 21
    
    # Market Regime Thresholds
    IV_SPIKE_THRESHOLD = 5.0
    IV_CRUSH_THRESHOLD = -5.0
    IV_STABLE_THRESHOLD = 2.0
    OI_SPIKE_THRESHOLD = 15.0
    OI_CRUSH_THRESHOLD = -15.0
    
    # Percentile Configuration
    PERCENTILE_WINDOW_HOURS = 24
    PERCENTILE_EXTREME_THRESHOLD = 80  # Above 80th percentile = extreme
    PERCENTILE_LOW_THRESHOLD = 20      # Below 20th percentile = low
    
    # Volatility Analysis
    RV_PERIODS = {
        '15m': 4,   # 4 periods of 15m = 1 hour
        '1h': 24,   # 24 hours
        '4h': 96    # 4 days
    }
    
    # Max Pain
    MAX_PAIN_PROXIMITY = 600
    
    # Data Storage
    DATA_DIR = "options_data_v4"
    JOURNAL_FILE = os.path.join(DATA_DIR, "trading_journal.csv")
    HISTORY_DIR = os.path.join(DATA_DIR, "history")
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    
    # Debug
    DEBUG_MODE = False
    
    # Auto Refresh
    AUTO_REFRESH_INTERVALS = {
        '1 minute': 60,
        '3 minutes': 180,
        '5 minutes': 300,
        '10 minutes': 600
    }
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_ist_time() -> datetime:
    """Get current time in IST"""
    return datetime.now(Config.IST)

def format_ist_time(dt: Optional[datetime] = None, format_str: str = '%d-%m-%Y %H:%M:%S') -> str:
    """Format datetime to IST string"""
    if dt is None:
        dt = get_ist_time()
    elif isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    if dt.tzinfo is None:
        dt = Config.IST.localize(dt)
    else:
        dt = dt.astimezone(Config.IST)
    
    return dt.strftime(format_str)

def calculate_data_age(timestamp: str) -> Tuple[int, str]:
    """
    Calculate age of data in seconds and return appropriate CSS class
    
    Returns:
        (age_seconds, css_class)
    """
    data_time = pd.to_datetime(timestamp)
    if data_time.tzinfo is None:
        data_time = Config.IST.localize(data_time)
    
    age_seconds = int((get_ist_time() - data_time).total_seconds())
    
    if age_seconds < 60:
        css_class = "data-age-fresh"
    elif age_seconds < 300:
        css_class = "data-age-stale"
    else:
        css_class = "data-age-old"
    
    return age_seconds, css_class

def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes"""
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

# ═══════════════════════════════════════════════════════════════════
# DEBUG LOGGER
# ═══════════════════════════════════════════════════════════════════

class DebugLogger:
    """Structured debug logging system"""
    
    @staticmethod
    def log(title: str, data: Dict, level: str = "info"):
        """Log debug information if debug mode is enabled"""
        if not Config.DEBUG_MODE:
            return
        
        color_map = {
            "info": "#00d9ff",
            "success": "#00ff88",
            "warning": "#ffaa00",
            "error": "#ff6b6b"
        }
        
        color = color_map.get(level, "#00d9ff")
        
        with st.expander(f"🔍 DEBUG: {title}", expanded=False):
            st.markdown(f"""
            <div class="debug-box">
                <strong style="color: {color};">{title}</strong><br>
                <pre>{json.dumps(data, indent=2, default=str)}</pre>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════

class CacheManager:
    """Intelligent caching to prevent API overload"""
    
    @staticmethod
    def get_cache_path(cache_key: str) -> str:
        """Generate cache file path"""
        return os.path.join(Config.CACHE_DIR, f"{cache_key}.pkl")
    
    @staticmethod
    def save_cache(cache_key: str, data: any, ttl_seconds: int = 60):
        """Save data to cache with TTL"""
        try:
            cache_data = {
                'data': data,
                'timestamp': get_ist_time().isoformat(),
                'ttl': ttl_seconds
            }
            cache_path = CacheManager.get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            DebugLogger.log("Cache Save Error", {"error": str(e)}, "error")
    
    @staticmethod
    def load_cache(cache_key: str) -> Optional[any]:
        """Load data from cache if valid"""
        try:
            cache_path = CacheManager.get_cache_path(cache_key)
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            cache_time = pd.to_datetime(cache_data['timestamp'])
            if cache_time.tzinfo is None:
                cache_time = Config.IST.localize(cache_time)
            
            age = (get_ist_time() - cache_time).total_seconds()
            
            if age <= cache_data['ttl']:
                DebugLogger.log("Cache Hit", {
                    "key": cache_key,
                    "age_seconds": age,
                    "ttl": cache_data['ttl']
                }, "success")
                return cache_data['data']
            else:
                DebugLogger.log("Cache Expired", {
                    "key": cache_key,
                    "age_seconds": age,
                    "ttl": cache_data['ttl']
                }, "warning")
                return None
                
        except Exception as e:
            DebugLogger.log("Cache Load Error", {"error": str(e)}, "error")
            return None

# ═══════════════════════════════════════════════════════════════════
# HISTORICAL DATA MANAGER
# ═══════════════════════════════════════════════════════════════════

class HistoricalDataManager:
    """Manage historical data for percentile calculations"""
    
    @staticmethod
    def save_metric_data(asset: str, expiry: str, metric_name: str, value: float):
        """
        Save metric data point for percentile analysis
        
        Why: We need historical context to determine if current values are extreme
        """
        try:
            timestamp = get_ist_time()
            file_path = os.path.join(Config.HISTORY_DIR, f'{metric_name}_{asset}_{expiry}.pkl')
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    history = pickle.load(f)
            else:
                history = []
            
            history.append({
                'timestamp': timestamp.isoformat(),
                'value': value
            })
            
            # Keep only data within percentile window
            cutoff = get_ist_time() - timedelta(hours=Config.PERCENTILE_WINDOW_HOURS)
            history = [e for e in history if pd.to_datetime(e['timestamp']) > cutoff]
            
            with open(file_path, 'wb') as f:
                pickle.dump(history, f)
            
            return history
        except Exception as e:
            DebugLogger.log(f"Error saving {metric_name}", {"error": str(e)}, "error")
            return []
    
    @staticmethod
    def load_metric_data(asset: str, expiry: str, metric_name: str) -> List[Dict]:
        """Load historical metric data"""
        try:
            file_path = os.path.join(Config.HISTORY_DIR, f'{metric_name}_{asset}_{expiry}.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        return []
    
    @staticmethod
    def calculate_percentile(history: List[Dict], current_value: float) -> float:
        """
        Calculate percentile of current value within historical distribution
        
        Why: Percentile provides context - is this value normal or extreme?
        """
        if len(history) < 10:  # Need minimum data points
            return 50.0  # Return neutral percentile
        
        values = [e['value'] for e in history]
        values.append(current_value)
        
        percentile = (sum(v <= current_value for v in values) / len(values)) * 100
        
        return percentile
    
    @staticmethod
    def calculate_metric_changes(history: List[Dict], periods: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate metric changes over multiple periods
        
        Why: Rate of change matters as much as absolute values
        """
        if len(history) < 2:
            return {f'change_{k}': 0 for k in periods.keys()}
        
        now = get_ist_time()
        current_value = history[-1]['value']
        changes = {}
        
        for period_name, minutes in periods.items():
            period_data = [
                e for e in history 
                if pd.to_datetime(e['timestamp']) >= now - timedelta(minutes=minutes)
            ]
            
            if len(period_data) >= 2:
                old_value = period_data[0]['value']
                if old_value != 0:
                    change_pct = ((current_value - old_value) / old_value) * 100
                    changes[f'change_{period_name}'] = change_pct
                else:
                    changes[f'change_{period_name}'] = 0
            else:
                changes[f'change_{period_name}'] = 0
        
        return changes

# ═══════════════════════════════════════════════════════════════════
# API CLASS
# ═══════════════════════════════════════════════════════════════════

class DeltaAPI:
    """Delta Exchange API client with caching and error handling"""
    
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'OptionsHub/4.0'
        })
        
        DebugLogger.log("API Initialized", {
            "base_url": self.base_url,
            "headers": dict(self.session.headers)
        }, "info")
    
    def get_spot_price(self, symbol: str = 'BTCUSD') -> Optional[float]:
        """Fetch spot price with caching"""
        cache_key = f"spot_{symbol}"
        cached = CacheManager.load_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f'{self.base_url}/v2/tickers/{symbol}'
            
            DebugLogger.log("Fetching Spot Price", {
                "url": url,
                "symbol": symbol
            }, "info")
            
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'result' in data:
                    spot_price = float(data['result']['spot_price'])
                    CacheManager.save_cache(cache_key, spot_price, ttl_seconds=30)
                    return spot_price
            
            st.error(f"API Error: Status {response.status_code}")
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout - API is slow")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Connection error - Check internet")
        except Exception as e:
            st.error(f"Error fetching spot: {str(e)}")
            DebugLogger.log("Exception in get_spot_price", {
                "error": str(e),
                "type": type(e).__name__
            }, "error")
        
        return None
    
    def get_historical_candles(self, symbol: str = 'BTCUSD', 
                              resolution: str = '15m', 
                              count: int = 100) -> Optional[List[Dict]]:
        """
        Fetch historical OHLC candles with caching
        
        Why: We need historical price data for RV calculation
        """
        cache_key = f"candles_{symbol}_{resolution}_{count}"
        cached = CacheManager.load_cache(cache_key)
        if cached:
            return cached
        
        try:
            end_time = int(time.time())
            minutes_per_candle = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '1d': 1440
            }
            minutes = minutes_per_candle.get(resolution, 15)
            start_time = end_time - (count * minutes * 60)
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'start': start_time,
                'end': end_time
            }
            
            endpoints_to_try = [
                '/v2/history/candles',
                '/history/candles',
                '/v2/chart/history'
            ]
            
            for endpoint in endpoints_to_try:
                url = f'{self.base_url}{endpoint}'
                
                try:
                    response = self.session.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success') and 'result' in data:
                            candles = data['result']
                            CacheManager.save_cache(cache_key, candles, ttl_seconds=60)
                            return candles
                    elif response.status_code == 404:
                        continue
                        
                except Exception:
                    continue
            
            st.error("❌ Failed to fetch candles from all endpoints")
                
        except Exception as e:
            st.error(f"Error fetching candles: {str(e)}")
            DebugLogger.log("Critical Exception in get_historical_candles", {
                "error": str(e),
                "type": type(e).__name__
            }, "error")
        
        return None
    
    def get_options_chain(self, underlying: str = 'BTC', 
                         expiry_date: Optional[str] = None) -> Optional[List[Dict]]:
        """Fetch options chain with caching"""
        cache_key = f"options_{underlying}_{expiry_date}"
        cached = CacheManager.load_cache(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                'contract_types': 'call_options,put_options',
                'underlying_asset_symbols': underlying
            }
            
            if expiry_date:
                try:
                    dt = datetime.strptime(expiry_date, '%d-%m-%Y')
                    formatted_date = dt.strftime('%Y-%m-%d')
                    params['expiry_after'] = formatted_date
                    params['expiry_before'] = formatted_date
                except:
                    pass
            
            url = f'{self.base_url}/v2/tickers'
            
            response = self.session.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'result' in data:
                    options = data['result']
                    CacheManager.save_cache(cache_key, options, ttl_seconds=30)
                    return options
            
            st.error(f"Options API Error: Status {response.status_code}")
            
        except Exception as e:
            st.error(f"Error fetching options: {str(e)}")
            DebugLogger.log("Exception in get_options_chain", {
                "error": str(e),
                "type": type(e).__name__
            }, "error")
        
        return None
    
    def get_all_expiries(self, underlying: str = 'BTC') -> List[str]:
        """Fetch all available expiries"""
        cache_key = f"expiries_{underlying}"
        cached = CacheManager.load_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f'{self.base_url}/v2/products'
            params = {'contract_types': 'call_options', 'states': 'live'}
            
            response = self.session.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                products = data.get('result', [])
                
                expiries = set()
                for p in products:
                    if p.get('underlying_asset', {}).get('symbol') == underlying:
                        settlement = p.get('settlement_time')
                        if settlement:
                            try:
                                exp_dt = datetime.strptime(settlement, '%Y-%m-%dT%H:%M:%SZ')
                                exp_dt_ist = pytz.utc.localize(exp_dt).astimezone(Config.IST)
                                if exp_dt_ist > get_ist_time():
                                    expiries.add(exp_dt_ist.strftime('%d-%m-%Y'))
                            except:
                                continue
                
                expiry_list = sorted(list(expiries), key=lambda x: datetime.strptime(x, '%d-%m-%Y'))
                CacheManager.save_cache(cache_key, expiry_list, ttl_seconds=300)
                return expiry_list
                
        except Exception as e:
            st.error(f"Error finding expiries: {str(e)}")
        
        return []

# ═══════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class TechnicalAnalysis:
    """Technical analysis calculations"""
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data.mean() if len(data) > 0 else 0.0
        return data.ewm(span=period, adjust=False).mean().iloc[-1]
    
    @staticmethod
    def calculate_all_emas(candles: List[Dict]) -> Tuple[float, float, float, Dict]:
        """
        Calculate all EMAs from OHLC data
        
        Returns:
            (ema8, ema13, ema21, ema_series)
        """
        if not candles or len(candles) == 0:
            return 0.0, 0.0, 0.0, {}
        
        df = pd.DataFrame(candles)
        if 'close' not in df.columns:
            return 0.0, 0.0, 0.0, {}
        
        close_prices = pd.Series([float(c) for c in df['close']])
        
        ema8 = TechnicalAnalysis.calculate_ema(close_prices, Config.EMA_FAST)
        ema13 = TechnicalAnalysis.calculate_ema(close_prices, Config.EMA_MID)
        ema21 = TechnicalAnalysis.calculate_ema(close_prices, Config.EMA_SLOW)
        
        ema_series = {
            'ema8': close_prices.ewm(span=Config.EMA_FAST, adjust=False).mean().tolist(),
            'ema13': close_prices.ewm(span=Config.EMA_MID, adjust=False).mean().tolist(),
            'ema21': close_prices.ewm(span=Config.EMA_SLOW, adjust=False).mean().tolist()
        }
        
        return ema8, ema13, ema21, ema_series
    
    @staticmethod
    def get_ema_signal(spot: float, ema8: float, ema13: float, ema21: float) -> Tuple[str, str]:
        """
        Determine EMA-based market structure
        
        Returns:
            (signal_text, signal_type)
        """
        if spot > ema8 > ema13 > ema21:
            return "Strong Uptrend - Upper strikes under pressure", "bullish"
        elif spot < ema8 < ema13 < ema21:
            return "Strong Downtrend - Lower strikes under pressure", "bearish"
        elif ema8 > spot > ema21:
            return "Consolidation - Range-bound", "neutral"
        else:
            return "Mixed Structure - No clear trend", "mixed"
    
    @staticmethod
    def calculate_realized_volatility(candles: List[Dict], periods: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate Realized Volatility over multiple periods
        
        Why: RV shows actual price movement, essential for IV-RV comparison
        
        Formula: RV = sqrt(sum(log_returns^2) * (252/n)) * 100
        """
        if not candles or len(candles) < 2:
            return {f'rv_{k}': 0 for k in periods.keys()}
        
        df = pd.DataFrame(candles)
        df['close'] = df['close'].astype(float)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        rv_results = {}
        
        for period_name, period_count in periods.items():
            if len(df) >= period_count:
                recent_returns = df['log_return'].tail(period_count)
                variance = (recent_returns ** 2).sum()
                # Annualize: 252 trading days, adjust for intraday
                rv = np.sqrt(variance * (252 / period_count)) * 100
                rv_results[f'rv_{period_name}'] = rv
            else:
                rv_results[f'rv_{period_name}'] = 0
        
        return rv_results

# ═══════════════════════════════════════════════════════════════════
# OPTIONS ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class OptionsAnalysis:
    """Options-specific analysis functions"""
    
    @staticmethod
    def find_atm_strike(spot: float, options_chain: List[Dict]) -> Optional[float]:
        """Find At-The-Money strike"""
        if not options_chain:
            return None
        
        strikes = set()
        for opt in options_chain:
            strike = opt.get('strike_price')
            if strike:
                strikes.add(float(strike))
        
        if not strikes:
            return None
        
        return min(strikes, key=lambda x: abs(x - spot))
    
    @staticmethod
    def get_atm_iv(options_chain: List[Dict], atm_strike: float) -> Optional[float]:
        """
        Get ATM Implied Volatility (average of call and put)
        
        Why: ATM IV is the market's expectation of future volatility
        """
        if not options_chain or not atm_strike:
            return None
        
        call_iv = None
        put_iv = None
        
        for opt in options_chain:
            strike = float(opt.get('strike_price', 0))
            if abs(strike - atm_strike) < 0.01:
                contract_type = opt.get('contract_type', '')
                quotes = opt.get('quotes', {})
                mark_iv = quotes.get('mark_iv')
                
                if mark_iv:
                    iv_value = float(mark_iv)
                    if 'call' in contract_type.lower():
                        call_iv = iv_value
                    elif 'put' in contract_type.lower():
                        put_iv = iv_value
        
        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        elif call_iv:
            return call_iv
        elif put_iv:
            return put_iv
        
        return None
    
    @staticmethod
    def calculate_max_pain(strikes_data: Dict) -> Optional[float]:
        """
        Calculate Max Pain strike
        
        Why: Max Pain is where option sellers have minimum loss at expiry
        """
        if not strikes_data:
            return None
        
        pain_values = {}
        strikes = sorted(strikes_data.keys())
        
        for test_strike in strikes:
            total_pain = 0
            for strike, data in strikes_data.items():
                if strike > test_strike:
                    total_pain += data.get('call_oi', 0) * (strike - test_strike)
                if strike < test_strike:
                    total_pain += data.get('put_oi', 0) * (test_strike - strike)
            pain_values[test_strike] = total_pain
        
        if pain_values:
            return min(pain_values, key=pain_values.get)
        
        return None
    
    @staticmethod
    def process_options_chain(options_chain: List[Dict], spot_price: float) -> Tuple[Dict, Dict]:
        """
        Process options chain into structured data
        
        Returns:
            (strikes_data, totals)
        """
        strikes_data = defaultdict(lambda: {
            'call_oi': 0, 'put_oi': 0, 
            'call_volume': 0, 'put_volume': 0,
            'call_mark_iv': 0, 'put_mark_iv': 0
        })
        
        totals = {
            'total_call_oi': 0, 'total_put_oi': 0, 
            'total_call_volume': 0, 'total_put_volume': 0
        }
        
        lower_bound = spot_price * (1 - Config.SPOT_RANGE_PERCENT / 100)
        upper_bound = spot_price * (1 + Config.SPOT_RANGE_PERCENT / 100)
        
        for opt in options_chain:
            strike = float(opt.get('strike_price', 0))
            if strike < lower_bound or strike > upper_bound:
                continue
            
            contract_type = opt.get('contract_type', '')
            oi = float(opt.get('oi', 0))
            volume = float(opt.get('volume', 0))
            quotes = opt.get('quotes', {})
            mark_iv = float(quotes.get('mark_iv', 0)) if quotes.get('mark_iv') else 0
            
            if 'call' in contract_type.lower():
                strikes_data[strike]['call_oi'] = oi
                strikes_data[strike]['call_volume'] = volume
                strikes_data[strike]['call_mark_iv'] = mark_iv
                totals['total_call_oi'] += oi
                totals['total_call_volume'] += volume
            elif 'put' in contract_type.lower():
                strikes_data[strike]['put_oi'] = oi
                strikes_data[strike]['put_volume'] = volume
                strikes_data[strike]['put_mark_iv'] = mark_iv
                totals['total_put_oi'] += oi
                totals['total_put_volume'] += volume
        
        totals['pcr_oi'] = (totals['total_put_oi'] / totals['total_call_oi'] 
                           if totals['total_call_oi'] > 0 else 0)
        totals['pcr_volume'] = (totals['total_put_volume'] / totals['total_call_volume'] 
                               if totals['total_call_volume'] > 0 else 0)
        
        return dict(strikes_data), totals
    
    @staticmethod
    def calculate_atm_oi_concentration(strikes_data: Dict, atm_strike: float, 
                                      total_oi: float) -> float:
        """
        Calculate OI concentration at ATM
        
        Why: High ATM concentration indicates strong gamma risk
        """
        if not strikes_data or not atm_strike or total_oi == 0:
            return 0.0
        
        atm_data = strikes_data.get(atm_strike, {})
        atm_oi = atm_data.get('call_oi', 0) + atm_data.get('put_oi', 0)
        
        concentration = (atm_oi / total_oi) * 100
        
        return concentration

# ═══════════════════════════════════════════════════════════════════
# VOLATILITY ANALYZER
# ═══════════════════════════════════════════════════════════════════

class VolatilityAnalyzer:
    """Advanced volatility analysis framework"""
    
    @staticmethod
    def calculate_iv_metrics(iv_history: List[Dict]) -> Dict:
        """
        Calculate IV velocity and acceleration
        
        Why: Rate of IV change matters more than absolute level
        """
        if len(iv_history) < 3:
            return {
                'iv_velocity': 0,
                'iv_acceleration': 0,
                'iv_trend': 'insufficient_data'
            }
        
        # Get last 3 data points
        recent = iv_history[-3:]
        iv_values = [e['value'] for e in recent]
        
        # Velocity: first derivative (rate of change)
        velocity = iv_values[-1] - iv_values[-2]
        
        # Acceleration: second derivative (rate of rate of change)
        prev_velocity = iv_values[-2] - iv_values[-3]
        acceleration = velocity - prev_velocity
        
        # Determine trend
        if velocity > 0.5:
            trend = 'rising'
        elif velocity < -0.5:
            trend = 'falling'
        else:
            trend = 'stable'
        
        return {
            'iv_velocity': velocity,
            'iv_acceleration': acceleration,
            'iv_trend': trend
        }
    
    @staticmethod
    def determine_volatility_regime(atm_iv: float, rv_1h: float, 
                                   iv_rv_spread: float) -> Dict:
        """
        Classify market into volatility regime
        
        Regimes:
        1. Premium Rich: Low RV, High IV → Good for selling
        2. Trap: High RV, Low IV → Dangerous for selling
        3. Dangerous: High RV, High IV → Avoid selling
        4. Dead Zone: Low RV, Low IV → Low opportunity
        
        Why: Different regimes require different strategies
        """
        
        # Define thresholds (these are relative, adjust based on asset)
        rv_threshold = 50  # Annualized %
        iv_threshold = 50  # Annualized %
        
        rv_high = rv_1h > rv_threshold
        iv_high = atm_iv > iv_threshold
        
        if not rv_high and iv_high:
            regime = "Premium Rich"
            regime_class = "vol-premium-rich"
            description = "IV elevated while RV low - favorable for premium selling"
            risk_level = "Low"
            color = "#00ff88"
        elif rv_high and not iv_high:
            regime = "Volatility Trap"
            regime_class = "vol-trap"
            description = "RV high but IV not pricing it - dangerous for sellers"
            risk_level = "High"
            color = "#ff6b6b"
        elif rv_high and iv_high:
            regime = "High Volatility"
            regime_class = "vol-dangerous"
            description = "Both IV and RV elevated - extreme caution required"
            risk_level = "Extreme"
            color = "#ff0066"
        else:
            regime = "Low Volatility"
            regime_class = "vol-dead"
            description = "Both IV and RV low - limited opportunity"
            risk_level = "Low"
            color = "#666666"
        
        return {
            'regime': regime,
            'regime_class': regime_class,
            'description': description,
            'risk_level': risk_level,
            'color': color,
            'iv_rv_spread': iv_rv_spread,
            'spread_interpretation': VolatilityAnalyzer._interpret_iv_rv_spread(iv_rv_spread)
        }
    
    @staticmethod
    def _interpret_iv_rv_spread(spread: float) -> str:
        """Interpret IV-RV spread"""
        if spread > 20:
            return "IV significantly overpricing realized moves - premium rich"
        elif spread > 10:
            return "IV moderately elevated vs realized - decent premium"
        elif spread > -10:
            return "IV fairly pricing realized moves - neutral"
        elif spread > -20:
            return "IV underpricing realized moves - caution"
        else:
            return "IV severely underpricing realized moves - danger zone"
    
    @staticmethod
    def calculate_expected_move(atm_iv: float, spot_price: float, 
                               days_to_expiry: float) -> Dict:
        """
        Calculate expected price move based on IV
        
        Formula: Expected Move = Spot * IV * sqrt(DTE/365)
        
        Why: Helps set realistic expectations for price range
        """
        if atm_iv == 0 or days_to_expiry == 0:
            return {
                'expected_move_1sd': 0,
                'upper_bound_1sd': spot_price,
                'lower_bound_1sd': spot_price,
                'expected_move_2sd': 0,
                'upper_bound_2sd': spot_price,
                'lower_bound_2sd': spot_price
            }
        
        # Convert IV from percentage to decimal
        iv_decimal = atm_iv / 100
        
        # Calculate 1 standard deviation move
        move_1sd = spot_price * iv_decimal * np.sqrt(days_to_expiry / 365)
        
        # Calculate 2 standard deviation move
        move_2sd = move_1sd * 2
        
        return {
            'expected_move_1sd': move_1sd,
            'upper_bound_1sd': spot_price + move_1sd,
            'lower_bound_1sd': spot_price - move_1sd,
            'expected_move_2sd': move_2sd,
            'upper_bound_2sd': spot_price + move_2sd,
            'lower_bound_2sd': spot_price - move_2sd
        }

# ═══════════════════════════════════════════════════════════════════
# MARKET REGIME ANALYZER
# ═══════════════════════════════════════════════════════════════════

class MarketRegimeAnalyzer:
    """
    Comprehensive market regime analysis
    
    Scoring System (0-100):
    - IV Analysis: 45%
    - OI Analysis: 45%
    - PCR Analysis: 10%
    """
    
    @staticmethod
    def analyze_regime(iv_changes: Dict, oi_changes: Dict, pcr_oi: float,
                      iv_percentile: float, oi_percentile: float) -> Dict:
        """
        Analyze market regime with percentile context
        
        Why: Absolute values are meaningless without historical context
        """
        
        # Get changes
        iv_change_15m = iv_changes.get('change_15m', 0)
        iv_change_30m = iv_changes.get('change_30m', 0)
        oi_change_15m = oi_changes.get('change_15m', 0)
        oi_change_30m = oi_changes.get('change_30m', 0)
        
        # Average changes
        avg_iv_change = (iv_change_15m + iv_change_30m) / 2
        avg_oi_change = (oi_change_15m + oi_change_30m) / 2
        
        # ═══════════════════════════════════════════════════════════
        # IV SCORE (45% weight)
        # ═══════════════════════════════════════════════════════════
        iv_score = 0
        
        # Percentile component (60% of IV score)
        if iv_percentile < Config.PERCENTILE_LOW_THRESHOLD:
            iv_percentile_score = 27  # Low IV = good for selling
            iv_status = "IV CRUSHED"
            iv_color = "#00ff88"
        elif iv_percentile > Config.PERCENTILE_EXTREME_THRESHOLD:
            iv_percentile_score = 0  # High IV = dangerous
            iv_status = "IV EXTREME"
            iv_color = "#ff6b6b"
        else:
            iv_percentile_score = 15  # Normal IV
            iv_status = "IV NORMAL"
            iv_color = "#ffaa00"
        
        # Change component (40% of IV score)
        if avg_iv_change < Config.IV_CRUSH_THRESHOLD:
            iv_change_score = 18  # IV falling = good
            iv_trend = "FALLING"
        elif avg_iv_change > Config.IV_SPIKE_THRESHOLD:
            iv_change_score = 0  # IV rising = bad
            iv_trend = "RISING"
        else:
            iv_change_score = 9  # IV stable
            iv_trend = "STABLE"
        
        iv_score = iv_percentile_score + iv_change_score
        
        # ═══════════════════════════════════════════════════════════
        # OI SCORE (45% weight)
        # ═══════════════════════════════════════════════════════════
        oi_score = 0
        
        # Percentile component (60% of OI score)
        if oi_percentile > Config.PERCENTILE_EXTREME_THRESHOLD:
            oi_percentile_score = 27  # High OI = strong interest
            oi_status = "OI EXTREME"
            oi_color = "#00ff88"
        elif oi_percentile < Config.PERCENTILE_LOW_THRESHOLD:
            oi_percentile_score = 0  # Low OI = weak interest
            oi_status = "OI LOW"
            oi_color = "#ff6b6b"
        else:
            oi_percentile_score = 15  # Normal OI
            oi_status = "OI NORMAL"
            oi_color = "#ffaa00"
        
        # Change component (40% of OI score)
        if avg_oi_change > Config.OI_SPIKE_THRESHOLD:
            oi_change_score = 18  # OI building = good
            oi_trend = "BUILDING"
        elif avg_oi_change < Config.OI_CRUSH_THRESHOLD:
            oi_change_score = 0  # OI declining = bad
            oi_trend = "DECLINING"
        else:
            oi_change_score = 9  # OI stable
            oi_trend = "STABLE"
        
        oi_score = oi_percentile_score + oi_change_score
        
        # ═══════════════════════════════════════════════════════════
        # PCR SCORE (10% weight)
        # ═══════════════════════════════════════════════════════════
        pcr_score = 0
        
        if 0.8 <= pcr_oi <= 1.2:
            pcr_score = 10  # Balanced
            pcr_status = "BALANCED"
            pcr_color = "#00ff88"
        elif pcr_oi > 1.5:
            pcr_score = 5  # Put heavy
            pcr_status = "PUT HEAVY"
            pcr_color = "#ffaa00"
        elif pcr_oi < 0.5:
            pcr_score = 5  # Call heavy
            pcr_status = "CALL HEAVY"
            pcr_color = "#ffaa00"
        else:
            pcr_score = 7  # Moderate
            pcr_status = "MODERATE"
            pcr_color = "#ffaa00"
        
        # ═══════════════════════════════════════════════════════════
        # TOTAL SCORE & REGIME CLASSIFICATION
        # ═══════════════════════════════════════════════════════════
        total_score = iv_score + oi_score + pcr_score
        
        if total_score >= 85:
            regime = "OPTIMAL SHORT STRADDLE"
            regime_class = "regime-optimal"
            regime_emoji = "🟢"
            recommendation = "Aggressive short straddle - High probability setup with favorable risk/reward"
        elif total_score >= 70:
            regime = "FAVORABLE SHORT STRADDLE"
            regime_class = "regime-favorable"
            regime_emoji = "🟢"
            recommendation = "Short straddle recommended - Good setup with acceptable risk"
        elif total_score >= 50:
            regime = "NEUTRAL MARKET"
            regime_class = "regime-neutral"
            regime_emoji = "🟡"
            recommendation = "Cautious short straddle - Monitor closely and use tight stops"
        elif total_score >= 30:
            regime = "UNFAVORABLE MARKET"
            regime_class = "regime-unfavorable"
            regime_emoji = "🔴"
            recommendation = "Avoid short straddle - Risk/reward unfavorable"
        else:
            regime = "DANGEROUS MARKET"
            regime_class = "regime-dangerous"
            regime_emoji = "⛔"
            recommendation = "Do not trade - Extreme volatility and unfavorable conditions"
        
        return {
            'regime': regime,
            'regime_class': regime_class,
            'regime_emoji': regime_emoji,
            'recommendation': recommendation,
            'total_score': total_score,
            'iv_score': iv_score,
            'oi_score': oi_score,
            'pcr_score': pcr_score,
            'iv_status': iv_status,
            'iv_trend': iv_trend,
            'oi_status': oi_status,
            'oi_trend': oi_trend,
            'pcr_status': pcr_status,
            'iv_color': iv_color,
            'oi_color': oi_color,
            'pcr_color': pcr_color,
            'avg_iv_change': avg_iv_change,
            'avg_oi_change': avg_oi_change,
            'iv_percentile': iv_percentile,
            'oi_percentile': oi_percentile
        }

# ═══════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════

class ScoringEngine:
    """
    Strike-level scoring engine
    
    Scoring Components:
    - Market Regime: 90%
    - Compression: 5%
    - Proximity: 5%
    """
    
    @staticmethod
    def calculate_compression_score(strike: float, strikes_data: Dict, 
                                   spot_price: float, max_pain: Optional[float]) -> Dict:
        """
        Calculate OI compression score for a strike
        
        Why: Symmetric OI distribution indicates equilibrium
        """
        data = strikes_data[strike]
        call_oi = data.get('call_oi', 0)
        put_oi = data.get('put_oi', 0)
        
        total_oi = call_oi + put_oi
        if total_oi == 0:
            return {
                'score': 0.0, 
                'symmetry': 0, 
                'intensity': 0, 
                'distance_spot_pct': 100
            }
        
        # Symmetry: how balanced are calls and puts
        symmetry = 1 - abs((call_oi / total_oi) - (put_oi / total_oi))
        
        # Intensity: relative to max OI in chain
        max_oi = max([s.get('call_oi', 0) + s.get('put_oi', 0) 
                     for s in strikes_data.values()])
        intensity = total_oi / max_oi if max_oi > 0 else 0
        
        # Combined compression
        compression = symmetry * intensity
        
        # Distance from spot
        distance_from_spot = abs(strike - spot_price) / spot_price * 100
        
        # Max pain proximity bonus
        distance_from_max_pain = abs(strike - max_pain) if max_pain else 1000
        proximity_multiplier = 1.2 if distance_from_max_pain <= Config.MAX_PAIN_PROXIMITY else 1.0
        
        final_score = compression * proximity_multiplier
        
        return {
            'score': min(final_score, 1.0),
            'symmetry': symmetry,
            'intensity': intensity,
            'distance_spot_pct': distance_from_spot
        }
    
    @staticmethod
    def calculate_unified_score(spot: float, ema8: float, ema13: float, ema21: float,
                               regime_score: float, compression_score: float, 
                               distance_from_spot: float) -> Dict:
        """
        Calculate unified strike score
        
        Weights:
        - Market Regime: 90%
        - Compression: 5%
        - Proximity: 5%
        """
        
        # Base score from regime (0-90)
        score = regime_score * 0.9
        
        breakdown = {}
        breakdown['regime'] = f'+{regime_score * 0.9:.1f} (Market Regime: {regime_score:.0f}/100)'
        
        # Compression Score (5%)
        comp_points = compression_score * 5
        score += comp_points
        breakdown['compression'] = f'+{comp_points:.1f} (Compression: {compression_score:.2f})'
        
        # Proximity (5%)
        if distance_from_spot < 0.5:
            prox_points = 5
            breakdown['proximity'] = '+5.0 (Very close to spot)'
        elif distance_from_spot < 1.0:
            prox_points = 3
            breakdown['proximity'] = '+3.0 (Close to spot)'
        else:
            prox_points = 0
            breakdown['proximity'] = '+0.0 (Far from spot)'
        
        score += prox_points
        
        final_score = max(0, min(100, score))
        
        # Rating
        if final_score >= 85:
            rating = '⭐⭐⭐⭐⭐ EXCELLENT'
            action = 'AGGRESSIVE SHORT STRADDLE'
        elif final_score >= 70:
            rating = '⭐⭐⭐⭐ HIGH'
            action = 'SHORT STRADDLE'
        elif final_score >= 50:
            rating = '⭐⭐⭐ MODERATE'
            action = 'CAUTIOUS SHORT STRADDLE'
        elif final_score >= 30:
            rating = '⭐⭐ LOW'
            action = 'MINIMAL EXPOSURE'
        else:
            rating = '⭐ AVOID'
            action = 'DO NOT TRADE'
        
        return {
            'score': final_score, 
            'rating': rating, 
            'action': action, 
            'breakdown': breakdown
        }

# ═══════════════════════════════════════════════════════════════════
# TRADING JOURNAL
# ═══════════════════════════════════════════════════════════════════

class TradingJournal:
    """Advanced trading journal with performance analytics"""
    
    @staticmethod
    def load_journal() -> pd.DataFrame:
        """Load trading journal from CSV"""
        if os.path.exists(Config.JOURNAL_FILE):
            try:
                df = pd.read_csv(Config.JOURNAL_FILE)
                df['entry_date'] = pd.to_datetime(df['entry_date'])
                df['exit_date'] = pd.to_datetime(df['exit_date'])
                return df
            except:
                return pd.DataFrame()
        return pd.DataFrame()
    
    @staticmethod
    def save_journal(df: pd.DataFrame):
        """Save trading journal to CSV"""
        df.to_csv(Config.JOURNAL_FILE, index=False)
    
    @staticmethod
    def add_trade(trade_data: Dict) -> pd.DataFrame:
        """Add new trade to journal"""
        df = TradingJournal.load_journal()
        new_trade = pd.DataFrame([trade_data])
        df = pd.concat([df, new_trade], ignore_index=True)
        TradingJournal.save_journal(df)
        return df
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame, period: str = 'all') -> Dict:
        """Calculate comprehensive trading statistics"""
        if df.empty:
            return {
                'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'max_drawdown': 0, 'sharpe_ratio': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0,
                'win_streak': 0, 'loss_streak': 0, 'expectancy': 0
            }
        
        # Filter by period
        if period != 'all':
            now = get_ist_time()
            if period == 'day':
                df = df[df['entry_date'] >= now - timedelta(days=1)]
            elif period == 'week':
                df = df[df['entry_date'] >= now - timedelta(weeks=1)]
            elif period == 'month':
                df = df[df['entry_date'] >= now - timedelta(days=30)]
        
        if df.empty:
            return {
                'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'max_drawdown': 0, 'sharpe_ratio': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0,
                'win_streak': 0, 'loss_streak': 0, 'expectancy': 0
            }
        
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = df['pnl'].mean()
        total_pnl = df['pnl'].sum()
        
        # Drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio
        sharpe_ratio = (df['pnl'].mean() / df['pnl'].std()) * np.sqrt(252) if df['pnl'].std() > 0 else 0
        
        # Profit Factor
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Win/Loss Stats
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        largest_win = wins['pnl'].max() if len(wins) > 0 else 0
        largest_loss = losses['pnl'].min() if len(losses) > 0 else 0
        
        # Streaks
        win_streak = TradingJournal._calculate_max_streak(df, 'win')
        loss_streak = TradingJournal._calculate_max_streak(df, 'loss')
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'win_streak': win_streak,
            'loss_streak': loss_streak,
            'expectancy': expectancy
        }
    
    @staticmethod
    def _calculate_max_streak(df: pd.DataFrame, streak_type: str) -> int:
        """Calculate maximum winning or losing streak"""
        if df.empty:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for pnl in df['pnl']:
            if streak_type == 'win' and pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            elif streak_type == 'loss' and pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    @staticmethod
    def calculate_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly P&L statistics"""
        if df.empty:
            return pd.DataFrame()
        
        df['month'] = pd.to_datetime(df['entry_date']).dt.to_period('M')
        monthly = df.groupby('month').agg({
            'pnl': ['sum', 'count', 'mean']
        }).reset_index()
        
        monthly.columns = ['month', 'total_pnl', 'trades', 'avg_pnl']
        monthly['month'] = monthly['month'].astype(str)
        
        return monthly

# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

def fetch_market_data(asset: str, expiry_date: str) -> Optional[Dict]:
    """
    Fetch and process all market data
    
    This is the main data pipeline that orchestrates all API calls
    and analysis calculations
    """
    try:
        api = DeltaAPI()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Spot Price
        # ═══════════════════════════════════════════════════════════
        status_text.text("📊 Fetching spot price...")
        progress_bar.progress(10)
        
        symbol = f'{asset}USD'
        spot_price = api.get_spot_price(symbol)
        
        if not spot_price:
            st.error("❌ Failed to fetch spot price")
            progress_bar.empty()
            status_text.empty()
            return None
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Historical Candles
        # ═══════════════════════════════════════════════════════════
        status_text.text("📈 Fetching historical candles...")
        progress_bar.progress(20)
        
        candles = api.get_historical_candles(
            symbol, 
            Config.CANDLE_RESOLUTION, 
            Config.HISTORICAL_CANDLES
        )
        
        if not candles:
            st.error("❌ Failed to fetch candles")
            progress_bar.empty()
            status_text.empty()
            return None
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Technical Analysis
        # ═══════════════════════════════════════════════════════════
        status_text.text("🔢 Calculating technical indicators...")
        progress_bar.progress(30)
        
        ema8, ema13, ema21, ema_series = TechnicalAnalysis.calculate_all_emas(candles)
        
        # Calculate Realized Volatility
        rv_metrics = TechnicalAnalysis.calculate_realized_volatility(
            candles,
            {'15m': 4, '1h': 24, '4h': 96}
        )
        
        # ═══════════════════════════════════════════════════════════
        # Step 4: Options Chain
        # ═══════════════════════════════════════════════════════════
        status_text.text("⚙️ Fetching options chain...")
        progress_bar.progress(45)
        
        options_chain = api.get_options_chain(asset, expiry_date)
        
        if not options_chain:
            st.error("❌ Failed to fetch options chain")
            progress_bar.empty()
            status_text.empty()
            return None
        
        # ═══════════════════════════════════════════════════════════
        # Step 5: Options Analysis
        # ═══════════════════════════════════════════════════════════
        status_text.text("🧮 Processing options data...")
        progress_bar.progress(55)
        
        strikes_data, totals = OptionsAnalysis.process_options_chain(
            options_chain, 
            spot_price
        )
        
        atm_strike = OptionsAnalysis.find_atm_strike(spot_price, options_chain)
        atm_iv = OptionsAnalysis.get_atm_iv(options_chain, atm_strike)
        max_pain = OptionsAnalysis.calculate_max_pain(strikes_data)
        
        # ATM OI Concentration
        total_oi = totals['total_call_oi'] + totals['total_put_oi']
        atm_oi_concentration = OptionsAnalysis.calculate_atm_oi_concentration(
            strikes_data, 
            atm_strike, 
            total_oi
        )
        
        # ═══════════════════════════════════════════════════════════
        # Step 6: Historical Data & Percentiles
        # ═══════════════════════════════════════════════════════════
        status_text.text("📊 Calculating percentiles...")
        progress_bar.progress(65)
        
        # Save and load IV data
        if atm_iv:
            iv_history = HistoricalDataManager.save_metric_data(
                asset, expiry_date, 'iv', atm_iv
            )
            iv_percentile = HistoricalDataManager.calculate_percentile(
                iv_history, atm_iv
            )
            iv_changes = HistoricalDataManager.calculate_metric_changes(
                iv_history,
                {'5m': 5, '15m': 15, '30m': 30, '1h': 60}
            )
        else:
            iv_history = []
            iv_percentile = 50.0
            iv_changes = {'change_5m': 0, 'change_15m': 0, 'change_30m': 0, 'change_1h': 0}
        
        # Save and load OI data
        oi_history = HistoricalDataManager.save_metric_data(
            asset, expiry_date, 'oi', total_oi
        )
        oi_percentile = HistoricalDataManager.calculate_percentile(
            oi_history, total_oi
        )
        oi_changes = HistoricalDataManager.calculate_metric_changes(
            oi_history,
            {'5m': 5, '15m': 15, '30m': 30, '1h': 60}
        )
        
        # Save and load PCR data
        pcr_history = HistoricalDataManager.save_metric_data(
            asset, expiry_date, 'pcr', totals['pcr_oi']
        )
        pcr_percentile = HistoricalDataManager.calculate_percentile(
            pcr_history, totals['pcr_oi']
        )
        
        # ═══════════════════════════════════════════════════════════
        # Step 7: Volatility Analysis
        # ═══════════════════════════════════════════════════════════
        status_text.text("🌊 Analyzing volatility regime...")
        progress_bar.progress(75)
        
        # IV metrics
        iv_metrics = VolatilityAnalyzer.calculate_iv_metrics(iv_history)
        
        # IV-RV spread
        rv_1h = rv_metrics.get('rv_1h', 0)
        iv_rv_spread = (atm_iv - rv_1h) if atm_iv else 0
        
        # Volatility regime
        vol_regime = VolatilityAnalyzer.determine_volatility_regime(
            atm_iv if atm_iv else 0,
            rv_1h,
            iv_rv_spread
        )
        
        # Expected move
        # Calculate days to expiry
        try:
            expiry_dt = datetime.strptime(expiry_date, '%d-%m-%Y')
            expiry_dt = Config.IST.localize(expiry_dt.replace(hour=15, minute=30))
            days_to_expiry = (expiry_dt - get_ist_time()).total_seconds() / 86400
            days_to_expiry = max(0.01, days_to_expiry)  # Prevent division by zero
        except:
            days_to_expiry = 1
        
        expected_move = VolatilityAnalyzer.calculate_expected_move(
            atm_iv if atm_iv else 0,
            spot_price,
            days_to_expiry
        )
        
        # ═══════════════════════════════════════════════════════════
        # Step 8: Market Regime Analysis
        # ═══════════════════════════════════════════════════════════
        status_text.text("🎯 Analyzing market regime...")
        progress_bar.progress(85)
        
        regime_analysis = MarketRegimeAnalyzer.analyze_regime(
            iv_changes,
            oi_changes,
            totals['pcr_oi'],
            iv_percentile,
            oi_percentile
        )
        
        # ═══════════════════════════════════════════════════════════
        # Step 9: Strike Scoring
        # ═══════════════════════════════════════════════════════════
        status_text.text("⭐ Scoring strikes...")
        progress_bar.progress(95)
        
        compression_scores = {}
        for strike in strikes_data.keys():
            comp_data = ScoringEngine.calculate_compression_score(
                strike, strikes_data, spot_price, max_pain
            )
            compression_scores[strike] = comp_data
        
        # ═══════════════════════════════════════════════════════════
        # Complete
        # ═══════════════════════════════════════════════════════════
        progress_bar.progress(100)
        status_text.text("✅ Data loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'timestamp': get_ist_time().isoformat(),
            'asset': asset,
            'expiry_date': expiry_date,
            'days_to_expiry': days_to_expiry,
            
            # Price Data
            'spot_price': spot_price,
            'ema8': ema8,
            'ema13': ema13,
            'ema21': ema21,
            'ema_series': ema_series,
            'candles': candles,
            
            # Options Data
            'atm_strike': atm_strike,
            'atm_iv': atm_iv,
            'max_pain': max_pain,
            'strikes_data': strikes_data,
            'totals': totals,
            'atm_oi_concentration': atm_oi_concentration,
            
            # Historical & Percentiles
            'iv_history': iv_history,
            'iv_percentile': iv_percentile,
            'iv_changes': iv_changes,
            'oi_history': oi_history,
            'oi_percentile': oi_percentile,
            'oi_changes': oi_changes,
            'pcr_history': pcr_history,
            'pcr_percentile': pcr_percentile,
            
            # Volatility Analysis
            'rv_metrics': rv_metrics,
            'iv_metrics': iv_metrics,
            'iv_rv_spread': iv_rv_spread,
            'vol_regime': vol_regime,
            'expected_move': expected_move,
            
            # Regime & Scoring
            'regime_analysis': regime_analysis,
            'compression_scores': compression_scores
        }
        
    except Exception as e:
        st.error(f"❌ Error fetching market data: {str(e)}")
        st.exception(e)
        return None

# ═══════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def render_market_overview(data: Dict):
    """Render Market Overview tab"""
    
    st.markdown("### 📊 Market Snapshot")
    
    # Price Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Spot Price", f"${data['spot_price']:,.2f}")
    
    with col2:
        ema_signal, ema_type = TechnicalAnalysis.get_ema_signal(
            data['spot_price'], data['ema8'], data['ema13'], data['ema21']
        )
        st.metric("EMA 8", f"${data['ema8']:,.2f}")
    
    with col3:
        st.metric("EMA 13", f"${data['ema13']:,.2f}")
    
    with col4:
        st.metric("EMA 21", f"${data['ema21']:,.2f}")
    
    # EMA Signal
    ema_signal, ema_type = TechnicalAnalysis.get_ema_signal(
        data['spot_price'], data['ema8'], data['ema13'], data['ema21']
    )
    
    signal_colors = {
        'bullish': '#00ff88',
        'bearish': '#ff6b6b',
        'neutral': '#ffaa00',
        'mixed': '#a0a0a0'
    }
    
    st.markdown(f"""
    <div class="info-box" style="border-left-color: {signal_colors.get(ema_type, '#00d9ff')};">
        <strong>Market Structure:</strong> {ema_signal}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Options Metrics
    st.markdown("### 📈 Options Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "ATM IV", 
            f"{data['atm_iv']:.2f}%" if data['atm_iv'] else "N/A",
            delta=f"{data['iv_changes'].get('change_15m', 0):+.2f}% (15m)"
        )
    
    with col2:
        st.metric(
            "PCR (OI)", 
            f"{data['totals']['pcr_oi']:.2f}",
            help="Put-Call Ratio based on Open Interest"
        )
    
    with col3:
        st.metric(
            "Max Pain", 
            f"${data['max_pain']:,.0f}" if data['max_pain'] else "N/A",
            help="Strike with minimum option seller pain"
        )
    
    with col4:
        st.metric(
            "ATM OI %", 
            f"{data['atm_oi_concentration']:.1f}%",
            help="OI concentration at ATM strike"
        )
    
    st.markdown("---")
    
    # Candlestick Chart with EMAs
    st.markdown("### 📊 Price Action (15-Minute)")
    
    candles = data.get('candles', [])
    if candles:
        df_candles = pd.DataFrame(candles)
        df_candles['timestamp'] = pd.to_datetime(
            df_candles['time'], unit='s', utc=True
        ).dt.tz_convert(Config.IST)
        
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df_candles['timestamp'],
            open=df_candles['open'],
            high=df_candles['high'],
            low=df_candles['low'],
            close=df_candles['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff6b6b'
        ))
        
        # EMAs
        ema_series = data.get('ema_series', {})
        if ema_series:
            fig.add_trace(go.Scatter(
                x=df_candles['timestamp'], 
                y=ema_series['ema8'], 
                mode='lines', 
                name='EMA 8', 
                line=dict(color='#ff6b6b', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_candles['timestamp'], 
                y=ema_series['ema13'], 
                mode='lines', 
                name='EMA 13', 
                line=dict(color='#4ecdc4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df_candles['timestamp'], 
                y=ema_series['ema21'], 
                mode='lines', 
                name='EMA 21', 
                line=dict(color='#ffe66d', width=2)
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            xaxis_title='Time (IST)',
            yaxis_title='Price',
            height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_market_regime(data: Dict):
    """Render Market Regime tab"""
    
    regime = data['regime_analysis']
    
    st.markdown("### 🎯 MARKET REGIME ANALYSIS")
    
    # Main Regime Display
    st.markdown(f"""
    <div class="{regime['regime_class']} regime-card">
        <h1 style='font-size: 56px; margin: 0;'>{regime['regime_emoji']} {regime['regime']}</h1>
        <h2 style='font-size: 80px; color: #00d9ff; margin: 16px 0;'>{regime['total_score']:.0f}/100</h2>
        <p style='font-size: 18px; margin: 0; color: #e0e0e0;'>{regime['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Score Breakdown
    st.markdown("### 📊 Score Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style='border-left: 4px solid {regime['iv_color']};'>
            <h3 style='color: {regime['iv_color']};'>IV ANALYSIS</h3>
            <h2 style='font-size: 48px; color: {regime['iv_color']};'>{regime['iv_score']:.0f}/45</h2>
            <p style='color: {regime['iv_color']}; font-weight: 600;'>{regime['iv_status']}</p>
            <p style='color: #a0a0a0;'>Trend: {regime['iv_trend']}</p>
            <p style='color: #a0a0a0;'>Change: {regime['avg_iv_change']:+.2f}%</p>
            <p style='color: #a0a0a0;'>Percentile: {regime['iv_percentile']:.0f}th</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style='border-left: 4px solid {regime['oi_color']};'>
            <h3 style='color: {regime['oi_color']};'>OI ANALYSIS</h3>
            <h2 style='font-size: 48px; color: {regime['oi_color']};'>{regime['oi_score']:.0f}/45</h2>
            <p style='color: {regime['oi_color']}; font-weight: 600;'>{regime['oi_status']}</p>
            <p style='color: #a0a0a0;'>Trend: {regime['oi_trend']}</p>
            <p style='color: #a0a0a0;'>Change: {regime['avg_oi_change']:+.2f}%</p>
            <p style='color: #a0a0a0;'>Percentile: {regime['oi_percentile']:.0f}th</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style='border-left: 4px solid {regime['pcr_color']};'>
            <h3 style='color: {regime['pcr_color']};'>PCR ANALYSIS</h3>
            <h2 style='font-size: 48px; color: {regime['pcr_color']};'>{regime['pcr_score']:.0f}/10</h2>
            <p style='color: {regime['pcr_color']}; font-weight: 600;'>{regime['pcr_status']}</p>
            <p style='color: #a0a0a0;'>PCR: {data['totals']['pcr_oi']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Historical Trends
    st.markdown("### 📈 Historical Context (24h)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### IV Trend")
        iv_history = data.get('iv_history', [])
        
        if len(iv_history) > 1:
            df_iv = pd.DataFrame(iv_history)
            df_iv['timestamp'] = pd.to_datetime(df_iv['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_iv['timestamp'],
                y=df_iv['value'],
                mode='lines+markers',
                name='ATM IV',
                line=dict(color='#00d9ff', width=3),
                marker=dict(size=6)
            ))
            
            # Add percentile reference lines
            if len(df_iv) >= 10:
                p80 = df_iv['value'].quantile(0.8)
                p20 = df_iv['value'].quantile(0.2)
                
                fig.add_hline(
                    y=p80, 
                    line_dash="dash", 
                    line_color="#ff6b6b",
                    annotation_text="80th Percentile"
                )
                fig.add_hline(
                    y=p20, 
                    line_dash="dash", 
                    line_color="#00ff88",
                    annotation_text="20th Percentile"
                )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0f0f0f',
                xaxis_title='Time',
                yaxis_title='IV (%)',
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Collecting IV data... (need 24h history)")
    
    with col2:
        st.markdown("#### OI Trend")
        oi_history = data.get('oi_history', [])
        
        if len(oi_history) > 1:
            df_oi = pd.DataFrame(oi_history)
            df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_oi['timestamp'],
                y=df_oi['value'],
                mode='lines+markers',
                name='Total OI',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.1)'
            ))
            
            # Add percentile reference lines
            if len(df_oi) >= 10:
                p80 = df_oi['value'].quantile(0.8)
                p20 = df_oi['value'].quantile(0.2)
                
                fig.add_hline(
                    y=p80, 
                    line_dash="dash", 
                    line_color="#00ff88",
                    annotation_text="80th Percentile"
                )
                fig.add_hline(
                    y=p20, 
                    line_dash="dash", 
                    line_color="#ff6b6b",
                    annotation_text="20th Percentile"
                )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0f0f0f',
                xaxis_title='Time',
                yaxis_title='OI',
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Collecting OI data... (need 24h history)")

def render_oi_distribution(data: Dict):
    """Render OI Distribution tab with percentile analysis"""
    
    st.markdown("### 📊 OPEN INTEREST DISTRIBUTION")
    
    # Percentile Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_oi = data['totals']['total_call_oi'] + data['totals']['total_put_oi']
        st.metric(
            "Total OI",
            format_large_number(total_oi),
            help="Total Open Interest across all strikes"
        )
    
    with col2:
        oi_percentile = data['oi_percentile']
        percentile_color = (
            "#00ff88" if oi_percentile < Config.PERCENTILE_LOW_THRESHOLD
            else "#ff6b6b" if oi_percentile > Config.PERCENTILE_EXTREME_THRESHOLD
            else "#ffaa00"
        )
        st.markdown(f"""
        <div class="metric-card">
            <p style='color: #a0a0a0; font-size: 13px; margin: 0;'>OI PERCENTILE</p>
            <h2 style='color: {percentile_color}; font-size: 48px; margin: 8px 0;'>{oi_percentile:.0f}th</h2>
            <p style='color: #a0a0a0; font-size: 12px; margin: 0;'>24h rolling window</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric(
            "ATM Concentration",
            f"{data['atm_oi_concentration']:.1f}%",
            help="Percentage of total OI at ATM strike"
        )
    
    with col4:
        pcr_percentile = data['pcr_percentile']
        st.metric(
            "PCR Percentile",
            f"{pcr_percentile:.0f}th",
            help="Put-Call Ratio percentile"
        )
    
    # Percentile Bar
    st.markdown("#### OI Percentile Context")
    
    percentile_position = data['oi_percentile']
    
    st.markdown(f"""
    <div class="percentile-container">
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='color: #00ff88;'>Low (Favorable)</span>
            <span style='color: #ffaa00;'>Normal</span>
            <span style='color: #ff6b6b;'>Extreme (Caution)</span>
        </div>
        <div class="percentile-bar">
            <div class="percentile-marker" style="left: {percentile_position}%;"></div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 8px; color: #a0a0a0; font-size: 12px;'>
            <span>0th</span>
            <span>50th</span>
            <span>100th</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation
    if percentile_position < Config.PERCENTILE_LOW_THRESHOLD:
        interpretation = "OI is at historically low levels - crowd risk is minimal"
        interp_color = "#00ff88"
    elif percentile_position > Config.PERCENTILE_EXTREME_THRESHOLD:
        interpretation = "OI is at extreme levels - crowd risk is elevated, exercise caution"
        interp_color = "#ff6b6b"
    else:
        interpretation = "OI is at normal levels - moderate crowd participation"
        interp_color = "#ffaa00"
    
    st.markdown(f"""
    <div class="info-box" style="border-left-color: {interp_color};">
        <strong>Interpretation:</strong> {interpretation}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # OI Change Metrics
    st.markdown("### 📈 OI Dynamics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    oi_changes = data['oi_changes']
    
    with col1:
        change_5m = oi_changes.get('change_5m', 0)
        st.metric("5m Change", f"{change_5m:+.2f}%")
    
    with col2:
        change_15m = oi_changes.get('change_15m', 0)
        st.metric("15m Change", f"{change_15m:+.2f}%")
    
    with col3:
        change_30m = oi_changes.get('change_30m', 0)
        st.metric("30m Change", f"{change_30m:+.2f}%")
    
    with col4:
        change_1h = oi_changes.get('change_1h', 0)
        st.metric("1h Change", f"{change_1h:+.2f}%")
    
    # OI Build/Unwind Status
    avg_change = (change_15m + change_30m) / 2
    
    if avg_change > Config.OI_SPIKE_THRESHOLD:
        oi_status = "🟢 OI BUILDING - Strong interest, positions being added"
        status_color = "#00ff88"
    elif avg_change < Config.OI_CRUSH_THRESHOLD:
        oi_status = "🔴 OI UNWINDING - Positions being closed, interest declining"
        status_color = "#ff6b6b"
    else:
        oi_status = "🟡 OI STABLE - Moderate activity, no extreme moves"
        status_color = "#ffaa00"
    
    st.markdown(f"""
    <div class="info-box" style="border-left-color: {status_color};">
        <strong>{oi_status}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # OI Distribution Chart
    st.markdown("### 📊 Strike-wise OI Distribution")
    
    spot = data['spot_price']
    max_pain = data['max_pain']
    atm_strike = data['atm_strike']
    strikes_data = data['strikes_data']
    
    strikes = sorted(strikes_data.keys())
    call_oi = [strikes_data[s]['call_oi'] for s in strikes]
    put_oi = [strikes_data[s]['put_oi'] for s in strikes]
    
    fig = go.Figure()
    
    # Call OI
    fig.add_trace(go.Bar(
        x=strikes,
        y=call_oi,
        name='Call OI',
        marker_color='#00ff88',
        hovertemplate='Strike: %{x}<br>Call OI: %{y:,.0f}<extra></extra>'
    ))
    
    # Put OI (negative for visual separation)
    fig.add_trace(go.Bar(
        x=strikes,
        y=[-p for p in put_oi],
        name='Put OI',
        marker_color='#ff6b6b',
        hovertemplate='Strike: %{x}<br>Put OI: %{y:,.0f}<extra></extra>'
    ))
    
    # Spot Price
    fig.add_vline(
        x=spot,
        line_dash="solid",
        line_color="#ffff00",
        line_width=2,
        annotation_text=f"Spot: ${spot:,.0f}",
        annotation_position="top"
    )
    
    # ATM Strike
    if atm_strike:
        fig.add_vline(
            x=atm_strike,
            line_dash="dash",
            line_color="#00d9ff",
            line_width=2,
            annotation_text=f"ATM: ${atm_strike:,.0f}",
            annotation_position="top"
        )
    
    # Max Pain
    if max_pain:
        fig.add_vline(
            x=max_pain,
            line_dash="dot",
            line_color="#ff00ff",
            line_width=2,
            annotation_text=f"Max Pain: ${max_pain:,.0f}",
            annotation_position="bottom"
        )
    
    fig.update_layout(
        barmode='relative',
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0f0f0f',
        title='Open Interest Distribution',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PCR Analysis
    st.markdown("---")
    st.markdown("### 📊 Put-Call Ratio Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pcr_oi = data['totals']['pcr_oi']
        pcr_volume = data['totals']['pcr_volume']
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>PCR (OI)</h4>
            <h2 style='font-size: 56px; color: #00d9ff;'>{pcr_oi:.2f}</h2>
            <p style='color: #a0a0a0;'>Put OI / Call OI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>PCR (Volume)</h4>
            <h2 style='font-size: 56px; color: #00d9ff;'>{pcr_volume:.2f}</h2>
            <p style='color: #a0a0a0;'>Put Volume / Call Volume</p>
        </div>
        """, unsafe_allow_html=True)
    
    # PCR Interpretation
    if 0.8 <= pcr_oi <= 1.2:
        pcr_interp = "Balanced market - No extreme directional bias"
        pcr_color = "#00ff88"
    elif pcr_oi > 1.5:
        pcr_interp = "Put heavy - Bearish positioning or hedging activity"
        pcr_color = "#ff6b6b"
    elif pcr_oi < 0.5:
        pcr_interp = "Call heavy - Bullish positioning or speculative activity"
        pcr_color = "#ffaa00"
    else:
        pcr_interp = "Moderate bias - Slight directional lean"
        pcr_color = "#ffaa00"
    
    st.markdown(f"""
    <div class="info-box" style="border-left-color: {pcr_color};">
        <strong>PCR Interpretation:</strong> {pcr_interp}
    </div>
    """, unsafe_allow_html=True)

def render_volatility_analysis(data: Dict):
    """Render Volatility Analysis tab with IV-RV framework"""
    
    st.markdown("### 🌊 VOLATILITY ANALYSIS")
    
    # Volatility Regime
    vol_regime = data['vol_regime']
    
    st.markdown(f"""
    <div class="regime-card" style="border-color: {vol_regime['color']};">
        <h2 style='font-size: 48px; margin: 0;'>{vol_regime['regime']}</h2>
        <p style='font-size: 16px; margin: 16px 0; color: #e0e0e0;'>{vol_regime['description']}</p>
        <p style='font-size: 14px; margin: 0;'><strong>Risk Level:</strong> {vol_regime['risk_level']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # IV vs RV Metrics
    st.markdown("### 📊 IV vs RV Comparison")
    
    atm_iv = data['atm_iv'] if data['atm_iv'] else 0
    rv_metrics = data['rv_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "ATM IV",
            f"{atm_iv:.2f}%",
            help="Implied Volatility at ATM strike"
        )
    
    with col2:
        rv_1h = rv_metrics.get('rv_1h', 0)
        st.metric(
            "RV (1h)",
            f"{rv_1h:.2f}%",
            help="Realized Volatility over 1 hour"
        )
    
    with col3:
        rv_4h = rv_metrics.get('rv_4h', 0)
        st.metric(
            "RV (4h)",
            f"{rv_4h:.2f}%",
            help="Realized Volatility over 4 hours"
        )
    
    with col4:
        iv_rv_spread = data['iv_rv_spread']
        spread_color = (
            "#00ff88" if iv_rv_spread > 10
            else "#ff6b6b" if iv_rv_spread < -10
            else "#ffaa00"
        )
        st.markdown(f"""
        <div class="metric-card">
            <p style='color: #a0a0a0; font-size: 13px; margin: 0;'>IV - RV SPREAD</p>
            <h2 style='color: {spread_color}; font-size: 48px; margin: 8px 0;'>{iv_rv_spread:+.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Spread Interpretation
    st.markdown(f"""
    <div class="info-box">
        <strong>Spread Interpretation:</strong> {vol_regime['spread_interpretation']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # IV-RV Chart
    st.markdown("### 📈 IV vs RV Trend")
    
    iv_history = data.get('iv_history', [])
    
    if len(iv_history) > 10:
        df_iv = pd.DataFrame(iv_history)
        df_iv['timestamp'] = pd.to_datetime(df_iv['timestamp'])
        
        # Calculate RV for each timestamp (simplified - using last 4 periods)
        # In production, you'd calculate RV at each historical point
        
        fig = go.Figure()
        
        # IV Line
        fig.add_trace(go.Scatter(
            x=df_iv['timestamp'],
            y=df_iv['value'],
            mode='lines',
            name='Implied Volatility (IV)',
            line=dict(color='#00d9ff', width=3)
        ))
        
        # RV Line (constant for now - ideally historical RV)
        fig.add_trace(go.Scatter(
            x=df_iv['timestamp'],
            y=[rv_1h] * len(df_iv),
            mode='lines',
            name='Realized Volatility (RV)',
            line=dict(color='#ff6b6b', width=3, dash='dash')
        ))
        
        # Spread Area
        fig.add_trace(go.Scatter(
            x=df_iv['timestamp'],
            y=df_iv['value'] - rv_1h,
            mode='lines',
            name='IV-RV Spread',
            line=dict(color='#00ff88', width=0),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            xaxis_title='Time',
            yaxis_title='Volatility (%)',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Collecting volatility data... (need more history)")
    
    st.markdown("---")
    
    # IV Dynamics
    st.markdown("### 📊 IV Dynamics")
    
    iv_metrics = data['iv_metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        velocity = iv_metrics['iv_velocity']
        velocity_color = (
            "#00ff88" if velocity < -0.5
            else "#ff6b6b" if velocity > 0.5
            else "#ffaa00"
        )
        st.markdown(f"""
        <div class="metric-card">
            <h4>IV Velocity</h4>
            <h2 style='font-size: 48px; color: {velocity_color};'>{velocity:+.2f}</h2>
            <p style='color: #a0a0a0;'>Rate of IV change</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        acceleration = iv_metrics['iv_acceleration']
        accel_color = (
            "#00ff88" if acceleration < 0
            else "#ff6b6b" if acceleration > 0
            else "#ffaa00"
        )
        st.markdown(f"""
        <div class="metric-card">
            <h4>IV Acceleration</h4>
            <h2 style='font-size: 48px; color: {accel_color};'>{acceleration:+.2f}</h2>
            <p style='color: #a0a0a0;'>Rate of velocity change</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend = iv_metrics['iv_trend']
        trend_display = {
            'rising': ('🔴 RISING', '#ff6b6b'),
            'falling': ('🟢 FALLING', '#00ff88'),
            'stable': ('🟡 STABLE', '#ffaa00'),
            'insufficient_data': ('⚪ INSUFFICIENT DATA', '#a0a0a0')
        }
        trend_text, trend_color = trend_display.get(trend, ('UNKNOWN', '#a0a0a0'))
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>IV Trend</h4>
            <h2 style='font-size: 32px; color: {trend_color};'>{trend_text}</h2>
            <p style='color: #a0a0a0;'>Current direction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Expected Move
    st.markdown("### 📏 Expected Price Move")
    
    expected_move = data['expected_move']
    spot = data['spot_price']
    days_to_expiry = data['days_to_expiry']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>1 Standard Deviation Move</h4>
            <h2 style='font-size: 48px; color: #00d9ff;'>±${expected_move['expected_move_1sd']:,.0f}</h2>
            <p style='color: #a0a0a0;'>68% probability range</p>
            <p style='color: #00ff88;'>Upper: ${expected_move['upper_bound_1sd']:,.0f}</p>
            <p style='color: #ff6b6b;'>Lower: ${expected_move['lower_bound_1sd']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>2 Standard Deviation Move</h4>
            <h2 style='font-size: 48px; color: #00d9ff;'>±${expected_move['expected_move_2sd']:,.0f}</h2>
            <p style='color: #a0a0a0;'>95% probability range</p>
            <p style='color: #00ff88;'>Upper: ${expected_move['upper_bound_2sd']:,.0f}</p>
            <p style='color: #ff6b6b;'>Lower: ${expected_move['lower_bound_2sd']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Context:</strong> Based on current IV of {atm_iv:.2f}% and {days_to_expiry:.1f} days to expiry, 
        the market expects price to stay within ±${expected_move['expected_move_1sd']:,.0f} with 68% probability.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Volatility Regime Matrix
    st.markdown("### 🎯 Volatility Regime Matrix")
    
    st.markdown("""
    <div class="vol-matrix">
        <div class="vol-cell vol-premium-rich">
            <h4>🟢 PREMIUM RICH</h4>
            <p>Low RV + High IV</p>
            <p style='font-size: 12px;'>Favorable for selling</p>
        </div>
        <div class="vol-cell vol-trap">
            <h4>🔴 VOLATILITY TRAP</h4>
            <p>High RV + Low IV</p>
            <p style='font-size: 12px;'>Dangerous for selling</p>
        </div>
        <div class="vol-cell vol-dead">
            <h4>⚪ LOW VOLATILITY</h4>
            <p>Low RV + Low IV</p>
            <p style='font-size: 12px;'>Limited opportunity</p>
        </div>
        <div class="vol-cell vol-dangerous">
            <h4>⛔ HIGH VOLATILITY</h4>
            <p>High RV + High IV</p>
            <p style='font-size: 12px;'>Extreme caution</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_straddle_analyzer(data: Dict):
    """Render Straddle Analyzer tab"""
    
    st.markdown("### 💡 STRADDLE OPPORTUNITY ANALYZER")
    
    spot = data['spot_price']
    ema8 = data['ema8']
    ema13 = data['ema13']
    ema21 = data['ema21']
    regime_analysis = data['regime_analysis']
    strikes_data = data['strikes_data']
    compression_scores = data['compression_scores']
    
    # Score all strikes
    scored_strikes = []
    for strike, comp_data in compression_scores.items():
        score_result = ScoringEngine.calculate_unified_score(
            spot, ema8, ema13, ema21, 
            regime_analysis['total_score'],
            comp_data['score'], 
            comp_data['distance_spot_pct']
        )
        scored_strikes.append({
            'strike': strike,
            'score': score_result['score'],
            'rating': score_result['rating'],
            'action': score_result['action'],
            'breakdown': score_result['breakdown'],
            'call_oi': strikes_data[strike]['call_oi'],
            'put_oi': strikes_data[strike]['put_oi'],
            'compression': comp_data['score'],
            'symmetry': comp_data['symmetry']
        })
    
    scored_strikes.sort(key=lambda x: x['score'], reverse=True)
    top_5 = scored_strikes[:5]
    
    if not top_5:
        st.warning("⚠️ No opportunities found in current market conditions")
        return
    
    # Best Opportunity
    best = top_5[0]
    
    st.markdown("### 🎯 BEST STRADDLE OPPORTUNITY")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        score_color = (
            "#00ff88" if best['score'] >= 85
            else "#00d9ff" if best['score'] >= 70
            else "#ffaa00" if best['score'] >= 50
            else "#ff6b6b"
        )
        
        st.markdown(f"""
        <div class="regime-card" style="border-color: {score_color};">
            <h1 style='font-size: 80px; color: {score_color}; margin: 0;'>{best['score']:.0f}</h1>
            <p style='color: #a0a0a0; font-size: 14px; margin: 8px 0;'>/100</p>
            <h3 style='margin: 16px 0;'>{best['rating']}</h3>
            <hr style='border-color: #2a2a2a; margin: 16px 0;'>
            <h4 style='margin: 8px 0;'>📍 Strike: ${best['strike']:,.0f}</h4>
            <h4 style='margin: 8px 0;'>💡 {best['action']}</h4>
            <p style='color: #a0a0a0; font-size: 12px; margin-top: 16px;'>
                Compression: {best['compression']:.2f} | Symmetry: {best['symmetry']:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top 5 Opportunities
    st.markdown("### 📊 Top 5 Opportunities")
    
    for i, opp in enumerate(top_5, 1):
        score_color = (
            "#00ff88" if opp['score'] >= 85
            else "#00d9ff" if opp['score'] >= 70
            else "#ffaa00" if opp['score'] >= 50
            else "#ff6b6b"
        )
        
        with st.expander(
            f"#{i} Strike: ${opp['strike']:,.0f} - Score: {opp['score']:.0f}/100",
            expanded=(i == 1)
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Strike Details</h4>
                    <p><strong>Rating:</strong> {opp['rating']}</p>
                    <p><strong>Action:</strong> {opp['action']}</p>
                    <p><strong>Call OI:</strong> {format_large_number(opp['call_oi'])}</p>
                    <p><strong>Put OI:</strong> {format_large_number(opp['put_oi'])}</p>
                    <p><strong>Compression:</strong> {opp['compression']:.2f}</p>
                    <p><strong>Symmetry:</strong> {opp['symmetry']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Score Breakdown</h4>
                """, unsafe_allow_html=True)
                
                for component, value in opp['breakdown'].items():
                    st.markdown(f"<p style='font-size: 13px;'>• {value}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Warning
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ RISK DISCLOSURE</h4>
        <p>Short straddles carry <strong>unlimited risk</strong> on the upside and substantial risk on the downside. 
        This tool provides decision support only and does not constitute trading advice. Always:</p>
        <ul>
            <li>Use appropriate position sizing</li>
            <li>Set stop losses</li>
            <li>Monitor positions actively</li>
            <li>Understand your maximum loss potential</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_trading_journal():
    """Render Trading Journal tab"""
    
    st.markdown("### 📓 TRADING JOURNAL")
    
    df = TradingJournal.load_journal()
    
    # Period Selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period = st.selectbox(
            "Analysis Period",
            ['all', 'month', 'week', 'day'],
            format_func=lambda x: {
                'all': 'All Time',
                'month': 'Last 30 Days',
                'week': 'Last 7 Days',
                'day': 'Today'
            }[x]
        )
    
    stats = TradingJournal.calculate_statistics(df, period)
    
    # Key Performance Metrics
    st.markdown("#### 📊 Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", stats['total_trades'])
    
    with col2:
        win_rate_color = (
            "#00ff88" if stats['win_rate'] >= 60
            else "#ffaa00" if stats['win_rate'] >= 50
            else "#ff6b6b"
        )
        st.markdown(f"""
        <div class="metric-card">
            <p style='color: #a0a0a0; font-size: 13px; margin: 0;'>WIN RATE</p>
            <h2 style='color: {win_rate_color}; font-size: 36px; margin: 8px 0;'>{stats['win_rate']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pnl_color = "#00ff88" if stats['total_pnl'] > 0 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-card">
            <p style='color: #a0a0a0; font-size: 13px; margin: 0;'>TOTAL P&L</p>
            <h2 style='color: {pnl_color}; font-size: 36px; margin: 8px 0;'>${stats['total_pnl']:,.0f}</h2>
            <p style='color: #a0a0a0; font-size: 11px;'>Avg: ${stats['avg_pnl']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    
    with col5:
        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    
    # Additional Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg Win", f"${stats['avg_win']:,.0f}")
    
    with col2:
        st.metric("Avg Loss", f"${stats['avg_loss']:,.0f}")
    
    with col3:
        st.metric("Largest Win", f"${stats['largest_win']:,.0f}")
    
    with col4:
        st.metric("Largest Loss", f"${stats['largest_loss']:,.0f}")
    
    with col5:
        st.metric("Expectancy", f"${stats['expectancy']:,.0f}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Drawdown", f"${stats['max_drawdown']:,.0f}")
    
    with col2:
        st.metric("Win Streak", stats['win_streak'])
    
    with col3:
        st.metric("Loss Streak", stats['loss_streak'])
    
    st.markdown("---")
    
    # Equity Curve
    if not df.empty:
        st.markdown("#### 📈 Equity Curve")
        
        df_sorted = df.sort_values('entry_date')
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_sorted['entry_date'],
            y=df_sorted['cumulative_pnl'],
            mode='lines',
            name='Equity',
            line=dict(color='#00d9ff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#666666",
            annotation_text="Break Even"
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0f0f0f',
            xaxis_title='Date',
            yaxis_title='Cumulative P&L ($)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Performance
        st.markdown("---")
        st.markdown("#### 📅 Monthly Performance")
        
        monthly_stats = TradingJournal.calculate_monthly_stats(df)
        
        if not monthly_stats.empty:
            fig = go.Figure()
            
            colors = ['#00ff88' if x > 0 else '#ff6b6b' for x in monthly_stats['total_pnl']]
            
            fig.add_trace(go.Bar(
                x=monthly_stats['month'],
                y=monthly_stats['total_pnl'],
                marker_color=colors,
                text=monthly_stats['total_pnl'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                hovertemplate='Month: %{x}<br>P&L: $%{y:,.0f}<br>Trades: %{customdata}<extra></extra>',
                customdata=monthly_stats['trades']
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0f0f0f',
                xaxis_title='Month',
                yaxis_title='P&L ($)',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly Stats Table
            st.dataframe(
                monthly_stats.style.format({
                    'total_pnl': '${:,.0f}',
                    'avg_pnl': '${:,.0f}'
                }),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Add New Trade
    st.markdown("### ➕ Add New Trade")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        entry_date = st.date_input("Entry Date", value=get_ist_time().date())
    
    with col2:
        exit_date = st.date_input("Exit Date", value=get_ist_time().date())
    
    with col3:
        strike = st.number_input("Strike", min_value=0.0, value=67000.0, step=100.0)
    
    with col4:
        pnl = st.number_input("P&L ($)", value=0.0, step=10.0)
    
    with col5:
        trade_type = st.selectbox(
            "Type",
            ["Short Straddle", "Short Strangle", "Iron Condor", "Other"]
        )
    
    notes = st.text_area("Notes (optional)")
    
    if st.button("Add Trade", type="primary", use_container_width=True):
        if strike and pnl != 0:
            trade_data = {
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'strike': float(strike),
                'pnl': float(pnl),
                'trade_type': trade_type,
                'notes': notes or ''
            }
            TradingJournal.add_trade(trade_data)
            st.success("✅ Trade added successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Please enter valid strike and P&L values")
    
    # Trade History
    if not df.empty:
        st.markdown("---")
        st.markdown("### 📋 Trade History")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            filter_type = st.multiselect(
                "Filter by Type",
                df['trade_type'].unique() if 'trade_type' in df.columns else []
            )
        
        with col2:
            filter_pnl = st.selectbox(
                "Filter by Result",
                ["All", "Wins Only", "Losses Only"]
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if filter_type:
            filtered_df = filtered_df[filtered_df['trade_type'].isin(filter_type)]
        
        if filter_pnl == "Wins Only":
            filtered_df = filtered_df[filtered_df['pnl'] > 0]
        elif filter_pnl == "Losses Only":
            filtered_df = filtered_df[filtered_df['pnl'] < 0]
        
        # Display
        st.dataframe(
            filtered_df.sort_values('entry_date', ascending=False).style.format({
                'strike': '${:,.0f}',
                'pnl': '${:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Export
        if st.button("📥 Export to CSV", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_journal_{get_ist_time().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #00d9ff; font-size: 48px; margin-bottom: 8px;'>
        🎯 Options Intelligence Hub
    </h1>
    <h4 style='text-align: center; color: #a0a0a0; margin-bottom: 4px;'>
        Decision-Support & Risk-Context Engine
    </h4>
    <p style='text-align: center; color: #666666; font-size: 14px;'>
        By Santanu Bez | v4.0 - Professional Edition
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════
    # SIDEBAR CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Debug Mode
        Config.DEBUG_MODE = st.checkbox(
            "🔍 Debug Mode",
            value=False,
            help="Show detailed API calls and system logs"
        )
        
        if Config.DEBUG_MODE:
            st.warning("⚠️ Debug mode enabled")
        
        st.markdown("---")
        
        # Asset Selection
        asset = st.selectbox(
            "Asset",
            options=['BTC', 'ETH', 'SOL'],
            format_func=lambda x: {
                'BTC': '₿ Bitcoin',
                'ETH': 'Ξ Ethereum',
                'SOL': '◎ Solana'
            }[x]
        )
        
        # Expiry Selection
        api = DeltaAPI()
        expiries = api.get_all_expiries(asset)
        
        if not expiries:
            st.error("❌ No expiries available")
            st.stop()
        
        expiry = st.selectbox(
            "Expiry",
            options=expiries,
            format_func=lambda x: f"📅 {x}"
        )
        
        st.markdown("---")
        
        # Auto Refresh Configuration
        st.markdown("### 🔄 Auto Refresh")
        
        enable_auto_refresh = st.checkbox(
            "Enable Auto Refresh",
            value=False,
            help="Automatically refresh data at specified intervals"
        )
        
        if enable_auto_refresh:
            refresh_interval_label = st.selectbox(
                "Refresh Interval",
                options=list(Config.AUTO_REFRESH_INTERVALS.keys()),
                index=2  # Default to 5 minutes
            )
            refresh_interval = Config.AUTO_REFRESH_INTERVALS[refresh_interval_label]
            
            st.info(f"⏱️ Auto-refreshing every {refresh_interval_label}")
        else:
            refresh_interval = None
        
        # Manual Refresh Button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Data Status
        st.markdown("### 📊 Data Status")
        
        # This will be populated after data fetch
        if 'market_data' in st.session_state and st.session_state.market_data:
            data_timestamp = st.session_state.market_data['timestamp']
            age_seconds, age_class = calculate_data_age(data_timestamp)
            
            if age_seconds < 60:
                age_display = f"{age_seconds}s ago"
            elif age_seconds < 3600:
                age_display = f"{age_seconds // 60}m ago"
            else:
                age_display = f"{age_seconds // 3600}h ago"
            
            st.markdown(f"""
            <div class="metric-card">
                <p style='color: #a0a0a0; font-size: 12px; margin: 0;'>LAST UPDATE</p>
                <p class='{age_class}' style='font-size: 16px; margin: 4px 0;'>{age_display}</p>
                <p style='color: #666666; font-size: 11px; margin: 0;'>{format_ist_time(data_timestamp, '%H:%M:%S IST')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # API Health
            api_health = "🟢 Healthy" if age_seconds < 120 else "🟡 Stale" if age_seconds < 600 else "🔴 Old"
            st.markdown(f"**API Status:** {api_health}")
        
        st.markdown("---")
        
        # Current Time
        st.markdown(f"""
        <div class="metric-card">
            <p style='color: #a0a0a0; font-size: 12px; margin: 0;'>CURRENT TIME</p>
            <p style='color: #00d9ff; font-size: 16px; margin: 4px 0;'>{format_ist_time(format_str='%H:%M:%S')}</p>
            <p style='color: #666666; font-size: 11px; margin: 0;'>IST (UTC+5:30)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Risk Warning
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ RISK WARNING</h4>
            <p style='font-size: 12px;'>
                Short straddles carry unlimited risk. This tool provides decision support only.
                Always use proper risk management.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Scoring Weights Info
        with st.expander("📊 Scoring Methodology"):
            st.markdown("""
            **Market Regime Score (0-100):**
            - IV Analysis: 45%
            - OI Analysis: 45%
            - PCR Analysis: 10%
            
            **Strike Score (0-100):**
            - Market Regime: 90%
            - Compression: 5%
            - Proximity: 5%
            
            **Percentile Analysis:**
            - 24-hour rolling window
            - <20th: Low (favorable)
            - >80th: Extreme (caution)
            """)
    
    # ═══════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ═══════════════════════════════════════════════════════════════
    
    # Check if we need to fetch new data
    fetch_new_data = False
    
    if 'market_data' not in st.session_state:
        fetch_new_data = True
    elif 'last_asset' not in st.session_state or st.session_state.last_asset != asset:
        fetch_new_data = True
    elif 'last_expiry' not in st.session_state or st.session_state.last_expiry != expiry:
        fetch_new_data = True
    elif enable_auto_refresh and 'last_refresh' in st.session_state:
        time_since_refresh = (get_ist_time() - st.session_state.last_refresh).total_seconds()
        if time_since_refresh >= refresh_interval:
            fetch_new_data = True
    
    # Fetch data if needed
    if fetch_new_data:
        data = fetch_market_data(asset, expiry)
        
        if not data:
            st.error("❌ Failed to load market data. Please check your connection and try again.")
            st.stop()
        
        st.session_state.market_data = data
        st.session_state.last_asset = asset
        st.session_state.last_expiry = expiry
        st.session_state.last_refresh = get_ist_time()
    else:
        data = st.session_state.market_data
    
    # Success message
    data_age_seconds, _ = calculate_data_age(data['timestamp'])
    st.success(f"✅ Data loaded | Last update: {data_age_seconds}s ago | {format_ist_time(data['timestamp'], '%H:%M:%S IST')}")
    
    # ═══════════════════════════════════════════════════════════════
    # TAB NAVIGATION
    # ═══════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Market Overview",
        "🎯 Market Regime",
        "📊 OI Distribution",
        "🌊 Volatility Analysis",
        "💡 Straddle Analyzer",
        "📓 Trading Journal"
    ])
    
    with tab1:
        render_market_overview(data)
    
    with tab2:
        render_market_regime(data)
    
    with tab3:
        render_oi_distribution(data)
    
    with tab4:
        render_volatility_analysis(data)
    
    with tab5:
        render_straddle_analyzer(data)
    
    with tab6:
        render_trading_journal()
    
    # ═══════════════════════════════════════════════════════════════
    # AUTO REFRESH IMPLEMENTATION
    # ═══════════════════════════════════════════════════════════════
    
    if enable_auto_refresh:
        # Use st.empty() to create a placeholder for countdown
        countdown_placeholder = st.empty()
        
        # Calculate time until next refresh
        if 'last_refresh' in st.session_state:
            time_since_refresh = (get_ist_time() - st.session_state.last_refresh).total_seconds()
            time_until_refresh = max(0, refresh_interval - time_since_refresh)
            
            countdown_placeholder.info(f"⏱️ Next refresh in {int(time_until_refresh)}s")
            
            # Trigger refresh when time is up
            if time_until_refresh == 0:
                time.sleep(1)
                st.rerun()
        
        # Use JavaScript to trigger refresh (alternative method)
        st.markdown(f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
