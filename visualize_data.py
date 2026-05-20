"""Interactive visualization for ALL regime labels and market data features."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import psycopg
from typing import Any
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_db_config() -> dict[str, Any]:
    """Get database configuration from environment variables."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'dbname': os.getenv('DB_NAME', 'narrux'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }


def fetch_regime_data(limit: int = 1000) -> pd.DataFrame:
    """Fetch ALL regime labels data from database.

    Args:
        limit: Number of rows to fetch (default: 1000)

    Returns:
        DataFrame with all regime data
    """
    print(f"Fetching ALL regime data (last {limit if limit else 'all'} bars)...")

    limit_clause = f"LIMIT {limit}" if limit else ""

    # Fetch ALL columns from the table
    query = f"""
        SELECT * FROM hmm_regime_labels
        ORDER BY time DESC
        {limit_clause}
    """

    try:
        conn = psycopg.connect(**get_db_config())
        df = pd.read_sql_query(query, conn)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df = df.sort_values('time').reset_index(drop=True)
        conn.close()
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def add_ohlcv_traces(fig, df: pd.DataFrame, timeframe: str, row: int, price_range: tuple = None, show_volume: bool = False):
    """Add candlestick trace to figure.

    Args:
        fig: Plotly figure
        df: DataFrame with data
        timeframe: Timeframe prefix (e.g., '15m', '1h')
        row: Row number for subplot
        price_range: Optional (min, max) price range for y-axis (None = auto from data)
        show_volume: If True, also add volume trace to same row with secondary y-axis
    """
    import plotly.graph_objects as go

    prefix = f'{timeframe}_'

    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df[f'{prefix}open'],
            high=df[f'{prefix}high'],
            low=df[f'{prefix}low'],
            close=df[f'{prefix}close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=row, col=1
    )

    # Volume in same subplot as secondary y-axis (only if requested)
    if show_volume:
        colors = pd.Series([df[f'{prefix}close'].iloc[i] >= df[f'{prefix}open'].iloc[i]
                           for i in range(len(df))])
        fig.add_trace(
            go.Bar(
                x=df['time'],
                y=df[f'{prefix}volume'],
                name='Volume',
                marker_color=['rgba(38, 166, 154, 0.5)' if c else 'rgba(239, 83, 80, 0.5)' for c in colors],
                yaxis='y2'
            ),
            row=row, col=1
        )

    # Set price range - auto-calculate from data if None
    if price_range is None:
        # Calculate min/max from OHLC data with padding
        price_cols = [f'{prefix}open', f'{prefix}high', f'{prefix}low', f'{prefix}close']
        all_prices = pd.concat([df[col] for col in price_cols if col in df.columns])
        min_price = all_prices.min()
        max_price = all_prices.max()
        # Add 2% padding for better visibility
        padding = (max_price - min_price) * 0.02
        price_range = (min_price - padding, max_price + padding)

    fig.update_yaxes(range=price_range, row=row, col=1)


def add_volume_trace(fig, df: pd.DataFrame, timeframe: str, row: int):
    """Add volume bar trace to figure.

    Args:
        fig: Plotly figure
        df: DataFrame with data
        timeframe: Timeframe prefix (e.g., '15m', '1h')
        row: Row number for subplot
    """
    import plotly.graph_objects as go

    prefix = f'{timeframe}_'

    # Color volume bars based on price direction
    colors = pd.Series([df[f'{prefix}close'].iloc[i] >= df[f'{prefix}open'].iloc[i]
                       for i in range(len(df))])

    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df[f'{prefix}volume'],
            name='Volume',
            marker_color=['rgba(38, 166, 154, 0.7)' if c else 'rgba(239, 83, 80, 0.7)' for c in colors]
        ),
        row=row, col=1
    )


def add_scatter_trace(fig, df: pd.DataFrame, col: str, row: int, color: str = '#2196f3', name: str = None):
    """Add a scatter trace for a single column.

    Args:
        fig: Plotly figure
        df: DataFrame with data
        col: Column name
        row: Row number for subplot
        color: Line color
        name: Trace name (defaults to column name)
    """
    import plotly.graph_objects as go

    if col not in df.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df[col],
            name=name or col,
            line=dict(color=color, width=1.5)
        ),
        row=row, col=1
    )


def add_spike_lines(fig: Any, n_rows: int):
    """Add vertical spike lines that follow mouse cursor across all subplots.

    Args:
        fig: Plotly figure
        n_rows: Number of rows in the subplot
    """
    for i in range(1, n_rows + 1):
        fig.update_xaxes(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikedash='solid',
            spikecolor='rgba(255, 255, 255, 0.5)',
            row=i, col=1
        )


def prepare_timeframe_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Prepare data for specific timeframe by removing duplicates.

    For 1h timeframe, keeps only one row per hour (1h_ columns are already hourly).
    For 15m timeframe, keeps all data.

    Args:
        df: DataFrame with data
        timeframe: '15m' or '1h'

    Returns:
        Filtered DataFrame
    """
    if timeframe == '1h':
        df_copy = df.copy()
        df_copy['hour_key'] = pd.to_datetime(df_copy['time']).dt.floor('h')
        # Drop duplicates keeping first row - 1h_ columns are already aggregated values
        df_unique = df_copy.drop_duplicates(subset='hour_key', keep='first')
        return df_unique.reset_index(drop=True)
    return df


def create_price_action_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create price action features chart.

    Shows: returns, candle_range, body_ratio, gap_up, gap_down, Volume, OHLCV (at bottom with rangeslider)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Prepare data - remove duplicate hourly candles for 1h
    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 7  # returns, candle_range, body_ratio, gap_up, gap_down, Volume, OHLCV
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28],
        subplot_titles=('Returns', 'Candle Range', 'Body Ratio', 'Gap Up', 'Gap Down', 'Volume', 'Price (OHLCV)')
    )

    # Add traces (Volume at row 6, OHLCV at row 7)
    add_scatter_trace(fig, df_plot, f'{prefix}returns', 1, '#26a69a')
    add_scatter_trace(fig, df_plot, f'{prefix}candle_range', 2, '#ff9800')
    add_scatter_trace(fig, df_plot, f'{prefix}body_ratio', 3, '#9c27b0')
    add_scatter_trace(fig, df_plot, f'{prefix}gap_up', 4, '#4caf50')
    add_scatter_trace(fig, df_plot, f'{prefix}gap_down', 5, '#f44336')
    add_volume_trace(fig, df_plot, timeframe, 6)
    add_ohlcv_traces(fig, df_plot, timeframe, 7, price_range=None, show_volume=False)

    # Add zero line for gap indicators
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=5, col=1)

    # Enable rangeslider on OHLCV chart (row 7 - at bottom)
    # Make chart wider for 1h timeframe to reduce space between candles
    figure_width = 1800 if timeframe == '1h' else 1400

    fig.update_layout(
        title=f'Price Action Features - {timeframe.upper()}',
        height=1100,
        width=figure_width,
        hovermode='x unified',
        template='plotly_dark',
        xaxis7_rangeslider_visible=True,  # Show rangeslider only on OHLCV row
        xaxis_rangeslider_visible=False    # Hide rangeslider on other x-axes
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_deltas_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create OHLCV deltas features chart.

    Shows: close_delta, high_delta, low_delta, volume_delta (and % versions), OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 9
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.094] * 8 + [0.25],
        subplot_titles=(
            'Close Delta', 'Close Delta %',
            'High Delta', 'High Delta %',
            'Low Delta', 'Low Delta %',
            'Volume Delta', 'Volume Delta %',
            'Price & Volume'
        )
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 9, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}close_delta', 1, '#26a69a')
    add_scatter_trace(fig, df_plot, f'{prefix}close_delta_pct', 2, '#4caf50')
    add_scatter_trace(fig, df_plot, f'{prefix}high_delta', 3, '#ff9800')
    add_scatter_trace(fig, df_plot, f'{prefix}high_delta_pct', 4, '#ff5722')
    add_scatter_trace(fig, df_plot, f'{prefix}low_delta', 5, '#9c27b0')
    add_scatter_trace(fig, df_plot, f'{prefix}low_delta_pct', 6, '#673ab7')
    add_scatter_trace(fig, df_plot, f'{prefix}volume_delta', 7, '#2196f3')
    add_scatter_trace(fig, df_plot, f'{prefix}volume_delta_pct', 8, '#03a9f4')

    fig.update_layout(
        title=f'OHLCV Deltas Features - {timeframe.upper()}',
        height=1200,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_volatility_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create volatility features chart.

    Shows: realized_vol (5,10,20), vol_ratio, spread_vol, price_velocity, downward_pressure, OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 9
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.094] * 8 + [0.25],
        subplot_titles=(
            'Realized Vol 5', 'Realized Vol 10', 'Realized Vol 20',
            'Vol Ratio 5/20', 'Spread Vol',
            'Price Velocity', 'Downward Pressure',
            'Price & Volume'
        )
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 9, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}realized_vol_combined_5', 1, '#f44336')
    add_scatter_trace(fig, df_plot, f'{prefix}realized_vol_combined_10', 2, '#ff9800')
    add_scatter_trace(fig, df_plot, f'{prefix}realized_vol_combined_20', 3, '#ffeb3b')
    add_scatter_trace(fig, df_plot, f'{prefix}vol_ratio_combined_5_20', 4, '#9c27b0')
    add_scatter_trace(fig, df_plot, f'{prefix}spread_vol', 5, '#3f51b5')
    add_scatter_trace(fig, df_plot, f'{prefix}price_velocity', 6, '#00bcd4')
    add_scatter_trace(fig, df_plot, f'{prefix}downward_pressure', 7, '#e91e63')

    fig.update_layout(
        title=f'Volatility Features - {timeframe.upper()}',
        height=1200,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_liquidity_features_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create liquidity features chart.

    Shows: log_depth, depth_relative, spread_relative, imbalance_change, OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 6
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.14] * 5 + [0.3],
        subplot_titles=('Log Depth', 'Depth Relative', 'Spread Relative', 'Imbalance Change', 'Price & Volume')
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 6, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}log_depth', 1, '#2196f3')
    add_scatter_trace(fig, df_plot, f'{prefix}depth_relative', 2, '#00bcd4')
    add_scatter_trace(fig, df_plot, f'{prefix}spread_relative', 3, '#009688')
    add_scatter_trace(fig, df_plot, f'{prefix}imbalance_change', 4, '#4caf50')

    fig.update_layout(
        title=f'Liquidity Features - {timeframe.upper()}',
        height=1000,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_advanced_ohlcv_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create advanced OHLCV indicators chart.

    Shows: atr_pct, rsi, bb_width_pct, volume_ratio, ema_distance_pct, momentum_accel, OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 8
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.11] * 7 + [0.22],
        subplot_titles=(
            'ATR %', 'RSI', 'BB Width %',
            'Volume Ratio', 'EMA Distance %', 'Momentum Acceleration',
            'Price & Volume'
        )
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 8, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}atr_pct', 1, '#ff9800')
    add_scatter_trace(fig, df_plot, f'{prefix}rsi', 2, '#9c27b0')

    # RSI zones
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(244, 67, 54, 0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(76, 175, 80, 0.5)", row=2, col=1)

    add_scatter_trace(fig, df_plot, f'{prefix}bb_width_pct', 3, '#f44336')
    add_scatter_trace(fig, df_plot, f'{prefix}volume_ratio', 4, '#2196f3')
    add_scatter_trace(fig, df_plot, f'{prefix}ema_distance_pct', 5, '#00bcd4')
    add_scatter_trace(fig, df_plot, f'{prefix}momentum_accel', 6, '#009688')

    fig.update_layout(
        title=f'Advanced OHLCV Indicators - {timeframe.upper()}',
        height=1100,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_orderbook_features_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create orderbook features chart with raw orderbook data.

    Shows: best_bid/ask, mid_price, spread_bps, total_depth, depth_imbalance,
           liquidity_density, order_imbalance_5/10, depth_skew, bid_ask_slope, OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 12
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
        row_heights=[0.067] * 11 + [0.2],
        subplot_titles=(
            'Best Bid', 'Best Ask', 'Mid Price',
            'Spread BPS', 'Total Depth 10', 'Depth Imbalance', 'Liquidity Density',
            'Order Imbalance 5', 'Order Imbalance 10', 'Depth Skew', 'Bid Ask Slope',
            'Price & Volume'
        )
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 12, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}best_bid', 1, '#4caf50')
    add_scatter_trace(fig, df_plot, f'{prefix}best_ask', 2, '#f44336')
    add_scatter_trace(fig, df_plot, f'{prefix}mid_price', 3, '#ffeb3b')
    add_scatter_trace(fig, df_plot, f'{prefix}spread_bps', 4, '#ff9800')
    add_scatter_trace(fig, df_plot, f'{prefix}total_depth_10', 5, '#2196f3')
    add_scatter_trace(fig, df_plot, f'{prefix}depth_imbalance', 6, '#00bcd4')
    add_scatter_trace(fig, df_plot, f'{prefix}liquidity_density', 7, '#009688')
    add_scatter_trace(fig, df_plot, f'{prefix}order_imbalance_5', 8, '#9c27b0')
    add_scatter_trace(fig, df_plot, f'{prefix}order_imbalance_10', 9, '#673ab7')
    add_scatter_trace(fig, df_plot, f'{prefix}depth_skew', 10, '#3f51b5')
    add_scatter_trace(fig, df_plot, f'{prefix}bid_ask_slope', 11, '#e91e63')

    # Zero lines for imbalance metrics
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=8, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=9, col=1)

    fig.update_layout(
        title=f'Orderbook Features (Raw + Derived) - {timeframe.upper()}',
        height=1400,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_funding_oi_features_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create funding rate and OI features chart.

    Shows: fr_value, funding_rate, funding_direction, oi_value,
           oi_change_pct_1h, oi_change_pct_24h, oi_price_divergence, OHLCV
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = prepare_timeframe_data(df, timeframe)
    prefix = f'{timeframe}_'

    n_rows = 9
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.094] * 8 + [0.25],
        subplot_titles=(
            'FR Value', 'Funding Rate', 'Funding Direction',
            'OI Value', 'OI Change % 1h', 'OI Change % 24h', 'OI Price Divergence',
            'Price & Volume'
        )
    )

    add_ohlcv_traces(fig, df_plot, timeframe, 9, price_range=None)
    add_scatter_trace(fig, df_plot, f'{prefix}fr_value', 1, '#ffeb3b')
    add_scatter_trace(fig, df_plot, f'{prefix}funding_rate', 2, '#4caf50')
    add_scatter_trace(fig, df_plot, f'{prefix}funding_direction', 3, '#f44336')
    add_scatter_trace(fig, df_plot, f'{prefix}oi_value', 4, '#2196f3')
    add_scatter_trace(fig, df_plot, f'{prefix}oi_change_pct_1h', 5, '#00bcd4')
    add_scatter_trace(fig, df_plot, f'{prefix}oi_change_pct_24h', 6, '#009688')
    add_scatter_trace(fig, df_plot, f'{prefix}oi_price_divergence', 7, '#9c27b0')

    # Zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=5, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=7, col=1)

    fig.update_layout(
        title=f'Funding Rate & Open Interest Features - {timeframe.upper()}',
        height=1200,
        hovermode='x unified',
        template='plotly_dark'
    )

    add_spike_lines(fig, n_rows)

    return fig


def create_regime_output_chart(df: pd.DataFrame, timeframe: str = '15m') -> Any:
    """Create HMM regime output chart with regime overlay on price.

    Shows: OHLCV with regime background colors, regime name, regime_confidence, confidence_liq
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Prepare data - remove duplicate hourly candles for 1h OHLCV
    df_ohlcv = prepare_timeframe_data(df, timeframe)
    # Use full df for regime overlay to show all regime data
    prefix = f'{timeframe}_'

    # Regime definitions with colors and y-positions for display
    regime_info = {
        'LIQ_CASCADE': {'color': 'rgba(244, 67, 54, 0.3)', 'border': 'rgba(244, 67, 54, 0.8)', 'y': 0, 'label': 'LIQ_CASCADE'},
        'VOL_EXPAND': {'color': 'rgba(255, 152, 0, 0.3)', 'border': 'rgba(255, 152, 0, 0.8)', 'y': 1, 'label': 'VOL_EXPAND'},
        'TREND_BEAR': {'color': 'rgba(139, 69, 19, 0.3)', 'border': 'rgba(139, 69, 19, 0.8)', 'y': 2, 'label': 'TREND_BEAR'},
        'RANGE': {'color': 'rgba(128, 128, 128, 0.3)', 'border': 'rgba(128, 128, 128, 0.8)', 'y': 3, 'label': 'RANGE'},
        'TRANSITION': {'color': 'rgba(156, 39, 176, 0.3)', 'border': 'rgba(156, 39, 176, 0.8)', 'y': 4, 'label': 'TRANSITION'},
        'TREND_BULL': {'color': 'rgba(76, 175, 80, 0.3)', 'border': 'rgba(76, 175, 80, 0.8)', 'y': 5, 'label': 'TREND_BULL'},
        'VOL_COMPRESS': {'color': 'rgba(33, 150, 243, 0.3)', 'border': 'rgba(33, 150, 243, 0.8)', 'y': 6, 'label': 'VOL_COMPRESS'},
    }

    n_rows = 5  # Regime, Regime Confidence, LIQ Confidence, Volume, OHLCV
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.18, 0.18, 0.18, 0.15, 0.31],
        subplot_titles=(
            'Regime', 'Regime Confidence', 'LIQ Cascade Confidence', 'Volume', 'Price (OHLCV with Regime Overlay)'
        )
    )

    # Add OHLCV at row 5 (bottom), with regime background
    add_ohlcv_traces(fig, df_ohlcv, timeframe, 5, price_range=None, show_volume=False)

    # Add regime background colors on price chart (using full df)
    if f'{prefix}regime' in df.columns:
        # Filter out NaN regimes and create continuous blocks
        df_valid = df[df[f'{prefix}regime'].notna()].copy()

        if len(df_valid) > 0:
            current_regime = None
            start_time = None

            for i in range(len(df_valid)):
                row = df_valid.iloc[i]
                regime = row[f'{prefix}regime']
                time = row['time']

                if pd.isna(regime):
                    continue

                if current_regime is None:
                    # First valid regime
                    current_regime = regime
                    start_time = time
                elif regime != current_regime:
                    # Regime changed - add rectangle for previous period
                    end_time = df_valid.iloc[i-1]['time']

                    # Extend end_time slightly to avoid gaps
                    if i < len(df_valid):
                        next_time = df_valid.iloc[i]['time']
                        end_time = end_time + (next_time - end_time) / 2
                    else:
                        # Last period - extend to end of data
                        end_time = df['time'].iloc[-1]

                    fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor=regime_info.get(current_regime, {}).get('color', 'rgba(128, 128, 128, 0.1)'),
                        layer="below", line_width=0, row=5, col=1  # Row 5 = OHLCV
                    )
                    current_regime = regime
                    start_time = time

            # Add final regime period
            if current_regime is not None:
                fig.add_vrect(
                    x0=start_time,
                    x1=df_ohlcv['time'].iloc[-1],
                    fillcolor=regime_info.get(current_regime, {}).get('color', 'rgba(128, 128, 128, 0.1)'),
                    layer="below", line_width=0, row=5, col=1  # Row 5 = OHLCV
                )

        # Add regime name as step plot on row 2 (use deduplicated data for cleaner display)
        df_regime = df_ohlcv.copy()
        df_regime['regime_y'] = df_regime[f'{prefix}regime'].map(lambda x: regime_info.get(x, {}).get('y', 3))

        # Create step plot for regime names
        fig.add_trace(
            go.Scatter(
                x=df_ohlcv['time'],
                y=df_regime['regime_y'],
                mode='lines',
                name='Regime',
                line=dict(shape='hv', color='white', width=2),
                hovertext=df_ohlcv[f'{prefix}regime'],
                hovertemplate='%{hovertext}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add regime labels on y-axis
        fig.update_yaxes(
            ticktext=list(regime_info.keys()),
            tickvals=[r['y'] for r in regime_info.values()],
            row=2, col=1
        )

    # Add traces for metrics (use deduplicated data)
    add_scatter_trace(fig, df_ohlcv, f'{prefix}regime_confidence', 3, '#9c27b0')
    add_scatter_trace(fig, df_ohlcv, f'{prefix}confidence_liq', 4, '#f44336')

    # Add volume trace
    add_volume_trace(fig, df_ohlcv, timeframe, 4)  # Volume at row 4

    # Add legend with regime colors as annotation
    legend_text = "<br>".join([f"<span style='color:{info['border'].replace('0.8', '1')}'>■</span> {name}"
                               for name, info in regime_info.items()])

    fig.add_annotation(
        text=legend_text,
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(30, 30, 30, 0.8)",
        bordercolor="gray",
        borderwidth=1
    )

    fig.update_layout(
        title=f'HMM Regime Output - {timeframe.upper()}',
        height=1000,
        hovermode='x unified',
        template='plotly_dark',
        margin=dict(r=150),  # Make room for legend
        xaxis5_rangeslider_visible=True,  # Show rangeslider only on OHLCV row (row 5)
        xaxis_rangeslider_visible=False    # Hide rangeslider on other x-axes
    )

    add_spike_lines(fig, n_rows)

    return fig


def visualize(df: pd.DataFrame):
    """Create and display ALL visualization windows.

    Args:
        df: DataFrame with ALL regime data
    """
    if df.empty:
        print("No data to visualize")
        return []

    print("\nCreating comprehensive visualizations for ALL features...")

    figures = []
    tf_list = ['15m', '1h']

    # For each timeframe, create all feature category charts
    for tf in tf_list:
        print(f"\n  Creating {tf.upper()} charts...")

        # 1. Price Action Features
        print(f"    - Price Action Features...")
        figures.append((f'{tf.upper()} - Price Action', create_price_action_chart(df, tf)))

        # 2. OHLCV Deltas
        print(f"    - OHLCV Deltas...")
        figures.append((f'{tf.upper()} - OHLCV Deltas', create_deltas_chart(df, tf)))

        # 3. Volatility Features
        print(f"    - Volatility Features...")
        figures.append((f'{tf.upper()} - Volatility', create_volatility_chart(df, tf)))

        # 4. Liquidity Features
        print(f"    - Liquidity Features...")
        figures.append((f'{tf.upper()} - Liquidity', create_liquidity_features_chart(df, tf)))

        # 5. Advanced OHLCV Indicators
        print(f"    - Advanced OHLCV Indicators...")
        figures.append((f'{tf.upper()} - Advanced Indicators', create_advanced_ohlcv_chart(df, tf)))

        # 6. Orderbook Features
        print(f"    - Orderbook Features...")
        figures.append((f'{tf.upper()} - Orderbook', create_orderbook_features_chart(df, tf)))

        # 7. Funding & OI Features
        print(f"    - Funding & OI Features...")
        figures.append((f'{tf.upper()} - Funding & OI', create_funding_oi_features_chart(df, tf)))

        # 8. HMM Regime Output
        print(f"    - HMM Regime Output...")
        figures.append((f'{tf.upper()} - Regime Output', create_regime_output_chart(df, tf)))

    # Display all figures
    print(f"\nOpening {len(figures)} interactive chart windows...")
    print("  - Each window shows OHLCV at top for context")
    print("  - Use mouse to zoom (click and drag)")
    print("  - Use pan tool to move around")
    print("  - Double-click to reset zoom")
    print("  - Close windows to exit\n")

    for title, fig in figures:
        fig.show()

    return figures


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize ALL regime labels and features')
    parser.add_argument('--limit', type=int, default=1000, help='Number of bars to fetch (default: 1000)')
    args = parser.parse_args()

    # Fetch ALL data from hmm_regime_labels table
    df = fetch_regime_data(limit=args.limit)

    if not df.empty:
        # Visualize ALL features
        visualize(df)
    else:
        print("No data found in hmm_regime_labels table")


if __name__ == '__main__':
    main()
