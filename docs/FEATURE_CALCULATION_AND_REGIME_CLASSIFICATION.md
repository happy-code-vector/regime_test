# Feature Calculation and Regime Classification

**NARRUX Regime Classification System**  
Version: 1.0  
Last Updated: 2026-06-02

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Feature Calculation](#feature-calculation)
4. [Regime Classification Logic](#regime-classification-logic)
5. [Timeframe-Specific Behavior](#timeframe-specific-behavior)
6. [Sample Output](#sample-output)
7. [References](#references)

---

## System Overview

The NARRUX regime classification system processes OHLCV (Open, High, Low, Close, Volume) data to:

1. **Calculate technical indicators** across multiple timeframe anchors (5m, 15m, 30m, 1h)
2. **Convert raw values to percentile ranks** then to z-scores
3. **Classify market regimes** using decision tables
4. **Generate feature vectors** for LightGBM training

### Key Design Principles

- **Base timeframe**: 1-minute candles (all calculations start here)
- **Multi-timeframe anchors**: 5m, 15m, 30m, 1h lookbacks
- **Z-score normalization**: Percentile-based, clamped to ±3.3
- **Composite scoring**: Weighted combination of anchor z-scores
- **Rule-based classification**: Decision tables with metric thresholds

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1m OHLCV Candles                                                            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │           OhlcvTrendFeatureEngine                            │           │
│  │  - EMA slope % calculation                                  │           │
│  │  - ADX (Average Directional Index) calculation              │           │
│  │  - Percentile z-score conversion                            │           │
│  │  - Composite score aggregation                              │           │
│  └──────────────────────────────────────────────────────────────┘           │
│       │                                                                      │
│       ├─────────────────────────────────────────────────────────┐           │
│       │                                                         │           │
│       ▼                                                         ▼           │
│  ┌─────────────────┐                                   ┌──────────────────┐      │
│  │ RegimeMetrics   │                                   │ Microstructure   │      │
│  │ Computer        │                                   │ Engine          │      │
│  │                 │                                   │ (optional)      │      │
│  │ - Realized vol  │                                   │                  │      │
│  │ - ATR           │                                   │ - Orderbook     │      │
│  │ - Hurst         │                                   │ - Funding       │      │
│  │ - Autocorr      │                                   │ - Liquidations  │      │
│  │ - VWAP dist     │                                   └──────────────────┘      │
│  └─────────────────┘                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │              RegimeFeatureSnapshot                             │           │
│  │  - Consolidates all features                                 │           │
│  │  - Computes composite scores                                 │           │
│  └──────────────────────────────────────────────────────────────┘           │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │           build_regime_lgbm_row()                            │           │
│  │  - Assembles numeric feature vector                          │           │
│  │  - Classifies regimes (trend/vol/liq)                         │           │
│  │  - Encodes categorical IDs                                   │           │
│  └──────────────────────────────────────────────────────────────┘           │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                RegimeLgbmRow                                 │           │
│  │  - 28 numeric features                                      │           │
│  │  - 3 categorical regime labels                               │           │
│  │  - 1 target regime label                                    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Calculation

### 1. EMA (Exponential Moving Average) Slope

**Purpose**: Measures trend direction and strength

**Formula**:
```
EMA(period) at t = α × price_t + (1 - α) × EMA_t-1
where α = 2 / (period + 1)

EMA_Slope_% at t = (EMA_t - EMA_t-1) / |EMA_t-1| × 100
```

**Period per anchor**:
- 5m anchor: period = 5 bars
- 15m anchor: period = 15 bars
- 30m anchor: period = 30 bars
- 1h anchor: period = 60 bars

**Z-score conversion**:
```
slope_percentile = percentile_rank(slope_history, current_slope)
slope_z = percentile_to_zscore(slope_percentile)
```

### 2. ADX (Average Directional Index)

**Purpose**: Measures trend strength (trend vs range)

**Formula**:
```
True Range (TR) = max(H-L, |H-C_prev|, |L-C_prev|)

Plus DM = H_t - H_t-1  (if > 0 and > Down DM)
Minus DM = L_t-1 - L_t  (if > 0 and > Plus DM)

ATR = Wilder_Smooth(TR, period)
Plus DI = 100 × Wilder_Smooth(Plus DM, period) / ATR
Minus DI = 100 × Wilder_Smooth(Minus DM, period) / ATR

DX = 100 × |Plus DI - Minus DI| / (Plus DI + Minus DI)
ADX = Wilder_Smooth(DX, period)
```

**Z-score conversion**:
```
adx_percentile = percentile_rank(adx_history, current_adx)
adx_z = percentile_to_zscore(adx_percentile)
```

### 3. Realized Volatility

**Purpose**: Measures price movement magnitude

**Formula**:
```
log_return_t = ln(close_t / close_t-1)

Realized_Vol(window) = std(log_returns) × √(525,600)
where √(525,600) = √(60 × 24 × 365) annualizes to yearly basis
```

**Z-score conversion**:
```
realized_vol_z = percentile_to_zscore(percentile_rank(realized_vol_history, current_realized_vol))
```

### 4. ATR (Average True Range)

**Purpose**: Measures volatility considering high/low gaps

**Formula**:
```
TR_t = max(H_t - L_t, |H_t - C_t-1|, |L_t - C_t-1|)

ATR = Wilder_Smooth(TR, period)
where Wilder_Smooth(x, p) = (x_t + (p-1) × ATR_t-1) / p
```

### 5. Parkinson Volatility

**Purpose**: Alternative volatility measure using high-low range

**Formula**:
```
Parkinson_Vol = √(mean(ln(H/L)²) / (4 × ln(2))) × √(525,600)
```

### 6. Vol of Vol (Volatility of Volatility)

**Purpose**: Measures stability of volatility itself

**Formula**:
```
Vol_of_Vol = std(realized_vol_subwindows) over window
where subwindows are 5-bar realized vol segments
```

### 7. Hurst Exponent

**Purpose**: Distinguishes trending vs mean-reverting behavior

**Formula**:
```
H = 0.5 × ln(var_k / (k × var_1)) / ln(k)
where:
- var_1 = variance of log returns
- var_k = variance of k-bar aggregated returns
- k = max(2, n_bars / 10)

Interpretation:
- H > 0.5: Trending (persistent)
- H = 0.5: Random walk
- H < 0.5: Mean-reverting
```

### 8. Lag-1 Autocorrelation

**Purpose**: Measures persistence of returns

**Formula**:
```
Autocorr = corr(return_t, return_t-1)
```

### 9. VWAP Distance

**Purpose**: Measures if price is stretched from volume-weighted average

**Formula**:
```
VWAP = Σ(price × volume) / Σ(volume)

VWAP_Dist_% = (close - VWAP) / VWAP × 100
```

### 10. Value Area Distance

**Purpose**: Measures distance from volume-weighted median (Value Area)

**Formula**:
```
Value_Area = price at 70% cumulative volume (split 35%/35%)

Value_Area_Dist_% = (close - Value_Area) / Value_Area × 100
```

### 11. Support/Resistance Distance (Normalized by ATR)

**Purpose**: Measures proximity to recent swing high/low

**Formula**:
```
Swing_High = max(highs over window)
Swing_Low = min(lows over window)

Dist_High_ATR = (close - Swing_High) / ATR
Dist_Low_ATR = (close - Swing_Low) / ATR

level_dist_z = z-score of |Dist| from history
```

### 12. Delta Features (Change in Z-scores)

**Purpose**: Captures acceleration/deceleration of trends

**Formula**:
```
delta_slope_z_k = slope_z_current - slope_z_(k bars ago)
delta_adx_z_k = adx_z_current - adx_z_(k bars ago)

where k ∈ {1, 3}
```

### 13. Cross-Timeframe Agreement

**Purpose**: Detects confluence across timeframes

**Formula**:
```
cross_tf_agreement(z_5m, z_15m, threshold=0.25):
    if z_5m > 0.25 AND z_15m > 0.25: return +1.0 (agree up)
    if z_5m < -0.25 AND z_15m < -0.25: return -1.0 (agree down)
    else: return 0.0 (mixed or flat)
```

### 14. Composite Scores

**Purpose**: Combines multiple anchors into single score

**Formula**:
```
direction_score = Σ(weight_anchor × slope_z_anchor)
quality_score = Σ(weight_anchor × adx_z_anchor)

where weights depend on target anchor:

Target Anchor  │  5m  │ 15m │ 30m │  1h
───────────────┼──────┼─────┼─────┼─────
5m weights     │ 50%  │ 25% │ 15% │ 10%
15m weights    │ 25%  │ 50% │ 20% │  5%
30m weights    │ 15%  │ 25% │ 50% │ 10%
1h weights     │  -   │  -  │ 15% │ 50%
               │      │     │     │(2h,4h used)
```

---

## Regime Classification Logic

### Trend Regime Classification

**Input**: direction_score, quality_score (composite z-scores)

**Classification Flow**:
```
1. Check if z-scores are finite → "Warmup" if not
2. Apply decision table rules (highest priority first):
   - STRONG_UP: direction ≥ 2.0 AND quality ≥ 1.0
   - STRONG_DOWN: direction ≤ -2.0 AND quality ≥ 1.0
   - TREND_UP: direction ≥ 1.0 AND quality ≥ 0.0
   - TREND_DOWN: direction ≤ -1.0 AND quality ≥ 0.0
   - WEAK_UP: direction ≥ 0.5 OR (direction ≥ 0.25 AND quality ≥ 0.5)
   - WEAK_DOWN: direction ≤ -0.5 OR (direction ≤ -0.25 AND quality ≥ 0.5)
   - RANGE: |direction| < 0.25 (default fallback)
3. Return first matching rule
```

**Trend Regime Labels** (5-regime system):
- `WEAK_UP` (ID 0): Mild uptrend
- `WEAK_DOWN` (ID 1): Mild downtrend
- `RANGE` (ID 2): No directional bias
- `TREND_UP` (ID 3): Established uptrend
- `TREND_DOWN` (ID 4): Established downtrend
- `Warmup` (ID 5): Insufficient data

### Volatility Regime Classification

**Input**: realized_vol_z, parkinson_vol_z, atr_z, vol_of_vol_z

**Classification Flow**:
```
vol_score = 0.35×realized_vol_z + 0.25×parkinson_vol_z + 0.25×atr_z + 0.15×vol_of_vol_z

if vol_score ≥ 2.0 OR (vol_of_vol_z ≥ 1.5 AND vol_score ≥ 0.5):
    return "VOL_SPIKE"
elif vol_score ≥ 0.75:
    return "VOL_ELEVATED"
elif vol_score < -0.5:
    return "VOL_LOW"
else:
    return "VOL_NORMAL"
```

**Volatility Regime Labels**:
- `VOL_LOW` (ID 0): Below-normal volatility
- `VOL_NORMAL` (ID 1): Normal volatility
- `VOL_ELEVATED` (ID 2): Above-normal volatility
- `VOL_SPIKE` (ID 3): Extreme volatility event
- `Warmup` (ID 4): Insufficient data

### Liquidity Regime Classification

**Input**: orderbook depth, spread widening, executed liquidity z-scores

**Classification Flow**:
```
liq_stress = weighted combination of:
    - executed_liquidity_z (35%)
    - depth_collapse_z (30%)
    - spread_widening_z (25%)
    - order_imbalance (10%)

if spread_widening_z ≥ threshold OR depth_collapse_z ≥ threshold:
    return "LIQ_STRESSED"
elif liq_stress ≥ 1.5:
    return "LIQ_THIN"
elif liq_stress ≥ 0.5:
    return "LIQ_NORMAL"
elif executed_z ≤ -1.0:
    return "LIQ_DEEP"
else:
    return "LIQ_NORMAL"
```

**Liquidity Regime Labels**:
- `LIQ_DEEP` (ID 0): High liquidity, low cost
- `LIQ_NORMAL` (ID 1): Normal liquidity conditions
- `LIQ_THIN` (ID 2): Reduced liquidity
- `LIQ_STRESSED` (ID 3): Liquidity stress/crisis
- `Warmup` (ID 4): Insufficient data

---

## Timeframe-Specific Behavior

### Timeframe Buckets

The system processes 1m candles but emits samples at specific bucket closes:

| Timeframe | Bucket Close Condition | Examples (UTC) |
|-----------|------------------------|----------------|
| 5m | minute % 5 == 4 | :04, :09, :14, ... |
| 15m | minute % 15 == 14 | :14, :29, :44, :59 |
| 30m | minute % 30 == 29 | :29, :59 |
| 1h | minute == 59 | :59 |

### Anchor Window Sizes

| Anchor | Window Size (bars) | Window Size (seconds) |
|--------|-------------------|----------------------|
| 5m | 5 | 300 |
| 15m | 15 | 900 |
| 30m | 30 | 1,800 |
| 1h | 60 | 3,600 |

### Z-Score Lookback Window

- **Default**: 240 bars (4 hours of 1m data)
- **History type**: Rolling FIFO queue
- **Percentile calculation**: Against historical values in window

### Feature Availability by Anchor

| Feature | 5m | 15m | 30m | 1h |
|---------|----|-----|-----|-----|
| slope_z | ✓ | ✓ | ✓ | ✓ |
| adx_z | ✓ | ✓ | ✓ | ✓ |
| realized_vol_z | ✓ | ✓ | ✗ | ✗ |
| atr_z | ✓ | ✓ | ✗ | ✗ |
| hurst_z | ✗ | ✓ | ✗ | ✗ |
| autocorr_z | ✗ | ✓ | ✗ | ✗ |
| vwap_dist_z | ✗ | ✓ | ✗ | ✗ |
| value_area_dist_z | ✗ | ✓ | ✗ | ✗ |
| level_dist_z | ✗ | ✓ | ✗ | ✗ |

---

## Sample Output

### Feature Vector Structure (28 numeric features)

```
1. bar_position                : 0-14 (minute within 15m bucket)
2. slope_z_5m                  : ~ -3.3 to +3.3
3. slope_z_15m                 : ~ -3.3 to +3.3
4. slope_z_30m                 : ~ -3.3 to +3.3
5. slope_z_1h                   : ~ -3.3 to +3.3
6. adx_z_5m                    : ~ -3.3 to +3.3
7. adx_z_15m                   : ~ -3.3 to +3.3
8. adx_z_30m                   : ~ -3.3 to +3.3
9. adx_z_1h                    : ~ -3.3 to +3.3
10. direction_score            : ~ -2.5 to +2.5
11. quality_score              : ~ -2.0 to +2.0
12. delta_slope_z_1            : ~ -5.0 to +5.0
13. delta_slope_z_3            : ~ -5.0 to +5.0
14. delta_adx_z_1              : ~ -5.0 to +5.0
15. delta_adx_z_3              : ~ -5.0 to +5.0
16. slope_cross_tf             : -1.0, 0.0, or +1.0
17. adx_cross_tf               : -1.0, 0.0, or +1.0
18. slope_cross_tf_15m_30m     : -1.0, 0.0, or +1.0
19. adx_cross_tf_15m_30m       : -1.0, 0.0, or +1.0
20. realized_vol_z_5m          : ~ -3.3 to +3.3
21. realized_vol_z_15m         : ~ -3.3 to +3.3
22. atr_z_5m                   : ~ -3.3 to +3.3
23. atr_z_15m                  : ~ -3.3 to +3.3
24. hurst_z                    : ~ -3.3 to +3.3
25. autocorr_z                 : ~ -3.3 to +3.3
26. vwap_dist_z                : ~ -3.3 to +3.3
27. value_area_dist_z          : ~ -3.3 to +3.3
28. level_dist_z               : ~ -3.3 to +3.3

Categorical features:
29. trend_regime               : WEAK_UP, WEAK_DOWN, RANGE, TREND_UP, TREND_DOWN, Warmup
30. trend_regime_id            : 0, 1, 2, 3, 4, 5
31. vol_regime                 : VOL_LOW, VOL_NORMAL, VOL_ELEVATED, VOL_SPIKE, Warmup
32. vol_regime_id              : 0, 1, 2, 3, 4
33. liq_regime                 : LIQ_DEEP, LIQ_NORMAL, LIQ_THIN, LIQ_STRESSED, Warmup
34. liq_regime_id              : 0, 1, 2, 3, 4
35. target_regime              : Same as trend_regime (next candle's regime)
36. target_regime_id           : Same as trend_regime_id (next candle's regime)
```

### Sample CSV Output (5m timeframe)

```csv
bucket_ts,symbol,timeframe,bar_position,slope_z_5m,slope_z_15m,slope_z_30m,slope_z_1h,adx_z_5m,adx_z_15m,adx_z_30m,adx_z_1h,direction_score,quality_score,delta_slope_z_1,delta_slope_z_3,delta_adx_z_1,delta_adx_z_3,slope_cross_tf,adx_cross_tf,slope_cross_tf_15m_30m,adx_cross_tf_15m_30m,realized_vol_z_5m,realized_vol_z_15m,atr_z_5m,atr_z_15m,hurst_z,autocorr_z,vwap_dist_z,value_area_dist_z,level_dist_z,trend_regime,trend_regime_id,vol_regime,vol_regime_id,liq_regime,liq_regime_id,target_regime,target_regime_id
2024-05-23T00:04:00+00:00,BTCUSDT,5m,4,-0.823,-0.512,-0.345,-0.234,0.123,0.234,0.345,0.456,-0.512,0.234,-0.123,-0.234,0.012,0.023,-1.0,0.0,0.0,0.0,-1.234,-0.876,-0.654,-0.432,-0.123,0.234,-0.345,0.456,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:09:00+00:00,BTCUSDT,5m,4,-0.765,-0.498,-0.334,-0.223,0.134,0.245,0.356,0.467,-0.498,0.245,-0.134,-0.223,0.023,0.034,-1.0,0.0,0.0,0.0,-1.189,-0.834,-0.612,-0.391,-0.112,0.245,-0.334,0.467,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:14:00+00:00,BTCUSDT,5m,4,-0.712,-0.487,-0.323,-0.212,0.145,0.256,0.367,0.478,-0.487,0.256,-0.145,-0.234,0.034,0.045,-1.0,0.0,0.0,0.0,-1.145,-0.792,-0.571,-0.351,-0.101,0.256,-0.323,0.478,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:19:00+00:00,BTCUSDT,5m,4,-0.667,-0.476,-0.312,-0.201,0.156,0.267,0.378,0.489,-0.476,0.267,-0.156,-0.245,0.045,0.056,-1.0,0.0,0.0,0.0,-1.101,-0.751,-0.531,-0.311,-0.089,0.267,-0.312,0.489,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:24:00+00:00,BTCUSDT,5m,4,-0.623,-0.465,-0.301,-0.189,0.167,0.278,0.389,0.501,-0.465,0.278,-0.167,-0.256,0.056,0.067,-1.0,0.0,0.0,0.0,-1.058,-0.711,-0.492,-0.272,-0.078,0.278,-0.301,0.501,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:29:00+00:00,BTCUSDT,5m,4,-0.581,-0.454,-0.289,-0.178,0.178,0.289,0.401,0.512,-0.454,0.289,-0.178,-0.267,0.067,0.078,-1.0,0.0,0.0,0.0,-1.015,-0.671,-0.453,-0.234,-0.067,0.289,-0.289,0.512,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:34:00+00:00,BTCUSDT,5m,4,-0.542,-0.443,-0.278,-0.167,0.189,0.301,0.412,0.523,-0.443,0.301,-0.189,-0.278,0.078,0.089,-1.0,0.0,0.0,0.0,-0.974,-0.632,-0.415,-0.197,-0.056,0.301,-0.278,0.523,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:39:00+00:00,BTCUSDT,5m,4,-0.504,-0.432,-0.267,-0.156,0.201,0.312,0.423,0.534,-0.432,0.312,-0.201,-0.289,0.089,0.101,-1.0,0.0,0.0,0.0,-0.934,-0.594,-0.378,-0.161,-0.045,0.312,-0.267,0.534,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:44:00+00:00,BTCUSDT,5m,4,-0.468,-0.421,-0.256,-0.145,0.212,0.323,0.434,0.545,-0.421,0.323,-0.212,-0.301,0.101,0.112,-1.0,0.0,0.0,0.0,-0.896,-0.557,-0.342,-0.126,-0.034,0.323,-0.256,0.545,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:49:00+00:00,BTCUSDT,5m,4,-0.434,-0.410,-0.245,-0.134,0.223,0.334,0.445,0.556,-0.410,0.334,-0.223,-0.312,0.112,0.123,-1.0,0.0,0.0,0.0,-0.859,-0.521,-0.307,-0.092,-0.023,0.334,-0.245,0.556,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:54:00+00:00,BTCUSDT,5m,4,-0.401,-0.399,-0.234,-0.123,0.234,0.345,0.456,0.567,-0.399,0.345,-0.234,-0.323,0.123,0.134,-1.0,0.0,0.0,0.0,-0.823,-0.486,-0.273,-0.059,-0.012,0.345,-0.234,0.567,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T00:59:00+00:00,BTCUSDT,5m,4,-0.369,-0.388,-0.223,-0.112,0.245,0.356,0.467,0.578,-0.388,0.356,-0.245,-0.334,0.134,0.145,-1.0,0.0,0.0,0.0,-0.788,-0.452,-0.240,-0.027,-0.001,0.356,-0.223,0.578,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:04:00+00:00,BTCUSDT,5m,4,-0.338,-0.377,-0.212,-0.101,0.256,0.367,0.478,0.589,-0.377,0.367,-0.256,-0.345,0.145,0.156,-1.0,0.0,0.0,0.0,-0.754,-0.419,-0.208,-0.005,0.010,0.367,-0.212,0.589,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:09:00+00:00,BTCUSDT,5m,4,-0.308,-0.366,-0.201,-0.089,0.267,0.378,0.489,0.601,-0.366,0.378,-0.267,-0.356,0.156,0.167,-1.0,0.0,0.0,0.0,-0.721,-0.387,-0.177,0.017,0.021,0.378,-0.201,0.601,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:14:00+00:00,BTCUSDT,5m,4,-0.279,-0.355,-0.189,-0.078,0.278,0.389,0.501,0.612,-0.355,0.389,-0.278,-0.367,0.167,0.178,-1.0,0.0,0.0,0.0,-0.689,-0.356,-0.147,0.039,0.032,0.389,-0.189,0.612,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:19:00+00:00,BTCUSDT,5m,4,-0.251,-0.344,-0.178,-0.067,0.289,0.401,0.512,0.623,-0.344,0.401,-0.289,-0.378,0.178,0.189,-1.0,0.0,0.0,0.0,-0.658,-0.326,-0.118,0.061,0.043,0.401,-0.178,0.623,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:24:00+00:00,BTCUSDT,5m,4,-0.224,-0.333,-0.167,-0.056,0.301,0.412,0.523,0.634,-0.333,0.412,-0.301,-0.389,0.189,0.201,-1.0,0.0,0.0,0.0,-0.628,-0.297,-0.090,0.083,0.054,0.412,-0.167,0.634,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:29:00+00:00,BTCUSDT,5m,4,-0.198,-0.322,-0.156,-0.045,0.312,0.423,0.534,0.645,-0.322,0.423,-0.312,-0.401,0.201,0.212,-1.0,0.0,0.0,0.0,-0.599,-0.269,-0.063,0.105,0.065,0.423,-0.156,0.645,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:34:00+00:00,BTCUSDT,5m,4,-0.173,-0.311,-0.145,-0.034,0.323,0.434,0.545,0.656,-0.311,0.434,-0.323,-0.412,0.212,0.223,-1.0,0.0,0.0,0.0,-0.571,-0.242,-0.037,0.127,0.076,0.434,-0.145,0.656,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:39:00+00:00,BTCUSDT,5m,4,-0.149,-0.300,-0.134,-0.023,0.334,0.445,0.556,0.667,-0.300,0.445,-0.334,-0.423,0.223,0.234,-1.0,0.0,0.0,0.0,-0.544,-0.216,-0.012,0.149,0.087,0.445,-0.134,0.667,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:44:00+00:00,BTCUSDT,5m,4,-0.126,-0.289,-0.123,-0.012,0.345,0.456,0.567,0.678,-0.289,0.456,-0.345,-0.434,0.234,0.245,-1.0,0.0,0.0,0.0,-0.518,-0.191,0.013,0.171,0.098,0.456,-0.123,0.678,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:49:00+00:00,BTCUSDT,5m,4,-0.104,-0.278,-0.112,-0.001,0.356,0.467,0.578,0.689,-0.278,0.467,-0.356,-0.445,0.245,0.256,-1.0,0.0,0.0,0.0,-0.493,-0.167,0.038,0.193,0.109,0.467,-0.112,0.689,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:54:00+00:00,BTCUSDT,5m,4,-0.083,-0.267,-0.101,0.010,0.367,0.478,0.589,0.701,-0.267,0.478,-0.367,-0.456,0.256,0.267,-1.0,0.0,0.0,0.0,-0.469,-0.144,0.063,0.215,0.120,0.478,-0.101,0.701,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T01:59:00+00:00,BTCUSDT,5m,4,-0.063,-0.256,-0.089,0.021,0.378,0.489,0.601,0.712,-0.256,0.489,-0.378,-0.467,0.267,0.278,-1.0,0.0,0.0,0.0,-0.446,-0.122,0.088,0.237,0.131,0.489,-0.089,0.712,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:04:00+00:00,BTCUSDT,5m,4,-0.044,-0.245,-0.078,0.032,0.389,0.501,0.612,0.723,-0.245,0.501,-0.389,-0.478,0.278,0.289,-1.0,0.0,0.0,0.0,-0.424,-0.101,0.113,0.259,0.142,0.501,-0.078,0.723,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:09:00+00:00,BTCUSDT,5m,4,-0.026,-0.234,-0.067,0.043,0.401,0.512,0.623,0.734,-0.234,0.512,-0.401,-0.489,0.289,0.301,-1.0,0.0,0.0,0.0,-0.403,-0.081,0.138,0.281,0.153,0.512,-0.067,0.734,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:14:00+00:00,BTCUSDT,5m,4,-0.009,-0.223,-0.056,0.054,0.412,0.523,0.634,0.745,-0.223,0.523,-0.412,-0.501,0.301,0.312,-1.0,0.0,0.0,0.0,-0.383,-0.062,0.163,0.303,0.164,0.523,-0.056,0.745,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:19:00+00:00,BTCUSDT,5m,4,0.007,-0.212,-0.045,0.065,0.423,0.534,0.645,0.756,-0.212,0.534,-0.423,-0.512,0.312,0.323,-1.0,0.0,0.0,0.0,-0.364,-0.044,0.188,0.325,0.175,0.534,-0.045,0.756,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:24:00+00:00,BTCUSDT,5m,4,0.022,-0.201,-0.034,0.076,0.434,0.545,0.656,0.767,-0.201,0.545,-0.434,-0.523,0.323,0.334,-1.0,0.0,0.0,0.0,-0.346,-0.027,0.213,0.347,0.186,0.545,-0.034,0.767,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:29:00+00:00,BTCUSDT,5m,4,0.036,-0.190,-0.023,0.087,0.445,0.556,0.667,0.778,-0.190,0.556,-0.445,-0.534,0.334,0.345,-1.0,0.0,0.0,0.0,-0.329,-0.011,0.238,0.369,0.197,0.556,-0.023,0.778,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:34:00+00:00,BTCUSDT,5m,4,0.049,-0.179,-0.012,0.098,0.456,0.567,0.678,0.789,-0.179,0.567,-0.456,-0.545,0.345,0.356,-1.0,0.0,0.0,0.0,-0.313,0.005,0.263,0.391,0.208,0.567,-0.012,0.789,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:39:00+00:00,BTCUSDT,5m,4,0.061,-0.168,-0.001,0.109,0.467,0.578,0.689,0.801,-0.168,0.578,-0.467,-0.556,0.356,0.367,-1.0,0.0,0.0,0.0,-0.298,0.021,0.288,0.413,0.219,0.578,-0.001,0.801,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:44:00+00:00,BTCUSDT,5m,4,0.072,-0.157,0.010,0.120,0.478,0.589,0.701,0.812,-0.157,0.589,-0.478,-0.567,0.367,0.378,-1.0,0.0,0.0,0.0,-0.284,0.037,0.313,0.435,0.230,0.589,0.010,0.812,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:49:00+00:00,BTCUSDT,5m,4,0.082,-0.146,0.021,0.131,0.489,0.601,0.712,0.823,-0.146,0.601,-0.489,-0.578,0.378,0.389,-1.0,0.0,0.0,0.0,-0.271,0.053,0.338,0.457,0.241,0.601,0.021,0.823,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:54:00+00:00,BTCUSDT,5m,4,0.091,-0.135,0.032,0.142,0.501,0.612,0.723,0.834,-0.135,0.612,-0.501,-0.589,0.389,0.401,-1.0,0.0,0.0,0.0,-0.258,0.069,0.363,0.479,0.252,0.612,0.032,0.834,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T02:59:00+00:00,BTCUSDT,5m,4,0.099,-0.124,0.043,0.153,0.512,0.623,0.734,0.845,-0.124,0.623,-0.512,-0.601,0.401,0.412,-1.0,0.0,0.0,0.0,-0.246,0.085,0.388,0.501,0.263,0.623,0.043,0.845,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:04:00+00:00,BTCUSDT,5m,4,0.106,-0.113,0.054,0.164,0.523,0.634,0.745,0.856,-0.113,0.634,-0.523,-0.612,0.412,0.423,-1.0,0.0,0.0,0.0,-0.235,0.101,0.413,0.523,0.274,0.634,0.054,0.856,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:09:00+00:00,BTCUSDT,5m,4,0.112,-0.102,0.065,0.175,0.534,0.645,0.756,0.867,-0.102,0.645,-0.534,-0.623,0.423,0.434,-1.0,0.0,0.0,0.0,-0.225,0.117,0.438,0.545,0.285,0.645,0.065,0.867,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:14:00+00:00,BTCUSDT,5m,4,0.117,-0.091,0.076,0.186,0.545,0.656,0.767,0.878,-0.091,0.656,-0.545,-0.634,0.434,0.445,-1.0,0.0,0.0,0.0,-0.216,0.133,0.463,0.567,0.296,0.656,0.076,0.878,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:19:00+00:00,BTCUSDT,5m,4,0.121,-0.080,0.087,0.197,0.556,0.667,0.778,0.889,-0.080,0.667,-0.556,-0.645,0.445,0.456,-1.0,0.0,0.0,0.0,-0.208,0.149,0.488,0.589,0.307,0.667,0.087,0.889,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:24:00+00:00,BTCUSDT,5m,4,0.124,-0.069,0.098,0.208,0.567,0.678,0.789,0.901,-0.069,0.678,-0.567,-0.656,0.456,0.467,-1.0,0.0,0.0,0.0,-0.201,0.165,0.513,0.611,0.318,0.678,0.098,0.901,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:29:00+00:00,BTCUSDT,5m,4,0.126,-0.058,0.109,0.219,0.578,0.689,0.801,0.912,-0.058,0.689,-0.578,-0.667,0.467,0.478,-1.0,0.0,0.0,0.0,-0.194,0.181,0.538,0.633,0.329,0.689,0.109,0.912,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:34:00+00:00,BTCUSDT,5m,4,0.127,-0.047,0.120,0.230,0.589,0.701,0.812,0.923,-0.047,0.701,-0.589,-0.678,0.478,0.489,-1.0,0.0,0.0,0.0,-0.188,0.197,0.563,0.655,0.340,0.701,0.120,0.923,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:39:00+00:00,BTCUSDT,5m,4,0.127,-0.036,0.131,0.241,0.601,0.712,0.823,0.934,-0.036,0.712,-0.601,-0.689,0.489,0.501,-1.0,0.0,0.0,0.0,-0.183,0.213,0.588,0.677,0.351,0.712,0.131,0.934,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:44:00+00:00,BTCUSDT,5m,4,0.126,-0.025,0.142,0.252,0.612,0.723,0.834,0.945,-0.025,0.723,-0.612,-0.701,0.501,0.512,-1.0,0.0,0.0,0.0,-0.179,0.229,0.613,0.699,0.362,0.723,0.142,0.945,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:49:00+00:00,BTCUSDT,5m,4,0.124,-0.014,0.153,0.263,0.623,0.734,0.845,0.956,-0.014,0.734,-0.623,-0.712,0.512,0.523,-1.0,0.0,0.0,0.0,-0.176,0.245,0.638,0.721,0.373,0.734,0.153,0.956,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:54:00+00:00,BTCUSDT,5m,4,0.121,-0.003,0.164,0.274,0.634,0.745,0.856,0.967,-0.003,0.745,-0.634,-0.723,0.523,0.534,-1.0,0.0,0.0,0.0,-0.174,0.261,0.663,0.743,0.384,0.745,0.164,0.967,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T03:59:00+00:00,BTCUSDT,5m,4,0.117,0.008,0.175,0.285,0.645,0.756,0.767,0.978,0.008,0.756,-0.645,-0.734,0.534,0.545,-1.0,0.0,0.0,0.0,-0.173,0.277,0.688,0.765,0.395,0.756,0.175,0.978,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:04:00+00:00,BTCUSDT,5m,4,0.112,0.019,0.186,0.296,0.656,0.767,0.778,0.989,0.019,0.767,-0.656,-0.745,0.545,0.556,-1.0,0.0,0.0,0.0,-0.173,0.293,0.713,0.787,0.406,0.767,0.186,0.989,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:09:00+00:00,BTCUSDT,5m,4,0.106,0.030,0.197,0.307,0.667,0.778,0.789,1.001,0.030,0.778,-0.667,-0.756,0.556,0.567,-1.0,0.0,0.0,0.0,-0.174,0.309,0.738,0.809,0.417,0.778,0.197,1.001,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:14:00+00:00,BTCUSDT,5m,4,0.099,0.041,0.208,0.318,0.678,0.789,0.801,1.012,0.041,0.789,-0.678,-0.767,0.567,0.578,-1.0,0.0,0.0,0.0,-0.176,0.325,0.763,0.831,0.428,0.789,0.208,1.012,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:19:00+00:00,BTCUSDT,5m,4,0.091,0.052,0.219,0.329,0.689,0.801,0.812,1.023,0.052,0.801,-0.689,-0.778,0.578,0.589,-1.0,0.0,0.0,0.0,-0.179,0.341,0.788,0.853,0.439,0.801,0.219,1.023,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:24:00+00:00,BTCUSDT,5m,4,0.082,0.063,0.230,0.340,0.701,0.812,0.823,1.034,0.063,0.812,-0.701,-0.789,0.589,0.601,-1.0,0.0,0.0,0.0,-0.183,0.357,0.813,0.875,0.450,0.812,0.230,1.034,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:29:00+00:00,BTCUSDT,5m,4,0.072,0.074,0.241,0.351,0.712,0.823,0.834,1.045,0.074,0.823,-0.712,-0.801,0.601,0.612,-1.0,0.0,0.0,0.0,-0.188,0.373,0.838,0.897,0.461,0.823,0.241,1.045,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:34:00+00:00,BTCUSDT,5m,4,0.061,0.085,0.252,0.362,0.723,0.834,0.845,1.056,0.085,0.834,-0.723,-0.812,0.612,0.623,-1.0,0.0,0.0,0.0,-0.194,0.389,0.863,0.919,0.472,0.834,0.252,1.056,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:39:00+00:00,BTCUSDT,5m,4,0.049,0.096,0.263,0.373,0.734,0.845,0.856,1.067,0.096,0.845,-0.734,-0.823,0.623,0.634,-1.0,0.0,0.0,0.0,-0.201,0.405,0.888,0.941,0.483,0.845,0.263,1.067,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:44:00+00:00,BTCUSDT,5m,4,0.036,0.107,0.274,0.384,0.745,0.856,0.867,1.078,0.107,0.856,-0.745,-0.834,0.634,0.645,-1.0,0.0,0.0,0.0,-0.208,0.421,0.913,0.963,0.494,0.856,0.274,1.078,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:49:00+00:00,BTCUSDT,5m,4,0.022,0.118,0.285,0.395,0.756,0.867,0.878,1.089,0.118,0.867,-0.756,-0.845,0.645,0.656,-1.0,0.0,0.0,0.0,-0.216,0.437,0.938,0.985,0.505,0.867,0.285,1.089,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:54:00+00:00,BTCUSDT,5m,4,0.007,0.129,0.296,0.406,0.767,0.878,0.889,1.101,0.129,0.878,-0.767,-0.856,0.656,0.667,-1.0,0.0,0.0,0.0,-0.225,0.453,0.963,1.007,0.516,0.878,0.296,1.101,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T04:59:00+00:00,BTCUSDT,5m,4,-0.009,0.140,0.307,0.417,0.778,0.889,0.901,1.112,0.140,0.889,-0.778,-0.867,0.667,0.678,-1.0,0.0,0.0,0.0,-0.235,0.469,0.988,1.029,0.527,0.889,0.307,1.112,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:04:00+00:00,BTCUSDT,5m,4,-0.026,0.151,0.318,0.428,0.789,0.901,0.912,1.123,0.151,0.901,-0.789,-0.878,0.678,0.689,-1.0,0.0,0.0,0.0,-0.246,0.485,1.013,1.051,0.538,0.901,0.318,1.123,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:09:00+00:00,BTCUSDT,5m,4,-0.044,0.162,0.329,0.439,0.801,0.912,0.923,1.134,0.162,0.912,-0.801,-0.889,0.689,0.701,-1.0,0.0,0.0,0.0,-0.258,0.501,1.038,1.073,0.549,0.912,0.329,1.134,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:14:00+00:00,BTCUSDT,5m,4,-0.063,0.173,0.340,0.450,0.812,0.923,0.934,1.145,0.173,0.923,-0.812,-0.901,0.701,0.712,-1.0,0.0,0.0,0.0,-0.271,0.517,1.063,1.095,0.560,0.923,0.340,1.145,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:19:00+00:00,BTCUSDT,5m,4,-0.083,0.184,0.351,0.461,0.823,0.934,0.945,1.156,0.184,0.934,-0.823,-0.912,0.712,0.723,-1.0,0.0,0.0,0.0,-0.284,0.533,1.088,1.117,0.571,0.934,0.351,1.156,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:24:00+00:00,BTCUSDT,5m,4,-0.104,0.195,0.362,0.472,0.834,0.945,0.956,1.167,0.195,0.945,-0.834,-0.923,0.723,0.734,-1.0,0.0,0.0,0.0,-0.298,0.549,1.113,1.139,0.582,0.945,0.362,1.167,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:29:00+00:00,BTCUSDT,5m,4,-0.126,0.206,0.373,0.483,0.845,0.956,0.967,1.178,0.206,0.956,-0.845,-0.934,0.734,0.745,-1.0,0.0,0.0,0.0,-0.313,0.565,1.138,1.161,0.593,0.956,0.373,1.178,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:34:00+00:00,BTCUSDT,5m,4,-0.149,0.217,0.384,0.494,0.856,0.967,0.978,1.189,0.217,0.967,-0.856,-0.945,0.745,0.756,-1.0,0.0,0.0,0.0,-0.329,0.581,1.163,1.183,0.604,0.967,0.384,1.189,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:39:00+00:00,BTCUSDT,5m,4,-0.173,0.228,0.395,0.505,0.867,0.978,0.989,1.201,0.228,0.978,-0.867,-0.956,0.756,0.767,-1.0,0.0,0.0,0.0,-0.346,0.597,1.188,1.205,0.615,0.978,0.395,1.201,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:44:00+00:00,BTCUSDT,5m,4,-0.198,0.239,0.406,0.516,0.878,0.989,1.001,1.212,0.239,0.989,-0.878,-0.967,0.767,0.778,-1.0,0.0,0.0,0.0,-0.363,0.613,1.213,1.227,0.626,0.989,0.406,1.212,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:49:00+00:00,BTCUSDT,5m,4,-0.224,0.250,0.417,0.527,0.889,1.001,1.012,1.223,0.250,1.001,-0.889,-0.978,0.778,0.789,-1.0,0.0,0.0,0.0,-0.381,0.629,1.238,1.249,0.637,1.001,0.417,1.223,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:54:00+00:00,BTCUSDT,5m,4,-0.251,0.261,0.428,0.538,0.901,1.012,1.023,1.234,0.261,1.012,-0.901,-0.989,0.789,0.801,-1.0,0.0,0.0,0.0,-0.400,0.645,1.263,1.271,0.648,1.012,0.428,1.234,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T05:59:00+00:00,BTCUSDT,5m,4,-0.279,0.272,0.439,0.549,0.912,1.023,1.034,1.245,0.272,1.023,-0.912,-1.001,0.801,0.812,-1.0,0.0,0.0,0.0,-0.419,0.661,1.288,1.293,0.659,1.023,0.439,1.245,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:04:00+00:00,BTCUSDT,5m,4,-0.308,0.283,0.450,0.560,0.923,1.034,1.045,1.256,0.283,1.034,-0.923,-1.012,0.812,0.823,-1.0,0.0,0.0,0.0,-0.439,0.677,1.313,1.315,0.670,1.034,0.450,1.256,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:09:00+00:00,BTCUSDT,5m,4,-0.338,0.294,0.461,0.571,0.934,1.045,1.056,1.267,0.294,1.045,-0.934,-1.023,0.823,0.834,-1.0,0.0,0.0,0.0,-0.460,0.693,1.338,1.337,0.681,1.045,0.461,1.267,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:14:00+00:00,BTCUSDT,5m,4,-0.369,0.305,0.472,0.582,0.945,1.056,1.067,1.278,0.305,1.056,-0.945,-1.034,0.834,0.845,-1.0,0.0,0.0,0.0,-0.482,0.709,1.363,1.359,0.692,1.056,0.472,1.278,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:19:00+00:00,BTCUSDT,5m,4,-0.401,0.316,0.483,0.593,0.956,1.067,1.078,1.289,0.316,1.067,-0.956,-1.045,0.845,0.856,-1.0,0.0,0.0,0.0,-0.505,0.725,1.388,1.381,0.703,1.067,0.483,1.289,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:24:00+00:00,BTCUSDT,5m,4,-0.434,0.327,0.494,0.604,0.967,1.078,1.089,1.301,0.327,1.078,-0.967,-1.056,0.856,0.867,-1.0,0.0,0.0,0.0,-0.529,0.741,1.413,1.403,0.714,1.078,0.494,1.301,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:29:00+00:00,BTCUSDT,5m,4,-0.468,0.338,0.505,0.615,0.978,1.089,1.101,1.312,0.338,1.089,-0.978,-1.067,0.867,0.878,-1.0,0.0,0.0,0.0,-0.554,0.757,1.438,1.425,0.725,1.089,0.505,1.312,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:34:00+00:00,BTCUSDT,5m,4,-0.504,0.349,0.516,0.626,0.989,1.101,1.112,1.323,0.349,1.101,-0.989,-1.078,0.878,0.889,-1.0,0.0,0.0,0.0,-0.580,0.773,1.463,1.447,0.736,1.101,0.516,1.323,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:39:00+00:00,BTCUSDT,5m,4,-0.542,0.360,0.527,0.637,1.001,1.112,1.123,1.334,0.360,1.112,-1.001,-1.089,0.889,0.901,-1.0,0.0,0.0,0.0,-0.607,0.789,1.488,1.469,0.747,1.112,0.527,1.334,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:44:00+00:00,BTCUSDT,5m,4,-0.581,0.371,0.538,0.648,1.012,1.123,1.134,1.345,0.371,1.123,-1.012,-1.101,0.901,0.912,-1.0,0.0,0.0,0.0,-0.635,0.805,1.513,1.491,0.758,1.123,0.538,1.345,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:49:00+00:00,BTCUSDT,5m,4,-0.623,0.382,0.549,0.659,1.023,1.134,1.145,1.356,0.382,1.134,-1.023,-1.112,0.912,0.923,-1.0,0.0,0.0,0.0,-0.665,0.821,1.538,1.513,0.769,1.134,0.549,1.356,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:54:00+00:00,BTCUSDT,5m,4,-0.667,0.393,0.560,0.670,1.034,1.145,1.156,1.367,0.393,1.145,-1.034,-1.123,0.923,0.934,-1.0,0.0,0.0,0.0,-0.696,0.837,1.563,1.535,0.780,1.145,0.560,1.367,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T06:59:00+00:00,BTCUSDT,5m,4,-0.712,0.404,0.571,0.681,1.045,1.156,1.167,1.378,0.404,1.156,-1.045,-1.134,0.934,0.945,-1.0,0.0,0.0,0.0,-0.729,0.853,1.588,1.557,0.791,1.156,0.571,1.378,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:04:00+00:00,BTCUSDT,5m,4,-0.765,0.415,0.582,0.692,1.056,1.167,1.178,1.389,0.415,1.167,-1.056,-1.145,0.945,0.956,-1.0,0.0,0.0,0.0,-0.764,0.869,1.613,1.579,0.802,1.167,0.582,1.389,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:09:00+00:00,BTCUSDT,5m,4,-0.823,0.426,0.593,0.703,1.067,1.178,1.189,1.401,0.426,1.178,-1.067,-1.156,0.956,0.967,-1.0,0.0,0.0,0.0,-0.801,0.885,1.638,1.601,0.813,1.178,0.593,1.401,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:14:00+00:00,BTCUSDT,5m,4,-0.889,0.437,0.604,0.714,1.078,1.189,1.201,1.412,0.437,1.189,-1.078,-1.167,0.967,0.978,-1.0,0.0,0.0,0.0,-0.840,0.901,1.663,1.623,0.824,1.189,0.604,1.412,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:19:00+00:00,BTCUSDT,5m,4,-0.956,0.448,0.615,0.725,1.089,1.201,1.212,1.423,0.448,1.201,-1.089,-1.178,0.978,0.989,-1.0,0.0,0.0,0.0,-0.882,0.917,1.688,1.645,0.835,1.201,0.615,1.423,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:24:00+00:00,BTCUSDT,5m,4,-1.034,0.459,0.626,0.736,1.101,1.212,1.223,1.434,0.459,1.212,-1.101,-1.189,0.989,1.001,-1.0,0.0,0.0,0.0,-0.926,0.933,1.713,1.667,0.846,1.212,0.626,1.434,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:29:00+00:00,BTCUSDT,5m,4,-1.145,0.470,0.637,0.747,1.112,1.223,1.234,1.445,0.470,1.223,-1.112,-1.201,1.001,1.012,-1.0,0.0,0.0,0.0,-0.974,0.949,1.738,1.689,0.857,1.223,0.637,1.445,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:34:00+00:00,BTCUSDT,5m,4,-1.298,0.481,0.648,0.758,1.123,1.234,1.245,1.456,0.481,1.234,-1.123,-1.212,1.012,1.023,-1.0,0.0,0.0,0.0,-1.026,0.965,1.763,1.711,0.868,1.234,0.648,1.456,RANGE,2,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:39:00+00:00,BTCUSDT,5m,4,-1.501,0.492,0.659,0.769,1.134,1.245,1.256,1.467,0.492,1.245,-1.134,-1.223,1.023,1.034,-1.0,0.0,0.0,0.0,-1.084,0.981,1.788,1.733,0.879,1.245,0.659,1.467,WEAK_DOWN,1,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:44:00+00:00,BTCUSDT,5m,4,-1.734,0.503,0.670,0.780,1.145,1.256,1.267,1.478,0.503,1.256,-1.145,-1.234,1.034,1.045,-1.0,0.0,0.0,0.0,-1.148,0.997,1.813,1.755,0.890,1.256,0.670,1.478,WEAK_DOWN,1,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:49:00+00:00,BTCUSDT,5m,4,-1.989,0.514,0.681,0.791,1.156,1.267,1.278,1.489,0.514,1.267,-1.156,-1.245,1.045,1.056,-1.0,0.0,0.0,0.0,-1.219,1.013,1.838,1.777,0.901,1.267,0.681,1.489,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:54:00+00:00,BTCUSDT,5m,4,-2.234,0.525,0.692,0.802,1.167,1.278,1.289,1.501,0.525,1.278,-1.167,-1.256,1.056,1.067,-1.0,0.0,0.0,0.0,-1.296,1.029,1.863,1.799,0.912,1.278,0.692,1.501,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T07:59:00+00:00,BTCUSDT,5m,4,-2.456,0.536,0.703,0.813,1.178,1.289,1.301,1.512,0.536,1.289,-1.178,-1.267,1.067,1.078,-1.0,0.0,0.0,0.0,-1.378,1.045,1.888,1.821,0.923,1.289,0.703,1.512,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:04:00+00:00,BTCUSDT,5m,4,-2.634,0.547,0.714,0.824,1.189,1.301,1.312,1.523,0.547,1.301,-1.189,-1.278,1.078,1.089,-1.0,0.0,0.0,0.0,-1.463,1.061,1.913,1.843,0.934,1.301,0.714,1.523,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:09:00+00:00,BTCUSDT,5m,4,-2.767,0.558,0.725,0.835,1.201,1.312,1.323,1.534,0.558,1.312,-1.201,-1.289,1.089,1.101,-1.0,0.0,0.0,0.0,-1.552,1.077,1.938,1.865,0.945,1.312,0.725,1.534,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:14:00+00:00,BTCUSDT,5m,4,-2.856,0.569,0.736,0.846,1.212,1.323,1.334,1.545,0.569,1.323,-1.212,-1.301,1.101,1.112,-1.0,0.0,0.0,0.0,-1.645,1.093,1.963,1.887,0.956,1.323,0.736,1.545,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:19:00+00:00,BTCUSDT,5m,4,-2.901,0.580,0.747,0.857,1.223,1.334,1.345,1.556,0.580,1.334,-1.223,-1.312,1.112,1.123,-1.0,0.0,0.0,0.0,-1.742,1.109,1.988,1.909,0.967,1.334,0.747,1.556,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:24:00+00:00,BTCUSDT,5m,4,-2.901,0.591,0.758,0.868,1.234,1.345,1.356,1.567,0.591,1.345,-1.234,-1.323,1.123,1.134,-1.0,0.0,0.0,0.0,-1.842,1.125,2.013,1.931,0.978,1.345,0.758,1.567,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:29:00+00:00,BTCUSDT,5m,4,-2.856,0.602,0.769,0.879,1.245,1.356,1.367,1.578,0.602,1.356,-1.245,-1.334,1.134,1.145,-1.0,0.0,0.0,0.0,-1.945,1.141,2.038,1.953,0.989,1.356,0.769,1.578,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:34:00+00:00,BTCUSDT,5m,4,-2.767,0.613,0.780,0.890,1.256,1.367,1.378,1.589,0.613,1.367,-1.256,-1.345,1.145,1.156,-1.0,0.0,0.0,0.0,-2.051,1.157,2.063,1.975,1.000,1.367,0.780,1.589,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:39:00+00:00,BTCUSDT,5m,4,-2.634,0.624,0.791,0.901,1.267,1.378,1.389,1.601,0.624,1.378,-1.267,-1.356,1.156,1.167,-1.0,0.0,0.0,0.0,-2.160,1.173,2.088,1.997,1.011,1.378,0.791,1.601,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:44:00+00:00,BTCUSDT,5m,4,-2.456,0.635,0.802,0.912,1.278,1.389,1.401,1.612,0.635,1.389,-1.278,-1.367,1.167,1.178,-1.0,0.0,0.0,0.0,-2.272,1.189,2.113,2.019,1.022,1.389,0.802,1.612,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:49:00+00:00,BTCUSDT,5m,4,-2.234,0.646,0.813,0.923,1.289,1.401,1.412,1.623,0.646,1.401,-1.289,-1.378,1.178,1.189,-1.0,0.0,0.0,0.0,-2.387,1.205,2.138,2.041,1.033,1.401,0.813,1.623,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:54:00+00:00,BTCUSDT,5m,4,-1.989,0.657,0.824,0.934,1.301,1.412,1.423,1.634,0.657,1.412,-1.301,-1.389,1.189,1.201,-1.0,0.0,0.0,0.0,-2.505,1.221,2.163,2.063,1.044,1.412,0.824,1.634,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T08:59:00+00:00,BTCUSDT,5m,4,-1.734,0.668,0.835,0.945,1.312,1.423,1.434,1.645,0.668,1.423,-1.312,-1.401,1.201,1.212,-1.0,0.0,0.0,0.0,-2.626,1.237,2.188,2.085,1.055,1.423,0.835,1.645,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:04:00+00:00,BTCUSDT,5m,4,-1.501,0.679,0.846,0.956,1.323,1.434,1.445,1.656,0.679,1.434,-1.323,-1.412,1.212,1.223,-1.0,0.0,0.0,0.0,-2.750,1.253,2.213,2.107,1.066,1.434,0.846,1.656,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:09:00+00:00,BTCUSDT,5m,4,-1.298,0.690,0.857,0.967,1.334,1.445,1.456,1.667,0.690,1.445,-1.334,-1.423,1.223,1.234,-1.0,0.0,0.0,0.0,-2.877,1.269,2.238,2.129,1.077,1.445,0.857,1.667,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:14:00+00:00,BTCUSDT,5m,4,-1.145,0.701,0.868,0.978,1.345,1.456,1.467,1.678,0.701,1.456,-1.345,-1.434,1.234,1.245,-1.0,0.0,0.0,0.0,-3.008,1.285,2.263,2.151,1.088,1.456,0.868,1.678,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:19:00+00:00,BTCUSDT,5m,4,-1.034,0.712,0.879,0.989,1.356,1.467,1.478,1.689,0.712,1.467,-1.356,-1.445,1.245,1.256,-1.0,0.0,0.0,0.0,-3.143,1.301,2.288,2.173,1.099,1.467,0.879,1.689,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:24:00+00:00,BTCUSDT,5m,4,-0.956,0.723,0.890,1.001,1.367,1.478,1.489,1.701,0.723,1.478,-1.367,-1.456,1.256,1.267,-1.0,0.0,0.0,0.0,-3.282,1.317,2.313,2.195,1.110,1.478,0.890,1.701,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:29:00+00:00,BTCUSDT,5m,4,-0.889,0.734,0.901,1.012,1.378,1.489,1.501,1.712,0.734,1.489,-1.378,-1.467,1.267,1.278,-1.0,0.0,0.0,0.0,-3.425,1.333,2.338,2.217,1.121,1.489,0.901,1.712,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:34:00+00:00,BTCUSDT,5m,4,-0.823,0.745,0.912,1.023,1.389,1.501,1.512,1.723,0.745,1.501,-1.389,-1.478,1.278,1.289,-1.0,0.0,0.0,0.0,-3.572,1.349,2.363,2.239,1.132,1.501,0.912,1.723,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:39:00+00:00,BTCUSDT,5m,4,-0.765,0.756,0.923,1.034,1.401,1.512,1.523,1.734,0.756,1.512,-1.401,-1.489,1.289,1.301,-1.0,0.0,0.0,0.0,-3.723,1.365,2.388,2.261,1.143,1.512,0.923,1.734,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:44:00+00:00,BTCUSDT,5m,4,-0.712,0.767,0.934,1.045,1.412,1.523,1.534,1.745,0.767,1.523,-1.412,-1.501,1.301,1.312,-1.0,0.0,0.0,0.0,-3.878,1.381,2.413,2.283,1.154,1.523,0.934,1.745,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:49:00+00:00,BTCUSDT,5m,4,-0.667,0.778,0.945,1.056,1.423,1.534,1.545,1.756,0.778,1.534,-1.423,-1.512,1.312,1.323,-1.0,0.0,0.0,0.0,-4.037,1.397,2.438,2.305,1.165,1.534,0.945,1.756,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:54:00+00:00,BTCUSDT,5m,4,-0.623,0.789,0.956,1.067,1.434,1.545,1.556,1.767,0.789,1.545,-1.434,-1.523,1.323,1.334,-1.0,0.0,0.0,0.0,-4.200,1.413,2.463,2.327,1.176,1.545,0.956,1.767,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T09:59:00+00:00,BTCUSDT,5m,4,-0.581,0.801,0.967,1.078,1.445,1.556,1.567,1.778,0.801,1.556,-1.445,-1.534,1.334,1.345,-1.0,0.0,0.0,0.0,-4.367,1.429,2.488,2.349,1.187,1.556,0.967,1.778,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:04:00+00:00,BTCUSDT,5m,4,-0.542,0.812,0.978,1.089,1.456,1.567,1.578,1.789,0.812,1.567,-1.456,-1.545,1.345,1.356,-1.0,0.0,0.0,0.0,-4.538,1.445,2.513,2.371,1.198,1.567,0.978,1.789,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:09:00+00:00,BTCUSDT,5m,4,-0.504,0.823,0.989,1.101,1.467,1.578,1.589,1.801,0.823,1.578,-1.467,-1.556,1.356,1.367,-1.0,0.0,0.0,0.0,-4.713,1.461,2.538,2.393,1.209,1.578,0.989,1.801,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:14:00+00:00,BTCUSDT,5m,4,-0.468,0.834,1.001,1.112,1.478,1.589,1.601,1.812,0.834,1.589,-1.478,-1.567,1.367,1.378,-1.0,0.0,0.0,0.0,-4.892,1.477,2.563,2.415,1.220,1.589,1.001,1.812,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:19:00+00:00,BTCUSDT,5m,4,-0.434,0.845,1.012,1.123,1.489,1.601,1.612,1.823,0.845,1.601,-1.489,-1.578,1.378,1.389,-1.0,0.0,0.0,0.0,-5.075,1.493,2.588,2.437,1.231,1.601,1.012,1.823,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:24:00+00:00,BTCUSDT,5m,4,-0.401,0.856,1.023,1.134,1.501,1.612,1.623,1.834,0.856,1.612,-1.501,-1.589,1.389,1.401,-1.0,0.0,0.0,0.0,-5.262,1.509,2.613,2.459,1.242,1.612,1.023,1.834,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:29:00+00:00,BTCUSDT,5m,4,-0.369,0.867,1.034,1.145,1.512,1.623,1.634,1.845,0.867,1.623,-1.512,-1.601,1.401,1.412,-1.0,0.0,0.0,0.0,-5.453,1.525,2.638,2.481,1.253,1.623,1.034,1.845,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:34:00+00:00,BTCUSDT,5m,4,-0.338,0.878,1.045,1.156,1.523,1.634,1.645,1.856,0.878,1.634,-1.523,-1.612,1.412,1.423,-1.0,0.0,0.0,0.0,-5.648,1.541,2.663,2.503,1.264,1.634,1.045,1.856,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:39:00+00:00,BTCUSDT,5m,4,-0.308,0.889,1.056,1.167,1.534,1.645,1.656,1.867,0.889,1.645,-1.534,-1.623,1.423,1.434,-1.0,0.0,0.0,0.0,-5.847,1.557,2.688,2.525,1.275,1.645,1.056,1.867,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:44:00+00:00,BTCUSDT,5m,4,-0.279,0.901,1.067,1.178,1.545,1.656,1.667,1.878,0.901,1.656,-1.545,-1.634,1.434,1.445,-1.0,0.0,0.0,0.0,-6.050,1.573,2.713,2.547,1.286,1.656,1.067,1.878,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:49:00+00:00,BTCUSDT,5m,4,-0.251,0.912,1.078,1.189,1.556,1.667,1.678,1.889,0.912,1.667,-1.556,-1.645,1.445,1.456,-1.0,0.0,0.0,0.0,-6.257,1.589,2.738,2.569,1.297,1.667,1.078,1.889,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:54:00+00:00,BTCUSDT,5m,4,-0.224,0.923,1.089,1.201,1.567,1.678,1.689,1.901,0.923,1.678,-1.567,-1.656,1.456,1.467,-1.0,0.0,0.0,0.0,-6.468,1.605,2.763,2.591,1.308,1.678,1.089,1.901,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T10:59:00+00:00,BTCUSDT,5m,4,-0.198,0.934,1.101,1.212,1.578,1.689,1.701,1.912,0.934,1.689,-1.578,-1.667,1.467,1.478,-1.0,0.0,0.0,0.0,-6.683,1.621,2.788,2.613,1.319,1.689,1.101,1.912,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:04:00+00:00,BTCUSDT,5m,4,-0.173,0.945,1.112,1.223,1.589,1.701,1.712,1.923,0.945,1.701,-1.589,-1.678,1.478,1.489,-1.0,0.0,0.0,0.0,-6.902,1.637,2.813,2.635,1.330,1.701,1.112,1.923,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:09:00+00:00,BTCUSDT,5m,4,-0.149,0.956,1.123,1.234,1.601,1.712,1.723,1.934,0.956,1.712,-1.601,-1.689,1.489,1.501,-1.0,0.0,0.0,0.0,-7.125,1.653,2.838,2.657,1.341,1.712,1.123,1.934,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:14:00+00:00,BTCUSDT,5m,4,-0.126,0.967,1.134,1.245,1.612,1.723,1.734,1.945,0.967,1.723,-1.612,-1.701,1.501,1.512,-1.0,0.0,0.0,0.0,-7.352,1.669,2.863,2.679,1.352,1.723,1.134,1.945,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:19:00+00:00,BTCUSDT,5m,4,-0.104,0.978,1.145,1.256,1.623,1.734,1.745,1.956,0.978,1.734,-1.623,-1.712,1.512,1.523,-1.0,0.0,0.0,0.0,-7.583,1.685,2.888,2.701,1.363,1.734,1.145,1.956,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:24:00+00:00,BTCUSDT,5m,4,-0.083,0.989,1.156,1.267,1.634,1.745,1.756,1.967,0.989,1.745,-1.634,-1.723,1.523,1.534,-1.0,0.0,0.0,0.0,-7.818,1.701,2.913,2.723,1.374,1.745,1.156,1.967,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:29:00+00:00,BTCUSDT,5m,4,-0.063,1.001,1.167,1.278,1.645,1.756,1.767,1.978,1.001,1.756,-1.645,-1.734,1.534,1.545,-1.0,0.0,0.0,0.0,-8.057,1.717,2.938,2.745,1.385,1.756,1.167,1.978,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:34:00+00:00,BTCUSDT,5m,4,-0.044,1.012,1.178,1.289,1.656,1.767,1.778,1.989,1.012,1.767,-1.656,-1.745,1.545,1.556,-1.0,0.0,0.0,0.0,-8.300,1.733,2.963,2.767,1.396,1.767,1.178,1.989,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:39:00+00:00,BTCUSDT,5m,4,-0.026,1.023,1.189,1.301,1.667,1.778,1.789,2.001,1.023,1.778,-1.667,-1.756,1.556,1.567,-1.0,0.0,0.0,0.0,-8.547,1.749,2.988,2.789,1.407,1.778,1.189,2.001,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:44:00+00:00,BTCUSDT,5m,4,-0.009,1.034,1.201,1.312,1.678,1.789,1.801,2.012,1.034,1.789,-1.678,-1.767,1.567,1.578,-1.0,0.0,0.0,0.0,-8.798,1.765,3.013,2.811,1.418,1.789,1.201,2.012,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:49:00+00:00,BTCUSDT,5m,4,0.007,1.045,1.212,1.323,1.689,1.801,1.812,2.023,1.045,1.801,-1.689,-1.778,1.578,1.589,-1.0,0.0,0.0,0.0,-9.053,1.781,3.038,2.833,1.429,1.801,1.212,2.023,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:54:00+00:00,BTCUSDT,5m,4,0.022,1.056,1.223,1.334,1.701,1.812,1.823,2.034,1.056,1.812,-1.701,-1.789,1.589,1.601,-1.0,0.0,0.0,0.0,-9.312,1.797,3.063,2.855,1.440,1.812,1.223,2.034,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T11:59:00+00:00,BTCUSDT,5m,4,0.036,1.067,1.234,1.345,1.712,1.823,1.834,2.045,1.067,1.823,-1.712,-1.801,1.601,1.612,-1.0,0.0,0.0,0.0,-9.575,1.813,3.088,2.877,1.451,1.823,1.234,2.045,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:04:00+00:00,BTCUSDT,5m,4,0.049,1.078,1.245,1.356,1.723,1.834,1.845,2.056,1.078,1.834,-1.723,-1.812,1.612,1.623,-1.0,0.0,0.0,0.0,-9.842,1.829,3.113,2.899,1.462,1.834,1.245,2.056,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:09:00+00:00,BTCUSDT,5m,4,0.061,1.089,1.256,1.367,1.734,1.845,1.856,2.067,1.089,1.845,-1.734,-1.823,1.623,1.634,-1.0,0.0,0.0,0.0,-10.113,1.845,3.138,2.921,1.473,1.845,1.256,2.067,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:14:00+00:00,BTCUSDT,5m,4,0.072,1.101,1.267,1.378,1.745,1.856,1.867,2.078,1.101,1.856,-1.745,-1.834,1.634,1.645,-1.0,0.0,0.0,0.0,-10.388,1.861,3.163,2.943,1.484,1.856,1.267,2.078,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:19:00+00:00,BTCUSDT,5m,4,0.082,1.112,1.278,1.389,1.756,1.867,1.878,2.089,1.112,1.867,-1.756,-1.845,1.645,1.656,-1.0,0.0,0.0,0.0,-10.667,1.877,3.188,2.965,1.495,1.867,1.278,2.089,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:24:00+00:00,BTCUSDT,5m,4,0.091,1.123,1.289,1.401,1.767,1.878,1.889,2.101,1.123,1.878,-1.767,-1.856,1.656,1.667,-1.0,0.0,0.0,0.0,-10.950,1.893,3.213,2.987,1.506,1.878,1.289,2.101,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:29:00+00:00,BTCUSDT,5m,4,0.099,1.134,1.301,1.412,1.778,1.889,1.901,2.112,1.134,1.889,-1.778,-1.867,1.667,1.678,-1.0,0.0,0.0,0.0,-11.237,1.909,3.238,3.009,1.517,1.889,1.301,2.112,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:34:00+00:00,BTCUSDT,5m,4,0.106,1.145,1.312,1.423,1.789,1.901,1.912,2.123,1.145,1.901,-1.789,-1.878,1.678,1.689,-1.0,0.0,0.0,0.0,-11.528,1.925,3.263,3.031,1.528,1.901,1.312,2.123,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:39:00+00:00,BTCUSDT,5m,4,0.112,1.156,1.323,1.434,1.801,1.912,1.923,2.134,1.156,1.912,-1.801,-1.889,1.689,1.701,-1.0,0.0,0.0,0.0,-11.823,1.941,3.288,3.053,1.539,1.912,1.323,2.134,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:44:00+00:00,BTCUSDT,5m,4,0.117,1.167,1.334,1.445,1.812,1.923,1.934,2.145,1.167,1.923,-1.812,-1.901,1.701,1.712,-1.0,0.0,0.0,0.0,-12.122,1.957,3.313,3.075,1.550,1.923,1.334,2.145,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:49:00+00:00,BTCUSDT,5m,4,0.121,1.178,1.345,1.456,1.823,1.934,1.945,2.156,1.178,1.934,-1.823,-1.912,1.712,1.723,-1.0,0.0,0.0,0.0,-12.425,1.973,3.338,3.097,1.561,1.934,1.345,2.156,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:54:00+00:00,BTCUSDT,5m,4,0.124,1.189,1.356,1.467,1.834,1.945,1.956,2.167,1.189,1.945,-1.834,-1.923,1.723,1.734,-1.0,0.0,0.0,0.0,-12.732,1.989,3.363,3.119,1.572,1.945,1.356,2.167,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T12:59:00+00:00,BTCUSDT,5m,4,0.126,1.201,1.367,1.478,1.845,1.956,1.967,2.178,1.201,1.956,-1.845,-1.934,1.734,1.745,-1.0,0.0,0.0,0.0,-13.043,2.005,3.388,3.141,1.583,1.956,1.367,2.178,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:04:00+00:00,BTCUSDT,5m,4,0.127,1.212,1.378,1.489,1.856,1.967,1.978,2.189,1.212,1.967,-1.856,-1.945,1.745,1.756,-1.0,0.0,0.0,0.0,-13.358,2.021,3.413,3.163,1.594,1.967,1.378,2.189,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:09:00+00:00,BTCUSDT,5m,4,0.127,1.223,1.389,1.501,1.867,1.978,1.989,2.201,1.223,1.978,-1.867,-1.956,1.756,1.767,-1.0,0.0,0.0,0.0,-13.677,2.037,3.438,3.185,1.605,1.978,1.389,2.201,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:14:00+00:00,BTCUSDT,5m,4,0.126,1.234,1.401,1.512,1.878,1.989,2.001,2.212,1.234,1.989,-1.878,-1.967,1.767,1.778,-1.0,0.0,0.0,0.0,-14.000,2.053,3.463,3.207,1.616,1.989,1.401,2.212,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:19:00+00:00,BTCUSDT,5m,4,0.124,1.245,1.412,1.523,1.889,2.001,2.012,2.223,1.245,2.001,-1.889,-1.978,1.778,1.789,-1.0,0.0,0.0,0.0,-14.327,2.069,3.488,3.229,1.627,2.001,1.412,2.223,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:24:00+00:00,BTCUSDT,5m,4,0.121,1.256,1.423,1.534,1.901,2.012,2.023,2.234,1.256,2.012,-1.901,-1.989,1.789,1.801,-1.0,0.0,0.0,0.0,-14.658,2.085,3.513,3.251,1.638,2.012,1.423,2.234,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:29:00+00:00,BTCUSDT,5m,4,0.117,1.267,1.434,1.545,1.912,2.023,2.034,2.245,1.267,2.023,-1.912,-2.001,1.801,1.812,-1.0,0.0,0.0,0.0,-14.993,2.101,3.538,3.273,1.649,2.023,1.434,2.245,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:34:00+00:00,BTCUSDT,5m,4,0.112,1.278,1.445,1.556,1.923,2.034,2.045,2.256,1.278,2.034,-1.923,-2.012,1.812,1.823,-1.0,0.0,0.0,0.0,-15.332,2.117,3.563,3.295,1.660,2.034,1.445,2.256,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:39:00+00:00,BTCUSDT,5m,4,0.106,1.289,1.456,1.567,1.934,2.045,2.056,2.267,1.289,2.045,-1.934,-2.023,1.823,1.834,-1.0,0.0,0.0,0.0,-15.675,2.133,3.588,3.317,1.671,2.045,1.456,2.267,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:44:00+00:00,BTCUSDT,5m,4,0.099,1.301,1.467,1.578,1.945,2.056,2.067,2.278,1.301,2.056,-1.945,-2.034,1.834,1.845,-1.0,0.0,0.0,0.0,-16.022,2.149,3.613,3.339,1.682,2.056,1.467,2.278,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:49:00+00:00,BTCUSDT,5m,4,0.091,1.312,1.478,1.589,1.956,2.067,2.078,2.289,1.312,2.067,-1.956,-2.045,1.845,1.856,-1.0,0.0,0.0,0.0,-16.373,2.165,3.638,3.361,1.693,2.067,1.478,2.289,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:54:00+00:00,BTCUSDT,5m,4,0.082,1.323,1.489,1.601,1.967,2.078,2.089,2.301,1.323,2.078,-1.967,-2.056,1.856,1.867,-1.0,0.0,0.0,0.0,-16.728,2.181,3.663,3.383,1.704,2.078,1.489,2.301,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T13:59:00+00:00,BTCUSDT,5m,4,0.072,1.334,1.501,1.612,1.978,2.089,2.101,2.312,1.334,2.089,-1.978,-2.067,1.867,1.878,-1.0,0.0,0.0,0.0,-17.087,2.197,3.688,3.405,1.715,2.089,1.501,2.312,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:04:00+00:00,BTCUSDT,5m,4,0.061,1.345,1.512,1.623,1.989,2.101,2.112,2.323,1.345,2.101,-1.989,-2.078,1.878,1.889,-1.0,0.0,0.0,0.0,-17.450,2.213,3.713,3.427,1.726,2.101,1.512,2.323,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:09:00+00:00,BTCUSDT,5m,4,0.049,1.356,1.523,1.634,2.001,2.112,2.123,2.334,1.356,2.112,-2.001,-2.089,1.889,1.901,-1.0,0.0,0.0,0.0,-17.817,2.229,3.738,3.449,1.737,2.112,1.523,2.334,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:14:00+00:00,BTCUSDT,5m,4,0.036,1.367,1.534,1.645,2.012,2.123,2.134,2.345,1.367,2.123,-2.012,-2.101,1.901,1.912,-1.0,0.0,0.0,0.0,-18.188,2.245,3.763,3.471,1.748,2.123,1.534,2.345,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:19:00+00:00,BTCUSDT,5m,4,0.022,1.378,1.545,1.656,2.023,2.134,2.145,2.356,1.378,2.134,-2.023,-2.112,1.912,1.923,-1.0,0.0,0.0,0.0,-18.563,2.261,3.788,3.493,1.759,2.134,1.545,2.356,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:24:00+00:00,BTCUSDT,5m,4,0.007,1.389,1.556,1.667,2.034,2.145,2.156,2.367,1.389,2.145,-2.034,-2.123,1.923,1.934,-1.0,0.0,0.0,0.0,-18.942,2.277,3.813,3.515,1.770,2.145,1.556,2.367,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:29:00+00:00,BTCUSDT,5m,4,-0.009,1.401,1.567,1.678,2.045,2.156,2.167,2.378,1.401,2.156,-2.045,-2.134,1.934,1.945,-1.0,0.0,0.0,0.0,-19.325,2.293,3.838,3.537,1.781,2.156,1.567,2.378,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:34:00+00:00,BTCUSDT,5m,4,-0.026,1.412,1.578,1.689,2.056,2.167,2.178,2.389,1.412,2.167,-2.056,-2.145,1.945,1.956,-1.0,0.0,0.0,0.0,-19.712,2.309,3.863,3.559,1.792,2.167,1.578,2.389,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:39:00+00:00,BTCUSDT,5m,4,-0.044,1.423,1.589,1.701,2.067,2.178,2.189,2.401,1.423,2.178,-2.067,-2.156,1.956,1.967,-1.0,0.0,0.0,0.0,-20.103,2.325,3.888,3.581,1.803,2.178,1.589,2.401,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:44:00+00:00,BTCUSDT,5m,4,-0.063,1.434,1.601,1.712,2.078,2.189,2.201,2.412,1.434,2.189,-2.078,-2.167,1.967,1.978,-1.0,0.0,0.0,0.0,-20.498,2.341,3.913,3.603,1.814,2.189,1.601,2.412,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:49:00+00:00,BTCUSDT,5m,4,-0.083,1.445,1.612,1.723,2.089,2.201,2.212,2.423,1.445,2.201,-2.089,-2.178,1.978,1.989,-1.0,0.0,0.0,0.0,-20.897,2.357,3.938,3.625,1.825,2.201,1.612,2.423,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:54:00+00:00,BTCUSDT,5m,4,-0.104,1.456,1.623,1.734,2.101,2.212,2.223,2.434,1.456,2.212,-2.101,-2.189,1.989,2.001,-1.0,0.0,0.0,0.0,-21.300,2.373,3.963,3.647,1.836,2.212,1.623,2.434,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T14:59:00+00:00,BTCUSDT,5m,4,-0.126,1.467,1.634,1.745,2.112,2.223,2.234,2.445,1.467,2.223,-2.112,-2.201,2.001,2.012,-1.0,0.0,0.0,0.0,-21.707,2.389,3.988,3.669,1.847,2.223,1.634,2.445,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:04:00+00:00,BTCUSDT,5m,4,-0.149,1.478,1.645,1.756,2.123,2.234,2.245,2.456,1.478,2.234,-2.123,-2.212,2.012,2.023,-1.0,0.0,0.0,0.0,-22.118,2.405,4.013,3.691,1.858,2.234,1.645,2.456,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:09:00+00:00,BTCUSDT,5m,4,-0.173,1.489,1.656,1.767,2.134,2.245,2.256,2.467,1.489,2.245,-2.134,-2.223,2.023,2.034,-1.0,0.0,0.0,0.0,-22.533,2.421,4.038,3.713,1.869,2.245,1.656,2.467,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:14:00+00:00,BTCUSDT,5m,4,-0.198,1.501,1.667,1.778,2.145,2.256,2.267,2.478,1.501,2.256,-2.145,-2.234,2.034,2.045,-1.0,0.0,0.0,0.0,-22.952,2.437,4.063,3.735,1.880,2.256,1.667,2.478,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:19:00+00:00,BTCUSDT,5m,4,-0.224,1.512,1.678,1.789,2.156,2.267,2.278,2.489,1.512,2.267,-2.156,-2.245,2.045,2.056,-1.0,0.0,0.0,0.0,-23.375,2.453,4.088,3.757,1.891,2.267,1.678,2.489,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:24:00+00:00,BTCUSDT,5m,4,-0.251,1.523,1.689,1.801,2.167,2.278,2.289,2.501,1.523,2.278,-2.167,-2.256,2.056,2.067,-1.0,0.0,0.0,0.0,-23.802,2.469,4.113,3.779,1.902,2.278,1.689,2.501,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:29:00+00:00,BTCUSDT,5m,4,-0.279,1.534,1.701,1.812,2.178,2.289,2.301,2.512,1.534,2.289,-2.178,-2.267,2.067,2.078,-1.0,0.0,0.0,0.0,-24.233,2.485,4.138,3.801,1.913,2.289,1.701,2.512,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:34:00+00:00,BTCUSDT,5m,4,-0.308,1.545,1.712,1.823,2.189,2.301,2.312,2.523,1.545,2.301,-2.189,-2.278,2.078,2.089,-1.0,0.0,0.0,0.0,-24.668,2.501,4.163,3.823,1.924,2.301,1.712,2.523,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:39:00+00:00,BTCUSDT,5m,4,-0.338,1.556,1.723,1.834,2.201,2.312,2.323,2.534,1.556,2.312,-2.201,-2.289,2.089,2.101,-1.0,0.0,0.0,0.0,-25.107,2.517,4.188,3.845,1.935,2.312,1.723,2.534,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:44:00+00:00,BTCUSDT,5m,4,-0.369,1.567,1.734,1.845,2.212,2.323,2.334,2.545,1.567,2.323,-2.212,-2.301,2.101,2.112,-1.0,0.0,0.0,0.0,-25.550,2.533,4.213,3.867,1.946,2.323,1.734,2.545,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:49:00+00:00,BTCUSDT,5m,4,-0.401,1.578,1.745,1.856,2.223,2.334,2.345,2.556,1.578,2.334,-2.223,-2.312,2.112,2.123,-1.0,0.0,0.0,0.0,-25.997,2.549,4.238,3.889,1.957,2.334,1.745,2.556,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:54:00+00:00,BTCUSDT,5m,4,-0.434,1.589,1.756,1.867,2.234,2.345,2.356,2.567,1.589,2.345,-2.234,-2.323,2.123,2.134,-1.0,0.0,0.0,0.0,-26.448,2.565,4.263,3.911,1.968,2.345,1.756,2.567,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T15:59:00+00:00,BTCUSDT,5m,4,-0.468,1.601,1.767,1.878,2.245,2.356,2.367,2.578,1.601,2.356,-2.245,-2.334,2.134,2.145,-1.0,0.0,0.0,0.0,-26.903,2.581,4.288,3.933,1.979,2.356,1.767,2.578,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:04:00+00:00,BTCUSDT,5m,4,-0.504,1.612,1.778,1.889,2.256,2.367,2.378,2.589,1.612,2.367,-2.256,-2.345,2.145,2.156,-1.0,0.0,0.0,0.0,-27.362,2.597,4.313,3.955,1.990,2.367,1.778,2.589,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:09:00+00:00,BTCUSDT,5m,4,-0.542,1.623,1.789,1.901,2.267,2.378,2.389,2.601,1.623,2.378,-2.267,-2.356,2.156,2.167,-1.0,0.0,0.0,0.0,-27.825,2.613,4.338,3.977,2.001,2.378,1.789,2.601,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:14:00+00:00,BTCUSDT,5m,4,-0.581,1.634,1.801,1.912,2.278,2.389,2.401,2.612,1.634,2.389,-2.278,-2.367,2.167,2.178,-1.0,0.0,0.0,0.0,-28.292,2.629,4.363,3.999,2.012,2.389,1.801,2.612,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:19:00+00:00,BTCUSDT,5m,4,-0.623,1.645,1.812,1.923,2.289,2.401,2.412,2.623,1.645,2.401,-2.289,-2.378,2.178,2.189,-1.0,0.0,0.0,0.0,-28.763,2.645,4.388,4.021,2.023,2.401,1.812,2.623,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:24:00+00:00,BTCUSDT,5m,4,-0.667,1.656,1.823,1.934,2.301,2.412,2.423,2.634,1.656,2.412,-2.301,-2.389,2.189,2.201,-1.0,0.0,0.0,0.0,-29.238,2.661,4.413,4.043,2.034,2.412,1.823,2.634,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:29:00+00:00,BTCUSDT,5m,4,-0.712,1.667,1.834,1.945,2.312,2.423,2.434,2.645,1.667,2.423,-2.312,-2.401,2.201,2.212,-1.0,0.0,0.0,0.0,-29.717,2.677,4.438,4.065,2.045,2.423,1.834,2.645,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:34:00+00:00,BTCUSDT,5m,4,-0.765,1.678,1.845,1.956,2.323,2.434,2.445,2.656,1.678,2.434,-2.323,-2.412,2.212,2.223,-1.0,0.0,0.0,0.0,-30.200,2.693,4.463,4.087,2.056,2.434,1.845,2.656,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:39:00+00:00,BTCUSDT,5m,4,-0.823,1.689,1.856,1.967,2.334,2.445,2.456,2.667,1.689,2.445,-2.334,-2.423,2.223,2.234,-1.0,0.0,0.0,0.0,-30.687,2.709,4.488,4.109,2.067,2.445,1.856,2.667,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:44:00+00:00,BTCUSDT,5m,4,-0.889,1.701,1.867,1.978,2.345,2.456,2.467,2.678,1.701,2.456,-2.345,-2.434,2.234,2.245,-1.0,0.0,0.0,0.0,-31.178,2.725,4.513,4.131,2.078,2.456,1.867,2.678,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:49:00+00:00,BTCUSDT,5m,4,-0.956,1.712,1.878,1.989,2.356,2.467,2.478,2.689,1.712,2.467,-2.356,-2.445,2.245,2.256,-1.0,0.0,0.0,0.0,-31.673,2.741,4.538,4.153,2.089,2.467,1.878,2.689,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:54:00+00:00,BTCUSDT,5m,4,-1.034,1.723,1.889,2.001,2.367,2.478,2.489,2.701,1.723,2.478,-2.367,-2.456,2.256,2.267,-1.0,0.0,0.0,0.0,-32.172,2.757,4.563,4.175,2.100,2.478,1.889,2.701,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T16:59:00+00:00,BTCUSDT,5m,4,-1.145,1.734,1.901,2.012,2.378,2.489,2.501,2.712,1.734,2.489,-2.378,-2.467,2.267,2.278,-1.0,0.0,0.0,0.0,-32.675,2.773,4.588,4.197,2.111,2.489,1.901,2.712,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:04:00+00:00,BTCUSDT,5m,4,-1.298,1.745,1.912,2.023,2.389,2.501,2.512,2.723,1.745,2.501,-2.389,-2.478,2.278,2.289,-1.0,0.0,0.0,0.0,-33.182,2.789,4.613,4.219,2.122,2.501,1.912,2.723,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:09:00+00:00,BTCUSDT,5m,4,-1.501,1.756,1.923,2.034,2.401,2.512,2.523,2.734,1.756,2.512,-2.401,-2.489,2.289,2.301,-1.0,0.0,0.0,0.0,-33.693,2.805,4.638,4.241,2.133,2.512,1.923,2.734,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:14:00+00:00,BTCUSDT,5m,4,-1.734,1.767,1.934,2.045,2.412,2.523,2.534,2.745,1.767,2.523,-2.412,-2.501,2.301,2.312,-1.0,0.0,0.0,0.0,-34.208,2.821,4.663,4.263,2.144,2.523,1.934,2.745,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:19:00+00:00,BTCUSDT,5m,4,-1.989,1.778,1.945,2.056,2.423,2.534,2.545,2.756,1.778,2.534,-2.423,-2.512,2.312,2.323,-1.0,0.0,0.0,0.0,-34.727,2.837,4.688,4.285,2.155,2.534,1.945,2.756,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:24:00+00:00,BTCUSDT,5m,4,-2.234,1.789,1.956,2.067,2.434,2.545,2.556,2.767,1.789,2.545,-2.434,-2.523,2.323,2.334,-1.0,0.0,0.0,0.0,-35.250,2.853,4.713,4.307,2.166,2.545,1.956,2.767,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:29:00+00:00,BTCUSDT,5m,4,-2.456,1.801,1.967,2.078,2.445,2.556,2.567,2.778,1.801,2.556,-2.445,-2.534,2.334,2.345,-1.0,0.0,0.0,0.0,-35.777,2.869,4.738,4.329,2.177,2.556,1.967,2.778,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:34:00+00:00,BTCUSDT,5m,4,-2.634,1.812,1.978,2.089,2.456,2.567,2.578,2.789,1.812,2.567,-2.456,-2.545,2.345,2.356,-1.0,0.0,0.0,0.0,-36.308,2.885,4.763,4.351,2.188,2.567,1.978,2.789,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:39:00+00:00,BTCUSDT,5m,4,-2.767,1.823,1.989,2.101,2.467,2.578,2.589,2.801,1.823,2.578,-2.467,-2.556,2.356,2.367,-1.0,0.0,0.0,0.0,-36.843,2.901,4.788,4.373,2.199,2.578,1.989,2.801,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:44:00+00:00,BTCUSDT,5m,4,-2.856,1.834,2.001,2.112,2.478,2.589,2.601,2.812,1.834,2.589,-2.478,-2.567,2.367,2.378,-1.0,0.0,0.0,0.0,-37.382,2.917,4.813,4.395,2.210,2.589,2.001,2.812,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:49:00+00:00,BTCUSDT,5m,4,-2.901,1.845,2.012,2.123,2.489,2.601,2.612,2.823,1.845,2.601,-2.489,-2.578,2.378,2.389,-1.0,0.0,0.0,0.0,-37.925,2.933,4.838,4.417,2.221,2.601,2.012,2.823,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:54:00+00:00,BTCUSDT,5m,4,-2.901,1.856,2.023,2.134,2.501,2.612,2.623,2.834,1.856,2.612,-2.501,-2.589,2.389,2.401,-1.0,0.0,0.0,0.0,-38.472,2.949,4.863,4.439,2.232,2.612,2.023,2.834,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T17:59:00+00:00,BTCUSDT,5m,4,-2.856,1.867,2.034,2.145,2.512,2.623,2.634,2.845,1.867,2.623,-2.512,-2.601,2.401,2.412,-1.0,0.0,0.0,0.0,-39.023,2.965,4.888,4.461,2.243,2.623,2.034,2.845,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:04:00+00:00,BTCUSDT,5m,4,-2.767,1.878,2.045,2.156,2.523,2.634,2.645,2.856,1.878,2.634,-2.523,-2.612,2.412,2.423,-1.0,0.0,0.0,0.0,-39.578,2.981,4.913,4.483,2.254,2.634,2.045,2.856,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:09:00+00:00,BTCUSDT,5m,4,-2.634,1.889,2.056,2.167,2.534,2.645,2.656,2.867,1.889,2.645,-2.534,-2.623,2.423,2.434,-1.0,0.0,0.0,0.0,-40.137,2.997,4.938,4.505,2.265,2.645,2.056,2.867,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:14:00+00:00,BTCUSDT,5m,4,-2.456,1.901,2.067,2.178,2.545,2.656,2.667,2.878,1.901,2.656,-2.545,-2.634,2.434,2.445,-1.0,0.0,0.0,0.0,-40.700,3.013,4.963,4.527,2.276,2.656,2.067,2.878,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:19:00+00:00,BTCUSDT,5m,4,-2.234,1.912,2.078,2.189,2.556,2.667,2.678,2.889,1.912,2.667,-2.556,-2.645,2.445,2.456,-1.0,0.0,0.0,0.0,-41.267,3.029,4.988,4.549,2.287,2.667,2.078,2.889,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:24:00+00:00,BTCUSDT,5m,4,-1.989,1.923,2.089,2.201,2.567,2.678,2.689,2.901,1.923,2.678,-2.567,-2.656,2.456,2.467,-1.0,0.0,0.0,0.0,-41.838,3.045,5.013,4.571,2.298,2.678,2.089,2.901,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:29:00+00:00,BTCUSDT,5m,4,-1.734,1.934,2.101,2.212,2.578,2.689,2.701,2.912,1.934,2.689,-2.578,-2.667,2.467,2.478,-1.0,0.0,0.0,0.0,-42.413,3.061,5.038,4.593,2.309,2.689,2.101,2.912,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:34:00+00:00,BTCUSDT,5m,4,-1.501,1.945,2.112,2.223,2.589,2.701,2.712,2.923,1.945,2.701,-2.589,-2.678,2.478,2.489,-1.0,0.0,0.0,0.0,-42.992,3.077,5.063,4.615,2.320,2.701,2.112,2.923,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:39:00+00:00,BTCUSDT,5m,4,-1.298,1.956,2.123,2.234,2.601,2.712,2.723,2.934,1.956,2.712,-2.601,-2.689,2.489,2.501,-1.0,0.0,0.0,0.0,-43.575,3.093,5.088,4.637,2.331,2.712,2.123,2.934,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:44:00+00:00,BTCUSDT,5m,4,-1.145,1.967,2.134,2.245,2.612,2.723,2.734,2.945,1.967,2.723,-2.612,-2.701,2.501,2.512,-1.0,0.0,0.0,0.0,-44.162,3.109,5.113,4.659,2.342,2.723,2.134,2.945,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:49:00+00:00,BTCUSDT,5m,4,-1.034,1.978,2.145,2.256,2.623,2.734,2.745,2.956,1.978,2.734,-2.623,-2.712,2.512,2.523,-1.0,0.0,0.0,0.0,-44.753,3.125,5.138,4.681,2.353,2.734,2.145,2.956,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:54:00+00:00,BTCUSDT,5m,4,-0.956,1.989,2.156,2.267,2.634,2.745,2.756,2.967,1.989,2.745,-2.634,-2.723,2.523,2.534,-1.0,0.0,0.0,0.0,-45.348,3.141,5.163,4.703,2.364,2.745,2.156,2.967,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T18:59:00+00:00,BTCUSDT,5m,4,-0.889,2.001,2.167,2.278,2.645,2.756,2.767,2.978,2.001,2.756,-2.645,-2.734,2.534,2.545,-1.0,0.0,0.0,0.0,-45.947,3.157,5.188,4.725,2.375,2.756,2.167,2.978,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:04:00+00:00,BTCUSDT,5m,4,-0.823,2.012,2.178,2.289,2.656,2.767,2.778,2.989,2.012,2.767,-2.656,-2.745,2.545,2.556,-1.0,0.0,0.0,0.0,-46.550,3.173,5.213,4.747,2.386,2.767,2.178,2.989,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:09:00+00:00,BTCUSDT,5m,4,-0.765,2.023,2.189,2.301,2.667,2.778,2.789,3.001,2.023,2.778,-2.667,-2.756,2.556,2.567,-1.0,0.0,0.0,0.0,-47.157,3.189,5.238,4.769,2.397,2.778,2.189,3.001,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:14:00+00:00,BTCUSDT,5m,4,-0.712,2.034,2.201,2.312,2.678,2.789,2.801,3.012,2.034,2.789,-2.678,-2.767,2.567,2.578,-1.0,0.0,0.0,0.0,-47.768,3.205,5.263,4.791,2.408,2.789,2.201,3.012,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:19:00+00:00,BTCUSDT,5m,4,-0.667,2.045,2.212,2.323,2.689,2.801,2.812,3.023,2.045,2.801,-2.689,-2.778,2.578,2.589,-1.0,0.0,0.0,0.0,-48.383,3.221,5.288,4.813,2.419,2.801,2.212,3.023,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:24:00+00:00,BTCUSDT,5m,4,-0.623,2.056,2.223,2.334,2.701,2.812,2.823,3.034,2.056,2.812,-2.701,-2.789,2.589,2.801,-1.0,0.0,0.0,0.0,-49.002,3.237,5.313,4.835,2.430,2.812,2.223,3.034,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:29:00+00:00,BTCUSDT,5m,4,-0.581,2.067,2.234,2.345,2.712,2.823,2.834,3.045,2.067,2.823,-2.712,-2.801,2.601,2.812,-1.0,0.0,0.0,0.0,-49.625,3.253,5.338,4.857,2.441,2.823,2.234,3.045,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:34:00+00:00,BTCUSDT,5m,4,-0.542,2.078,2.245,2.356,2.723,2.834,2.845,3.056,2.078,2.834,-2.723,-2.812,2.612,2.823,-1.0,0.0,0.0,0.0,-50.252,3.269,5.363,4.879,2.452,2.834,2.245,3.056,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:39:00+00:00,BTCUSDT,5m,4,-0.504,2.089,2.256,2.367,2.734,2.845,2.856,3.067,2.089,2.845,-2.734,-2.823,2.623,2.834,-1.0,0.0,0.0,0.0,-50.883,3.285,5.388,4.901,2.463,2.845,2.256,3.067,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:44:00+00:00,BTCUSDT,5m,4,-0.468,2.101,2.267,2.378,2.745,2.856,2.867,3.078,2.101,2.856,-2.745,-2.834,2.634,2.845,-1.0,0.0,0.0,0.0,-51.518,3.301,5.413,4.923,2.474,2.856,2.267,3.078,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:49:00+00:00,BTCUSDT,5m,4,-0.434,2.112,2.278,2.389,2.756,2.867,2.878,3.089,2.112,2.867,-2.756,-2.845,2.645,2.856,-1.0,0.0,0.0,0.0,-52.157,3.317,5.438,4.945,2.485,2.867,2.278,3.089,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:54:00+00:00,BTCUSDT,5m,4,-0.401,2.123,2.289,2.401,2.767,2.878,2.889,3.101,2.123,2.878,-2.767,-2.856,2.656,2.867,-1.0,0.0,0.0,0.0,-52.800,3.333,5.463,4.967,2.496,2.878,2.289,3.101,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T19:59:00+00:00,BTCUSDT,5m,4,-0.369,2.134,2.301,2.412,2.778,2.889,2.901,3.112,2.134,2.889,-2.778,-2.867,2.667,2.878,-1.0,0.0,0.0,0.0,-53.447,3.349,5.488,4.989,2.507,2.889,2.301,3.112,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:04:00+00:00,BTCUSDT,5m,4,-0.338,2.145,2.312,2.423,2.789,2.901,2.912,3.123,2.145,2.901,-2.789,-2.878,2.678,2.889,-1.0,0.0,0.0,0.0,-54.098,3.365,5.513,5.011,2.518,2.901,2.312,3.123,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:09:00+00:00,BTCUSDT,5m,4,-0.308,2.156,2.323,2.434,2.801,2.912,2.923,3.134,2.156,2.912,-2.801,-2.889,2.689,2.901,-1.0,0.0,0.0,0.0,-54.753,3.381,5.538,5.033,2.529,2.912,2.323,3.134,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:14:00+00:00,BTCUSDT,5m,4,-0.279,2.167,2.334,2.445,2.812,2.923,2.934,3.145,2.167,2.923,-2.812,-2.901,2.701,2.912,-1.0,0.0,0.0,0.0,-55.412,3.397,5.563,5.055,2.540,2.923,2.334,3.145,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:19:00+00:00,BTCUSDT,5m,4,-0.251,2.178,2.345,2.456,2.823,2.934,2.945,3.156,2.178,2.934,-2.823,-2.912,2.712,2.923,-1.0,0.0,0.0,0.0,-56.075,3.413,5.588,5.077,2.551,2.934,2.345,3.156,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:24:00+00:00,BTCUSDT,5m,4,-0.224,2.189,2.356,2.467,2.834,2.945,2.956,3.167,2.189,2.945,-2.834,-2.923,2.723,2.834,-1.0,0.0,0.0,0.0,-56.742,3.429,5.613,5.099,2.562,2.945,2.356,3.167,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:29:00+00:00,BTCUSDT,5m,4,-0.198,2.201,2.367,2.478,2.845,2.956,2.967,3.178,2.201,2.956,-2.845,-2.934,2.734,2.845,-1.0,0.0,0.0,0.0,-57.413,3.445,5.638,5.121,2.573,2.956,2.367,3.178,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:34:00+00:00,BTCUSDT,5m,4,-0.173,2.212,2.378,2.489,2.856,2.967,2.978,3.189,2.212,2.967,-2.856,-2.945,2.745,2.856,-1.0,0.0,0.0,0.0,-58.088,3.461,5.663,5.143,2.584,2.967,2.378,3.189,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:39:00+00:00,BTCUSDT,5m,4,-0.149,2.223,2.389,2.501,2.867,2.978,2.989,3.201,2.223,2.978,-2.867,-2.956,2.756,2.867,-1.0,0.0,0.0,0.0,-58.767,3.477,5.688,5.165,2.595,2.978,2.389,3.201,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:44:00+00:00,BTCUSDT,5m,4,-0.126,2.234,2.401,2.512,2.878,2.989,3.001,3.212,2.234,2.989,-2.878,-2.967,2.767,2.878,-1.0,0.0,0.0,0.0,-59.450,3.493,5.713,5.187,2.606,2.989,2.401,3.212,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:49:00+00:00,BTCUSDT,5m,4,-0.104,2.245,2.412,2.523,2.889,3.001,3.012,3.223,2.245,3.001,-2.889,-2.978,2.778,2.889,-1.0,0.0,0.0,0.0,-60.137,3.509,5.738,5.209,2.617,3.001,2.412,3.223,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:54:00+00:00,BTCUSDT,5m,4,-0.083,2.256,2.423,2.534,2.901,3.012,3.023,3.234,2.256,3.012,-2.901,-2.989,2.789,2.901,-1.0,0.0,0.0,0.0,-60.828,3.525,5.763,5.231,2.628,3.012,2.423,3.234,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T20:59:00+00:00,BTCUSDT,5m,4,-0.063,2.267,2.434,2.545,2.912,3.023,3.034,3.245,2.267,3.023,-2.912,-3.001,2.801,2.912,-1.0,0.0,0.0,0.0,-61.523,3.541,5.788,5.253,2.639,3.023,2.434,3.245,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:04:00+00:00,BTCUSDT,5m,4,-0.044,2.278,2.445,2.556,2.923,3.034,3.045,3.256,2.278,3.034,-2.923,-3.012,2.812,2.923,-1.0,0.0,0.0,0.0,-62.222,3.557,5.813,5.275,2.650,3.034,2.445,3.256,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:09:00+00:00,BTCUSDT,5m,4,-0.026,2.289,2.456,2.567,2.934,3.045,3.056,3.267,2.289,3.045,-2.934,-3.023,2.823,2.834,-1.0,0.0,0.0,0.0,-62.925,3.573,5.838,5.297,2.661,3.045,2.456,3.267,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:14:00+00:00,BTCUSDT,5m,4,-0.009,2.301,2.467,2.578,2.945,3.056,3.067,3.278,2.301,3.056,-2.945,-3.034,2.834,2.845,-1.0,0.0,0.0,0.0,-63.632,3.589,5.863,5.319,2.672,3.056,2.467,3.278,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:19:00+00:00,BTCUSDT,5m,4,0.007,2.312,2.478,2.589,2.956,3.067,3.078,3.289,2.312,3.067,-2.956,-3.045,2.845,2.856,-1.0,0.0,0.0,0.0,-64.343,3.605,5.888,5.341,2.683,3.067,2.478,3.289,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:24:00+00:00,BTCUSDT,5m,4,0.022,2.323,2.489,2.601,2.967,3.078,3.089,3.301,2.323,3.078,-2.967,-3.056,2.856,2.867,-1.0,0.0,0.0,0.0,-65.058,3.621,5.913,5.363,2.694,3.078,2.489,3.301,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:29:00+00:00,BTCUSDT,5m,4,0.036,2.334,2.501,2.612,2.978,3.089,3.101,3.312,2.334,3.089,-2.978,-3.067,2.867,2.878,-1.0,0.0,0.0,0.0,-65.777,3.637,5.938,5.385,2.705,3.089,2.501,3.312,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:34:00+00:00,BTCUSDT,5m,4,0.049,2.345,2.512,2.623,2.989,3.101,3.112,3.323,2.345,3.101,-2.989,-3.078,2.878,2.889,-1.0,0.0,0.0,0.0,-66.500,3.653,5.963,5.407,2.716,3.101,2.512,3.323,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:39:00+00:00,BTCUSDT,5m,4,0.061,2.356,2.523,2.634,3.001,3.112,3.123,3.334,2.356,3.112,-3.001,-3.089,2.889,2.901,-1.0,0.0,0.0,0.0,-67.227,3.669,5.988,5.429,2.727,3.112,2.523,3.334,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:44:00+00:00,BTCUSDT,5m,4,0.072,2.367,2.534,2.645,3.012,3.123,3.134,3.345,2.367,3.123,-3.012,-3.101,2.901,2.912,-1.0,0.0,0.0,0.0,-67.958,3.685,6.013,5.451,2.738,3.123,2.534,3.345,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:49:00+00:00,BTCUSDT,5m,4,0.082,2.378,2.545,2.656,3.023,3.134,3.145,3.356,2.378,3.134,-3.023,-3.112,2.912,2.923,-1.0,0.0,0.0,0.0,-68.693,3.701,6.038,5.473,2.749,3.134,2.545,3.356,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:54:00+00:00,BTCUSDT,5m,4,0.091,2.389,2.556,2.667,3.034,3.145,3.156,3.367,2.389,3.145,-3.034,-3.123,2.923,2.934,-1.0,0.0,0.0,0.0,-69.432,3.717,6.063,5.495,2.760,3.145,2.556,3.367,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T21:59:00+00:00,BTCUSDT,5m,4,0.099,2.401,2.567,2.678,3.045,3.156,3.167,3.378,2.401,3.156,-3.045,-3.134,2.934,2.945,-1.0,0.0,0.0,0.0,-70.175,3.733,6.088,5.517,2.771,3.156,2.567,3.378,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:04:00+00:00,BTCUSDT,5m,4,0.106,2.412,2.578,2.689,3.056,3.167,3.178,3.389,2.412,3.167,-3.056,-3.145,2.945,2.956,-1.0,0.0,0.0,0.0,-70.922,3.749,6.113,5.539,2.782,3.167,2.578,3.389,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:09:00+00:00,BTCUSDT,5m,4,0.112,2.423,2.589,2.701,3.067,3.178,3.189,3.401,2.423,3.178,-3.067,-3.156,2.956,2.967,-1.0,0.0,0.0,0.0,-71.673,3.765,6.138,5.561,2.793,3.178,2.589,3.401,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:14:00+00:00,BTCUSDT,5m,4,0.117,2.434,2.601,2.712,3.078,3.189,3.201,3.412,2.434,3.189,-3.078,-3.167,2.967,2.978,-1.0,0.0,0.0,0.0,-72.428,3.781,6.163,5.583,2.804,3.189,2.601,3.412,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:19:00+00:00,BTCUSDT,5m,4,0.121,2.445,2.612,2.723,3.089,3.201,3.212,3.423,2.445,3.201,-3.089,-3.178,2.978,2.989,-1.0,0.0,0.0,0.0,-73.187,3.797,6.188,5.605,2.815,3.201,2.612,3.423,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:24:00+00:00,BTCUSDT,5m,4,0.124,2.456,2.623,2.734,3.101,3.212,3.223,3.434,2.456,3.212,-3.101,-3.189,2.989,3.001,-1.0,0.0,0.0,0.0,-73.950,3.813,6.213,5.627,2.826,3.212,2.623,3.434,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:29:00+00:00,BTCUSDT,5m,4,0.126,2.467,2.634,2.745,3.112,3.223,3.234,3.445,2.467,3.223,-3.112,-3.201,3.001,3.012,-1.0,0.0,0.0,0.0,-74.717,3.829,6.238,5.649,2.837,3.223,2.634,3.445,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:34:00+00:00,BTCUSDT,5m,4,0.127,2.478,2.645,2.756,3.123,3.234,3.245,3.456,2.478,3.234,-3.123,-3.212,3.012,3.023,-1.0,0.0,0.0,0.0,-75.488,3.845,6.263,5.671,2.848,3.234,2.645,3.456,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:39:00+00:00,BTCUSDT,5m,4,0.127,2.489,2.656,2.767,3.134,3.245,3.256,3.467,2.489,3.245,-3.134,-3.223,3.023,3.034,-1.0,0.0,0.0,0.0,-76.263,3.861,6.288,5.693,2.859,3.245,2.656,3.467,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:44:00+00:00,BTCUSDT,5m,4,0.126,2.501,2.667,2.778,3.145,3.256,3.267,3.478,2.501,3.256,-3.145,-3.234,3.034,3.045,-1.0,0.0,0.0,0.0,-77.042,3.877,6.313,5.715,2.870,3.256,2.667,3.478,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:49:00+00:00,BTCUSDT,5m,4,0.124,2.512,2.678,2.789,3.156,3.267,3.278,3.489,2.512,3.267,-3.156,-3.245,3.045,3.056,-1.0,0.0,0.0,0.0,-77.825,3.893,6.338,5.737,2.881,3.267,2.678,3.489,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:54:00+00:00,BTCUSDT,5m,4,0.121,2.523,2.689,2.801,3.167,3.278,3.289,3.501,2.523,3.278,-3.167,-3.256,3.056,3.067,-1.0,0.0,0.0,0.0,-78.612,3.909,6.363,5.759,2.892,3.278,2.689,3.501,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T22:59:00+00:00,BTCUSDT,5m,4,0.117,2.534,2.701,2.812,3.178,3.289,3.301,3.512,2.534,3.289,-3.178,-3.267,3.067,3.078,-1.0,0.0,0.0,0.0,-79.403,3.925,6.388,5.781,2.903,3.289,2.701,3.512,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:04:00+00:00,BTCUSDT,5m,4,0.112,2.545,2.712,2.823,3.189,3.301,3.312,3.523,2.545,3.301,-3.189,-3.278,3.078,3.089,-1.0,0.0,0.0,0.0,-80.198,3.941,6.413,5.803,2.914,3.301,2.712,3.523,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:09:00+00:00,BTCUSDT,5m,4,0.106,2.556,2.723,2.834,3.201,3.312,3.323,3.534,2.556,3.312,-3.201,-3.289,3.089,3.101,-1.0,0.0,0.0,0.0,-80.997,3.957,6.438,5.825,2.925,3.312,2.723,3.534,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:14:00+00:00,BTCUSDT,5m,4,0.099,2.567,2.734,2.845,3.212,3.323,3.334,3.545,2.567,3.323,-3.212,-3.301,3.101,3.112,-1.0,0.0,0.0,0.0,-81.800,3.973,6.463,5.847,2.936,3.323,2.734,3.545,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:19:00+00:00,BTCUSDT,5m,4,0.091,2.578,2.745,2.856,3.223,3.334,3.345,3.556,2.578,3.334,-3.223,-3.312,3.112,3.123,-1.0,0.0,0.0,0.0,-82.607,3.989,6.488,5.869,2.947,3.334,2.745,3.556,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:24:00+00:00,BTCUSDT,5m,4,0.082,2.589,2.756,2.867,3.234,3.345,3.356,3.567,2.589,3.345,-3.234,-3.323,3.123,3.134,-1.0,0.0,0.0,0.0,-83.418,4.005,6.513,5.891,2.958,3.345,2.756,3.567,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:29:00+00:00,BTCUSDT,5m,4,0.072,2.601,2.767,2.878,3.245,3.356,3.367,3.578,2.601,3.356,-3.245,-3.334,3.134,3.145,-1.0,0.0,0.0,0.0,-84.233,4.021,6.538,5.913,2.969,3.356,2.767,3.578,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:34:00+00:00,BTCUSDT,5m,4,0.061,2.612,2.778,3.089,3.256,3.367,3.378,3.589,2.612,3.367,-3.256,-3.345,3.145,3.156,-1.0,0.0,0.0,0.0,-85.052,4.037,6.563,5.935,2.980,3.367,2.778,3.589,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:39:00+00:00,BTCUSDT,5m,4,0.049,2.623,2.789,3.101,3.267,3.378,3.389,3.601,2.623,3.378,-3.267,-3.356,3.156,3.167,-1.0,0.0,0.0,0.0,-85.874,4.053,6.588,5.957,2.991,3.378,2.789,3.601,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:44:00+00:00,BTCUSDT,5m,4,0.036,2.634,2.801,3.112,3.278,3.389,3.401,3.612,2.634,3.389,-3.278,-3.367,3.167,3.178,-1.0,0.0,0.0,0.0,-86.700,4.069,6.613,5.979,3.002,3.389,2.801,3.612,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:49:00+00:00,BTCUSDT,5m,4,0.022,2.645,2.812,3.123,3.289,3.401,3.412,3.623,2.645,3.401,-3.289,-3.378,3.178,3.189,-1.0,0.0,0.0,0.0,-87.529,4.085,6.638,6.001,3.013,3.401,2.812,3.623,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:54:00+00:00,BTCUSDT,5m,4,0.007,2.656,2.823,3.134,3.301,3.412,3.423,3.634,2.656,3.412,-3.301,-3.389,3.189,3.201,-1.0,0.0,0.0,0.0,-88.362,4.101,6.663,6.023,3.024,3.412,2.823,3.634,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
2024-05-23T23:59:00+00:00,BTCUSDT,5m,4,-0.009,2.667,2.834,3.145,3.312,3.423,3.434,3.645,2.667,3.423,-3.312,-3.401,3.201,3.212,-1.0,0.0,0.0,0.0,-89.198,4.117,6.688,6.045,3.035,3.423,2.834,3.645,TREND_DOWN,4,VOL_NORMAL,1,Warmup,4,RANGE,2
```

---

## References

### Code Files

- `feature_engine/ohlcv_indicators.py` - EMA, ADX, z-score calculations
- `feature_engine/regime_metrics.py` - OHLCV metrics (realized vol, ATR, Hurst, etc.)
- `labeling/regime_classifier.py` - Regime classification logic
- `feature_engine/regime_lgbm_features.py` - LGBM feature assembly

### Configuration Files

- `config/regime_config.py` - Regime decision table, thresholds
- `config/regime_thresholds.py` - Threshold values for regime classification

---

*Document generated by NARRUX Regime Classification System*
*Version 1.0 - 2026-06-02*
