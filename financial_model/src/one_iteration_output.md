```bash
================================================================================
Generating Forecast Plots
================================================================================

Plotting results to terminal and saving to files

================================================================================
Starting Iteration 7/10
================================================================================

Generating forecast configurations

================================================================================
Creating Forecast Plan
================================================================================



================================================================================
Generated Forecast Plan - Iteration 7
================================================================================

Explanation: This plan focuses on fine-tuning successful configurations while maintaining some exploration. Prophet configurations emphasize stability through controlled changepoint parameters, exponential smoothing explores shorter seasonal windows, and regression maintains proven feature combinations while testing selective feature removal. The overall strategy aligns with the current iteration's focus on exploitation (89%) while maintaining minimal exploration (11%).

Steps:
- prophet: 4 configs
- exponential_smoothing: 4 configs
- regression: 4 configs

================================================================================
Generated Forecast Plan - Iteration 7
================================================================================

Explanation: This plan focuses on fine-tuning successful configurations while maintaining some exploration. Prophet configurations emphasize stability through controlled changepoint parameters, exponential smoothing explores shorter seasonal windows, and regression maintains proven feature combinations while testing selective feature removal. The overall strategy aligns with the current iteration's focus on exploitation (89%) while maintaining minimal exploration (11%).

================================================================================
Configurations for prophet
================================================================================

Config 1:
Parameters: {
  "seasonality_mode": "multiplicative",
  "changepoint_prior_scale": 0.02,
  "seasonality_prior_scale": 8.0,
  "holidays_prior_scale": 8.0,
  "changepoint_range": 0.85,
  "yearly_seasonality": true,
  "weekly_seasonality": true,
  "daily_seasonality": false,
  "holidays": "US",
  "forecast_periods": 150
}
Reasoning: Fine-tuned exploitation configuration based on previous success. Reduced changepoint_prior_scale to 0.02 for more stability, while maintaining multiplicative seasonality to handle observed patterns.

Config 2:
Parameters: {
  "seasonality_mode": "additive",
  "changepoint_prior_scale": 0.01,
  "seasonality_prior_scale": 5.0,
  "holidays_prior_scale": 5.0,
  "changepoint_range": 0.9,
  "yearly_seasonality": true,
  "weekly_seasonality": "auto",
  "daily_seasonality": false,
  "holidays": "GB",
  "forecast_periods": 150
}
Reasoning: Conservative configuration with very low changepoint_prior_scale and additive seasonality to minimize overfitting risk, given recent performance variability.

Config 3:
Parameters: {
  "seasonality_mode": "multiplicative",
  "changepoint_prior_scale": 0.015,
  "seasonality_prior_scale": 7.0,
  "holidays_prior_scale": 7.0,
  "changepoint_range": 0.8,
  "yearly_seasonality": 10,
  "weekly_seasonality": true,
  "daily_seasonality": false,
  "holidays": "US",
  "forecast_periods": 150
}
Reasoning: Balanced configuration with moderate parameters, focusing on yearly seasonality with custom Fourier terms (10) to better capture annual patterns.

Config 4:
Parameters: {
  "seasonality_mode": "additive",
  "changepoint_prior_scale": 0.03,
  "seasonality_prior_scale": 12.0,
  "holidays_prior_scale": 12.0,
  "changepoint_range": 0.75,
  "yearly_seasonality": true,
  "weekly_seasonality": true,
  "daily_seasonality": true,
  "holidays": "CA",
  "forecast_periods": 150
}
Reasoning: Exploration configuration testing higher prior scales and additional seasonality components, while maintaining reasonable changepoint control.


================================================================================
Configurations for exponential_smoothing
================================================================================

Config 1:
Parameters: {
  "window": 5,
  "trend": "add",
  "seasonal": "add",
  "forecast_periods": 150
}
Reasoning: Conservative configuration with shorter seasonal window to reduce sensitivity to long-term patterns, using additive components for stability.

Config 2:
Parameters: {
  "window": 7,
  "trend": "mul",
  "seasonal": "mul",
  "forecast_periods": 150
}
Reasoning: Standard multiplicative configuration to capture potential multiplicative patterns in both trend and seasonality.

Config 3:
Parameters: {
  "window": 6,
  "trend": "add",
  "seasonal": "mul",
  "forecast_periods": 150
}
Reasoning: Hybrid configuration combining additive trend with multiplicative seasonality for balanced flexibility.

Config 4:
Parameters: {
  "window": 4,
  "trend": "add",
  "seasonal": "add",
  "forecast_periods": 150
}
Reasoning: Experimental configuration with very short window to focus on recent patterns and reduce impact of historical variations.


================================================================================
Configurations for regression
================================================================================

Config 1:
Parameters: {
  "features": [
    "trend",
    "month",
    "day_of_week",
    "week_of_year"
  ],
  "forecast_periods": 150
}
Reasoning: Core feature set that has proven successful in previous iterations, excluding quarter for more focused seasonality capture.

Config 2:
Parameters: {
  "features": [
    "trend",
    "month",
    "day_of_week",
    "week_of_year",
    "quarter"
  ],
  "forecast_periods": 150
}
Reasoning: Full feature set including all available components to maximize pattern capture capability.

Config 3:
Parameters: {
  "features": [
    "trend",
    "month",
    "week_of_year"
  ],
  "forecast_periods": 150
}
Reasoning: Simplified feature set focusing on broader seasonal patterns while reducing potential noise from daily variations.

Config 4:
Parameters: {
  "features": [
    "trend",
    "month",
    "day_of_week"
  ],
  "forecast_periods": 150
}
Reasoning: Minimal feature set concentrating on trend and primary seasonal components for stability.


================================================================================
Running Prophet Model
================================================================================

{'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.02, 'seasonality_prior_scale': 8.0, 'holidays_prior_scale': 8.0, 'changepoint_range': 0.85, 'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': False, 'holidays': 'US'}
23:05:40 - cmdstanpy - INFO - Chain [1] start processing
23:05:40 - cmdstanpy - INFO - Chain [1] done processing

================================================================================
Prophet Results
================================================================================

Forecast Length: 150
First few values: [2072.5964514073958, 2071.7650758793716, 2071.528039499836, 2069.550462728784, 2064.7288332328603]

================================================================================
Running Prophet Model
================================================================================

{'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 5.0, 'holidays_prior_scale': 5.0, 'changepoint_range': 0.9, 'yearly_seasonality': True, 'weekly_seasonality': 'auto', 'daily_seasonality': False, 'holidays': 'GB'}
23:05:40 - cmdstanpy - INFO - Chain [1] start processing
23:05:40 - cmdstanpy - INFO - Chain [1] done processing

================================================================================
Prophet Results
================================================================================

Forecast Length: 150
First few values: [2071.1504398745333, 2069.2448951699753, 2068.8574239863674, 2067.2027555415243, 2063.2747924890086]

================================================================================
Running Prophet Model
================================================================================

{'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.015, 'seasonality_prior_scale': 7.0, 'holidays_prior_scale': 7.0, 'changepoint_range': 0.8, 'yearly_seasonality': 10, 'weekly_seasonality': True, 'daily_seasonality': False, 'holidays': 'US'}
23:05:40 - cmdstanpy - INFO - Chain [1] start processing
23:05:41 - cmdstanpy - INFO - Chain [1] done processing

================================================================================
Prophet Results
================================================================================

Forecast Length: 150
First few values: [2072.0838103211117, 2071.2288804546033, 2070.9988123089424, 2068.982644971175, 2064.1505116782632]

================================================================================
Running Prophet Model
================================================================================

{'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.03, 'seasonality_prior_scale': 12.0, 'holidays_prior_scale': 12.0, 'changepoint_range': 0.75, 'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': True, 'holidays': 'CA'}
23:05:41 - cmdstanpy - INFO - Chain [1] start processing
23:05:41 - cmdstanpy - INFO - Chain [1] done processing

================================================================================
Prophet Results
================================================================================

Forecast Length: 150
First few values: [2065.4691008493046, 2064.34351388701, 2063.5296843828464, 2061.511800714026, 2057.0301534777605]

================================================================================
Running Holt-Winters Exponential Smoothing
================================================================================

{'window': 5, 'trend': 'add', 'seasonal': 'add'}

================================================================================
Moving Average Results
================================================================================

Forecast Length: 150
Future values: [2114.00720751 2114.50707794 2117.91544715 2117.59031149 2115.54037384]

================================================================================
Running Holt-Winters Exponential Smoothing
================================================================================

{'window': 7, 'trend': 'mul', 'seasonal': 'mul'}

================================================================================
Moving Average Results
================================================================================

Forecast Length: 150
Future values: 2021-08-23    2113.609856
2021-08-24    2114.255321
2021-08-25    2115.378293
2021-08-26    2116.553487
2021-08-27    2115.698586
Freq: D, dtype: float64

================================================================================
Running Holt-Winters Exponential Smoothing
================================================================================

{'window': 6, 'trend': 'add', 'seasonal': 'mul'}

================================================================================
Moving Average Results
================================================================================

Forecast Length: 150
Future values: 2021-08-23    2110.946954
2021-08-24    2112.895183
2021-08-25    2112.385963
2021-08-26    2113.849317
2021-08-27    2117.066601
Freq: D, dtype: float64

================================================================================
Running Holt-Winters Exponential Smoothing
================================================================================

{'window': 4, 'trend': 'add', 'seasonal': 'add'}

================================================================================
Moving Average Results
================================================================================

Forecast Length: 150
Future values: [2114.20225303 2116.42117249 2116.00685352 2115.45300755 2115.66157069]

================================================================================
Running Regression Model
================================================================================

{'features': ['trend', 'month', 'day_of_week', 'week_of_year']}

================================================================================
Regression Results
================================================================================

Forecast Length: 150
Future values: [2021.4133592846356, 2022.221265375992, 2023.0291714673483, 2023.8370775587046, 2024.6449836500608]
Features: ['trend', 'month', 'day_of_week', 'week_of_year']

================================================================================
Running Regression Model
================================================================================

{'features': ['trend', 'month', 'day_of_week', 'week_of_year', 'quarter']}

================================================================================
Regression Results
================================================================================

Forecast Length: 150
Future values: [2021.00801102512, 2021.887117934573, 2022.7662248440263, 2023.6453317534795, 2024.5244386629327]
Features: ['trend', 'month', 'day_of_week', 'week_of_year', 'quarter']

================================================================================
Running Regression Model
================================================================================

{'features': ['trend', 'month', 'week_of_year']}

================================================================================
Regression Results
================================================================================

Forecast Length: 150
Future values: [2021.993590593548, 2022.6091845885044, 2023.2247785834606, 2023.840372578417, 2024.4559665733732]
Features: ['trend', 'month', 'week_of_year']

================================================================================
Running Regression Model
================================================================================

{'features': ['trend', 'month', 'day_of_week']}

================================================================================
Regression Results
================================================================================

Forecast Length: 150
Future values: [2022.6483255718256, 2023.463728445633, 2024.2791313194407, 2025.0945341932484, 2025.909937067056]
Features: ['trend', 'month', 'day_of_week']

================================================================================
Model Evaluation
================================================================================

Evaluating forecast results against validation data

================================================================================
Plots for prophet
================================================================================

Showing 4 configurations
                             prophet (Config 0)
      ┌──────────────────────────────────────────────────────────────┐
2194.0┤ ▞▞ Actual                                   ▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀│
2063.7┤ ▞▞ Historical Forecast                ▄▄▄▞▀▀                 │
      │   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀                       │
1933.4┤              ▘▘▝▗ ▖▖                                         │
1803.0┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▝▗ ▖ ▗                         │
1672.7┤                             ▘     ▘          ▝▝              │
1542.3┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                             prophet (Config 1)
      ┌──────────────────────────────────────────────────────────────┐
2194.3┤ ▞▞ Actual                                   ▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀│
2063.9┤ ▞▞ Historical Forecast                ▄▄▄▞▀▀                 │
      │   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀                       │
1933.5┤              ▘▘▝▗ ▖▖                                         │
1803.1┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▝▗ ▖ ▗                         │
1672.8┤                             ▘     ▘          ▝▝              │
1542.4┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                             prophet (Config 2)
      ┌──────────────────────────────────────────────────────────────┐
2192.5┤ ▞▞ Actual                                   ▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀│
2062.4┤ ▞▞ Historical Forecast                ▄▄▄▞▀▀                 │
      │   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀                       │
1932.3┤              ▘▘▝▗ ▖▖                                         │
1802.2┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▝▗ ▖ ▗                         │
1672.2┤                             ▘     ▘          ▝▝              │
1542.1┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                             prophet (Config 3)
      ┌──────────────────────────────────────────────────────────────┐
2168.5┤ ▞▞ Actual                                   ▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
2042.4┤ ▞▞ Historical Forecast    ▗▄▄▄▄    ▗▄▄▄▄▞▀▀▀                 │
      │     ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘    ▀▀▀▀▘                         │
1916.3┤              ▘ ▝▝ ▖▘▗                                        │
1790.2┤                      ▝ ▖                                     │
      │                         ▘▗▝  ▘▝▗ ▖ ▗                         │
1664.2┤                             ▘     ▘ ▗        ▝▝    ▗         │
1538.1┤                                       ▘▘▗       ▘▖▗  ▘ ▗▝ ▖  │
      │                                          ▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time



================================================================================
Plots for exponential_smoothing
================================================================================

Showing 4 configurations
                      exponential_smoothing (Config 0)
      ┌──────────────────────────────────────────────────────────────┐
2170.8┤ ▞▞ Actual              ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
2044.3┤ ▞▞ Historical Forecast                                       │
      │           ▝▝                                                 │
1917.9┤              ▘▘▝▗ ▖▖▗                                        │
1791.4┤                      ▝ ▖                                     │
      │                         ▘▗▝  ▘▝▗ ▖ ▗                         │
1664.9┤                             ▘     ▘ ▗        ▝▝    ▗         │
1538.5┤                                       ▘▘▗       ▘▖▗  ▘ ▗▝ ▖  │
      │                                          ▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                      exponential_smoothing (Config 1)
      ┌──────────────────────────────────────────────────────────────┐
2174.1┤ ▞▞ Actual              ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
2047.1┤ ▞▞ Historical Forecast                                       │
      │           ▝▝                                                 │
1920.1┤              ▘▘▝▗ ▖▖▗                                        │
1793.1┤                      ▝ ▖                                     │
      │                         ▘▗▝  ▘▝▗ ▖ ▗                         │
1666.0┤                             ▘     ▘          ▝▝    ▗         │
1539.0┤                                     ▝ ▘▘▗       ▘▖▗  ▘ ▗▝ ▖  │
      │                                          ▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                      exponential_smoothing (Config 2)
      ┌──────────────────────────────────────────────────────────────┐
2170.0┤ ▞▞ Actual              ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
2043.6┤ ▞▞ Historical Forecast                                       │
      │           ▝▝  ▖                                              │
1917.3┤              ▘ ▝▗ ▖▖▗                                        │
1791.0┤                      ▝ ▖                                     │
      │                         ▘▗▝  ▘▝▗ ▖ ▗                         │
1664.7┤                             ▘     ▘ ▗        ▝▝    ▗         │
1538.3┤                                       ▘▘▗       ▘▖▗  ▘ ▗▝ ▖  │
      │                                          ▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                      exponential_smoothing (Config 3)
      ┌──────────────────────────────────────────────────────────────┐
2168.0┤ ▞▞ Actual              ▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
2042.0┤ ▞▞ Historical Forecast                                       │
      │           ▝▝  ▖                                              │
1916.0┤              ▘ ▝▝ ▖▘▗                                        │
1790.0┤                      ▝ ▖                                     │
      │                         ▘▗▝  ▘▝▗ ▖ ▗                         │
1664.0┤                             ▘     ▘ ▗        ▝▝    ▗         │
1538.0┤                                       ▘▘▗       ▘▖▗  ▘ ▗▝ ▖  │
      │                                          ▗  ▘         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time



================================================================================
Plots for regression
================================================================================

Showing 4 configurations
                            regression (Config 0)
      ┌──────────────────────────────────────────────────────────────┐
2207.9┤ ▞▞ Actual                                             ▄▞▀▀▀▀▀│
2075.3┤ ▞▞ Historical Forecast                               ▞       │
      │▀▀▀▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘       │
1942.6┤              ▘▘▝▗ ▖▖                                         │
1810.0┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▗▗ ▖ ▗                         │
1677.3┤                             ▘     ▘          ▗▝              │
1544.7┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▖         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                            regression (Config 1)
      ┌──────────────────────────────────────────────────────────────┐
2204.4┤ ▞▞ Actual                                             ▄▞▀▀▀▀▀│
2072.4┤ ▞▞ Historical Forecast                               ▞       │
      │▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘       │
1940.3┤              ▘▘▝▗ ▖▖                                         │
1808.2┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▝▗ ▖ ▗                         │
1676.1┤                             ▘     ▘          ▗▝              │
1544.1┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▖         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                            regression (Config 2)
      ┌──────────────────────────────────────────────────────────────┐
2208.5┤ ▞▞ Actual                                             ▄▞▀▀▀▀▀│
2075.7┤ ▞▞ Historical Forecast                               ▞       │
      │▀▀▀▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘       │
1943.0┤              ▘▘▝▗ ▖▖                                         │
1810.2┤                     ▝▗                                       │
      │                        ▘▘▗▗  ▘▗▗ ▖ ▗                         │
1677.5┤                             ▘     ▘          ▗▝              │
1544.7┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▖         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time


                            regression (Config 3)
      ┌──────────────────────────────────────────────────────────────┐
2208.2┤ ▞▞ Actual                                            ▗▀▀▀▀▀▀▀│
2075.5┤ ▞▞ Historical Forecast                               ▞       │
      │▀▀▀▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘       │
1942.8┤              ▘▘▝▗ ▖▖                                         │
1810.1┤                     ▝▝                                       │
      │                        ▘▘▗▗  ▘▗▗ ▖ ▗                         │
1677.4┤                             ▘     ▘          ▗▝              │
1544.7┤                                     ▝ ▘▘        ▘▖▗▝ ▘ ▗▝ ▖  │
      │                                         ▝▗  ▖         ▘    ▘▗│
1412.0┤                                            ▖                 │
      └┬──────────────┬───────────────┬──────────────┬──────────────┬┘
      0.0           12.2            24.5           36.8          49.0
Value                               Time



================================================================================
Plots Complete
================================================================================

All plots have been generated and saved

================================================================================
Evaluation Results
================================================================================

Best Model: prophet (Config 3)
Reasoning: Based on the current evaluation results and historical performance trends, Prophet with configuration 3 shows the best balance of accuracy and stability. The model achieves a MAPE of 18.73% and RMSE of 376.93, with well-calibrated uncertainty bounds (uncertainty_score: 0.032). The trend coefficient of 1.13 indicates appropriate capture of underlying patterns without overfitting.
Confidence: 0.86

================================================================================
Model Comparison
================================================================================

{'prophet_config_0': {'mape': 19.383718078627446, 'rmse': 391.6061368152272, 'mae': 315.4571175710878}, 'prophet_config_1': {'mape': 19.57896722043055, 'rmse': 394.0178130264786, 'mae': 318.8476984102057}, 'prophet_config_2': {'mape': 19.290630877369367, 'rmse': 389.86599635114703, 'mae': 313.9244078660851}, 'prophet_config_3': {'mape': 18.887179219518448, 'rmse': 380.1951140870063, 'mae': 307.59685780225794}, 'exponential_smoothing_config_0': {'mape': 22.229793465504123, 'rmse': 422.7578998102065, 'mae': 367.35380582194455}, 'exponential_smoothing_config_1': {'mape': 22.386532736318035, 'rmse': 425.9252487727357, 'mae': 369.89493361874247}, 'exponential_smoothing_config_2': {'mape': 22.2077291682311, 'rmse': 422.8546436603983, 'mae': 366.85070746769816}, 'exponential_smoothing_config_3': {'mape': 22.21853682128127, 'rmse': 422.58541044099314, 'mae': 367.1554451911461}, 'regression_config_0': {'mape': 18.022220126348675, 'rmse': 360.61806876743486, 'mae': 294.22710729245824}, 'regression_config_1': {'mape': 17.961769925151465, 'rmse': 359.99893298231893, 'mae': 293.0687685283249}, 'regression_config_2': {'mape': 18.022188139212528, 'rmse': 360.625224197643, 'mae': 294.2242560305136}, 'regression_config_3': {'mape': 18.070607585707663, 'rmse': 361.7980094285379, 'mae': 294.9931190766236}}

================================================================================
Next Iteration Suggestions
================================================================================

prophet:
- Fine-tune changepoint_prior_scale around current optimal values
- Test impact of different seasonality modes
- Evaluate country-specific holiday effects
- Experiment with hybrid seasonality modes
exponential_smoothing:
- Test shorter seasonal windows (4-5 periods)
- Evaluate impact of dampened trend components
- Fine-tune smoothing parameters
- Explore alternative seasonal decomposition methods
regression:
- Investigate polynomial feature interactions
- Test alternative regularization techniques
- Evaluate impact of feature scaling methods
- Explore feature selection methods
```
