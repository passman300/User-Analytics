# User Analytics & Predictive Modeling Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/pyspark-3.3.0-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.20.0-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-5.10.0-green.svg)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/pandas-1.5.0-yellow.svg)](https://pandas.pydata.org/)

A comprehensive data pipeline for processing, analyzing, and visualizing user behavior data from raw event logs to actionable insights.

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Processing Pipeline](#-processing-pipeline)
- [Data Insights](#-data-insights)
- [Analytics Models](#-analytics-models)
- [Visualization](#-visualization)

## ✨ Features

- Scalable data processing pipeline for large event datasets
- Memory-efficient processing through chunking
- Comprehensive data cleaning and feature engineering
- Geographic data imputation and standardization
- Temporal pattern analysis and time-based feature extraction
- Multiple predictive models for behavioral analysis
- Interactive visualizations and dashboards

## 📁 Project Structure

```
.
├── images/                  # Visualization images and assets
├── 2024.ipynb              # Jupyter notebook for 2024 data exploration
├── 2025.ipynb              # Jupyter notebook for 2025 data exploration
├── analysis_results.json   # Primary analysis output in JSON format
├── analysis_results_1.json # Additional analysis results
├── compile_2024.py         # Processes 2024 raw JSON data into CSV chunks
├── compile_2025.py         # Processes 2025 raw JSON data into CSV chunks
├── data_processing.py      # Streamlit app for data processing steps
├── predictive_analysis.py  # ML models for prediction and pattern recognition
├── read_spark.py           # PySpark analysis of processed data
├── requirements.txt        # Project dependencies
├── spark_analysis.log      # Processing logs from Spark operations
├── user_analytics.py       # User behavior analysis and insights generation
└── visualize_analysis.py   # Interactive visualizations using Plotly
```

Each file serves a specific purpose in the data processing and analytics pipeline:

- **Data Processing**: `compile_2024.py`, `compile_2025.py` process raw data into analyzable formats
- **Analysis**: `read_spark.py`, `user_analytics.py` perform in-depth data analysis
- **Visualization**: `visualize_analysis.py`, `data_processing.py` create interactive dashboards
- **Exploration**: `2024.ipynb`, `2025.ipynb` provide interactive data exploration
- **Output**: `analysis_results.json`, logs, and images store results and visualizations

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/user-analytics-pipeline.git
cd user-analytics-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data processing
python compile_2024.py
python compile_2025.py
python read_spark.py

# Launch visualization dashboard
streamlit run visualize_analysis.py
```

## 🔄 Processing Pipeline

1. **Data Ingestion** (`compile_2024.py`, `compile_2025.py`)
   - Processes raw JSON data using memory-efficient ijson
   - Chunks data into 100,000-record batches
   - Preserves essential columns and removes noise
   - Converts to CSV format for downstream processing

2. **Spark Processing** (`read_spark.py`)
   - Uses PySpark for scalable data processing
   - Memory usage monitoring and optimization
   - Schema enforcement and validation
   - Multi-faceted analysis with execution timing

3. **Analytics Processing** (`user_analytics.py`)
   - Data quality assessment and cleaning
   - Feature engineering and extraction
   - Geographic and temporal pattern analysis
   - User behavior insights generation

4. **Visualization** (`visualize_analysis.py`, `data_processing.py`)
   - Interactive dashboards with Streamlit
   - Comprehensive data visualizations with Plotly
   - Process transparency and explainability
   - Insights presentation and drill-down capabilities

## 📊 Data Insights

### Dataset Overview
- 1.85M events processed across 31 tracked dimensions
- Comprehensive event metadata including temporal, geographic, and device information
- 100% timestamp synchronization between client and server

### Key Metrics
- **Session Patterns**: Average session duration of 1h 27m, median of 53m
- **Platform Usage**: Windows (82%), Mac (16%), Linux (2.5%)
- **Geographic Distribution**: US (42%), UK (19%), Canada (12%), Australia (9%)
- **Performance**: Average server latency 234ms, 95th percentile 892ms

## 🧠 Analytics Models

### Available Analysis Types

1. **Markov Chain Analysis**
   - Event sequence transition analysis
   - Pattern visualization and probability mapping

2. **Hidden Markov Model (HMM)**
   - Uncovering hidden states in user behavior
   - State transition probability visualization

3. **Prophet Time Series Forecast**
   - Multi-metric forecasting with seasonality analysis
   - Confidence intervals for business planning

4. **ARIMA Time Series Analysis**
   - Short-term prediction and trend analysis
   - Cyclical pattern detection

5. **LSTM Neural Network**
   - Deep learning for sequence prediction
   - Complex pattern recognition in user journeys

6. **KMeans Clustering**
   - User segmentation based on behavior patterns
   - Cluster characteristic visualization

7. **XGBoost Prediction**
   - Next event prediction with feature importance
   - Performance and accuracy metrics

## 📈 Visualization

Run the visualization dashboard to explore insights:

```bash
streamlit run visualize_analysis.py
```

For data processing transparency:

```bash
streamlit run data_processing.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

