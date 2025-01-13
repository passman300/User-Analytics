import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import glob
import os
import tempfile
import shutil
import traceback
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle
import base64
import io
import json
from datetime import datetime

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# torch.classes.__path__ = []

@st.cache_data
def load_and_preprocess_data(year=2025):
    """Load all CSV chunks for a given year and combine them into a single DataFrame."""
    csv_files = glob.glob(f"./federato/{year}_csv/*_chunk_*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['event_time'] = pd.to_datetime(df['event_time'])
        dfs.append(df)
    
    if not dfs:
        sample_data = {
            'event_time': pd.date_range(start='2025-01-01', periods=100, freq='H'),
            'user_id': np.random.randint(1, 11, 100),
            'session_id': np.random.randint(1, 21, 100),
            'event_type': np.random.choice(['click', 'view', 'purchase', 'login'], 100),
            'device_family': np.random.choice(['desktop', 'mobile', 'tablet'], 100),
            'country': np.random.choice(['US', 'UK', 'CA', 'AU'], 100),
            'city': np.random.choice(['New York', 'London', 'Toronto', 'Sydney'], 100)
        }
        return pd.DataFrame(sample_data)
    
    return pd.concat(dfs, ignore_index=True)

def create_time_features(df):
    """Create time-based features for analysis."""
    df['hour'] = df['event_time'].dt.hour
    df['day'] = df['event_time'].dt.day
    df['weekday'] = df['event_time'].dt.dayofweek
    df['month'] = df['event_time'].dt.month
    df['year'] = df['event_time'].dt.year
    return df

def pad_sequences(sequences, maxlen):
    """Custom implementation of sequence padding."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_sequences.append(seq[:maxlen])
        else:
            padded_sequences.append(seq + [0] * (maxlen - len(seq)))
    return np.array(padded_sequences)

def perform_markov_chain_analysis(df):
    """Analyze event sequences using enhanced Markov Chain."""
    st.subheader("Markov Chain Analysis")
    st.write("""
    This analysis shows transition probabilities between events, considering both the event type
    and contextual information like time of day and device type.
    """)
    
    df = df.sort_values(['session_id', 'event_time'])
    
    df['time_of_day'] = pd.cut(df['hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['night', 'morning', 'afternoon', 'evening'])
    
    df['event_state'] = df['event_type'] + '|' + df['time_of_day'].astype(str)
    
    transitions = {}
    transition_times = {}
    
    for session_id, session_df in df.groupby('session_id'):
        events = session_df['event_state'].tolist()
        timestamps = session_df['event_time'].tolist()
        
        for i in range(len(events)-1):
            current = events[i]
            next_event = events[i+1]
            time_diff = (timestamps[i+1] - timestamps[i]).total_seconds()
            
            if current not in transitions:
                transitions[current] = {}
                transition_times[current] = {}
            if next_event not in transitions[current]:
                transitions[current][next_event] = 0
                transition_times[current][next_event] = []
            transitions[current][next_event] += 1
            transition_times[current][next_event].append(time_diff)
    
    transition_probs = {}
    avg_transition_times = {}
    
    for current in transitions:
        total = sum(transitions[current].values())
        transition_probs[current] = {}
        avg_transition_times[current] = {}
        
        for next_event in transitions[current]:
            transition_probs[current][next_event] = transitions[current][next_event] / total
            avg_transition_times[current][next_event] = np.mean(transition_times[current][next_event])
    
    edges = []
    for current in transition_probs:
        for next_event in transition_probs[current]:
            if transition_probs[current][next_event] > 0.05:
                edges.append({
                    'from': current.split('|')[0],
                    'to': next_event.split('|')[0],
                    'probability': transition_probs[current][next_event],
                    'avg_time': avg_transition_times[current][next_event]
                })
    
    edge_df = pd.DataFrame(edges)
    if not edge_df.empty:
        fig1 = px.scatter(edge_df, x='from', y='to', size='probability',
                         color='avg_time',
                         title="Event Transition Probabilities and Times",
                         labels={'from': 'From Event', 
                                'to': 'To Event',
                                'probability': 'Transition Probability',
                                'avg_time': 'Avg. Transition Time (s)'},
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Top Transition Patterns")
        top_transitions = edge_df.sort_values('probability', ascending=False).head(10)
        st.dataframe(top_transitions.round(3))
        
        st.subheader("Time-of-Day Transition Patterns")
        time_patterns = df.groupby('time_of_day')['event_type'].value_counts().unstack()
        fig2 = px.imshow(time_patterns,
                        title="Event Types by Time of Day",
                        labels={'x': 'Event Type', 
                               'y': 'Time of Day',
                               'color': 'Count'})
        st.plotly_chart(fig2, use_container_width=True)
    
    return transition_probs, avg_transition_times

def perform_hmm_analysis(df):
    """Analyze user behavior patterns using Hidden Markov Model."""
    st.subheader("Hidden Markov Model Analysis")
    st.write("""
    HMM analysis uncovers hidden states in user behavior, revealing underlying
    patterns that aren't immediately visible in the raw event sequence.
    """)
    
    le = LabelEncoder()
    df['event_encoded'] = le.fit_transform(df['event_type'])
    
    sessions = df.groupby('session_id')['event_encoded'].apply(list)
    
    max_len = 10
    sequences = pad_sequences(sessions, max_len)
    
    n_states = 5
    model = hmm.GaussianHMM(n_components=n_states, n_iter=100)
    model.fit(sequences.reshape(-1, 1))
    
    state_sequences = model.predict(sequences.reshape(-1, 1))
    
    state_transitions = pd.DataFrame(model.transmat_,
                                   columns=[f'To State {i}' for i in range(n_states)],
                                   index=[f'From State {i}' for i in range(n_states)])
    
    fig = go.Figure(data=go.Heatmap(
        z=model.transmat_,
        x=[f'State {i}' for i in range(n_states)],
        y=[f'State {i}' for i in range(n_states)],
        colorscale='Viridis'
    ))
    fig.update_layout(title='HMM State Transition Probabilities')
    st.plotly_chart(fig, use_container_width=True)
    
    # st.subheader("Test Prediction")
    # with st.expander("Make a test prediction"):
    #     test_sequence = st.multiselect(
    #         "Select a sequence of events",
    #         options=le.classes_,
    #         default=le.classes_[:3] if len(le.classes_) >= 3 else le.classes_
    #     )
        
    #     if st.button("Predict Next State") and test_sequence:
    #         try:
    #             encoded_sequence = [le.transform([event])[0] for event in test_sequence]
                
    #             sequence_array = np.array(encoded_sequence).reshape(-1, 1)
                
    #             state_sequence = model.predict(sequence_array)
    #             last_state = state_sequence[-1]
                
    #             st.success(f"Predicted hidden state: {last_state}")
                
    #             st.write("Transition probabilities from this state:")
    #             st.dataframe(pd.DataFrame({
    #                 'To State': [f'State {i}' for i in range(n_states)],
    #                 'Probability': model.transmat_[last_state]
    #             }))
    #         except Exception as e:
    #             st.error(f"Error making prediction: {str(e)}")
    
    return model, le

def perform_prophet_forecast(df):
    """Enhanced forecast using Facebook Prophet with multiple metrics."""
    st.subheader("Enhanced Prophet Time Series Forecast")
    st.write("""
    Multi-metric forecasting that predicts various aspects of user behavior:
    - Event volumes by type
    - Session durations
    - Processing latencies
    - Geographic activity patterns
    """)
    
    metrics = {
        'event_volume': df.groupby('event_time').size(),
        'avg_session_duration': df.groupby('event_time')['session_duration'].mean(),
        'avg_processing_latency': df.groupby('event_time')['processing_latency'].mean(),
        'unique_sessions': df.groupby('event_time')['session_id'].nunique()
    }
    
    forecasts = {}
    for metric_name, series in metrics.items():
        hourly_data = series.resample('H').mean().fillna(method='ffill')
        
        prophet_df = pd.DataFrame({
            'ds': hourly_data.index,
            'y': hourly_data.values
        })
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        model.add_seasonality(
            name='business_hours',
            period=24,
            fourier_order=5,
            condition_name='is_business_hour'
        )
        
        prophet_df['is_business_hour'] = (
            (prophet_df['ds'].dt.hour >= 9) & 
            (prophet_df['ds'].dt.hour < 17)
        ).astype(int)
        
        model.fit(prophet_df)
        
        future_dates = model.make_future_dataframe(
            periods=24*7,
            freq='H'
        )
        future_dates['is_business_hour'] = (
            (future_dates['ds'].dt.hour >= 9) & 
            (future_dates['ds'].dt.hour < 17)
        ).astype(int)
        
        forecast = model.predict(future_dates)
        forecasts[metric_name] = (model, forecast)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            name='Actual',
            mode='markers+lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Predicted',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'{metric_name.replace("_", " ").title()} Forecast',
            xaxis_title='Date',
            yaxis_title='Value'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        components = model.plot_components(forecast)
        st.pyplot(components)
        
        st.subheader(f"Key Insights for {metric_name.replace('_', ' ').title()}")
        
        growth_rate = ((forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / 
                      forecast['yhat'].iloc[0] * 100)
        
        peak_times = forecast.nlargest(3, 'yhat')[['ds', 'yhat']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Projected Growth Rate", f"{growth_rate:.1f}%")
            st.write("Peak Times:")
            st.dataframe(peak_times)
        
        with col2:
            st.write("Seasonal Patterns:")
            st.write("- Daily: ", "Strong" if model.daily_seasonality else "Weak")
            st.write("- Weekly: ", "Strong" if model.weekly_seasonality else "Weak")
            st.write("- Yearly: ", "Strong" if model.yearly_seasonality else "Weak")
    
    return forecasts

def perform_arima_analysis(df):
    """Enhanced ARIMA analysis with multiple time series components."""
    st.subheader("Enhanced ARIMA Time Series Analysis")
    st.write("""
    Advanced time series analysis using ARIMA models to capture:
    - Short-term event patterns
    - Session behavior trends
    - Performance metrics
    """)
    
    metrics = {
        'event_volume': df.groupby('event_time').size(),
        'avg_session_duration': df.groupby('event_time')['session_duration'].apply(
            lambda x: x.dt.total_seconds().mean() if isinstance(x.iloc[0], pd.Timedelta) else x.mean()
        ),
        'unique_sessions': df.groupby('event_time')['session_id'].nunique()
    }
    
    analysis_results = {}
    for metric_name, series in metrics.items():
        st.write(f"\nAnalyzing {metric_name}...")
        
        hourly_data = series.resample('H').mean().fillna(method='ffill')
        
        if not np.issubdtype(hourly_data.dtype, np.number):
            st.warning(f"Converting {metric_name} to numeric values")
            hourly_data = pd.to_numeric(hourly_data, errors='coerce')
            hourly_data = hourly_data.fillna(method='ffill')
        
        hourly_data = hourly_data.last('30D')
        
        try:
            model = ARIMA(hourly_data, order=(1, 1, 1))
            results = model.fit()
            
            last_date = hourly_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=168,
                freq='H'
            )
            
            forecast = results.forecast(steps=168)
            forecast_series = pd.Series(forecast, index=future_dates)
            
            pred_int = results.get_forecast(steps=168).conf_int()
            lower_series = pd.Series(pred_int.iloc[:, 0], index=future_dates)
            upper_series = pd.Series(pred_int.iloc[:, 1], index=future_dates)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hourly_data.index,
                y=hourly_data.values,
                name='Actual',
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_series,
                name='Forecast',
                mode='lines',
                line=dict(dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=upper_series.tolist() + lower_series.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'{metric_name.replace("_", " ").title()} - ARIMA Forecast',
                xaxis_title='Date',
                yaxis_title='Value'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(f"Model Statistics for {metric_name}")
            model_stats = pd.DataFrame({
                'AIC': [results.aic],
                'BIC': [results.bic]
            })
            st.dataframe(model_stats)
            
            residuals = results.resid
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            error_metrics = pd.DataFrame({
                'MSE': [mse],
                'RMSE': [rmse],
                'MAE': [mae]
            })
            st.write("Error Metrics:")
            st.dataframe(error_metrics)
            
            analysis_results[metric_name] = {
                'forecast': forecast_series.values,
                'dates': future_dates,
                'lower_ci': lower_series.values,
                'upper_ci': upper_series.values,
                'error_metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
            }
            
        except Exception as e:
            st.error(f"Error in forecasting for {metric_name}: {str(e)}")
            continue
    
    return analysis_results

class EventSequenceDataset(Dataset):
    """Dataset for event sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences if isinstance(sequences, np.ndarray) else np.array(sequences)
        self.labels = labels if isinstance(labels, np.ndarray) else np.array(labels)
        
        self.seq_shape = self.sequences.shape
        self.num_samples = len(self.sequences)
        
        st.write(f"Debug: Dataset initialized with {self.num_samples} samples")
        st.write(f"Debug: Sequence shape: {self.seq_shape}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        try:
            seq = torch.FloatTensor(self.sequences[idx])
            label = torch.LongTensor([self.labels[idx]])[0]
            return seq, label
        except Exception as e:
            st.error(f"Error loading item {idx}: {str(e)}")
            st.write(f"Debug: sequence shape at {idx}: {self.sequences[idx].shape}")
            st.write(f"Debug: label at {idx}: {self.labels[idx]}")
            raise e

class LSTMPredictor(nn.Module):
    """LSTM model for event prediction."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, 
                           bidirectional=True,
                           dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes) 
        # *2 for bidirectional
        
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
                                 else "cpu")
        
        self.to(self.device)
        st.info(f"Using device: {self.device}")
    
    def forward(self, x):
        if x.size(0) == 0:
            raise ValueError("Empty batch received")
            
        x = x.float()
        
        x = x.to(self.device)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def perform_lstm_prediction(df):
    """Predict user engagement using LSTM."""
    st.subheader("LSTM Neural Network Analysis")
    
    st.write("Starting LSTM prediction process...")
    
    st.sidebar.subheader("LSTM Configuration")
    sample_size = st.sidebar.slider("Sample Size (thousands)", 
                                  min_value=5, 
                                  max_value=50, 
                                  value=10) * 1000
    batch_size = st.sidebar.select_slider("Batch Size", 
                                        options=[16, 32, 64, 128, 256], 
                                        value=32)
    sequence_length = st.sidebar.slider("Sequence Length", 
                                      min_value=3, 
                                      max_value=10, 
                                      value=5)
    
    try:
        with st.spinner("Preparing sequences..."):
            st.write("Debug: Initial DataFrame shape:", df.shape)
            
            if df.empty:
                st.error("Empty DataFrame received")
                return None, None, None
                
            if 'event_type' not in df.columns or 'session_id' not in df.columns:
                st.error("Required columns missing from DataFrame")
                return None, None, None
            
            st.write("Debug: Encoding events...")
            le = LabelEncoder()
            df['event_encoded'] = le.fit_transform(df['event_type'])
            num_classes = len(le.classes_)
            st.write(f"Debug: Number of unique events: {num_classes}")
            
            unique_sessions = df['session_id'].unique()
            if len(unique_sessions) > sample_size:
                st.write(f"Debug: Sampling {sample_size} sessions from {len(unique_sessions)} total sessions")
                sampled_sessions = np.random.choice(unique_sessions, size=sample_size, replace=False)
                df = df[df['session_id'].isin(sampled_sessions)].copy()
            
            max_sequences_per_session = 25
            total_sessions = df['session_id'].nunique()
            max_total_sequences = min(total_sessions * max_sequences_per_session, sample_size)
            
            st.write(f"Debug: Will create maximum {max_total_sequences} sequences")
            
            temp_dir = tempfile.mkdtemp()
            sequences_file = os.path.join(temp_dir, 'sequences.npy')
            labels_file = os.path.join(temp_dir, 'labels.npy')
            
            try:
                sequences_mmap = np.lib.format.open_memmap(
                    sequences_file, 
                    mode='w+', 
                    dtype=np.float32,
                    shape=(max_total_sequences, sequence_length)
                )
                
                labels_mmap = np.lib.format.open_memmap(
                    labels_file,
                    mode='w+',
                    dtype=np.int64,
                    shape=(max_total_sequences,)
                )
                
                sequence_count = 0
                chunk_size = min(500, total_sessions)
                session_progress = st.progress(0)
                
                for chunk_idx in range(0, total_sessions, chunk_size):
                    chunk_sessions = df['session_id'].unique()[chunk_idx:chunk_idx + chunk_size]
                    chunk_df = df[df['session_id'].isin(chunk_sessions)]
                    
                    for idx, (session_id, group) in enumerate(chunk_df.groupby('session_id')):
                        events = group['event_encoded'].values
                        if len(events) > sequence_length:
                            seq = np.lib.stride_tricks.sliding_window_view(events, sequence_length+1)
                            if len(seq) > max_sequences_per_session:
                                indices = np.random.choice(len(seq), max_sequences_per_session, replace=False)
                                seq = seq[indices]
                            
                            space_left = max_total_sequences - sequence_count
                            sequences_to_add = min(len(seq), space_left)
                            
                            if sequences_to_add > 0:
                                end_idx = sequence_count + sequences_to_add
                                sequences_mmap[sequence_count:end_idx] = seq[:sequences_to_add, :-1]
                                labels_mmap[sequence_count:end_idx] = seq[:sequences_to_add, -1]
                                sequence_count += sequences_to_add
                            
                            if space_left <= 0:
                                break
                        
                        global_idx = chunk_idx + idx
                        if global_idx % 50 == 0:
                            session_progress.progress(min(1.0, global_idx / total_sessions))
                            st.write(f"Debug: Processed {global_idx}/{total_sessions} sessions, created {sequence_count} sequences")
                    
                    if space_left <= 0:
                        break
                
                session_progress.progress(1.0)
                
                if sequence_count == 0:
                    st.error("No valid sequences could be created")
                    return None, None, None
                
                sequences = sequences_mmap[:sequence_count].copy()
                labels = labels_mmap[:sequence_count].copy()
                
                del sequences_mmap
                del labels_mmap
                
                if len(sequences) > sample_size:
                    st.write("Debug: Subsampling sequences to fit memory constraints")
                    indices = np.random.choice(len(sequences), sample_size, replace=False)
                    sequences = sequences[indices]
                    labels = labels[indices]
                
                st.write(f"Debug: Final sequence shape: {sequences.shape}")
                st.write(f"Debug: Final labels shape: {labels.shape}")
                st.info(f"Prepared {len(sequences)} sequences for training")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    sequences, labels, 
                    test_size=0.1,
                    random_state=42
                )
                
                del sequences
                del labels
                
                with st.spinner("Training model..."):
                    train_dataset = EventSequenceDataset(X_train, y_train)
                    test_dataset = EventSequenceDataset(X_test, y_test)
                    
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=False,
                        num_workers=0
                    )
                    
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=False,
                        num_workers=0
                    )
                    
                    input_size = 1
                    hidden_size = 64
                    num_layers = 2
                    dropout_rate = 0.3
                    
                    model = LSTMPredictor(input_size, hidden_size, num_layers, num_classes, dropout_rate)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
                    
                    num_epochs = 10
                    train_losses = []
                    val_losses = []
                    train_accuracies = []
                    val_accuracies = []
                    best_val_loss = float('inf')
                    
                    epoch_progress = st.progress(0)
                    metrics_placeholder = st.empty()
                    loss_chart = st.empty()
                    accuracy_chart = st.empty()
                    
                    for epoch in range(num_epochs):
                        model.train()
                        total_train_loss = 0
                        correct_train = 0
                        total_train = 0
                        
                        batch_progress = st.progress(0)
                        for batch_idx, (batch_sequences, batch_labels) in enumerate(train_loader):
                            optimizer.zero_grad()
                            
                            batch_sequences = batch_sequences.unsqueeze(-1)
                            outputs = model(batch_sequences)
                            loss = criterion(outputs, batch_labels.to(model.device))
                            
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            total_train_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total_train += batch_labels.size(0)
                            correct_train += (predicted.cpu() == batch_labels.cpu()).sum().item()
                            
                            batch_progress.progress((batch_idx + 1) / len(train_loader))
                        
                        model.eval()
                        total_val_loss = 0
                        correct_val = 0
                        total_val = 0
                        
                        with torch.no_grad():
                            for batch_sequences, batch_labels in test_loader:
                                batch_sequences = batch_sequences.unsqueeze(-1)
                                outputs = model(batch_sequences)
                                loss = criterion(outputs, batch_labels.to(model.device))
                                
                                total_val_loss += loss.item()
                                _, predicted = torch.max(outputs.data, 1)
                                total_val += batch_labels.size(0)
                                correct_val += (predicted.cpu() == batch_labels.cpu()).sum().item()
                        
                        avg_train_loss = total_train_loss / len(train_loader)
                        avg_val_loss = total_val_loss / len(test_loader)
                        train_accuracy = 100 * correct_train / total_train
                        val_accuracy = 100 * correct_val / total_val
                        
                        train_losses.append(avg_train_loss)
                        val_losses.append(avg_val_loss)
                        train_accuracies.append(train_accuracy)
                        val_accuracies.append(val_accuracy)
                        
                        scheduler.step(avg_val_loss)
                        
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                        
                        epoch_progress.progress((epoch + 1) / num_epochs)
                        metrics_placeholder.write(f"""
                        Epoch {epoch+1}/{num_epochs}
                        Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%
                        Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%
                        Best Val Loss: {best_val_loss:.4f}
                        """)
                        
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=train_losses, name='Training Loss'))
                        fig_loss.add_trace(go.Scatter(y=val_losses, name='Validation Loss'))
                        fig_loss.update_layout(title='Training and Validation Loss',
                                             xaxis_title='Epoch',
                                             yaxis_title='Loss')
                        loss_chart.plotly_chart(fig_loss, use_container_width=True)
                        
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(y=train_accuracies, name='Training Accuracy'))
                        fig_acc.add_trace(go.Scatter(y=val_accuracies, name='Validation Accuracy'))
                        fig_acc.update_layout(title='Training and Validation Accuracy',
                                            xaxis_title='Epoch',
                                            yaxis_title='Accuracy (%)')
                        accuracy_chart.plotly_chart(fig_acc, use_container_width=True)
                        
                        if train_accuracy > 95 and val_accuracy > 90:
                            st.success(f"Reached high accuracy at epoch {epoch+1}, stopping training")
                            break
                    
                    st.success("Training completed!")
                    
                    st.subheader("Model Performance Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Training Accuracy", f"{train_accuracies[-1]:.2f}%")
                        st.metric("Final Training Loss", f"{train_losses[-1]:.4f}")
                    with col2:
                        st.metric("Final Validation Accuracy", f"{val_accuracies[-1]:.2f}%")
                        st.metric("Final Validation Loss", f"{val_losses[-1]:.4f}")
                    
                    return model, (train_losses, val_losses, train_accuracies, val_accuracies), le
                    
            except Exception as e:
                st.error(f"Error in sequence processing: {str(e)}")
                st.write("Debug: Full traceback:", traceback.format_exc())
                return None, None, None
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    st.error(f"Error cleaning up temporary files: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")
        st.write("Debug: Full traceback:", traceback.format_exc())
        return None, None, None

def perform_kmeans_clustering(df):
    """Cluster users based on behavior patterns."""
    st.subheader("User Behavior Clustering")
    st.write("""
    KMeans clustering groups events with similar patterns,
    helping identify distinct behavioral segments.
    """)
    
    session_features = df.groupby('session_id').agg({
        'event_type': 'count',
        'device_family': lambda x: x.mode().iloc[0] if not x.empty else None,
        'country': lambda x: x.mode().iloc[0] if not x.empty else None,
        'hour': 'mean',
        'session_duration': lambda x: x.iloc[0].total_seconds() if not x.empty else 0
    }).reset_index()
    
    le_device = LabelEncoder()
    le_country = LabelEncoder()
    session_features['device_encoded'] = le_device.fit_transform(session_features['device_family'])
    session_features['country_encoded'] = le_country.fit_transform(session_features['country'])
    
    features_for_clustering = [
        'event_type',
        'device_encoded',
        'country_encoded',
        'hour',
        'session_duration'
    ]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(session_features[features_for_clustering])
    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    session_features['Cluster'] = clusters
    cluster_stats = session_features.groupby('Cluster').agg({
        'session_id': 'count',
        'event_type': 'mean',
        'session_duration': 'mean',
        'hour': 'mean'
    }).round(2)
    
    cluster_stats.columns = ['Number of Sessions', 'Avg Events per Session', 
                           'Avg Session Duration (s)', 'Avg Hour of Day']
    
    fig1 = px.scatter(session_features, 
                     x='event_type', 
                     y='session_duration',
                     color='Cluster',
                     title='Session Clusters by Events and Duration',
                     labels={'event_type': 'Number of Events',
                            'session_duration': 'Session Duration (s)'})
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(session_features,
                     x='hour',
                     y='event_type',
                     color='Cluster',
                     title='Session Clusters by Time of Day',
                     labels={'hour': 'Hour of Day',
                            'event_type': 'Number of Events'})
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Cluster Statistics")
    st.dataframe(cluster_stats)
    
    st.subheader("Cluster Characteristics")
    for i in range(n_clusters):
        cluster_data = session_features[session_features['Cluster'] == i]
        st.write(f"\nCluster {i}:")
        st.write(f"- Number of sessions: {len(cluster_data)}")
        st.write(f"- Average events per session: {cluster_data['event_type'].mean():.1f}")
        st.write(f"- Average session duration: {cluster_data['session_duration'].mean():.1f} seconds")
        st.write(f"- Most common device: {cluster_data['device_family'].mode().iloc[0]}")
        st.write(f"- Most common country: {cluster_data['country'].mode().iloc[0]}")
        st.write(f"- Peak activity hour: {cluster_data['hour'].mean():.1f}")
    
    return kmeans, cluster_stats

def perform_xgboost_prediction(df):
    st.subheader("Next Event Prediction")
    
    st.sidebar.subheader("XGBoost Configuration")
    sample_size = st.sidebar.slider("Sample Size (thousands)", 
                                min_value=5, 
                                max_value=100, 
                                value=20) * 1000
    
    with st.spinner("Preparing features..."):
        if len(df) > sample_size:
            st.write(f"Sampling {sample_size:,} events from {len(df):,} total events")
            df = df.sample(n=sample_size, random_state=42)
        
        df['hour'] = df['event_time'].dt.hour
        df['weekday'] = df['event_time'].dt.dayofweek
        df['prev_event'] = df.groupby('session_id')['event_type'].shift(1)
        
        event_encoder = LabelEncoder()
        all_events_encoded = event_encoder.fit_transform(df['event_type'])
        
        event_counts = pd.Series(all_events_encoded).value_counts()
        min_samples = 5
        valid_events = event_counts[event_counts >= min_samples].index
        
        mask = np.isin(all_events_encoded, valid_events)
        df_filtered = df[mask].copy()
        
        event_encoder = LabelEncoder()
        df_filtered['event_encoded'] = event_encoder.fit_transform(df_filtered['event_type'])
        
        df_filtered['prev_event'] = df_filtered.groupby('session_id')['event_type'].shift(1)
        df_filtered['prev_event'] = df_filtered['prev_event'].fillna(df_filtered['event_type'].iloc[0])
        df_filtered['prev_event_encoded'] = event_encoder.transform(df_filtered['prev_event'])
        
        le_device = LabelEncoder()
        df_filtered['device_encoded'] = le_device.fit_transform(df_filtered['device_family'])
        
        features = ['hour', 'weekday', 'prev_event_encoded', 'device_encoded']
        X = df_filtered[features].values
        y = df_filtered['event_encoded'].values
        
        st.info(f"Prepared {len(features)} features for {len(X):,} samples")
        st.info(f"Number of unique event types after filtering: {len(event_encoder.classes_)}")
        
        class_dist = pd.DataFrame({
            'Event Type': event_encoder.classes_,
            'Count': pd.Series(y).value_counts().sort_index().values
        })
        st.write("Event Distribution:")
        st.dataframe(class_dist)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        train_dist = pd.Series(y_train).value_counts()
        test_dist = pd.Series(y_test).value_counts()
        st.write("Minimum samples per class in training:", train_dist.min())
        st.write("Minimum samples per class in test:", test_dist.min())
        
        with st.spinner("Training XGBoost model..."):
            params = {
                'objective': 'multi:softmax',
                'num_class': len(event_encoder.classes_),
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'random_state': 42,
                'eval_metric': ['mlogloss', 'merror']
            }
            
            eval_set = [(X_train, y_train), (X_test, y_test)]
            
            progress_bar = st.progress(0)
            metrics_placeholder = st.empty()
            train_chart = st.empty()
            
            train_metrics = {'mlogloss': [], 'merror': []}
            val_metrics = {'mlogloss': [], 'merror': []}
            
            class ProgressCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    iteration = epoch + 1
                    total_iterations = model.get_params()['n_estimators']
                    
                    progress = iteration / total_iterations
                    progress_bar.progress(progress)
                    
                    train_metrics_current = evals_log['validation_0']
                    val_metrics_current = evals_log['validation_1']
                    
                    for metric in ['mlogloss', 'merror']:
                        if metric in train_metrics_current:
                            train_metrics[metric].append(train_metrics_current[metric][-1])
                            val_metrics[metric].append(val_metrics_current[metric][-1])
                    
                    if 'mlogloss' in train_metrics_current and 'merror' in train_metrics_current:
                        metrics_placeholder.write(f"""
                        Iteration {iteration}/{total_iterations}
                        Training Loss: {train_metrics_current['mlogloss'][-1]:.4f}, Error: {train_metrics_current['merror'][-1]:.4f}
                        Validation Loss: {val_metrics_current['mlogloss'][-1]:.4f}, Error: {val_metrics_current['merror'][-1]:.4f}
                        """)
                    
                    if len(train_metrics['mlogloss']) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=train_metrics['mlogloss'], name='Train Loss'))
                        fig.add_trace(go.Scatter(y=val_metrics['mlogloss'], name='Val Loss'))
                        fig.add_trace(go.Scatter(y=train_metrics['merror'], name='Train Error'))
                        fig.add_trace(go.Scatter(y=val_metrics['merror'], name='Val Error'))
                        fig.update_layout(title='Training Progress',
                                        xaxis_title='Iteration',
                                        yaxis_title='Metric Value')
                        train_chart.plotly_chart(fig, use_container_width=True)
                    
                    return False
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_accuracy = np.mean(train_pred == y_train)
            test_accuracy = np.mean(test_pred == y_test)
            
            st.success("XGBoost training completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_accuracy:.2%}")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy:.2%}")
            
            st.subheader("Most Common Predictions")
            pred_df = pd.DataFrame({
                'Actual': [event_encoder.classes_[int(y)] for y in y_test],
                'Predicted': [event_encoder.classes_[int(p)] for p in test_pred]
            })
            confusion = pd.crosstab(pred_df['Actual'], pred_df['Predicted'])
            st.write("Top 5 most common actual vs predicted events:")
            st.dataframe(confusion.head())
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        title='Feature Importance in Next Event Prediction',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            # st.subheader("Test Prediction")
            # with st.expander("Make a test prediction"):
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         test_hour = st.slider("Hour of Day", 0, 23, 12)
            #         test_weekday = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 2)
            #     with col2:
            #         test_prev_event = st.selectbox("Previous Event", event_encoder.classes_)
            #         test_device = st.selectbox("Device", le_device.classes_)
                
            #     if st.button("Predict"):
            #         try:
            #             prev_event_encoded = event_encoder.transform([test_prev_event])[0]
            #             device_encoded = le_device.transform([test_device])[0]
                        
            #             X_test = np.array([[test_hour, test_weekday, prev_event_encoded, device_encoded]])
                        
            #             prediction = model.predict(X_test)[0]
            #             prediction_proba = model.predict_proba(X_test)[0]
                        
            #             top_indices = prediction_proba.argsort()[-5:][::-1]
            #             top_events = [event_encoder.classes_[i] for i in top_indices]
            #             top_probs = [prediction_proba[i] for i in top_indices]
                        
            #             st.success(f"Predicted next event: **{event_encoder.classes_[prediction]}**")
                        
            #             result_df = pd.DataFrame({
            #                 "Event": top_events,
            #                 "Probability": top_probs
            #             })
            #             st.dataframe(result_df)
            #         except Exception as e:
            #             st.error(f"Error making prediction: {str(e)}")
            
            return model, importance_df, event_encoder, le_device
            
    except Exception as e:
        st.error(f"Error in XGBoost training: {str(e)}")
        st.write("Full error details:", str(e))
        return None, None, None, None

def show_model_explanation(model_name):
    """Show detailed explanation for each model type."""
    explanations = {
        "Markov Chain": {
            "title": "Markov Chain Analysis",
            "description": "Analyzes event sequences using enhanced Markov Chain.",
            "use_cases": [
                "User journey optimization",
                "UX/UI improvement",
                "Feature recommendation",
                "Conversion funnel analysis"
            ]
        },
        "Hidden Markov Model": {
            "title": "Hidden Markov Model (HMM) Analysis",
            "description": "Analyzes user behavior patterns using Hidden Markov Model.",
            "use_cases": [
                "User intent prediction",
                "Behavioral pattern discovery",
                "Anomaly detection",
                "Session quality analysis"
            ]
        },
        "Prophet Forecast": {
            "title": "Prophet Time Series Forecast",
            "description": "Multi-metric forecasting for various aspects of user behavior.",
            "use_cases": [
                "Capacity planning",
                "Resource allocation",
                "Usage trend analysis",
                "Seasonal pattern detection"
            ]
        },
        "ARIMA Analysis": {
            "title": "ARIMA Time Series Analysis",
            "description": "Time series analysis using ARIMA model.",
            "use_cases": [
                "Short-term forecasting",
                "Trend analysis",
                "Cyclical pattern detection",
                "Anomaly detection"
            ]
        },
        "LSTM Prediction": {
            "title": "LSTM Neural Network Analysis",
            "description": "Neural network analysis for sequence prediction.",
            "use_cases": [
                "Complex pattern recognition",
                "Next event predictions",
                "User behavior modeling",
                "Sequence prediction"
            ]
        },
        "KMeans Clustering": {
            "title": "User Behavior Clustering",
            "description": "Groups events with similar patterns.",
            "use_cases": [
                "User segmentation",
                "Targeted marketing",
                "Feature personalization",
                "Customer insights"
            ]
        },
        "XGBoost Prediction": {
            "title": "XGBoost Next Event Prediction",
            "description": "Predicts next likely events using XGBoost.",
            "use_cases": [
                "Next action prediction",
                "Feature importance analysis",
                "User behavior understanding",
                "Personalization"
            ]
        }
    }
    
    if model_name in explanations:
        info = explanations[model_name]
        st.header(info["title"])
        st.write(info["description"])
        st.markdown("---")

def show_dashboard_overview():
    """Show dashboard overview and instructions."""
    st.sidebar.header("Dashboard Controls")
    st.sidebar.subheader("Analysis Selection")

def show_data_summary(df):
    """Show summary statistics of the dataset."""
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", f"{len(df):,}")
    with col2:
        if 'user_id' in df.columns:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        else:
            st.metric("Unique Users", "N/A")
    with col3:
        st.metric("Unique Sessions", f"{df['session_id'].nunique():,}")
    with col4:
        st.metric("Event Types", f"{df['event_type'].nunique():,}")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Date Range")
        min_date = df['event_time'].min()
        max_date = df['event_time'].max()
        st.write(f"From: {min_date.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"To: {max_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.subheader("Geographic Coverage")
        n_countries = df['country'].nunique()
        n_cities = df['city'].nunique()
        st.write(f"Countries: {n_countries}")
        st.write(f"Cities: {n_cities}")

def preprocess_data(df):
    """
    Comprehensive data preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    st.subheader("🔄 Data Preprocessing")
    
    with st.spinner("Cleaning and preprocessing data..."):
        original_shape = df.shape
        original_columns = set(df.columns)
        st.write(f"Original dataset shape: {original_shape}")
        st.write("Original columns:", sorted(list(original_columns)))
        
        st.write("\nInitial data sample (first 10 rows):")
        st.dataframe(df.head(10))
        
        columns_after_each_step = {}
        
        essential_columns = ['session_id', 'event_type', 'event_time', 'device_family', 'city', 'region', 'country', 'user_id']
        single_value_cols = df.nunique()[df.nunique() == 1].index
        single_value_cols = [col for col in single_value_cols if col not in essential_columns]
        df = df.drop(columns=single_value_cols)
        columns_after_each_step['after_single_value_drop'] = set(df.columns)
        st.write(f"Dropped {len(single_value_cols)} columns with only 1 unique value: {sorted(list(single_value_cols))}")

        initial_rows = len(df)
        df = df[df['event_time'] >= '2025-01-01']
        columns_after_each_step['after_date_filter'] = set(df.columns)
        dropped_rows = initial_rows - len(df)
        st.write(f"Dropped {dropped_rows:,} rows with event_time before 2025-01-01")
        
        columns_to_drop = [
            'insert_id', '$insert_id', 'amplitude_id', 'device_id', 'uuid', 'user_properties',
            'device_type', 'dma', 'data', 'event_properties', 'language'
        ]
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            columns_after_each_step['after_unnecessary_drop'] = set(df.columns)
            st.write(f"Dropped {len(existing_cols)} unnecessary columns: {sorted(existing_cols)}")
        
        current_columns = set(df.columns)
        removed_columns = original_columns - current_columns
        st.write("\nColumns removed during preprocessing:", sorted(list(removed_columns)))
        st.write("Remaining columns after dropping:", sorted(list(current_columns)))
        
        st.write("\nHandling missing location data:")
        
        city_null_count = df['city'].isnull().sum()
        st.write(f"Initial null values in city column: {city_null_count}")
        st.write(f"Percentage of null values: {(city_null_count/len(df))*100:.2f}%")
        
        if city_null_count > 0:
            location_features = ['region', 'country']
            
            encoders = {}
            encoded_df = df[location_features].copy()
            for col in location_features:
                encoders[col] = LabelEncoder()
                encoded_df[col] = encoders[col].fit_transform(df[col].fillna('Unknown'))
            
            city_encoder = LabelEncoder()
            encoded_df['city'] = city_encoder.fit_transform(df['city'].fillna('Unknown'))
            
            imputer = KNNImputer(n_neighbors=3, weights='distance')
            encoded_df = pd.DataFrame(imputer.fit_transform(encoded_df), columns=encoded_df.columns)
            
            df['city'] = city_encoder.inverse_transform(encoded_df['city'].astype(int))
            
            df['city'] = df['city'].fillna('Unknown')
            
            st.write(f"Remaining null values in city column after imputation: {df['city'].isnull().sum()}")
        
        if 'region' in df.columns and 'city' in df.columns:
            st.write("\nHandling city-region mapping:")
            st.write(f"Initial null regions: {df['region'].isnull().sum()}")
            
            city_region_map = df[df['region'].notna()].groupby('city')['region'].agg(lambda x: x.mode()[0]).to_dict()
            
            df.loc[df['region'].isnull(), 'region'] = df.loc[df['region'].isnull(), 'city'].map(city_region_map)
            
            if df['region'].isnull().any():
                most_common_region = df['region'].mode()[0]
                df['region'] = df['region'].fillna(most_common_region)
            
            st.write(f"Final null regions: {df['region'].isnull().sum()}")
        
        mobile_count = len(df[df['device_family'].isin(['iOS', 'Android'])])
        df = df[~df['device_family'].isin(['iOS', 'Android'])]
        st.write(f"Removed {mobile_count} mobile device rows (iOS/Android)")
        
        null_users = df['user_id'].isnull().sum()
        df = df.dropna(subset=['user_id'])
        st.write(f"Removed {null_users} rows with null user_ids")
        
        time_cols = [col for col in [
            'client_event_time', 'client_upload_time', 'processed_time',
            'server_received_time', 'server_upload_time', 'event_time'
        ] if col in df.columns]
        
        for col in time_cols:
            df[col] = pd.to_datetime(df[col])
        
        df['session_duration'] = df.groupby('session_id')['event_time'].transform(
            lambda x: (x.max() - x.min())
        )
        columns_after_each_step['after_duration_added'] = set(df.columns)
        
        z_scores = np.abs((df['session_duration'].dt.total_seconds() - 
                          df['session_duration'].dt.total_seconds().mean()) / 
                         df['session_duration'].dt.total_seconds().std())
        outliers_count = len(z_scores[z_scores >= 3])
        df = df[z_scores < 3]
        st.write(f"Removed {outliers_count} outlier sessions (z-score > 3)")
        
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.day_name()
        columns_after_each_step['after_time_features'] = set(df.columns)
        
        if all(col in df.columns for col in ['processed_time', 'client_upload_time', 'server_upload_time']):
            df['server_latency'] = (df['processed_time'] - df['client_upload_time']).dt.total_seconds()
            df['processing_latency'] = (df['processed_time'] - df['server_upload_time']).dt.total_seconds()
            columns_after_each_step['after_latency_features'] = set(df.columns)
        
        categorical_columns = ['device_family', 'os_name', 'region', 'city', 'day_of_week']
        existing_cat_cols = [col for col in categorical_columns if col in df.columns]
        for col in existing_cat_cols:
            df[col] = df[col].astype('category')
        
        final_shape = df.shape
        st.write("\nFinal dataset statistics:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Rows", f"{final_shape[0]:,}", 
                     f"{((final_shape[0] - original_shape[0])/original_shape[0]*100):.1f}%")
        with col2:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        with col3:
            st.metric("Unique Sessions", f"{df['session_id'].nunique():,}")
        
        st.write("\nColumn changes during preprocessing:")
        for step, cols in columns_after_each_step.items():
            st.write(f"\n{step}:", sorted(list(cols)))
        
        if 'user_id' in df.columns:
            df = df.drop(columns=['user_id'])
            st.write("\nDropped user_id column as final step")
        
        st.write("\nFinal columns in dataset:", sorted(list(df.columns)))
        
        st.write("\nProcessed data sample (first 10 rows):")
        st.dataframe(df.head(10))
        
        remaining_nulls = df.isnull().sum()[df.isnull().sum() > 0]
        if not remaining_nulls.empty:
            st.write("\nRemaining null values:")
            st.write(remaining_nulls)
        
        return df

def export_model(model, model_type, additional_data=None):
    """
    Export a trained model as a downloadable file.
    
    Args:
        model: The trained model to export
        model_type: String indicating the type of model
        additional_data: Dictionary containing any additional data needed for the model
    
    Returns:
        A download link for the serialized model
    """
    try:
        buffer = io.BytesIO()
        
        export_data = {
            'model': model,
            'model_type': model_type,
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'additional_data': additional_data
        }
        
        pickle.dump(export_data, buffer)
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.read()).decode()
        filename = f"{model_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {model_type} Model</a>'
        
        return href
    except Exception as e:
        st.error(f"Error exporting model: {str(e)}")
        return None

def create_prediction_playground(model, model_type, additional_data=None):
    """
    Create an interactive playground for predicting the next action based on a sequence of events.
    
    Args:
        model: The trained model to use for predictions
        model_type: String indicating the type of model
        additional_data: Dictionary containing any additional data needed for the model
    """
    st.write("Select a sequence of events and see what the model predicts will happen next.")
    
    event_options = [
        'account-lines::layout:render', 'account-lines:::view',
        'account-lines::widget:render',
        'account-lines::configurable-table:render',
        'application-window-opened', '::nav-header:action-center-click',
        'action-center:::view', 'action-center:action-details::view',
        'action-center:action-details:response-form:submit-click',
        'account:::view', 'session_start', 'session_end',
        'action-center:::close-click', ':all-accounts:layout:render',
        ':all-accounts::view', ':all-accounts:configurable-table:render',
        ':all-accounts:widget:render', '::configurable-table:render',
        '::widget:render', '::layout:render',
        'submissions:exposures-create::submit-click',
        'submissions:all-exposures:configurable-table:render',
        'submissions:all-exposures::view',
        'submissions:exposures-create::view',
        'submissions:all-policy:configurable-table:render',
        'submissions:all-policy::view',
        'account-lines:::change-rating-click',
        'account-property-rating:perils:configurable-table:render',
        'account-property-rating:perils::view',
        'dashboard:my-book:layout:render', 'dashboard:my-book::view',
        'dashboard:my-book:widget:render',
        'dashboard:my-book:configurable-table:render',
        'triaged-submission-list:my-book:configurable-table:render',
        'triaged-submission-list:my-book::view',
        'submissions:all-renewal::view',
        'submissions:renewal-create::view',
        'submissions:renewal-create::submit-click',
        'submissions:all-renewal:configurable-table:render',
        'submissions:renewal-definition::view',
        'submissions:triaged_submissions-definition::view',
        'triaged-submission:triaged_submissions-definition:layout:render',
        'triaged-submission:triaged_submissions-definition::view',
        'triaged-submission:triaged_submissions-definition:widget:render',
        'triaged-submission-list:triaged_submissions-definition:configurable-table:render',
        'triaged-submission-list:triaged_submissions-definition::view',
        'submissions:policy-definition::view',
        'submissions:policy-definition:configurable-table:render',
        'submissions:policy-definition::submit-click',
        'account-property-rating:perils:perils-table:add-click',
        'account-property-rating:perils:perils-table:edit-click',
        'action-center:::submit-click',
        'submissions:all-ingest_policy_through_pd:configurable-table:render',
        'submissions:all-ingest_policy_through_pd::view',
        'submissions:ingest_policy_through_pd-create::view',
        'account-lines::templeton-docs:create-document-click',
        'submissions:policy-create::view',
        'dashboard:portfolio-insights:layout:render',
        'dashboard:portfolio-insights::view',
        'dashboard:portfolio-insights:widget:render',
        'account-lines::construction-excess-rater:modify-existing-quote-click',
        'submissions:policy-create:configurable-table:render',
        'submissions:policy-create::submit-click',
        'dashboard:my-book::action-click', 'EMPTY',
        'submissions:all-financial_lines::view',
        'submissions:all-account:configurable-table:render',
        'submissions:all-account::view',
        'submissions:exposures-definition::view',
        'dashboard:my-book:recent-actions-table:account-click',
        'account-property-rating:perils:model-request-details:save-click',
        'dashboard:my-book:recent-actions-table:action-click',
        'assigned-email-thread:::email-thread-expansion',
        'assigned-email-thread:::document-download-click',
        'account-auto-rating:::view',
        'account-auto-rating::configurable-table:render',
        '::nav-header:help-menu-opened',
        'all-accounts:renewals:layout:render',
        'all-accounts:renewals::view',
        'all-accounts:renewals:configurable-table:render',
        'all-accounts:new-business:layout:render',
        'all-accounts:new-business::view',
        'all-accounts:new-business:configurable-table:render'
    ]
    
    event_options.sort()
    
    if 'playground_sequence' not in st.session_state:
        st.session_state.playground_sequence = []
    
    sequence_container = st.container()
    
    with sequence_container:
        st.write("Select events for your sequence:")
        selected_event = st.selectbox("Add an event to the sequence", [""] + event_options)
        
        if selected_event and st.button("Add Event"):
            # if len(st.session_state.playground_sequence) < 5:
            st.session_state.playground_sequence.append(selected_event)
            # else:
            #     st.warning("Maximum sequence length reached (5 events).")
        
        if st.session_state.playground_sequence:
            st.write("Current sequence:")
            for i, event in enumerate(st.session_state.playground_sequence):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.write(f"{i+1}. {event}")
                with col2:
                    if st.button("✖", key=f"remove_{i}"):
                        st.session_state.playground_sequence.pop(i)
                        st.rerun()
    
    if st.button("Predict Next Event"):
        sequence = st.session_state.playground_sequence
        if not sequence:
            st.warning("Please select at least one event in the sequence.")
        else:
            st.write(f"Selected sequence: {sequence}")
            
            if model_type == "Markov Chain":
                if additional_data and 'transition_probs' in additional_data:
                    transition_probs = additional_data['transition_probs']
                    
                    last_event = sequence[-1]
                    
                    next_events = []
                    
                    for time_of_day in ['morning', 'afternoon', 'evening', 'night']:
                        event_state = f"{last_event}|{time_of_day}"
                        if event_state in transition_probs:
                            for next_event, prob in transition_probs[event_state].items():
                                next_event_name = next_event.split('|')[0]
                                next_events.append((next_event_name, prob))
                    
                    if next_events:
                        next_events.sort(key=lambda x: x[1], reverse=True)
                        
                        st.subheader("Top 5 Most Likely Next Events")
                        result_df = pd.DataFrame(next_events[:5], columns=["Event", "Probability"])
                        st.dataframe(result_df)
                        
                        fig = px.bar(result_df, x="Probability", y="Event", orientation='h',
                                    title="Next Event Probabilities",
                                    labels={"Probability": "Probability", "Event": "Event"})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No transition data available for event '{last_event}'.")
                else:
                    st.error("Transition probability data not available. Please retrain the model.")
            
            elif model_type == "XGBoost Prediction":
                if model is not None and additional_data and 'event_encoder' in additional_data and 'le_device' in additional_data:
                    event_encoder = additional_data['event_encoder']
                    le_device = additional_data['le_device']
                    
                    hour = 12
                    weekday = 2
                    
                    device_options = le_device.classes_.tolist()
                    default_device = device_options[0] if device_options else None
                    
                    with st.expander("Customize Context Features"):
                        hour = st.slider("Hour of Day", 0, 23, hour)
                        weekday = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, weekday)
                        device = st.selectbox("Device", device_options, 0) if device_options else None
                    
                    try:
                        if not sequence or not device_options:
                            st.warning("Please ensure you have selected at least one event and device options are available.")
                            return
                            
                        prev_event_encoded = event_encoder.transform([sequence[-1]])[0]
                        device_encoded = le_device.transform([device])[0]
                        
                        X_test = np.array([[hour, weekday, prev_event_encoded, device_encoded]])
                        
                        prediction = model.predict(X_test)[0]
                        prediction_proba = model.predict_proba(X_test)[0]
                        
                        top_indices = prediction_proba.argsort()[-5:][::-1]
                        top_events = [event_encoder.classes_[i] for i in top_indices]
                        top_probs = [prediction_proba[i] for i in top_indices]
                        
                        st.subheader("Predicted Next Event")
                        st.success(f"**{event_encoder.classes_[prediction]}**")
                        
                        st.subheader("Top 5 Most Likely Next Events")
                        result_df = pd.DataFrame({
                            "Event": top_events,
                            "Probability": top_probs
                        })
                        st.dataframe(result_df)
                        
                        fig = px.bar(result_df, x="Probability", y="Event", orientation='h',
                                    title="Next Event Probabilities",
                                    labels={"Probability": "Probability", "Event": "Event"})
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write(f"Error details: {traceback.format_exc()}")
                else:
                    st.error("Model or encoder data not available. Please retrain the model.")
            
            elif model_type == "LSTM Prediction":
                if model is not None and additional_data and 'event_encoder' in additional_data:
                    event_encoder = additional_data['event_encoder']
                    
                    try:
                        encoded_sequence = [event_encoder.transform([event])[0] for event in sequence]
                        
                        max_len = 10
                        if len(encoded_sequence) < max_len:
                            encoded_sequence = [0] * (max_len - len(encoded_sequence)) + encoded_sequence
                        else:
                            encoded_sequence = encoded_sequence[-max_len:]
                        
                        sequence_tensor = torch.FloatTensor(encoded_sequence).unsqueeze(0).unsqueeze(-1)
                        
                        model.eval()
                        with torch.no_grad():
                            output = model(sequence_tensor)
                            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        
                        top_indices = probabilities.argsort()[-5:][::-1]
                        top_events = [event_encoder.classes_[i] for i in top_indices]
                        top_probs = [probabilities[i] for i in top_indices]
                        
                        st.subheader("Predicted Next Event")
                        st.success(f"**{top_events[0]}**")
                        
                        st.subheader("Top 5 Most Likely Next Events")
                        result_df = pd.DataFrame({
                            "Event": top_events,
                            "Probability": top_probs
                        })
                        st.dataframe(result_df)
                        
                        fig = px.bar(result_df, x="Probability", y="Event", orientation='h',
                                    title="Next Event Probabilities",
                                    labels={"Probability": "Probability", "Event": "Event"})
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write(f"Error details: {traceback.format_exc()}")
                else:
                    st.error("Model or encoder data not available. Please retrain the model.")
            
            elif model_type == "Hidden Markov Model":
                if model is not None and additional_data and 'event_encoder' in additional_data:
                    event_encoder = additional_data['event_encoder']
                    
                    try:
                        encoded_sequence = [event_encoder.transform([event])[0] for event in sequence]
                        
                        sequence_array = np.array(encoded_sequence).reshape(-1, 1)
                        
                        state_sequence = model.predict(sequence_array)
                        last_state = state_sequence[-1]
                        
                        next_state_probs = model.transmat_[last_state]
                        
                        top_state_indices = next_state_probs.argsort()[-5:][::-1]
                        
                        event_probs = {}
                        n_samples = 1000
                        
                        for state_idx in top_state_indices:
                            samples = np.random.normal(
                                model.means_[state_idx][0], 
                                np.sqrt(model.covars_[state_idx][0][0]), 
                                n_samples
                            )
                            
                            samples = np.clip(np.round(samples).astype(int), 0, len(event_encoder.classes_) - 1)
                            
                            unique, counts = np.unique(samples, return_counts=True)
                            
                            for event_idx, count in zip(unique, counts):
                                event_prob = (count / n_samples) * next_state_probs[state_idx]
                                if event_idx in event_probs:
                                    event_probs[event_idx] += event_prob
                                else:
                                    event_probs[event_idx] = event_prob
                        
                        top_events_with_probs = sorted(event_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_indices = [idx for idx, _ in top_events_with_probs]
                        top_probs = [prob for _, prob in top_events_with_probs]
                        
                        total_prob = sum(top_probs)
                        if total_prob > 0:
                            top_probs = [p/total_prob for p in top_probs]
                        
                        top_events = [event_encoder.classes_[i] for i in top_indices]
                        
                        st.subheader("Predicted Next Event")
                        st.success(f"**{top_events[0]}**")
                        
                        st.subheader("Top 5 Most Likely Next Events")
                        result_df = pd.DataFrame({
                            "Event": top_events,
                            "Probability": top_probs
                        })
                        st.dataframe(result_df)
                        
                        fig = px.bar(result_df, x="Probability", y="Event", orientation='h',
                                    title="Next Event Probabilities",
                                    labels={"Probability": "Probability", "Event": "Event"})
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write(f"Error details: {traceback.format_exc()}")
                else:
                    st.error("Model or encoder data not available. Please retrain the model.")
            else:
                st.info(f"Real-time prediction is not supported for {model_type} models.")
    
    if st.button("Clear Sequence"):
        st.session_state.playground_sequence = []
        st.rerun()

def main():
    st.set_page_config(
        page_title="Predictive Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Predictive Analytics Dashboard")
    st.write("Comprehensive predictive analyses of user behavior and events.")

    show_dashboard_overview()
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.df_processed = None
        st.session_state.current_model = None
        st.session_state.current_model_type = None
        st.session_state.additional_data = None
        st.session_state.model_trained = False
    
    tabs = st.tabs(["Analysis", "Playground", "Export"])
    
    with tabs[0]:
        try:
            if not st.session_state.data_loaded:
                with st.spinner("Loading and preprocessing data..."):
                    df = load_and_preprocess_data()
                    st.session_state.df = df
                    
                    st.subheader("Initial Dataset Overview")
                    show_data_summary(df)
                    
                    df = create_time_features(df)
                    
                    df_processed = preprocess_data(df)
                    st.session_state.df_processed = df_processed
                    
                    st.subheader("Preprocessed Dataset Overview")
                    show_data_summary(df_processed)
                    
                    st.session_state.data_loaded = True
            else:
                df = st.session_state.df
                df_processed = st.session_state.df_processed
            
            analysis_type = st.sidebar.selectbox(
                "Choose Analysis Type",
                ["Markov Chain", "Hidden Markov Model", "Prophet Forecast",
                 "ARIMA Analysis", "LSTM Prediction", "KMeans Clustering",
                 "XGBoost Prediction"]
            )
            
            show_model_explanation(analysis_type)
            
            force_retrain = st.sidebar.button("Train/Retrain Model")
            
            run_analysis = False
            if force_retrain:
                run_analysis = True
                st.session_state.model_trained = False
            elif st.session_state.current_model_type != analysis_type:
                st.session_state.model_trained = False

            if not st.session_state.model_trained:
                st.info(f"No {analysis_type} model has been trained yet. Click 'Train/Retrain Model' to train.")
            
            if run_analysis:
                progress_bar = st.progress(0)
                st.write("Running analysis...")
                
                if analysis_type == "Markov Chain":
                    transition_probs, avg_transition_times = perform_markov_chain_analysis(df_processed)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = None
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = {
                        'transition_probs': transition_probs,
                        'avg_transition_times': avg_transition_times
                    }
                    st.session_state.model_trained = True
                    
                elif analysis_type == "Hidden Markov Model":
                    df_hmm = df_processed.copy()
                    
                    le = LabelEncoder()
                    df_hmm['event_encoded'] = le.fit_transform(df_hmm['event_type'])
                    
                    model, le = perform_hmm_analysis(df_hmm)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = model
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = {
                        'event_encoder': le
                    }
                    st.session_state.model_trained = True
                    
                elif analysis_type == "Prophet Forecast":
                    forecasts = perform_prophet_forecast(df_processed)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = forecasts
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = None
                    st.session_state.model_trained = True
                    
                elif analysis_type == "ARIMA Analysis":
                    results = perform_arima_analysis(df_processed)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = results
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = None
                    st.session_state.model_trained = True
                    
                elif analysis_type == "LSTM Prediction":
                    df_lstm = df_processed.copy()
                    
                    le = LabelEncoder()
                    df_lstm['event_encoded'] = le.fit_transform(df_lstm['event_type'])
                    
                    model, metrics, le = perform_lstm_prediction(df_lstm)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = model
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = {
                        'event_encoder': le,
                        'metrics': metrics
                    }
                    st.session_state.model_trained = True
                    
                elif analysis_type == "KMeans Clustering":
                    model, stats = perform_kmeans_clustering(df_processed)
                    progress_bar.progress(100)
                    
                    st.subheader("Cluster Statistics")
                    st.dataframe(stats)
                    
                    st.session_state.current_model = model
                    st.session_state.current_model_type = analysis_type
                    st.session_state.additional_data = {
                        'stats': stats
                    }
                    st.session_state.model_trained = True
                    
                elif analysis_type == "XGBoost Prediction":
                    model, importance, event_encoder, le_device = perform_xgboost_prediction(df_processed)
                    progress_bar.progress(100)
                    
                    st.session_state.current_model = model
                    st.session_state.current_model_type = analysis_type
                    
                    if model is not None and event_encoder is not None and le_device is not None:
                        st.session_state.additional_data = {
                            'importance': importance,
                            'event_encoder': event_encoder,
                            'le_device': le_device
                        }
                        st.session_state.model_trained = True
                    else:
                        st.session_state.additional_data = None
                        st.session_state.model_trained = False
                
                if st.session_state.model_trained:
                    st.success(f"{analysis_type} model trained successfully!")
            
            if st.sidebar.button("Clear Cache and Reload Data"):
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.session_state.df_processed = None
                st.session_state.current_model = None
                st.session_state.current_model_type = None
                st.session_state.additional_data = None
                st.session_state.model_trained = False
                st.rerun()
        
        except Exception as e:
            st.error(f"Error performing analysis: {str(e)}")
            st.write("Please ensure all required data files are present and properly formatted.")
            st.write("Full error details:", traceback.format_exc())
    
    with tabs[1]:
        st.header("Real-Time Prediction Playground")
        
        if not st.session_state.data_loaded:
            st.warning("Please run an analysis first to use the playground.")
        elif not st.session_state.model_trained:
            st.warning(f"Please train a {st.session_state.current_model_type if st.session_state.current_model_type else 'model'} first to use the playground.")
        elif st.session_state.current_model_type not in ["Markov Chain", "XGBoost Prediction", "LSTM Prediction", "Hidden Markov Model"]:
            st.info(f"Real-time prediction is not supported for {st.session_state.current_model_type} models.")
        else:
            create_prediction_playground(
                st.session_state.current_model,
                st.session_state.current_model_type,
                st.session_state.additional_data
            )
    
    with tabs[2]:
        st.header("Export Model")
        
        if not st.session_state.data_loaded:
            st.warning("Please run an analysis first to export a model.")
        elif not st.session_state.model_trained:
            st.warning("No model has been trained yet. Please train a model first.")
        else:
            st.write(f"Current model: **{st.session_state.current_model_type}**")
            
            export_format = st.selectbox(
                "Export Format",
                ["Pickle (.pkl)", "JSON (.json)"]
            )
            
            if st.button("Generate Export"):
                if export_format == "Pickle (.pkl)":
                    export_link = export_model(
                        st.session_state.current_model,
                        st.session_state.current_model_type,
                        st.session_state.additional_data
                    )
                    
                    if export_link:
                        st.markdown(export_link, unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate export.")
                
                elif export_format == "JSON (.json)":
                    try:
                        if st.session_state.current_model_type == "Markov Chain":
                            export_data = {
                                'model_type': st.session_state.current_model_type,
                                'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'transition_probs': {
                                    k: {k2: float(v2) for k2, v2 in v.items()}
                                    for k, v in st.session_state.additional_data['transition_probs'].items()
                                }
                            }
                            
                            json_str = json.dumps(export_data, indent=2)
                            b64 = base64.b64encode(json_str.encode()).decode()
                            filename = f"markov_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download Markov Chain Model (JSON)</a>'
                            
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.warning("JSON export is currently only supported for Markov Chain models.")
                    except Exception as e:
                        st.error(f"Error exporting model to JSON: {str(e)}")
            
            with st.expander("How to Use Exported Models"):
                st.markdown("""
                ### Using Exported Models
                
                #### Pickle Format (.pkl)
                Pickle files contain the serialized model and can be loaded in Python:
                
                ```python
                import pickle
                
                # Load the model
                with open('model.pkl', 'rb') as f:
                    export_data = pickle.load(f)
                
                model = export_data['model']
                model_type = export_data['model_type']
                additional_data = export_data['additional_data']
                
                # Use the model for predictions
                # (The exact code depends on the model type)
                ```
                
                #### JSON Format (.json)
                JSON files contain a simplified version of the model and can be loaded in any language:
                
                ```python
                import json
                
                # Load the model
                with open('model.json', 'r') as f:
                    export_data = json.load(f)
                
                model_type = export_data['model_type']
                
                # For Markov Chain models
                if model_type == 'Markov Chain':
                    transition_probs = export_data['transition_probs']
                    
                    # Example: Get the most likely next event
                    def predict_next_event(current_event, time_of_day):
                        event_state = f"{current_event}|{time_of_day}"
                        if event_state in transition_probs:
                            next_events = transition_probs[event_state]
                            return max(next_events.items(), key=lambda x: x[1])[0]
                        return None
                ```
                """)

if __name__ == "__main__":
    main() 