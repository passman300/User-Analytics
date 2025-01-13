import pandas as pd
import glob
import json

def load_data(year=2025):
    """Load all CSV chunks for a given year and combine them into a single DataFrame."""
    csv_files = glob.glob(f"./federato/{year}_csv/*_chunk_*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def analyze_data_quality(df):
    """Analyze data quality and patterns for preprocessing decisions."""
    quality_analysis = {
        'missing_values': {},
        'unique_value_counts': {},
        'data_types': {},
        'value_ranges': {},
        'potential_outliers': {},
        'temporal_patterns': {},
        'categorical_imbalance': {},
        'correlation_analysis': {}
    }
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    quality_analysis['missing_values'] = {
        'counts': missing_counts.to_dict(),
        'percentages': missing_percentages.to_dict()
    }
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_counts = df[col].nunique()
        value_distribution = df[col].value_counts().head(10).to_dict()
        quality_analysis['unique_value_counts'][col] = {
            'unique_count': unique_counts,
            'top_values': value_distribution
        }
    
    if 'date' in df.columns:
        unique_counts = df['date'].nunique()
        value_distribution = df['date'].value_counts().head(10)
        value_distribution_dict = {date.strftime('%Y-%m-%d'): int(count) for date, count in value_distribution.items()}
        quality_analysis['unique_value_counts']['date'] = {
            'unique_count': unique_counts,
            'top_values': value_distribution_dict
        }
    
    quality_analysis['data_types'] = df.dtypes.astype(str).to_dict()
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        stats = df[col].describe()
        quality_analysis['value_ranges'][col] = stats.to_dict()
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
        quality_analysis['potential_outliers'][col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100
        }
    
    if 'client_event_time' in df.columns:
        client_event_time = pd.to_datetime(df['client_event_time'])
        daily_counts = client_event_time.dt.date.value_counts().sort_index()
        daily_counts_dict = {date.strftime('%Y-%m-%d'): int(count) for date, count in daily_counts.items()}
        
        quality_analysis['temporal_patterns'] = {
            'daily_event_counts': daily_counts_dict,
            'weekday_distribution': client_event_time.dt.dayofweek.value_counts().sort_index().to_dict()
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1
        quality_analysis['categorical_imbalance'][col] = {
            'imbalance_ratio': float(imbalance_ratio),
            'dominant_class_percentage': float((value_counts.max() / len(df)) * 100)
        }
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        strong_correlations = {}
        for col1 in correlation_matrix.columns:
            strong_corr = correlation_matrix[col1][abs(correlation_matrix[col1]) > 0.5]
            if len(strong_corr) > 1:
                strong_correlations[col1] = {str(k): float(v) for k, v in strong_corr.to_dict().items()}
        quality_analysis['correlation_analysis'] = strong_correlations
    
    return quality_analysis

def analyze_event_patterns(df):
    """Analyze event patterns and user behavior for preprocessing decisions."""
    event_analysis = {
        'session_patterns': {},
        'user_patterns': {},
        'event_sequences': {},
        'time_between_events': {}
    }
    
    session_stats = df.groupby('session_id').size().describe()
    event_analysis['session_patterns'] = {
        'events_per_session': {k: float(v) for k, v in session_stats.to_dict().items()},
        'session_duration': {
            k: float(v) for k, v in df.groupby('session_id').agg({
                'client_event_time': lambda x: (max(x) - min(x)).total_seconds()
            }).describe().to_dict()['client_event_time'].items()
        }
    }
    
    user_stats = df.groupby('user_id').size().describe()
    session_stats = df.groupby('user_id')['session_id'].nunique().describe()
    event_analysis['user_patterns'] = {
        'events_per_user': {k: float(v) for k, v in user_stats.to_dict().items()},
        'sessions_per_user': {k: float(v) for k, v in session_stats.to_dict().items()}
    }
    
    df_sorted = df.sort_values(['user_id', 'client_event_time'])
    event_pairs = df_sorted.groupby('user_id')['event_type'].apply(
        lambda x: pd.Series(zip(x, x.shift(-1)))
    ).value_counts().head(10)
    
    event_pairs_dict = {f"{pair[0]} -> {pair[1]}": int(count) for pair, count in event_pairs.items()}
    event_analysis['event_sequences'] = {
        'common_event_pairs': event_pairs_dict
    }
    
    df_sorted['time_to_next'] = df_sorted.groupby('user_id')['client_event_time'].diff().dt.total_seconds()
    time_stats = df_sorted['time_to_next'].describe()
    event_type_means = df_sorted.groupby('event_type')['time_to_next'].mean().sort_values(ascending=False).head(10)
    
    event_analysis['time_between_events'] = {
        'overall_stats': {k: float(v) for k, v in time_stats.to_dict().items()},
        'by_event_type': {k: float(v) for k, v in event_type_means.to_dict().items()}
    }
    
    return event_analysis

def analyze_user_journey(df):
    """Analyze user journeys and event sequences."""
    journey_analysis = {
        'event_flows': {
            'sequences_2': {},
            'sequences_3': {},
            'sequences_4': {},
            'sequences_5': {}
        },
        'path_analysis': {},
        'user_segments': {},
        'conversion_funnels': {}
    }
    
    df_sorted = df.sort_values(['session_id', 'client_event_time'])
    
    event_blacklist = {
        'session_start', 'session_end', '::nav-header:user_signed-out', 
        'user_signed_out', 'user_signed_in', 'page_view', 'page_load'
    }
    df_filtered = df_sorted[~df_sorted['event_type'].isin(event_blacklist)]
    
    session_sequences = df_filtered.groupby('session_id')['event_type'].agg(list).reset_index()
    
    sequence_counts = {2: {}, 3: {}, 4: {}, 5: {}}
    
    for _, row in session_sequences.iterrows():
        events = row['event_type']
        if len(events) >= 2:
            for length in [2, 3, 4, 5]:
                if len(events) >= length:
                    for i in range(len(events) - length + 1):
                        seq = " → ".join(events[i:i+length])
                        sequence_counts[length][seq] = sequence_counts[length].get(seq, 0) + 1
    
    for length in [2, 3, 4, 5]:
        top_limit = 20 if length == 2 else 10
        top_sequences = sorted(
            sequence_counts[length].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_limit]
        
        journey_analysis['event_flows'][f'sequences_{length}'] = {
            sequence: count for sequence, count in top_sequences
        }
    
    entry_points = {}
    exit_points = {}
    
    for _, row in session_sequences.iterrows():
        events = row['event_type']
        if events:
            entry = events[0]
            exit = events[-1]
            entry_points[entry] = entry_points.get(entry, 0) + 1
            exit_points[exit] = exit_points.get(exit, 0) + 1
    
    journey_analysis['path_analysis'] = {
        'common_entry_points': dict(sorted(entry_points.items(), key=lambda x: x[1], reverse=True)[:10]),
        'common_exit_points': dict(sorted(exit_points.items(), key=lambda x: x[1], reverse=True)[:10])
    }
    
    user_event_counts = df.groupby('user_id')['event_type'].value_counts().unstack(fill_value=0)
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    top_events = df['event_type'].value_counts().head(20).index
    X = user_event_counts[top_events]
    X_scaled = StandardScaler().fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_profiles = {}
    for cluster_id in range(5):
        cluster_mask = clusters == cluster_id
        cluster_size = cluster_mask.sum()
        cluster_events = X[cluster_mask].mean()
        top_events_cluster = cluster_events.sort_values(ascending=False).head(5)
        
        cluster_profiles[f"Segment_{cluster_id}"] = {
            'size': int(cluster_size),
            'percentage': float(cluster_size / len(clusters) * 100),
            'top_events': {str(k): float(v) for k, v in top_events_cluster.to_dict().items()}
        }
    
    journey_analysis['user_segments'] = cluster_profiles
    
    common_funnels = {
        'account_creation': [
            'session_start',
            'account:::view',
            'account-lines:::view',
        ],
        'action_completion': [
            'action-center:::view',
            'action-center:action-details::view',
            'action-center:::submit-click'
        ],
        'dashboard_navigation': [
            'dashboard:my-book::view',
            'dashboard:my-book:widget:render',
            'dashboard:my-book::action-click'
        ],
        'submission_workflow': [
            'submissions:all-account::view',
            'submissions:all-policy::view',
            'submissions:policy-create::view',
            'submissions:policy-create::submit-click'
        ],
        'property_rating': [
            'account-lines:::view',
            'account-property-rating:perils::view',
            'account-property-rating:perils:perils-table:edit-click',
            'account-property-rating:perils:model-request-details:save-click'
        ],
        'renewal_process': [
            'submissions:all-renewal::view',
            'submissions:renewal-create::view',
            'submissions:renewal-create::submit-click'
        ]
    }
    
    funnel_analysis = {}
    for funnel_name, funnel_steps in common_funnels.items():
        step_counts = []
        for step in funnel_steps:
            users_at_step = df[df['event_type'] == step]['user_id'].nunique()
            step_counts.append(int(users_at_step))

        print("Adding funnel: ", funnel_name)
        
        funnel_analysis[funnel_name] = {
            'steps': funnel_steps,
            'user_counts': step_counts,
            'conversion_rates': [
                float(step_counts[i] / step_counts[i-1] * 100) if i > 0 else 100.0
                for i in range(len(step_counts))
            ]
        }
    
    journey_analysis['conversion_funnels'] = funnel_analysis
    
    return journey_analysis

def analyze_temporal_relationships(df):
    """Analyze relationships between time-based factors and user behavior."""
    temporal_relationships = {
        'hourly_patterns': {},
        'weekday_patterns': {},
        'monthly_patterns': {},
        'daily_patterns': {},
        'peak_activity': {},
        'session_timing': {}
    }
    
    df['client_event_time'] = pd.to_datetime(df['client_event_time'])
    
    df['hour'] = df['client_event_time'].dt.hour
    df['weekday'] = df['client_event_time'].dt.day_name()
    df['month'] = df['client_event_time'].dt.month
    df['date'] = df['client_event_time'].dt.date
    df['year_month'] = df['client_event_time'].dt.strftime('%Y-%m')
    
    hourly_counts = df.groupby('hour').size()
    temporal_relationships['hourly_patterns'] = {
        'distribution': {str(k): int(v) for k, v in hourly_counts.to_dict().items()},
        'peak_hours': {str(k): int(v) for k, v in hourly_counts.nlargest(5).to_dict().items()},
        'quiet_hours': {str(k): int(v) for k, v in hourly_counts.nsmallest(5).to_dict().items()}
    }
    
    weekday_hour_counts = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    temporal_relationships['weekday_patterns'] = {
        'weekday_distribution': {str(k): int(v) for k, v in df.groupby('weekday').size().to_dict().items()},
        'weekday_hour_matrix': {
            str(weekday): {str(hour): int(count) for hour, count in hours.items()}
            for weekday, hours in weekday_hour_counts.to_dict().items()
        },
        'peak_weekdays': {str(k): int(v) for k, v in df.groupby('weekday').size().nlargest(3).to_dict().items()}
    }
    
    monthly_counts = df.groupby('year_month').size()
    temporal_relationships['monthly_patterns'] = {
        'distribution': {str(k): int(v) for k, v in monthly_counts.to_dict().items()},
        'peak_months': {str(k): int(v) for k, v in monthly_counts.nlargest(3).to_dict().items()},
        'month_over_month_growth': {str(k): float(v) for k, v in (monthly_counts.pct_change() * 100).dropna().to_dict().items()}
    }
    
    daily_counts = df.groupby('date').size()
    temporal_relationships['daily_patterns'] = {
        'distribution': {d.strftime('%Y-%m-%d'): int(count) for d, count in daily_counts.items()},
        'top_10_days': {d.strftime('%Y-%m-%d'): int(count) for d, count in daily_counts.nlargest(10).items()},
        'daily_stats': {
            'mean': float(daily_counts.mean()),
            'median': float(daily_counts.median()),
            'std': float(daily_counts.std())
        }
    }
    
    peak_activity = {}
    
    for weekday in df['weekday'].unique():
        weekday_data = df[df['weekday'] == weekday]
        peak_hours = weekday_data.groupby('hour').size().nlargest(3)
        peak_activity[str(weekday)] = {
            'peak_hours': {str(k): int(v) for k, v in peak_hours.to_dict().items()},
            'total_events': int(len(weekday_data))
        }
    
    temporal_relationships['peak_activity'] = {
        'by_weekday': peak_activity,
        'overall_peak_day': daily_counts.idxmax().strftime('%Y-%m-%d'),
        'overall_peak_day_count': int(daily_counts.max())
    }
    
    df_sorted = df.sort_values(['session_id', 'client_event_time'])
    session_times = df_sorted.groupby('session_id').agg({
        'client_event_time': ['first', 'last']
    })
    session_times.columns = ['start_time', 'end_time']
    session_times['duration'] = (session_times['end_time'] - session_times['start_time']).dt.total_seconds()
    
    temporal_relationships['session_timing'] = {
        'avg_duration_by_hour': {str(k): float(v) for k, v in session_times.groupby(session_times['start_time'].dt.hour)['duration'].mean().to_dict().items()},
        'avg_duration_by_weekday': {str(k): float(v) for k, v in session_times.groupby(session_times['start_time'].dt.day_name())['duration'].mean().to_dict().items()},
        'avg_duration_by_month': {str(k): float(v) for k, v in session_times.groupby(session_times['start_time'].dt.strftime('%Y-%m'))['duration'].mean().to_dict().items()}
    }
    
    return temporal_relationships

def analyze_geographic_relationships(df):
    """Analyze relationships between geographic factors and user behavior."""
    geographic_relationships = {
        'country_device_distribution': {},
        'regional_preferences': {},
        'city_patterns': {}
    }
    
    country_device = pd.crosstab(df['country'], df['device_family'])
    geographic_relationships['country_device_distribution'] = {
        str(country): {str(k): int(v) for k, v in devices.to_dict().items()}
        for country, devices in country_device.iterrows()
    }
    
    region_events = df.groupby(['region', 'event_type']).size().unstack(fill_value=0)
    top_events = df['event_type'].value_counts().head(10).index
    
    geographic_relationships['regional_preferences'] = {
        str(region): {str(k): int(v) for k, v in events[top_events].to_dict().items()}
        for region, events in region_events.iterrows()
    }
    
    city_metrics = df.groupby('city').agg({
        'user_id': 'nunique',
        'session_id': 'nunique',
        'event_type': 'count'
    }).reset_index()
    
    city_metrics.columns = ['city', 'unique_users', 'unique_sessions', 'total_events']
    city_metrics['events_per_user'] = city_metrics['total_events'] / city_metrics['unique_users']
    city_metrics['sessions_per_user'] = city_metrics['unique_sessions'] / city_metrics['unique_users']
    
    city_patterns = []
    for record in city_metrics.sort_values('total_events', ascending=False).head(20).to_dict('records'):
        city_patterns.append({
            'city': str(record['city']),
            'unique_users': int(record['unique_users']),
            'unique_sessions': int(record['unique_sessions']),
            'total_events': int(record['total_events']),
            'events_per_user': float(record['events_per_user']),
            'sessions_per_user': float(record['sessions_per_user'])
        })
    
    geographic_relationships['city_patterns'] = city_patterns
    
    return geographic_relationships

def analyze_data(df):
    """Perform analysis on the DataFrame."""
    print("\nStarting data analysis...")
    results = {}
    
    try:
        print("Converting timestamp columns...")
        timestamp_cols = ['client_event_time', 'client_upload_time', 'event_time', 
                         'processed_time', 'server_received_time', 'server_upload_time']
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col])

        print("Calculating basic stats...")
        results['basic_stats'] = {
            'total_events': int(len(df)),
            'unique_users': int(df['user_id'].nunique()),
            'unique_sessions': int(df['session_id'].nunique()),
            'date_range': {
                'start': df['client_event_time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['client_event_time'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        print("Analyzing device information...")
        results['device_analysis'] = {
            'device_distribution': {str(k): int(v) for k, v in df['device_family'].value_counts().to_dict().items()},
            'platform_distribution': {str(k): int(v) for k, v in df['platform'].value_counts().to_dict().items()},
            'os_distribution': {str(k): int(v) for k, v in df['os_name'].value_counts().to_dict().items()}
        }

        print("Analyzing geographic information...")
        results['geographic_analysis'] = {
            'country_distribution': {str(k): int(v) for k, v in df['country'].value_counts().to_dict().items()},
            'top_10_cities': {str(k): int(v) for k, v in df['city'].value_counts().head(10).to_dict().items()}
        }

        print("Analyzing event distributions...")
        df['hour'] = df['client_event_time'].dt.hour
        results['event_analysis'] = {
            'event_type_distribution': {str(k): int(v) for k, v in df['event_type'].value_counts().to_dict().items()},
            'hourly_distribution': {str(k): int(v) for k, v in df['hour'].value_counts().sort_index().to_dict().items()}
        }

        print("Validating initial results...")
        validate_results(results)

        print("Analyzing user journeys...")
        results['journey_analysis'] = analyze_user_journey(df)
        validate_results({'journey_analysis': results['journey_analysis']})

        print("Analyzing temporal relationships...")
        results['temporal_relationships'] = analyze_temporal_relationships(df)
        validate_results({'temporal_relationships': results['temporal_relationships']})

        print("Analyzing geographic relationships...")
        results['geographic_relationships'] = analyze_geographic_relationships(df)
        validate_results({'geographic_relationships': results['geographic_relationships']})
        
        print("Analyzing data quality...")
        results['data_quality_analysis'] = analyze_data_quality(df)
        validate_results({'data_quality_analysis': results['data_quality_analysis']})

        print("Analyzing event patterns...")
        results['event_pattern_analysis'] = analyze_event_patterns(df)
        validate_results({'event_pattern_analysis': results['event_pattern_analysis']})

        print("Data analysis completed successfully!")
        return results

    except Exception as e:
        print(f"\nERROR during data analysis:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Current results state:")
        for key in results:
            print(f"- {key}")
        raise

def validate_json_type(value, path=""):
    """Validate that a value is JSON serializable and return detailed error if not."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True, None
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            is_valid, error = validate_json_type(item, f"{path}[{i}]")
            if not is_valid:
                return False, error
        return True, None
    elif isinstance(value, dict):
        for key, val in value.items():
            if not isinstance(key, (str, int, float, bool)) or key is None:
                return False, f"Invalid key type at {path}: {type(key).__name__} (key: {key})"
            is_valid, error = validate_json_type(val, f"{path}.{key}")
            if not is_valid:
                return False, error
        return True, None
    else:
        return False, f"Invalid value type at {path}: {type(value).__name__} (value: {value})"

def validate_results(results):
    """Validate that all results are JSON serializable."""
    print("\nValidating results for JSON serializability...")
    for key, value in results.items():
        print(f"Checking section: {key}")
        is_valid, error = validate_json_type(value, key)
        if not is_valid:
            raise TypeError(f"Validation failed in section '{key}': {error}")
    print("All results validated successfully!")

def save_results(results, output_file='analysis_results.json'):
    """Save analysis results to a JSON file."""
    try:
        validate_results(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    except TypeError as e:
        print(f"\nERROR: Failed to save results due to non-serializable data:")
        print(str(e))
        raise
    except Exception as e:
        print(f"\nERROR: Unexpected error while saving results:")
        print(str(e))
        raise

def main():
    print("Loading data...")
    df = load_data()
    
    print("Analyzing data...")
    results = analyze_data(df)
    
    print("Saving results...")
    save_results(results)
    print("Analysis complete. Results saved to analysis_results.json")

if __name__ == "__main__":
    main()
