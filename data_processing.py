import streamlit as st
import pandas as pd
import glob
import plotly.express as px

st.set_page_config(page_title="Data Processing Steps", layout="wide")

st.title("🔄 Data Processing Steps")
st.write("Progressive data cleaning steps showing column removal rationale and impact.")

@st.cache_data
def load_data(year=2025):
    """Load all CSV chunks for a given year and combine them into a single DataFrame."""
    csv_files = glob.glob(f"./federato/{year}_csv/*_chunk_*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def show_shape_change(old_shape, new_shape, step_name):
    """Show the difference in shape after a processing step."""
    rows_diff = new_shape[0] - old_shape[0]
    cols_diff = new_shape[1] - old_shape[1]
    
    if rows_diff != 0:
        st.write(f"🔢 Rows: {rows_diff:+d} ({new_shape[0]} remaining)")

    if cols_diff != 0:
        st.write(f"📊 Columns: {cols_diff:+d} ({new_shape[1]} remaining)")

def show_column_info(df, column_name, show_viz=True):
    """Display information about a specific column."""
    if column_name in ['user_properties', 'event_properties', 'data'] or not show_viz or len(df[column_name].value_counts()) > 20:
        st.write(f"**{column_name}**")
        if column_name in ['user_properties', 'event_properties', 'data']:
            non_null_count = df[column_name].count()
            null_count = len(df) - non_null_count
            st.write(f"Non-null values: {non_null_count:,} ({(non_null_count/len(df)*100):.1f}%)")
            st.write(f"Null values: {null_count:,} ({(null_count/len(df)*100):.1f}%)")
            
            unique_count = df[column_name].nunique()
            st.write(f"Unique values: {unique_count:,}")
            
            value_counts = df[column_name].value_counts()
            percentages = (value_counts / len(df) * 100).round(2)
            distribution_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': percentages
            })
            rows_to_show = min(5, len(distribution_df))
            if rows_to_show > 0:
                st.write(f"Top {rows_to_show} most common structures:")
                table_height = (rows_to_show + 1) * 35
                st.dataframe(distribution_df.head(rows_to_show), height=table_height, use_container_width=True)
            
            st.write("Sample structure:")
            st.json(df[column_name].iloc[0])
            return
        
        value_counts = df[column_name].value_counts()
        percentages = (value_counts / len(df) * 100).round(2)
        distribution_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages
        })
    
        num_rows = len(distribution_df)
        rows_to_show = num_rows if num_rows <= 5 else 5
        st.write(f"Showing {rows_to_show} of {num_rows:,} unique values")
        table_height = (rows_to_show + 1) * 35
        st.dataframe(distribution_df.head(rows_to_show), height=table_height, use_container_width=True)
        return
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**{column_name}**")
        value_counts = df[column_name].value_counts()
        percentages = (value_counts / len(df) * 100).round(2)
        distribution_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages
        })
    
        num_rows = len(distribution_df)
        rows_to_show = num_rows if num_rows <= 5 else 5
        st.write(f"Showing {rows_to_show} of {num_rows:,} unique values")
        table_height = (rows_to_show + 1) * 35
        st.dataframe(distribution_df.head(rows_to_show), height=table_height, use_container_width=True)
    
    with col2:
        fig = px.bar(distribution_df.head(10), 
                    title=f"Distribution of {column_name}",
                    labels={'index': column_name, 'Count': 'Count'},
                    height=200)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

def show_code_snippet(step_info, is_duplicate_removal=False):
    """Display the code snippet for a processing step."""
    if is_duplicate_removal:
        code = """df = df.drop_duplicates()"""
    else:
        cols = step_info["columns"]
        code = f"""df = df.drop(columns={cols})"""
    st.code(code, language="python")

def show_nullity_analysis(df):
    st.write("### 📊 Nullity Analysis")
    
    nullity_stats = pd.DataFrame({
        'Non-null Count': df.count(),
        'Null Count': df.isna().sum(),
        'Null Percentage': (df.isna().sum() / len(df) * 100).round(2)
    }).sort_values('Null Percentage', ascending=False)
    
    nullity_stats['Total Rows'] = len(df)
    
    nullity_stats = nullity_stats[['Total Rows', 'Non-null Count', 'Null Count', 'Null Percentage']]
    
    st.write("Nullity statistics for each column:")
    table_height = (len(nullity_stats) + 1) * 35
    st.dataframe(nullity_stats, height=table_height, use_container_width=True)
    
    cols_with_nulls = nullity_stats[nullity_stats['Null Count'] > 0]
    if not cols_with_nulls.empty:
        st.write(f"Found {len(cols_with_nulls)} columns with null values:")
        for idx, (col, row) in enumerate(cols_with_nulls.iterrows(), 1):
            st.write(f"{idx}. `{col}`: {row['Null Percentage']:.1f}% null ({row['Null Count']:,} rows)")

def show_unique_analysis(df):
    st.write("### 🔍 Unique Values Analysis")
    
    unique_stats = pd.DataFrame({
        'Unique Values': df.nunique(),
        'Total Values': df.count(),
        'Uniqueness %': (df.nunique() / df.count() * 100).round(2)
    }).sort_values('Uniqueness %', ascending=False)
    
    st.write("Unique value statistics for each column:")
    table_height = (len(unique_stats) + 1) * 35
    st.dataframe(unique_stats, height=table_height, use_container_width=True)
    
    # Categorize columns by cardinality
    high_cardinality = unique_stats[unique_stats['Uniqueness %'] > 90].index.tolist()
    single_value = unique_stats[unique_stats['Unique Values'] == 1].index.tolist()
    low_cardinality = unique_stats[(unique_stats['Unique Values'] > 1) & (unique_stats['Unique Values'] <= 5)].index.tolist()
    
    # Show summaries
    if high_cardinality:
        st.write("🔴 High cardinality columns (>90% unique):")
        for col in high_cardinality:
            st.write(f"- `{col}`: {unique_stats.loc[col, 'Unique Values']:,} unique values")
    
    if single_value:
        st.write("⚪ Single value columns:")
        for col in single_value:
            value = df[col].iloc[0]
            st.write(f"- `{col}`: constant value = '{value}'")
    
    if low_cardinality:
        st.write("🟢 Low cardinality columns (2-5 unique values):")
        for col in low_cardinality:
            values = df[col].value_counts().index.tolist()
            st.write(f"- `{col}`: {values}")

def main():
    try:
        df = load_data()
        if df is None:
            return
        
        st.write(f"📊 Initial dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        show_nullity_analysis(df)
        show_unique_analysis(df)
        
        st.write("### 🕒 Timestamp Filtering")
        cutoff_date = pd.Timestamp('2025-01-01')
        initial_rows = len(df)
        df = df[pd.to_datetime(df['event_time']) >= cutoff_date]
        rows_removed = initial_rows - len(df)
        st.write(f"Removed {rows_removed:,} rows with event_time before 2025 ({(rows_removed/initial_rows*100):.1f}% of data)")
        st.write(f"Remaining rows: {len(df):,}")
        
        st.write("### 📊 Session Analysis")
        session_df = df.copy()
        session_df['event_time'] = pd.to_datetime(session_df['event_time'])
        
        session_analysis = session_df.groupby('session_id').agg({
            'event_time': ['min', 'max'],
            'event_type': ['first', 'last']
        }).reset_index()
        
        session_analysis.columns = ['session_id', 'start_time', 'end_time', 'first_event', 'last_event']
        session_analysis['duration_minutes'] = (session_analysis['end_time'] - session_analysis['start_time']).dt.total_seconds() / 60
        
        st.write("#### Session Event Analysis")
        st.write(f"Total unique sessions: {len(session_analysis):,}")
        
        first_events = session_analysis['first_event'].value_counts()
        st.write("**Most Common First Events:**")
        first_events_df = pd.DataFrame({
            'Event Type': first_events.index,
            'Count': first_events.values,
            'Percentage': (first_events.values / len(session_analysis) * 100).round(2)
        })
        st.dataframe(first_events_df.head(), use_container_width=True)
        
        last_events = session_analysis['last_event'].value_counts()
        st.write("**Most Common Last Events:**")
        last_events_df = pd.DataFrame({
            'Event Type': last_events.index,
            'Count': last_events.values,
            'Percentage': (last_events.values / len(session_analysis) * 100).round(2)
        })
        st.dataframe(last_events_df.head(), use_container_width=True)
        
        st.write("**Session Duration Statistics (minutes):**")
        duration_stats = session_analysis['duration_minutes'].describe().round(2)
        st.dataframe(pd.DataFrame(duration_stats), use_container_width=True)
        
        st.write("### ⏰ Event Time Analysis")
        total_rows = len(df)
        matching_rows = len(df[df['event_time'] == df['client_event_time']])
        st.write(f"Total rows: {total_rows:,}")
        st.write(f"Rows where event_time = client_event_time: {matching_rows:,}")
        if total_rows == matching_rows:
            st.info("💡 event_time and client_event_time are identical. We can safely drop client_event_time and other redundant time columns.")
        
        st.write("### 🧹 Removing Duplicates")
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            st.write(f"Removed {duplicates_removed:,} duplicate rows ({(duplicates_removed/initial_rows*100):.1f}% of data)")
        else:
            st.write("No duplicates found in the dataset")
        
        duplicate_reason = """Apparently no duplicates were found."""
        
        with st.expander("1️⃣ Remove duplicates", expanded=True):
            st.write("📝 Code:")
            show_code_snippet({}, is_duplicate_removal=True)
            if duplicate_reason:
                st.info(f"💭 {duplicate_reason}")
            st.write(f"🔢 Rows: {duplicates_removed:+d} ({len(df):,} remaining)")
        
        steps = {
            "2️⃣ Remove user IDs (sensitive, high cardinality)": {
                "columns": ['user_id', 'device_id', '$insert_id', 'uuid', 'amplitude_id', 'session_id'],
                "reason": """User IDs have high cardinality (many unique values) that aren't useful for aggregate analysis."""
            },
            "3️⃣ Remove redundant info": {
                "columns": ['platform', 'library', 'device_type', 'app', 'client_event_time', 'processed_time', 
                          'client_upload_time', 'server_received_time', 'server_upload_time'],
                "reason": """- Platform info: platform, library, and app columns have 1 unique value each, device_type is redundant with device_family
- Time columns: all these timestamps are redundant with event_time which we'll keep as our source of truth"""
            },
            "4️⃣ Remove nested data (complex)": {
                "columns": ['user_properties', 'event_properties', 'data'],
                "reason": """These columns contain complex nested JSON structures that are specific to each event type.
We'll analyze these separately as they require custom parsing for each event category."""
            }
        }
        
        df_cleaned = df.copy()
        for step_name, step_info in steps.items():
            with st.expander(f"{step_name}", expanded=True):
                old_shape = df_cleaned.shape
                cols_to_drop = [col for col in step_info['columns'] if col in df_cleaned.columns]
                
                if cols_to_drop:
                    st.write("📝 Code:")
                    show_code_snippet({"columns": cols_to_drop})
                    if step_info.get("reason"):
                        st.info(f"💭 {step_info['reason']}")
                    st.write("📊 Column distributions:")
                    for col in cols_to_drop:
                        show_column_info(df_cleaned, col)
                
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                show_shape_change(old_shape, df_cleaned.shape, step_name)
        
        st.header("Final Dataset")
        st.write(f"Final shape: {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
        st.write("Remaining columns:", ", ".join(df_cleaned.columns))
        
        show_nullity_analysis(df_cleaned)
        show_unique_analysis(df_cleaned)
        
        st.write("### Sample Data")
        st.dataframe(df_cleaned.head())
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Please ensure the CSV files are present in the correct location.")

if __name__ == "__main__":
    main() 