import streamlit as st
import json
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CUSTOM_COLORS = [
    '#1f77b4',  # Blue
    '#d62728',  # Red
    '#2ca02c',  # Green
    '#9467bd',  # Purple
    '#ff7f0e',  # Orange
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf'   # Cyan
]

def show_code(code):
    """Helper function to display code snippets."""
    st.code(code, language='python')

def load_analysis_results(file_path='./federato/analysis_results.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_device_analysis_plots(results):
    st.header("Device Analysis")
    st.write("Analysis of user devices, platforms, and operating systems.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Device Family Distribution")
        device_df = pd.DataFrame({
            'Device': results['device_analysis']['device_distribution'].keys(),
            'Count': results['device_analysis']['device_distribution'].values()
        })
        device_df['Percentage'] = device_df['Count'] / device_df['Count'].sum() * 100
        fig = px.pie(device_df, values='Count', names='Device',
                    title="Device Family",
                    hover_data=['Percentage'],
                    labels={'Percentage': 'Percentage (%)'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Platform Distribution")
        platform_df = pd.DataFrame({
            'Platform': results['device_analysis']['platform_distribution'].keys(),
            'Count': results['device_analysis']['platform_distribution'].values()
        })
        platform_df['Percentage'] = platform_df['Count'] / platform_df['Count'].sum() * 100
        fig = px.pie(platform_df, values='Count', names='Platform',
                    title="Platform",
                    hover_data=['Percentage'],
                    labels={'Percentage': 'Percentage (%)'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("OS Distribution")
        os_df = pd.DataFrame({
            'OS': results['device_analysis']['os_distribution'].keys(),
            'Count': results['device_analysis']['os_distribution'].values()
        })
        os_df['Percentage'] = os_df['Count'] / os_df['Count'].sum() * 100
        threshold = 1
        small_os = os_df[os_df['Percentage'] < threshold]
        if not small_os.empty:
            other_sum = small_os['Count'].sum()
            other_row = pd.DataFrame({
                'OS': ['Other'],
                'Count': [other_sum],
                'Percentage': [small_os['Percentage'].sum()]
            })
            os_df = pd.concat([os_df[os_df['Percentage'] >= threshold], other_row])
        
        fig = px.pie(os_df, values='Count', names='OS',
                    title="Operating System",
                    hover_data=['Percentage'],
                    labels={'Percentage': 'Percentage (%)'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detailed Distribution (Absolute Numbers)")
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Device Family',
        x=list(results['device_analysis']['device_distribution'].keys()),
        y=list(results['device_analysis']['device_distribution'].values()),
        text=list(results['device_analysis']['device_distribution'].values()),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Platform',
        x=list(results['device_analysis']['platform_distribution'].keys()),
        y=list(results['device_analysis']['platform_distribution'].values()),
        text=list(results['device_analysis']['platform_distribution'].values()),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='OS',
        x=list(results['device_analysis']['os_distribution'].keys()),
        y=list(results['device_analysis']['os_distribution'].values()),
        text=list(results['device_analysis']['os_distribution'].values()),
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='group',
        title="Device, Platform, and OS Distribution (Absolute Numbers)",
        xaxis_tickangle=-45,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def create_geographic_analysis_plots(results):
    st.header("Geographic Analysis")
    
    st.subheader("Country Distribution")
    
    col1, col2 = st.columns(2)
    
    country_df = pd.DataFrame({
        'Country': results['geographic_analysis']['country_distribution'].keys(),
        'Events': results['geographic_analysis']['country_distribution'].values()
    })
    
    with col1:
        st.write("Events by Country (Bar Chart)")
        fig_bar = px.bar(country_df, x='Country', y='Events',
                     title="Events by Country")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.write("Events by Country (Percentage)")
        country_df['Percentage'] = country_df['Events'] / country_df['Events'].sum() * 100
        fig_pie = px.pie(country_df, values='Events', names='Country',
                        title="Country Distribution",
                        hover_data=['Percentage'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("Top 10 Cities")
    
    col1, col2 = st.columns(2)
    
    city_df = pd.DataFrame({
        'City': results['geographic_analysis']['top_10_cities'].keys(),
        'Events': results['geographic_analysis']['top_10_cities'].values()
    })
    
    with col1:
        st.write("Events by City (Bar Chart)")
        fig_bar = px.bar(city_df, x='City', y='Events',
                     title="Events by City", color_discrete_sequence=['indianred'])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.write("Events by City (Percentage)")
        city_df['Percentage'] = city_df['Events'] / city_df['Events'].sum() * 100
        fig_pie = px.pie(city_df, values='Events', names='City',
                        title="City Distribution",
                        hover_data=['Percentage'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

def create_temporal_analysis_plots(results):
    st.header("Temporal Analysis")
    
    st.subheader("Daily Event Counts")
    daily_counts = pd.DataFrame.from_dict(
        results['data_quality_analysis']['temporal_patterns']['daily_event_counts'],
        orient='index',
        columns=['count']
    )
    daily_counts.index = pd.to_datetime(daily_counts.index)
    # Remove all dates before 2025-01-01
    daily_counts = daily_counts[daily_counts.index >= '2025-01-01']
    
    # Function to format numbers (e.g., 150000 -> "150k")
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.0f}M"
        elif num >= 1_000:
            return f"{num/1_000:.0f}k"
        else:
            return str(int(num))
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_counts.index,
            y=daily_counts['count'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            name='Daily Events'
        )
    )
    
    fig.update_layout(
        title="Daily Event Counts Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Events",
        xaxis=dict(
            tickangle=-45,
            tickformat='%Y-%m-%d',
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            tickformat=',d'
        ),
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Events by Hour of Day")
    hourly_df = pd.DataFrame({
        'Hour': [int(hour) for hour in results['event_analysis']['hourly_distribution'].keys()],
        'Events': list(results['event_analysis']['hourly_distribution'].values())
    }).sort_values('Hour')
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly_df['Hour'],
            y=hourly_df['Events'],
            marker_color='#2ecc71',
            text=hourly_df['Events'].apply(format_number),
            textposition='outside'
        )
    )
    
    fig.update_layout(
        title="Hourly Event Distribution",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Events",
        xaxis=dict(
            tickmode='array',
            ticktext=[f"{str(i).zfill(2)}:00" for i in range(24)],
            tickvals=list(range(24)),
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            tickformat=',d'
        ),
        showlegend=False,
        bargap=0.2
    )
    st.plotly_chart(fig, use_container_width=True)

def create_event_analysis_plots(results):
    st.header("Event Analysis")
    
    st.subheader("Top 20 Event Types")
    event_types = pd.DataFrame.from_dict(
        results['event_analysis']['event_type_distribution'],
        orient='index',
        columns=['count']
    ).sort_values('count', ascending=True).tail(20)
    
    fig = px.bar(event_types, x='count', y=event_types.index,
                 title="Top 20 Event Types",
                 orientation='h')
    st.plotly_chart(fig, use_container_width=True)

def create_user_session_analysis_plots(results):
    st.header("User & Session Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Events per Session")
        eps = results['event_pattern_analysis']['session_patterns']['events_per_session']
        eps_df = pd.DataFrame({
            'value': [eps['min'], eps['25%'], eps['50%'], eps['75%'], eps['max']],
            'metric': ['min', '25%', 'median', '75%', 'max']
        })
        eps_df = eps_df[eps_df['value'] != eps_df['value'].max()]
        fig = px.box(eps_df, y='value', title="Events per Session Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Events per User")
        epu = results['event_pattern_analysis']['user_patterns']['events_per_user']
        epu_df = pd.DataFrame({
            'value': [epu['min'], epu['25%'], epu['50%'], epu['75%'], epu['max']],
            'metric': ['min', '25%', 'median', '75%', 'max']
        })
        epu_df = epu_df[epu_df['value'] != epu_df['value'].max()]
        fig = px.box(epu_df, y='value', title="Events per User Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Session Duration")
        sd = results['event_pattern_analysis']['session_patterns']['session_duration']
        sd_df = pd.DataFrame({
            'value': [sd['min'], sd['25%'], sd['50%'], sd['75%'], sd['max']],
            'metric': ['min', '25%', 'median', '75%', 'max']
        })
        sd_df = sd_df[sd_df['value'] != sd_df['value'].max()]
        fig = px.box(sd_df, y='value', title="Session Duration Distribution (seconds)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sessions per User")
        spu = results['event_pattern_analysis']['user_patterns']['sessions_per_user']
        spu_df = pd.DataFrame({
            'value': [spu['min'], spu['25%'], spu['50%'], spu['75%'], spu['max']],
            'metric': ['min', '25%', 'median', '75%', 'max']
        })
        spu_df = spu_df[spu_df['value'] != spu_df['value'].max()]
        fig = px.box(spu_df, y='value', title="Sessions per User Distribution")
        st.plotly_chart(fig, use_container_width=True)

def create_event_sequence_plot(results):
    st.header("Event Sequence Analysis")
    
    st.subheader("Most Common Event Sequences")
    event_pairs = pd.DataFrame.from_dict(
        results['event_pattern_analysis']['event_sequences']['common_event_pairs'],
        orient='index',
        columns=['count']
    ).sort_values('count', ascending=True)
    
    fig = px.bar(event_pairs, x='count', y=event_pairs.index,
                 title="Most Common Event Sequences",
                 orientation='h')
    st.plotly_chart(fig, use_container_width=True)

def create_user_journey_plots(results):
    st.header("User Journey Analysis")
    
    st.subheader("Top Event Sequences")
    
    sequences = results.get('journey_analysis', {}).get('event_flows', {})
    if not sequences or not any(sequences.values()):
        st.warning("No event sequence data available.")
        return
        
    for length in [2, 3, 4, 5]:
        sequence_key = f'sequences_{length}'
        if sequence_key in sequences and sequences[sequence_key]:
            st.write(f"**Top {length}-Event Sequences:**")
            sequences_df = pd.DataFrame([
                {'Sequence': seq, 'Count': count}
                for seq, count in sequences[sequence_key].items()
            ]).sort_values('Count', ascending=False)
            
            fig = px.bar(sequences_df.head(10), 
                        x='Count', 
                        y='Sequence',
                        title=f"Most Common {length}-Event Sequences",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    if 'path_analysis' in results['journey_analysis']:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Common Entry Points")
            entry_df = pd.DataFrame({
                'Event': results['journey_analysis']['path_analysis']['common_entry_points'].keys(),
                'Count': results['journey_analysis']['path_analysis']['common_entry_points'].values()
            })
            fig = px.pie(entry_df, values='Count', names='Event',
                        title="First Events in Sessions")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Common Exit Points")
            exit_df = pd.DataFrame({
                'Event': results['journey_analysis']['path_analysis']['common_exit_points'].keys(),
                'Count': results['journey_analysis']['path_analysis']['common_exit_points'].values()
            })
            fig = px.pie(exit_df, values='Count', names='Event',
                        title="Last Events in Sessions")
            st.plotly_chart(fig, use_container_width=True)
    
    if 'user_segments' in results['journey_analysis']:
        st.subheader("User Segments Analysis")
        segments = results['journey_analysis']['user_segments']
        
        segment_summary = []
        for segment_id, data in segments.items():
            top_events = list(data['top_events'].items())
            summary = {
                'Segment': segment_id,
                'Size': data['size'],
                'Percentage': f"{data['percentage']:.1f}%",
                'Top Event 1': f"{top_events[0][0]} ({top_events[0][1]:.1f})",
                'Top Event 2': f"{top_events[1][0]} ({top_events[1][1]:.1f})",
                'Top Event 3': f"{top_events[2][0]} ({top_events[2][1]:.1f})"
            }
            segment_summary.append(summary)
        
        st.dataframe(pd.DataFrame(segment_summary))
    
    if 'conversion_funnels' in results['journey_analysis']:
        st.subheader("Conversion Funnels")
        for funnel_name, funnel_data in results['journey_analysis']['conversion_funnels'].items():
            st.write(f"**{funnel_name.replace('_', ' ').title()}**")
            
            funnel_df = pd.DataFrame({
                'Step': funnel_data['steps'],
                'Users': funnel_data['user_counts'],
                'Conversion Rate': funnel_data['conversion_rates']
            })
            
            fig = go.Figure(go.Funnel(
                y=funnel_df['Step'],
                x=funnel_df['Users'],
                textinfo="value+percent previous"
            ))
            fig.update_layout(title=f"{funnel_name.replace('_', ' ').title()} Funnel")
            st.plotly_chart(fig, use_container_width=True)
    

def create_temporal_relationship_plots(results):
    st.header("Temporal Relationships")
    
    st.subheader("Hourly Activity Patterns")
    hourly_patterns = results['temporal_relationships']['hourly_patterns']
    
    all_hours = pd.DataFrame({'Hour': range(24)})
    hourly_dist = hourly_patterns['distribution']
    data_df = pd.DataFrame({
        'Hour': [int(hour) for hour in hourly_dist.keys()],
        'Events': list(hourly_dist.values())
    })
    
    hourly_df = pd.merge(all_hours, data_df, on='Hour', how='left').fillna(0)
    hourly_df = hourly_df.sort_values('Hour')
    
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.0f}M"
        elif num >= 1_000:
            return f"{num/1_000:.0f}k"
        else:
            return str(int(num))
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly_df['Hour'],
            y=hourly_df['Events'],
            mode='lines+markers+text',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            text=hourly_df['Events'].apply(format_number),
            textposition='top center',
            textfont=dict(color='#1f77b4', size=10)
        )
    )
    
    fig.update_layout(
        title="Activity by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Events",
        xaxis=dict(
            tickmode='array',
            ticktext=[f"{str(i).zfill(2)}:00" for i in range(24)],
            tickvals=list(range(24)),
            range=[-0.5, 23.5],
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            tickformat=',d'
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Peak Hours")
        peak_hours = hourly_patterns['peak_hours']
        peak_df = pd.DataFrame({
            'Hour': [f"{str(int(hour)).zfill(2)}:00" for hour in peak_hours.keys()],
            'Events': list(peak_hours.values())
        }).sort_values('Events', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=peak_df['Hour'],
                y=peak_df['Events'],
                marker_color='#2ecc71',
                text=peak_df['Events'].apply(format_number),
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title="Top 5 Busiest Hours",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Events",
            height=450,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                type='category',
                categoryorder='array',
                categoryarray=peak_df['Hour'].tolist(),
                tickangle=45,
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)',
                tickformat=',d'
            ),
            showlegend=False,
            bargap=0.15
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quiet Hours")
        quiet_hours = hourly_patterns['quiet_hours']
        quiet_df = pd.DataFrame({
            'Hour': [int(hour) for hour in quiet_hours.keys()],
            'Events': list(quiet_hours.values())
        }).sort_values('Hour')
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=quiet_df['Hour'],
                y=quiet_df['Events'],
                marker_color='#e74c3c',
                text=quiet_df['Events'].apply(format_number),
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title="5 Quietest Hours",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Events",
            height=500,
            xaxis=dict(
                tickmode='array',
                ticktext=[f"{str(i).zfill(2)}:00" for i in quiet_df['Hour']],
                tickvals=quiet_df['Hour'],
                tickangle=45,
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)',
                tickformat=',d'
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Weekly Patterns")
    weekday_patterns = results['temporal_relationships']['weekday_patterns']
    weekday_dist = weekday_patterns['weekday_distribution']
    weekday_df = pd.DataFrame({
        'Weekday': list(weekday_dist.keys()),
        'Events': list(weekday_dist.values())
    })
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekday_df['Weekday'],
            y=weekday_df['Events'],
            marker_color='#3498db',
            text=weekday_df['Events'].apply(format_number),
            textposition='outside'
        )
    )
    
    fig.update_layout(
        title="Events by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Number of Events",
        height=500,
        xaxis=dict(
            tickangle=0,
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            tickformat=',d'
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def create_geographic_relationship_plots(results):
    st.header("Geographic Relationships")
    
    st.subheader("Device Usage by Country")
    country_devices = results['geographic_relationships']['country_device_distribution']
    
    top_countries = sorted(
        country_devices.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:10]
    
    device_data = []
    for country, devices in top_countries:
        total = sum(devices.values())
        for device, count in devices.items():
            device_data.append({
                'Country': country,
                'Device': device,
                'Percentage': (count / total) * 100
            })
    
    device_df = pd.DataFrame(device_data)
    fig = px.bar(device_df, x='Country', y='Percentage', color='Device',
                 title="Device Distribution by Country",
                 labels={'Percentage': 'Percentage of Users'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("City-Level Analytics")
    city_patterns = pd.DataFrame(results['geographic_relationships']['city_patterns'])
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("Total Events", "Unique Users",
                                     "Events per User", "Sessions per User"))
    
    fig.add_trace(
        go.Bar(x=city_patterns['city'], y=city_patterns['total_events'],
               name="Total Events"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=city_patterns['city'], y=city_patterns['unique_users'],
               name="Unique Users"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=city_patterns['city'], y=city_patterns['events_per_user'],
               name="Events/User"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=city_patterns['city'], y=city_patterns['sessions_per_user'],
               name="Sessions/User"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="City-Level Metrics Comparison")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def create_user_flow_sankey(results):
    st.header("User Flow Analysis")
    st.write("Sankey diagram visualization of user journeys showing event sequence frequencies.")
    
    try:
        sequences = results.get('journey_analysis', {}).get('event_flows', {})
        if not sequences:
            st.warning("No event sequence data available for Sankey diagrams.")
            return
        
        def clean_event_name(name, position=None):
            """Clean and format event names for display."""

            name = str(name)
            name = name.replace('account-lines:::', '')
            name = name.replace('account:::', '')
            name = name.replace('action-center:::', '')
            name = name.replace(':::', '::')
            name = name.replace('::', ':')
            
            if position is not None:
                name = f"{name} ({position})"
            
            return name
        
        def create_sankey_for_length(sequences_data, length):
            if not sequences_data:
                return None
            
            nodes = []
            node_indices = {}
            links = []
            values = []
            
            for sequence, count in sequences_data.items():
                events = sequence.split(" → ")
                if len(events) != length:
                    continue
                
                for i, event in enumerate(events):
                    node_name = clean_event_name(event, i+1)
                    
                    if node_name not in node_indices:
                        node_indices[node_name] = len(nodes)
                        nodes.append(node_name)
                    
                    if i < len(events) - 1:
                        next_node = clean_event_name(events[i+1], i+2)
                        source = node_indices[node_name]
                        target = node_indices.get(next_node)
                        if target is None:
                            node_indices[next_node] = len(nodes)
                            nodes.append(next_node)
                            target = node_indices[next_node]
                        links.append((source, target))
                        values.append(count)
            
            if not links:
                return None
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=25,
                    thickness=15,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    label=nodes,
                    color=[CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(nodes))]
                ),
                link=dict(
                    source=[link[0] for link in links],
                    target=[link[1] for link in links],
                    value=values,
                    color=['rgba(52, 152, 219, 0.2)'] * len(links),
                ),
                textfont=dict(
                    family="Times New Roman",
                    size=14,
                    color="black",
                    shadow="none",
                )
            )])
            
            fig.update_layout(
                title=f"Top {'20' if length == 2 else '10'} Most Common {length}-Event Sequences",
                height=1000 if length == 2 else 800,
                font=dict(
                    size=16,
                    color='black'
                ),
                title_font_size=18
            )
            return fig
        
        for length in [2, 3, 4, 5]:
            sequence_key = f'sequences_{length}'
            if sequence_key in sequences and sequences[sequence_key]:
                st.subheader(f"{length}-Event Sequences")
                fig = create_sankey_for_length(sequences[sequence_key], length)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"**Detailed {length}-Event Sequences:**")
                    sequence_df = pd.DataFrame([
                        {
                            'Sequence': sequence,
                            'Count': count
                        }
                        for sequence, count in sequences[sequence_key].items()
                    ]).sort_values('Count', ascending=False)
                    
                    st.dataframe(sequence_df, use_container_width=True)
                else:
                    st.warning(f"No valid {length}-event sequences available.")
    
    except Exception as e:
        st.error(f"Error creating Sankey diagrams: {str(e)}")
        st.write("Please check if the event sequence data is in the correct format.")

def create_temporal_heatmap(results):
    st.header("Temporal Patterns")
    st.write("Time-based analysis of user activity patterns.")
    
    st.subheader("Hourly Activity Patterns")
    
    hourly_dist = results['temporal_relationships']['hourly_patterns']['distribution']
    
    all_hours = pd.DataFrame({'Hour': range(24)})
    data_df = pd.DataFrame({
        'Hour': [int(hour) for hour in hourly_dist.keys()],
        'Events': list(hourly_dist.values())
    })
    
    hourly_df = pd.merge(all_hours, data_df, on='Hour', how='left').fillna(0)
    hourly_df = hourly_df.sort_values('Hour')
    
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.0f}M"
        elif num >= 1_000:
            return f"{num/1_000:.0f}k"
        else:
            return str(int(num))
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly_df['Hour'],
            y=hourly_df['Events'],
            mode='lines+markers+text',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            text=hourly_df['Events'].apply(format_number),
            textposition='top center',
            textfont=dict(color='#1f77b4', size=10)
        )
    )
    
    fig.update_layout(
        title="Activity by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Number of Events",
        xaxis=dict(
            tickmode='array',
            ticktext=[str(i).zfill(2) for i in range(24)],
            tickvals=list(range(24)),
            range=[-0.5, 23.5],
            tickangle=45
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # with col2:
    #     st.write("**Peak Activity Hours**")
    #     peak_hours = results['temporal_relationships']['hourly_patterns']['peak_hours']
    #     peak_df = pd.DataFrame({
    #         'Hour': list(peak_hours.keys()),
    #         'Events': list(peak_hours.values())
    #     })
    #     fig = px.bar(peak_df, x='Hour', y='Events',
    #                 title="Top 5 Busiest Hours",
    #                 color_discrete_sequence=['#2ecc71'])
    #     st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Weekly Activity Patterns")
    
    weekday_hour_matrix = results['temporal_relationships']['weekday_patterns']['weekday_hour_matrix']
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    matrix_data = [[0 for _ in range(24)] for _ in range(7)]
    
    for hour_str, weekday_data in weekday_hour_matrix.items():
        hour = int(hour_str)
        for weekday, value in weekday_data.items():
            weekday_idx = weekday_order.index(weekday)
            matrix_data[weekday_idx][hour] = value
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=[f"{str(i).zfill(2)}:00" for i in range(24)],
        y=weekday_order,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text="Number of Events",
                side="right"
            )
        ),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Events: %{z:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Activity Heatmap by Weekday and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        xaxis=dict(
            tickmode='array',
            ticktext=[f"{str(i).zfill(2)}:00" for i in range(24)],
            tickvals=list(range(24)),
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        yaxis=dict(
            tickmode='array',
            ticktext=weekday_order,
            tickvals=list(range(len(weekday_order))),
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        height=500,
        margin=dict(l=50, r=100, t=50, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Peak Activity Days")
    st.write("**Top 10 Most Active Days**")
    
    top_days = results['temporal_relationships']['daily_patterns']['top_10_days']
    top_days_df = pd.DataFrame({
        'Date': list(top_days.keys()),
        'Events': list(top_days.values())
    })
    
    top_days_df = top_days_df.nlargest(10, 'Events')
    top_days_df = top_days_df.sort_values('Events', ascending=True)
    
    dates_ordered = top_days_df['Date'].tolist()
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_days_df['Events'],
            y=dates_ordered,
            orientation='h',
            marker_color='#1f77b4',
            text=top_days_df['Events'].apply(format_number),
            textposition='outside',
            textfont=dict(size=12)
        )
    )
    
    max_events = max(top_days_df['Events'])
    
    fig.update_layout(
        title="Top 10 Most Active Days",
        xaxis_title="Number of Events",
        yaxis_title="Date",
        height=400,
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            range=[0, max_events * 1.1]
        ),
        yaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=dates_ordered,
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        bargap=0.2,
        margin=dict(l=20, r=100, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    daily_stats = results['temporal_relationships']['daily_patterns']['daily_stats']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Daily Events", f"{daily_stats['mean']:,.0f}")
    with col2:
        st.metric("Median Daily Events", f"{daily_stats['median']:,.0f}")
    with col3:
        st.metric("Daily Events Std Dev", f"{daily_stats['std']:,.0f}")
    
    st.subheader("Peak Activity Summary")
    peak_activity = results['temporal_relationships']['peak_activity']
    
    st.write(f"**Overall Peak Day:** {peak_activity['overall_peak_day']} with {peak_activity['overall_peak_day_count']:,} events")
    
    peak_by_weekday = peak_activity['by_weekday']
    weekday_summary = []
    for weekday, data in peak_by_weekday.items():
        peak_hours_str = ", ".join([f"{hour}:00" for hour in data['peak_hours'].keys()])
        weekday_summary.append({
            'Weekday': weekday,
            'Total Events': data['total_events'],
            'Peak Hours': peak_hours_str
        })
    
    weekday_summary_df = pd.DataFrame(weekday_summary)
    st.write("Peak Hours by Weekday")
    st.dataframe(weekday_summary_df, use_container_width=True)

# def create_user_segments_analysis(results):
#     st.header("User Segments")
#     st.write("Behavioral segmentation analysis of users.")
    
#     segments = results['journey_analysis']['user_segments']
#     if not segments:
#         st.warning("No user segments data available.")
#         return
    
#     fig = go.Figure()
    
#     for segment_id, data in segments.items():
#         if not data.get('top_events'):
#             continue
            
#         top_events = list(data['top_events'].items())
#         if not top_events:
#             continue
            
#         while len(top_events) < 5:
#             top_events.append((f"Event_{len(top_events)+1}", 0))
            
#         fig.add_trace(go.Scatterpolar(
#             r=[v for _, v in top_events[:5]],
#             theta=[k[:20] + '...' if len(k) > 20 else k for k, _ in top_events[:5]],
#             name=f"{segment_id} ({data.get('percentage', 0):.1f}%)",
#             line=dict(color=CUSTOM_COLORS[int(segment_id.split('_')[1]) % len(CUSTOM_COLORS)])
#         ))
    
#     if not fig.data:
#         st.warning("No segment data available for visualization.")
#         return
        
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, max([v for s in segments.values() if 'top_events' in s 
#                              for v in s['top_events'].values()], default=1)]
#             )
#         ),
#         showlegend=True,
#         title="User Segment Characteristics"
#     )
#     st.plotly_chart(fig, use_container_width=True)

def create_geographic_insights(results):
    st.header("Geographic Insights")
    st.write("Geographic distribution analysis of users and device usage.")
    
    country_data = pd.DataFrame({
        'Country': results['geographic_analysis']['country_distribution'].keys(),
        'Events': results['geographic_analysis']['country_distribution'].values()
    })
    
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Events',
        color_continuous_scale='Viridis',
        title='Global User Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="User Analytics Dashboard",
        page_icon=None,
        layout="wide"
    )
    
    st.title("User Analytics Dashboard")
    st.write("User behavior and analytics visualization dashboard")
    
    try:
        results = load_analysis_results()
        
        st.sidebar.title("Navigation")
        
        sections = {
            "Overview": None,
            "User Behavior Analysis": [
                "Event Analysis",
                "User Sessions",
                "User Journeys",
                "Event Sequences",
                # "User Segments"
            ],
            "Technical Environment": [
                "Device Distribution",
                "Operating Systems"
            ],
            "Geographic Analysis": [
                "Country Distribution",
                "City Analysis",
                "Geographic Insights"
            ],
            "Temporal Analysis": [
                "Time Patterns",
                "Activity Distribution"
            ],
            "Platform Snapshots": None
        }
        
        main_section = st.sidebar.radio("Main Sections", list(sections.keys()))
        
        selected_subsection = None
        if sections[main_section]:
            selected_subsection = st.sidebar.radio(
                f"{main_section} Sections",
                sections[main_section]
            )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Stats")
        st.sidebar.metric("Total Events", f"{results['basic_stats']['total_events']:,}")
        st.sidebar.metric("Unique Users", f"{results['basic_stats']['unique_users']:,}")
        st.sidebar.metric("Active Sessions", f"{results['basic_stats']['unique_sessions']:,}")
        
        if main_section == "Overview":
            # st.write(f"### Date Range: {results['basic_stats']['date_range']['start']} to {results['basic_stats']['date_range']['end']}")
            create_device_analysis_plots(results)
            create_geographic_analysis_plots(results)
            create_temporal_analysis_plots(results)
            create_event_analysis_plots(results)
            
        elif main_section == "User Behavior Analysis":
            if selected_subsection == "Event Analysis":
                create_event_analysis_plots(results)
            elif selected_subsection == "User Sessions":
                create_user_session_analysis_plots(results)
            elif selected_subsection == "User Journeys":
                if 'journey_analysis' in results:
                    create_user_journey_plots(results)
                else:
                    st.warning("User journey analysis data not available.")
            elif selected_subsection == "Event Sequences":
                create_event_sequence_plot(results)
                create_user_flow_sankey(results)
            # elif selected_subsection == "User Segments":
            #     try:
            #         if 'journey_analysis' in results and 'user_segments' in results['journey_analysis']:
            #             create_user_segments_analysis(results)
            #         else:
            #             st.warning("User segments data not available.")
            #     except Exception as e:
            #         st.warning(f"Could not create user segments visualization: {str(e)}")
            
        elif main_section == "Technical Environment":
            if selected_subsection == "Device Distribution":
                st.header("Device Distribution")
                device_df = pd.DataFrame({
                    'Device': results['device_analysis']['device_distribution'].keys(),
                    'Count': results['device_analysis']['device_distribution'].values()
                })
                device_df['Percentage'] = device_df['Count'] / device_df['Count'].sum() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(device_df, values='Count', names='Device',
                               title="Device Family Distribution",
                               hover_data=['Percentage'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(device_df, x='Device', y='Count',
                               title="Device Distribution (Absolute Numbers)")
                    st.plotly_chart(fig, use_container_width=True)
                
            elif selected_subsection == "Operating Systems":
                st.header("Operating Systems")
                
                col1, col2 = st.columns(2)
                with col1:
                    platform_df = pd.DataFrame({
                        'Platform': results['device_analysis']['platform_distribution'].keys(),
                        'Count': results['device_analysis']['platform_distribution'].values()
                    })
                    platform_df['Percentage'] = platform_df['Count'] / platform_df['Count'].sum() * 100
                    
                    fig = px.pie(platform_df, values='Count', names='Platform',
                               title="Platform Distribution",
                               hover_data=['Percentage'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    os_df = pd.DataFrame({
                        'OS': results['device_analysis']['os_distribution'].keys(),
                        'Count': results['device_analysis']['os_distribution'].values()
                    })
                    os_df['Percentage'] = os_df['Count'] / os_df['Count'].sum() * 100
                    
                    threshold = 1
                    small_os = os_df[os_df['Percentage'] < threshold]
                    if not small_os.empty:
                        other_sum = small_os['Count'].sum()
                        other_row = pd.DataFrame({
                            'OS': ['Other'],
                            'Count': [other_sum],
                            'Percentage': [small_os['Percentage'].sum()]
                        })
                        os_df = pd.concat([os_df[os_df['Percentage'] >= threshold], other_row])
                    
                    fig = px.pie(os_df, values='Count', names='OS',
                               title="Operating System Distribution",
                               hover_data=['Percentage'])
                    st.plotly_chart(fig, use_container_width=True)
        
        elif main_section == "Geographic Analysis":
            if selected_subsection == "Country Distribution":
                create_geographic_analysis_plots(results)
            elif selected_subsection == "City Analysis":
                create_geographic_relationship_plots(results)
            elif selected_subsection == "Geographic Insights":
                create_geographic_insights(results)
            
        elif main_section == "Temporal Analysis":
            if selected_subsection == "Time Patterns":
                create_temporal_analysis_plots(results)
                create_temporal_heatmap(results)
            elif selected_subsection == "Activity Distribution":
                create_temporal_relationship_plots(results)
        
        elif main_section == "Platform Snapshots":
            st.image("federato/images/accounts.png")
            st.image("federato/images/action-center.png")
            st.image("federato/images/dashboard.png")
            
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("Please make sure the analysis_results.json file is present in the same directory.")

if __name__ == "__main__":
    main() 