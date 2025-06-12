import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from functools import lru_cache

# Install required packages if not already installed
try:
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

try:
    from scipy.optimize import curve_fit
    from scipy import stats
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.optimize import curve_fit
    from scipy import stats

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:
    import openai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

# Configure Streamlit page
st.set_page_config(page_title="SFE Assistant", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 0.5rem;}
    div.stTitle {margin-top: -1rem; padding-bottom: 0;}
    header[data-testid="stHeader"] {height: 0.5rem;}
    [data-testid="stVerticalBlock"] {gap: 0.5rem;}
    div.element-container {margin: 0.25rem 0;}
    .ai-insights {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .ai-insights h3 {
        color: white !important;
        margin: 0 0 15px 0;
    }
    .ai-key-point {
        background: rgba(255, 255, 255, 0.1);
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# Required sheet structure
REQUIRED_SHEETS = {
    "Sales": ["Year", "Geo brick ID", "Competitive Market", "Product", "Volume", "Value"],
    "Universe": ["Customer ID", "Customer name", "Customer Speciality", "Geo brick ID"],
    "Visits": ["Call Date", "Customer ID", "Customer name", "Rep ID", "Rep name", "Customer Speciality", "Geo brick ID"]
}

# OpenAI Configuration
def setup_openai_sidebar():
    """Setup OpenAI configuration in sidebar."""
    with st.sidebar:
        st.markdown("### 🤖 AI Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            help="Enter your OpenAI API key to enable AI-powered insights",
            key="openai_api_key"
        )
        
        # Model selection
        available_models = [
            "gpt-4o-mini",
            "gpt-4o", 
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]
        
        selected_model = st.selectbox(
            "Select AI Model",
            available_models,
            index=0,  # Default to gpt-4o-mini
            key="openai_model"
        )
        
        # Test API connection
        if api_key:
            if st.button("🔧 Test API Connection"):
                with st.spinner("Testing API connection..."):
                    if test_openai_connection(api_key, selected_model):
                        st.success("✅ API connection successful!")
                    else:
                        st.error("❌ API connection failed. Please check your API key.")
        
        # AI Settings
        st.markdown("### ⚙️ AI Settings")
        ai_temperature = st.slider(
            "AI Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower values = more focused, Higher values = more creative"
        )
        
        return api_key, selected_model, ai_temperature

def test_openai_connection(api_key, model):
    """Test OpenAI API connection."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return False

def get_ai_insights(data_summary, context, api_key, model, temperature=0.3):
    """Generate AI insights using OpenAI API."""
    if not api_key:
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        You are a pharmaceutical sales force effectiveness (SFE) analyst. Analyze the following data and provide key business insights:

        Context: {context}
        
        Data Summary:
        {data_summary}
        
        IMPORTANT: Do NOT analyze or comment on correlations between Evolution Index (EI) and Sales Growth. Focus on other business metrics and relationships.
        
        Please provide:
        1. Top 3 key insights (each in 1-2 sentences)
        2. Main opportunities identified
        3. Potential risks or concerns
        4. Strategic recommendations
        
        Format your response as JSON with the following structure:
        {{
            "key_insights": ["insight1", "insight2", "insight3"],
            "opportunities": ["opportunity1", "opportunity2"],
            "risks": ["risk1", "risk2"],
            "recommendations": ["recommendation1", "recommendation2"]
        }}
        
        Focus on actionable business insights relevant to pharmaceutical sales management, excluding any EI vs Sales Growth analysis.
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=temperature
        )
        
        # Parse JSON response
        insights_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            start_idx = insights_text.find('{')
            end_idx = insights_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = insights_text[start_idx:end_idx]
                insights = json.loads(json_str)
                return insights
            else:
                # If no JSON found, return raw text
                return {"raw_insights": insights_text}
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text
            return {"raw_insights": insights_text}
            
    except Exception as e:
        st.error(f"AI Insights Error: {str(e)}")
        return None

def display_ai_insights(insights_data, context, api_key, model, temperature):
    """Display AI-powered insights section."""
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🤖 AI-Powered Key Insights")
    with col2:
        if api_key:
            generate_insights = st.button("🔄 Generate Insights", key=f"ai_insights_{context}")
        else:
            st.warning("⚠️ Configure OpenAI API key in sidebar")
            generate_insights = False
    
    if api_key and generate_insights:
        with st.spinner("🤖 AI is analyzing your data..."):
            insights = get_ai_insights(insights_data, context, api_key, model, temperature)
            
            if insights:
                st.markdown('<div class="ai-insights">', unsafe_allow_html=True)
                
                # Display structured insights if available
                if "key_insights" in insights:
                    st.markdown("<h3>🎯 Key Insights</h3>", unsafe_allow_html=True)
                    for i, insight in enumerate(insights["key_insights"], 1):
                        st.markdown(f'<div class="ai-key-point">📌 {insight}</div>', unsafe_allow_html=True)
                    
                    if "opportunities" in insights:
                        st.markdown("<h3>🚀 Opportunities</h3>", unsafe_allow_html=True)
                        for opportunity in insights["opportunities"]:
                            st.markdown(f'<div class="ai-key-point">💡 {opportunity}</div>', unsafe_allow_html=True)
                    
                    if "risks" in insights:
                        st.markdown("<h3>⚠️ Risks & Concerns</h3>", unsafe_allow_html=True)
                        for risk in insights["risks"]:
                            st.markdown(f'<div class="ai-key-point">🚨 {risk}</div>', unsafe_allow_html=True)
                    
                    if "recommendations" in insights:
                        st.markdown("<h3>💼 Strategic Recommendations</h3>", unsafe_allow_html=True)
                        for recommendation in insights["recommendations"]:
                            st.markdown(f'<div class="ai-key-point">✅ {recommendation}</div>', unsafe_allow_html=True)
                
                elif "raw_insights" in insights:
                    st.markdown("<h3>🎯 Analysis Results</h3>", unsafe_allow_html=True)
                    st.markdown(f'<div class="ai-key-point">{insights["raw_insights"]}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Failed to generate AI insights. Please try again.")
    
    elif not api_key:
        st.info("🔧 Configure your OpenAI API key in the sidebar to enable AI-powered insights that will analyze your data and provide strategic recommendations.")

def prepare_data_summary_for_ai(df, context_type):
    """Prepare data summary for AI analysis."""
    if df.empty:
        return "No data available for analysis."
    
    summary = f"Dataset has {len(df)} records.\n\n"
    
    # Add specific summaries based on context
    if context_type == "frequency_segmentation":
        if 'Segment' in df.columns and 'Frequency' in df.columns:
            segment_stats = df.groupby('Segment').agg({
                'Frequency': ['count', 'mean'],
                'Total Visits': 'sum' if 'Total Visits' in df.columns else 'count',
                'Coverage, %': 'mean' if 'Coverage, %' in df.columns else 'count'
            }).round(2)
            summary += f"Frequency Segmentation Analysis:\n{segment_stats.to_string()}\n\n"
    
    elif context_type == "response_curve":
        if 'Frequency' in df.columns and 'Sales_Growth' in df.columns:
            summary += f"Response Curve Analysis:\n"
            summary += f"Frequency range: {df['Frequency'].min():.1f} - {df['Frequency'].max():.1f}\n"
            summary += f"Sales Growth range: {df['Sales_Growth'].min():.1f}% - {df['Sales_Growth'].max():.1f}%\n"
            summary += f"Correlation: {df['Frequency'].corr(df['Sales_Growth']):.3f}\n\n"
    
    elif context_type == "correlation":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude EI (Evolution Index) and Sales Growth correlation analysis
        excluded_correlations = []
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()
            
            # Filter out EI vs Sales Growth correlations
            filtered_correlations = correlations.copy()
            
            # Remove EI-Sales Growth correlations specifically
            ei_cols = [col for col in correlations.columns if 'EI' in str(col) or 'Evolution' in str(col)]
            growth_cols = [col for col in correlations.columns if 'Growth' in str(col) and 'Sales' in str(col)]
            
            for ei_col in ei_cols:
                for growth_col in growth_cols:
                    if ei_col in filtered_correlations.index and growth_col in filtered_correlations.columns:
                        excluded_correlations.append(f"{ei_col} vs {growth_col}")
            
            # Create summary without EI-Sales Growth correlations
            important_correlations = []
            for col1 in filtered_correlations.columns:
                for col2 in filtered_correlations.columns:
                    if col1 != col2:
                        # Skip EI vs Sales Growth correlations
                        if not ((any(ei in col1 for ei in ['EI', 'Evolution']) and any(growth in col2 for growth in ['Growth']) and 'Sales' in col2) or
                               (any(ei in col2 for ei in ['EI', 'Evolution']) and any(growth in col1 for growth in ['Growth']) and 'Sales' in col1)):
                            corr_value = filtered_correlations.loc[col1, col2]
                            if abs(corr_value) > 0.3:  # Only show meaningful correlations
                                important_correlations.append(f"{col1} vs {col2}: {corr_value:.3f}")
            
            if important_correlations:
                summary += f"Key Correlations (excluding EI vs Sales Growth):\n"
                summary += "\n".join(important_correlations[:10])  # Show top 10
                summary += "\n\n"
            
            if excluded_correlations:
                summary += f"Note: EI vs Sales Growth correlations excluded from analysis as requested.\n\n"
    
    elif context_type == "clusters":
        if 'Cluster' in df.columns:
            cluster_distribution = df['Cluster'].value_counts()
            summary += f"Cluster Distribution:\n{cluster_distribution.to_string()}\n\n"
    
    # Add basic statistics for numeric columns (excluding EI vs Sales Growth analysis)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        basic_stats = df[numeric_cols].describe()
        summary += f"Statistical Summary:\n{basic_stats.to_string()}"
    
    return summary

# Helper functions (keeping all existing functions from the original code)
@st.cache_data
def validate_excel_file(file):
    """Validates Excel file structure."""
    try:
        excel_data = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        missing_sheets = set(REQUIRED_SHEETS.keys()) - set(excel_data.keys())
        if missing_sheets:
            return False, f"Missing required sheets: {', '.join(missing_sheets)}", None

        dataframes = {}
        for sheet_name, required_columns in REQUIRED_SHEETS.items():
            df = excel_data[sheet_name]
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return False, f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing_columns)}", None
            
            # Ensure Geo brick ID is consistent data type across all sheets
            if 'Geo brick ID' in df.columns:
                df['Geo brick ID'] = df['Geo brick ID'].astype(str).str.strip()
            
            # Also ensure Customer ID is consistent if it exists
            if 'Customer ID' in df.columns:
                df['Customer ID'] = df['Customer ID'].astype(str).str.strip()
                
            # Ensure Rep ID is consistent if it exists
            if 'Rep ID' in df.columns:
                df['Rep ID'] = df['Rep ID'].astype(str).str.strip()
            
            dataframes[sheet_name] = df

        return True, "Validation successful!", dataframes
    except Exception as e:
        return False, f"Error processing Excel file: {str(e)}", None

def calculate_single_year_ms(year_sales_df, brick_id, selected_product=None, metric_type="Value"):
    """Calculate market share for a single year/brick combination."""
    brick_sales = year_sales_df[year_sales_df['Geo brick ID'] == brick_id]
    if brick_sales.empty:
        return 0

    if selected_product:
        product_markets = brick_sales[brick_sales['Product'] == selected_product]['Competitive Market'].unique()
        if product_markets.size == 0:
            return 0
        brick_sales = brick_sales[brick_sales['Competitive Market'].isin(product_markets)]

    competitors_mask = brick_sales['Product'] == 'COMPETITOR'
    own_product_mask = brick_sales['Product'] == selected_product if selected_product else ~competitors_mask

    competitor_metric = brick_sales.loc[competitors_mask, metric_type].sum()
    own_product_metric = brick_sales.loc[own_product_mask, metric_type].sum()
    total_metric = own_product_metric + competitor_metric
    return (own_product_metric / total_metric * 100) if total_metric > 0 else 0

def calculate_market_share(sales_df, brick_list, selected_product=None, metric_type="Value"):
    """Calculate market share for multiple bricks."""
    all_years = sales_df['Year'].unique()
    latest_year = max(all_years)
    previous_year = sorted(all_years)[-2] if len(all_years) >= 2 else None

    ms_column_name = f"MS {latest_year}, %"
    ei_column_name = "Ei"
    growth_column_name = "Growth, %"

    result_df = pd.DataFrame({'Geo brick ID': brick_list})
    result_df[ms_column_name] = 0.0
    result_df[ei_column_name] = 0
    result_df[growth_column_name] = 0.0

    latest_year_sales = sales_df[sales_df['Year'] == latest_year]
    previous_year_sales = sales_df[sales_df['Year'] == previous_year] if previous_year else pd.DataFrame()

    for brick_id in brick_list:
        latest_ms = calculate_single_year_ms(latest_year_sales, brick_id, selected_product, metric_type)
        previous_ms = calculate_single_year_ms(previous_year_sales, brick_id, selected_product, metric_type) if previous_year else 0
        
        # Fix EI calculation to handle division by zero and inf values
        if previous_ms > 0:
            ei = (latest_ms / previous_ms) * 100
            # Handle potential inf/nan values
            if np.isfinite(ei):
                ei = int(round(ei))
            else:
                ei = 0
        else:
            ei = 0

        brick_latest_sales = latest_year_sales[latest_year_sales['Geo brick ID'] == brick_id]
        if selected_product:
            brick_latest_sales = brick_latest_sales[brick_latest_sales['Product'] == selected_product]
        else:
            brick_latest_sales = brick_latest_sales[brick_latest_sales['Product'] != 'COMPETITOR']
        latest_sales = brick_latest_sales[metric_type].sum()

        previous_sales = 0
        if previous_year:
            brick_previous_sales = previous_year_sales[previous_year_sales['Geo brick ID'] == brick_id]
            if selected_product:
                brick_previous_sales = brick_previous_sales[brick_previous_sales['Product'] == selected_product]
            else:
                brick_previous_sales = brick_previous_sales[brick_previous_sales['Product'] != 'COMPETITOR']
            previous_sales = brick_previous_sales[metric_type].sum()

        growth = ((latest_sales / previous_sales) - 1) * 100 if previous_sales > 0 else 0
        # Handle potential inf/nan values for growth
        if not np.isfinite(growth):
            growth = 0

        result_df.loc[result_df['Geo brick ID'] == brick_id, [ms_column_name, ei_column_name, growth_column_name]] = [
            round(latest_ms, 1), ei, round(growth, 1)
        ]

    return result_df, ms_column_name, ei_column_name, growth_column_name

def prepare_brick_data(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None):
    """Prepare brick data for segmentation."""
    # Fixed filtering logic to avoid KeyError
    filtered_visits = visits_df.copy()
    
    # Apply specialty filter
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    
    # Apply rep filter
    if selected_reps:
        filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]

    unique_brick_customers = filtered_visits[['Geo brick ID', 'Customer ID']].drop_duplicates()
    customers_per_brick = unique_brick_customers.groupby('Geo brick ID').size().reset_index(name='Covered HCP')
    visits_per_brick = filtered_visits.groupby('Geo brick ID').size().reset_index(name='Total Visits')
    universe_brick_hcp = universe_df.groupby('Geo brick ID')['Customer ID'].nunique().reset_index(name='Total HCP')

    brick_data = universe_brick_hcp.merge(customers_per_brick, on='Geo brick ID', how='left').merge(
        visits_per_brick, on='Geo brick ID', how='left'
    ).fillna({'Covered HCP': 0, 'Total Visits': 0, 'Total HCP': 0}).astype({'Covered HCP': int, 'Total Visits': int, 'Total HCP': int})

    brick_data['Frequency'] = brick_data['Total Visits'] / brick_data['Covered HCP'].replace(0, np.nan)
    brick_data['Frequency'] = brick_data['Frequency'].fillna(0).round(2)
    brick_data['Coverage, %'] = (brick_data['Covered HCP'] / brick_data['Total HCP'].replace(0, np.nan) * 100).clip(upper=100).fillna(0).round(2)

    market_share_df, ms_column_name, ei_column_name, growth_column_name = calculate_market_share(
        sales_df, brick_data['Geo brick ID'].tolist(), product, metric_type
    )
    brick_data = brick_data.merge(market_share_df, on='Geo brick ID', how='left').fillna({
        ms_column_name: 0, ei_column_name: 0, growth_column_name: 0
    })
    # Ensure EI column is properly handled for NaN/inf values
    brick_data[ei_column_name] = brick_data[ei_column_name].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

    return brick_data, ms_column_name, ei_column_name, growth_column_name

def segment_data(df, segmentation_column, segment_labels=None, exclude_zeros=False):
    """Segment data based on a specified column."""
    if segment_labels is None:
        segment_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

    # Filter out zeros if requested
    if exclude_zeros:
        df_for_segmentation = df[df[segmentation_column] > 0].copy()
        if df_for_segmentation.empty:
            # If no non-zero data, return original dataframe with all segments as 'Very Low'
            df['Segment'] = segment_labels[0]
            return df, [], segment_labels
    else:
        df_for_segmentation = df.copy()

    quantiles = df_for_segmentation[segmentation_column].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    unique_quantiles = []
    prev = -float('inf')
    for q in quantiles:
        unique_quantiles.append(max(q, prev + 0.00001))
        prev = unique_quantiles[-1]

    bins = [-float('inf')] + unique_quantiles + [float('inf')]
    
    if exclude_zeros:
        # For exclude_zeros mode, assign segments only to non-zero values
        df['Segment'] = segment_labels[0]  # Default all to 'Very Low'
        non_zero_mask = df[segmentation_column] > 0
        if non_zero_mask.any():
            df.loc[non_zero_mask, 'Segment'] = pd.cut(
                df.loc[non_zero_mask, segmentation_column], 
                bins=bins, labels=segment_labels, include_lowest=True
            )
    else:
        df['Segment'] = pd.cut(df[segmentation_column], bins=bins, labels=segment_labels, include_lowest=True)
    
    return df, unique_quantiles, segment_labels

def handle_response_curve_tab(df_visits, df_sales, specialties_options, products_options, reps_options, api_key, model, temperature):
    """Handle Response Curve tab content."""
    st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_specialties = st.multiselect("Customer Speciality", specialties_options, key="response_specialties")
    with col2:
        selected_product = st.selectbox("Product", products_options, key="response_product")
        selected_product = None if selected_product == 'All' else selected_product
    with col3:
        selected_reps = st.multiselect("Rep Name", reps_options, key="response_reps")
    with col4:
        metric_type = st.radio("Metric for Sales", ["Value", "Volume"], horizontal=True, key="response_metric")

    # Prepare data
    response_data, message = prepare_response_curve_data(
        df_visits, df_sales, selected_specialties, selected_product, metric_type, selected_reps
    )
    
    if response_data.empty:
        st.warning(f"No data available for response curve analysis. {message}")
        return
    
    st.info(message)
    
    # Create and display chart in a smaller column
    chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])
    
    with chart_col2:
        fig, optimal_freq = create_response_curve_chart(response_data)
        if fig is not None:
            st.pyplot(fig)
    
    if optimal_freq is not None:
        st.markdown("<h3>Analysis Summary</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Optimal Visit Frequency", f"{optimal_freq:.1f}")
            st.metric("Total Data Points", len(response_data))
        
        with col2:
            freq_stats = response_data['Frequency'].describe()
            growth_stats = response_data['Sales_Growth'].describe()
            st.write("**Frequency Statistics:**")
            st.write(f"Mean: {freq_stats['mean']:.1f}, Median: {freq_stats['50%']:.1f}")
            st.write("**Growth Statistics:**")
            st.write(f"Mean: {growth_stats['mean']:.1f}%, Median: {growth_stats['50%']:.1f}%")

    # Display data table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Response Curve Data</h3>", unsafe_allow_html=True)
    
    # Prepare display data
    display_data = response_data.copy()
    display_data = display_data.round(2)
    display_data.columns = ['Geo Brick ID', 'Frequency', 'Sales Growth %', 'Total Visits', 'Unique Customers', 'Latest Sales', 'Previous Sales']
    
    st.dataframe(display_data, use_container_width=True)
    create_download_button(display_data, "response_curve_data")
    
    # AI Insights
    data_summary = prepare_data_summary_for_ai(response_data, "response_curve")
    display_ai_insights(data_summary, "response_curve", api_key, model, temperature)

@st.cache_data(ttl=3600)
def segment_bricks_by_frequency(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None, exclude_zeros=False):
    """Segment bricks based on visit frequency."""
    brick_data, ms_column_name, ei_column_name, growth_column_name = prepare_brick_data(
        visits_df, universe_df, sales_df, specialties, product, metric_type, selected_reps
    )
    segmented_data, quantiles, segment_labels = segment_data(brick_data, 'Frequency', exclude_zeros=exclude_zeros)
    return segmented_data.sort_values('Frequency', ascending=False), quantiles, segment_labels, ms_column_name, ei_column_name, growth_column_name

@st.cache_data(ttl=3600)
def segment_bricks_by_ei(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None, exclude_zeros=False):
    """Segment bricks based on Evolution Index."""
    brick_data, ms_column_name, ei_column_name, growth_column_name = prepare_brick_data(
        visits_df, universe_df, sales_df, specialties, product, metric_type, selected_reps
    )
    brick_data = brick_data[brick_data[ei_column_name] > 0]
    if brick_data.empty:
        return brick_data, [], [], ms_column_name, ei_column_name, growth_column_name
    segmented_data, quantiles, segment_labels = segment_data(brick_data, ei_column_name, exclude_zeros=exclude_zeros)
    return segmented_data.sort_values(ei_column_name, ascending=False), quantiles, segment_labels, ms_column_name, ei_column_name, growth_column_name

@st.cache_data(ttl=3600)
def segment_bricks_by_coverage(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None, exclude_zeros=False):
    """Segment bricks based on coverage percentage."""
    brick_data, ms_column_name, ei_column_name, growth_column_name = prepare_brick_data(
        visits_df, universe_df, sales_df, specialties, product, metric_type, selected_reps
    )
    brick_data = brick_data[brick_data['Total HCP'] > 0]
    if brick_data.empty:
        return brick_data, [], [], ms_column_name, ei_column_name, growth_column_name
    segmented_data, quantiles, segment_labels = segment_data(brick_data, 'Coverage, %', exclude_zeros=exclude_zeros)
    return segmented_data.sort_values('Coverage, %', ascending=False), quantiles, segment_labels, ms_column_name, ei_column_name, growth_column_name

def calculate_segment_statistics(brick_segments, segment_labels, ms_column_name, ei_column_name, growth_column_name, primary_column=None):
    """Calculate segment statistics."""
    segment_agg = {
        'Geo brick ID': 'count', 'Total Visits': 'sum', 'Frequency': 'mean',
        'Total HCP': 'sum', 'Covered HCP': 'sum', 'Coverage, %': 'mean',
        ms_column_name: 'mean', ei_column_name: 'mean', growth_column_name: 'mean'
    }
    segment_stats = brick_segments.groupby('Segment').agg(segment_agg).reset_index()
    segment_stats.columns = ['Segment', 'Nb bricks', 'Total Visits', 'Frequency', 'Total HCP',
                            'Covered HCP', 'Coverage, %', ms_column_name, ei_column_name, growth_column_name]

    segment_stats['Coverage, %'] = (segment_stats['Covered HCP'] / segment_stats['Total HCP'] * 100).clip(upper=100).fillna(0).round(2)
    # Fix NaN/inf handling for EI column
    segment_stats[ei_column_name] = segment_stats[ei_column_name].fillna(0).replace([np.inf, -np.inf], 0).round(0).astype(int)
    segment_stats[ms_column_name] = segment_stats[ms_column_name].fillna(0).round(1)
    segment_stats[growth_column_name] = segment_stats[growth_column_name].fillna(0).round(1)
    segment_stats['Frequency'] = segment_stats['Frequency'].fillna(0).round(0).astype(int)

    segment_order = {label: i for i, label in enumerate(segment_labels)}
    segment_stats['Order'] = segment_stats['Segment'].map(segment_order)
    segment_stats = segment_stats.sort_values('Order').drop(columns=['Order'])

    display_columns = {
        'Frequency': ['Segment', 'Frequency', 'Nb bricks', 'Total Visits', 'Total HCP', 'Covered HCP', 'Coverage, %', ms_column_name, growth_column_name, ei_column_name],
        ei_column_name: ['Segment', ei_column_name, 'Nb bricks', 'Total Visits', 'Total HCP', 'Covered HCP', 'Coverage, %', ms_column_name, growth_column_name, 'Frequency'],
        'Coverage, %': ['Segment', 'Coverage, %', 'Nb bricks', 'Total Visits', 'Total HCP', 'Covered HCP', 'Frequency', ms_column_name, growth_column_name, ei_column_name]
    }.get(primary_column, segment_stats.columns)

    return segment_stats[display_columns]

def calculate_correlations(segment_stats, primary_column):
    """Calculate correlations between primary column and other numeric columns."""
    # Get numeric columns excluding the primary column itself
    numeric_columns = segment_stats.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != primary_column and col != 'Nb bricks']
    
    correlations = {}
    for col in numeric_columns:
        if len(segment_stats) > 2:  # Need at least 3 points for meaningful correlation
            corr = segment_stats[primary_column].corr(segment_stats[col])
            if not np.isnan(corr):
                correlations[col] = corr
    
    return correlations

def format_correlation_text(correlations, primary_column_name):
    """Format correlation analysis into readable text."""
    if not correlations:
        return "Insufficient data for correlation analysis."
    
    # Sort correlations by absolute value
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Find highest positive and negative correlations
    positive_corrs = [(col, corr) for col, corr in sorted_corrs if corr > 0]
    negative_corrs = [(col, corr) for col, corr in sorted_corrs if corr < 0]
    
    text_parts = []
    
    if positive_corrs:
        strongest_positive = positive_corrs[0]
        text_parts.append(f"**Strongest Positive Correlation**: {strongest_positive[0]} (r = {strongest_positive[1]:.3f})")
        
        if len(positive_corrs) > 1:
            other_positive = [f"{col} (r = {corr:.3f})" for col, corr in positive_corrs[1:3]]  # Show up to 2 more
            if other_positive:
                text_parts.append(f"**Other Positive Correlations**: {', '.join(other_positive)}")
    
    if negative_corrs:
        strongest_negative = negative_corrs[0]
        text_parts.append(f"**Strongest Negative Correlation**: {strongest_negative[0]} (r = {strongest_negative[1]:.3f})")
        
        if len(negative_corrs) > 1:
            other_negative = [f"{col} (r = {corr:.3f})" for col, corr in negative_corrs[1:3]]  # Show up to 2 more
            if other_negative:
                text_parts.append(f"**Other Negative Correlations**: {', '.join(other_negative)}")
    
    if not positive_corrs and not negative_corrs:
        return f"No significant correlations found with {primary_column_name}."
    
    return "\n\n".join(text_parts)

def create_rep_frequency_table(visits_df):
    """Create a table showing percentage of visits for each frequency value per Rep."""
    rep_customer_visits = visits_df.groupby(['Rep ID', 'Rep name', 'Customer ID']).size().reset_index(name='Frequency')
    rep_freq_distribution = rep_customer_visits.groupby(['Rep ID', 'Rep name', 'Frequency']).size().reset_index(name='Customers')
    rep_totals = rep_freq_distribution.groupby(['Rep ID', 'Rep name'])['Customers'].sum().reset_index(name='Total Customers')
    rep_freq_distribution['Total Visits'] = rep_freq_distribution['Frequency'] * rep_freq_distribution['Customers']
    rep_visits_total = rep_freq_distribution.groupby(['Rep ID', 'Rep name'])['Total Visits'].sum().reset_index()

    rep_freq_distribution = rep_freq_distribution.merge(rep_totals, on=['Rep ID', 'Rep name']).merge(rep_visits_total, on=['Rep ID', 'Rep name'])
    rep_freq_distribution['Percentage'] = (rep_freq_distribution['Customers'] / rep_freq_distribution['Total Customers'] * 100).round(1)

    pivot_table = rep_freq_distribution.pivot_table(
        index=['Rep ID', 'Rep name'], columns='Frequency', values='Percentage', fill_value=0
    ).reset_index().merge(rep_visits_total, on=['Rep ID', 'Rep name'])

    pivot_table = pivot_table.sort_values('Total Visits', ascending=False).rename(
        columns={'Rep name': 'Rep Name', 'Total Visits': 'Total # of Visits'}
    ).drop(columns=['Rep ID'])

    return pivot_table

def display_frequency_distribution_with_rep_table(brick_segments, visits_df):
    """Display frequency distribution chart and rep frequency table."""
    # Create centered layout for the chart like Response Curve
    chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])
    
    with chart_col2:
        fig = frequency_distribution_chart(brick_segments)
        st.pyplot(fig)
    
    st.markdown("<h4>Rep Frequency Distribution (%)</h4>", unsafe_allow_html=True)
    rep_table = create_rep_frequency_table(visits_df)
    st.dataframe(rep_table)
    create_download_button(rep_table, "rep_frequency_distribution")

def to_excel_bytes(df):
    """Convert DataFrame to Excel for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def create_download_button(df, prefix="data"):
    """Create download buttons with Excel as default."""
    try:
        excel_bytes = to_excel_bytes(df)
        st.download_button(
            label=f"Download {prefix} (Excel)", data=excel_bytes,
            file_name=f"{prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.warning(f"Excel export failed: {str(e)}. Using CSV instead.")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {prefix} (CSV)", data=csv_bytes,
            file_name=f"{prefix}.csv", mime="text/csv"
        )

@st.cache_data
def frequency_distribution_chart(brick_segments):
    """Create frequency distribution chart."""
    fig, ax = plt.subplots(figsize=(5, 3))
    brick_segments['Frequency_rounded'] = brick_segments['Frequency'].round().astype(int)
    freq_distribution = brick_segments.groupby('Frequency_rounded')['Covered HCP'].sum().reindex(
        range(1, min(21, brick_segments['Frequency_rounded'].max() + 1)), fill_value=0
    ).reset_index()

    ax.bar(freq_distribution['Frequency_rounded'], freq_distribution['Covered HCP'], color='skyblue', width=0.7)
    ax.set_xlabel('Visit Frequency', fontsize=9)
    ax.set_ylabel('Number of Customers', fontsize=9)
    ax.set_title('Customer Distribution by Visit Frequency', fontsize=10, fontweight='bold')
    ax.set_xticks(freq_distribution['Frequency_rounded'])
    ax.legend(handles=[Line2D([0], [0], color='skyblue', lw=4, label='Customers')], loc='upper right', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def sigmoid_function(x, a, b, c, d):
    """Sigmoid function for curve fitting."""
    return a / (1 + np.exp(-b * (x - c))) + d

def remove_outliers_iqr(df, column):
    """Remove outliers using Interquartile Range method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

@st.cache_data(ttl=3600)
def prepare_response_curve_data(visits_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None):
    """Prepare data for response curve analysis."""
    # Apply filters to visits
    filtered_visits = visits_df.copy()
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    if selected_reps:
        filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]
    
    # Get available years and find latest and previous year
    all_years = sorted(sales_df['Year'].unique())
    if len(all_years) < 2:
        return pd.DataFrame(), "Insufficient years of data (need at least 2 years)"
    
    latest_year = all_years[-1]
    previous_year = all_years[-2]
    
    # Filter sales data
    filtered_sales = sales_df[sales_df['Product'] != 'COMPETITOR']
    if product:
        filtered_sales = filtered_sales[filtered_sales['Product'] == product]
    
    # Calculate frequency per brick for latest year
    frequency_data = filtered_visits.groupby('Geo brick ID').size().reset_index(name='Total_Visits')
    customers_per_brick = filtered_visits[['Geo brick ID', 'Customer ID']].drop_duplicates().groupby('Geo brick ID').size().reset_index(name='Unique_Customers')
    
    frequency_data = frequency_data.merge(customers_per_brick, on='Geo brick ID', how='left')
    frequency_data['Frequency'] = frequency_data['Total_Visits'] / frequency_data['Unique_Customers'].replace(0, np.nan)
    frequency_data['Frequency'] = frequency_data['Frequency'].fillna(0)
    
    # Calculate sales for latest and previous year
    latest_sales = filtered_sales[filtered_sales['Year'] == latest_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name='Latest_Sales')
    previous_sales = filtered_sales[filtered_sales['Year'] == previous_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name='Previous_Sales')
    
    # Merge all data
    response_data = frequency_data.merge(latest_sales, on='Geo brick ID', how='inner')
    response_data = response_data.merge(previous_sales, on='Geo brick ID', how='inner')
    
    # Calculate sales growth percentage
    response_data['Sales_Growth'] = ((response_data['Latest_Sales'] / response_data['Previous_Sales'].replace(0, np.nan)) - 1) * 100
    response_data['Sales_Growth'] = response_data['Sales_Growth'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Remove bricks with zero frequency or zero previous sales
    response_data = response_data[(response_data['Frequency'] > 0) & (response_data['Previous_Sales'] > 0)]
    
    if response_data.empty:
        return pd.DataFrame(), "No valid data points after filtering"
    
    # Remove outliers using IQR method
    original_count = len(response_data)
    response_data = remove_outliers_iqr(response_data, 'Frequency')
    response_data = remove_outliers_iqr(response_data, 'Sales_Growth')
    final_count = len(response_data)
    
    outliers_removed = original_count - final_count
    
    return response_data[['Geo brick ID', 'Frequency', 'Sales_Growth', 'Total_Visits', 'Unique_Customers', 'Latest_Sales', 'Previous_Sales']], f"Removed {outliers_removed} outliers from {original_count} data points"

def create_response_curve_chart(response_data):
    """Create response curve chart with sigmoid fit."""
    if response_data.empty:
        return None, "No data available for plotting"
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    x_data = response_data['Frequency'].values
    y_data = response_data['Sales_Growth'].values
    
    # Create scatter plot
    ax.scatter(x_data, y_data, alpha=0.6, color='steelblue', s=15, label='Data Points')
    
    try:
        # Fit sigmoid curve
        # Initial parameter guesses
        p0 = [np.max(y_data) - np.min(y_data), 1, np.median(x_data), np.min(y_data)]
        
        # Fit the curve
        popt, pcov = curve_fit(sigmoid_function, x_data, y_data, p0=p0, maxfev=5000)
        
        # Generate smooth curve for plotting
        x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
        y_smooth = sigmoid_function(x_smooth, *popt)
        
        # Plot the fitted curve
        ax.plot(x_smooth, y_smooth, 'red', linewidth=1.5, label='Sigmoid Curve Fit')
        
        # Calculate R-squared
        y_pred = sigmoid_function(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Find optimal frequency (point of maximum slope/inflection point)
        optimal_freq = popt[2]  # c parameter is the inflection point
        optimal_growth = sigmoid_function(optimal_freq, *popt)
        
        # Mark optimal point
        ax.axvline(x=optimal_freq, color='orange', linestyle='--', alpha=0.7, linewidth=1, label=f'Optimal: {optimal_freq:.1f}')
        ax.scatter([optimal_freq], [optimal_growth], color='orange', s=40, zorder=5)
        
        # Add performance zones
        freq_25 = np.percentile(x_data, 25)
        freq_75 = np.percentile(x_data, 75)
        
        ax.axvspan(0, freq_25, alpha=0.1, color='red', label='Low')
        ax.axvspan(freq_25, freq_75, alpha=0.1, color='yellow', label='Medium')
        ax.axvspan(freq_75, x_data.max(), alpha=0.1, color='green', label='High')
        
        fit_info = f"R² = {r_squared:.3f}\nOptimal = {optimal_freq:.1f}"
        
    except Exception as e:
        fit_info = f"Fit failed\nScatter only"
        optimal_freq = None
    
    ax.set_xlabel('Visit Frequency', fontsize=9)
    ax.set_ylabel('Sales Growth (%)', fontsize=9)
    ax.set_title('Response Curve', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)
    
    # Make axis labels smaller - reduce number of ticks and make font smaller
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Set fewer ticks on both axes
    x_ticks = np.arange(0, int(x_data.max()) + 2, 2)  # Every 2 units: 0, 2, 4, 6, 8...
    y_ticks = np.arange(int(y_data.min()/50)*50, int(y_data.max()/50)*50 + 50, 50)  # Every 50 units
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Add statistics text box
    stats_text = f"Points: {len(response_data)}\n{fit_info}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    return fig, optimal_freq

@st.cache_data(ttl=3600)
def prepare_correlation_data(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value", selected_reps=None):
    """Prepare data for correlation analysis between coverage/frequency and sales growth/market share."""
    # Apply filters to visits
    filtered_visits = visits_df.copy()
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    if selected_reps:
        filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]
    
    # Get available years and find latest and previous year
    all_years = sorted(sales_df['Year'].unique())
    if len(all_years) < 2:
        return pd.DataFrame(), "Insufficient years of data (need at least 2 years)", None
    
    latest_year = all_years[-1]
    previous_year = all_years[-2]
    
    # Filter sales data
    filtered_sales = sales_df[sales_df['Product'] != 'COMPETITOR']
    if product:
        filtered_sales = filtered_sales[filtered_sales['Product'] == product]
    
    # Calculate coverage per brick
    covered_customers = filtered_visits[['Geo brick ID', 'Customer ID']].drop_duplicates().groupby('Geo brick ID').size().reset_index(name='Covered_HCP')
    total_customers = universe_df.groupby('Geo brick ID')['Customer ID'].nunique().reset_index(name='Total_HCP')
    
    coverage_data = total_customers.merge(covered_customers, on='Geo brick ID', how='left').fillna({'Covered_HCP': 0})
    coverage_data['Coverage'] = (coverage_data['Covered_HCP'] / coverage_data['Total_HCP'].replace(0, np.nan) * 100).fillna(0)
    
    # Calculate frequency per brick
    frequency_data = filtered_visits.groupby('Geo brick ID').size().reset_index(name='Total_Visits')
    customers_per_brick = filtered_visits[['Geo brick ID', 'Customer ID']].drop_duplicates().groupby('Geo brick ID').size().reset_index(name='Unique_Customers')
    
    frequency_data = frequency_data.merge(customers_per_brick, on='Geo brick ID', how='left')
    frequency_data['Frequency'] = frequency_data['Total_Visits'] / frequency_data['Unique_Customers'].replace(0, np.nan)
    frequency_data['Frequency'] = frequency_data['Frequency'].fillna(0)
    
    # Calculate sales for latest and previous year
    latest_sales = filtered_sales[filtered_sales['Year'] == latest_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name='Latest_Sales')
    previous_sales = filtered_sales[filtered_sales['Year'] == previous_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name='Previous_Sales')
    
    # Calculate market share for latest year
    brick_list = coverage_data['Geo brick ID'].tolist()
    market_share_df, ms_column_name, ei_column_name, growth_column_name = calculate_market_share(
        sales_df, brick_list, product, metric_type
    )
    
    # Merge all data
    correlation_data = coverage_data.merge(frequency_data, on='Geo brick ID', how='inner')
    correlation_data = correlation_data.merge(latest_sales, on='Geo brick ID', how='inner')
    correlation_data = correlation_data.merge(previous_sales, on='Geo brick ID', how='inner')
    correlation_data = correlation_data.merge(market_share_df[['Geo brick ID', ms_column_name, ei_column_name]], on='Geo brick ID', how='inner')
    
    # Calculate sales growth percentage
    correlation_data['Sales_Growth'] = ((correlation_data['Latest_Sales'] / correlation_data['Previous_Sales'].replace(0, np.nan)) - 1) * 100
    correlation_data['Sales_Growth'] = correlation_data['Sales_Growth'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Add market share and EI columns with proper names
    correlation_data['Market_Share'] = correlation_data[ms_column_name]
    correlation_data['EI'] = correlation_data[ei_column_name]
    
    # Remove bricks with zero coverage/frequency or zero previous sales
    correlation_data = correlation_data[(correlation_data['Coverage'] > 0) & (correlation_data['Frequency'] > 0) & (correlation_data['Previous_Sales'] > 0)]
    
    if correlation_data.empty:
        return pd.DataFrame(), "No valid data points after filtering", latest_year
    
    # Remove outliers using IQR method
    original_count = len(correlation_data)
    correlation_data = remove_outliers_iqr(correlation_data, 'Coverage')
    correlation_data = remove_outliers_iqr(correlation_data, 'Frequency')
    correlation_data = remove_outliers_iqr(correlation_data, 'Sales_Growth')
    correlation_data = remove_outliers_iqr(correlation_data, 'Market_Share')
    # Only remove EI outliers if EI column has valid data
    if 'EI' in correlation_data.columns and correlation_data['EI'].notna().any():
        correlation_data = remove_outliers_iqr(correlation_data, 'EI')
    final_count = len(correlation_data)
    
    outliers_removed = original_count - final_count
    
    return correlation_data[['Geo brick ID', 'Coverage', 'Frequency', 'Sales_Growth', 'Market_Share', 'EI', 'Covered_HCP', 'Total_HCP', 'Latest_Sales', 'Previous_Sales']], f"Removed {outliers_removed} outliers from {original_count} data points", latest_year

def create_correlation_chart(correlation_data, x_column, y_column, x_label, y_label, chart_title):
    """Create correlation chart with linear regression."""
    if correlation_data.empty:
        return None, "No data available for plotting"
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    x_data = correlation_data[x_column].values
    y_data = correlation_data[y_column].values
    
    # Create scatter plot
    ax.scatter(x_data, y_data, alpha=0.6, color='steelblue', s=15, label='Data Points')
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        
        # Generate line points
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, 'red', linewidth=1.5, label='Linear Regression')
        
        # Calculate R-squared
        r_squared = r_value ** 2
        
        regression_info = f"R² = {r_squared:.3f}\nSlope = {slope:.2f}"
        
    except Exception as e:
        regression_info = f"Regression failed: {str(e)}"
        r_squared = None
    
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(chart_title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)
    
    # Make axis labels smaller
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Set fewer ticks on both axes
    if x_column == 'Coverage':
        x_ticks = np.arange(0, int(x_data.max()) + 10, 10)  # Every 10 units for coverage
    else:  # Frequency
        x_ticks = np.arange(0, int(x_data.max()) + 2, 2)  # Every 2 units for frequency
    
    if y_column == 'Sales_Growth':
        y_ticks = np.arange(int(y_data.min()/50)*50, int(y_data.max()/50)*50 + 50, 50)  # Every 50 units for growth
    elif y_column == 'Market_Share':
        y_ticks = np.arange(0, int(y_data.max()) + 5, 5)  # Every 5 units for market share
    else:  # EI
        y_ticks = np.arange(0, int(y_data.max()) + 20, 20)  # Every 20 units for EI
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Add statistics text box
    stats_text = f"Points: {len(correlation_data)}\n{regression_info}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    return fig, r_squared

def handle_correlation_tab(df_visits, df_universe, df_sales, specialties_options, products_options, reps_options, api_key, model, temperature):
    """Handle Correlation tab content."""
    st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_specialties = st.multiselect("Customer Speciality", specialties_options, key="correlation_specialties")
    with col2:
        selected_product = st.selectbox("Product", products_options, key="correlation_product")
        selected_product = None if selected_product == 'All' else selected_product
    with col3:
        selected_reps = st.multiselect("Rep Name", reps_options, key="correlation_reps")
    with col4:
        metric_type = st.radio("Metric for Sales", ["Value", "Volume"], horizontal=True, key="correlation_metric")

    # Prepare data
    correlation_data, message, latest_year = prepare_correlation_data(
        df_visits, df_universe, df_sales, selected_specialties, selected_product, metric_type, selected_reps
    )
    
    if correlation_data.empty:
        st.warning(f"No data available for correlation analysis. {message}")
        return
    
    st.info(message)
    
    # Debug information
    if not correlation_data.empty:
        st.write(f"**Debug Info:** Dataset has {len(correlation_data)} rows and columns: {list(correlation_data.columns)}")
        if 'EI' in correlation_data.columns:
            ei_stats = correlation_data['EI'].describe()
            st.write(f"**EI Stats:** Min={ei_stats['min']:.1f}, Max={ei_stats['max']:.1f}, Mean={ei_stats['mean']:.1f}")
        else:
            st.warning("EI column missing from correlation data")
    
    # Create and display first row of charts (Coverage-based)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig1, r_squared1 = create_correlation_chart(
            correlation_data, 'Coverage', 'Sales_Growth', 'Coverage (%)', 'Sales Growth (%)', 'Coverage vs Sales Growth'
        )
        if fig1 is not None:
            st.pyplot(fig1)
    
    with chart_col2:
        fig2, r_squared2 = create_correlation_chart(
            correlation_data, 'Coverage', 'Market_Share', 'Coverage (%)', 'Market Share (%)', f'Coverage vs Market Share ({latest_year})'
        )
        if fig2 is not None:
            st.pyplot(fig2)
    
    # Create and display second row of charts (Frequency-based)
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        fig3, r_squared3 = create_correlation_chart(
            correlation_data, 'Frequency', 'Sales_Growth', 'Frequency', 'Sales Growth (%)', 'Frequency vs Sales Growth'
        )
        if fig3 is not None:
            st.pyplot(fig3)
    
    with chart_col4:
        fig4, r_squared4 = create_correlation_chart(
            correlation_data, 'Frequency', 'Market_Share', 'Frequency', 'Market Share (%)', f'Frequency vs Market Share ({latest_year})'
        )
        if fig4 is not None:
            st.pyplot(fig4)
    
    # Create and display third row of charts (EI-based) - NEW ADDITION
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    chart_col5, chart_col6 = st.columns(2)
    
    with chart_col5:
        # Debug: Check if EI data exists
        if 'EI' in correlation_data.columns and not correlation_data['EI'].empty:
            fig5, r_squared5 = create_correlation_chart(
                correlation_data, 'Frequency', 'EI', 'Frequency', 'Evolution Index', 'Frequency vs Evolution Index'
            )
            if fig5 is not None:
                st.pyplot(fig5)
            else:
                st.warning("Could not create Frequency vs EI chart")
        else:
            st.warning("EI data not available for Frequency vs EI chart")
    
    with chart_col6:
        # Debug: Check if EI data exists
        if 'EI' in correlation_data.columns and not correlation_data['EI'].empty:
            fig6, r_squared6 = create_correlation_chart(
                correlation_data, 'Coverage', 'EI', 'Coverage (%)', 'Evolution Index', 'Coverage vs Evolution Index'
            )
            if fig6 is not None:
                st.pyplot(fig6)
            else:
                st.warning("Could not create Coverage vs EI chart")
        else:
            st.warning("EI data not available for Coverage vs EI chart")

    # Display data table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Correlation Data</h3>", unsafe_allow_html=True)
    
    # Prepare display data
    display_data = correlation_data.copy()
    display_data = display_data.round(2)
    display_data.columns = ['Geo Brick ID', 'Coverage %', 'Frequency', 'Sales Growth %', f'Market Share {latest_year} %', 'Evolution Index', 'Covered HCP', 'Total HCP', 'Latest Sales', 'Previous Sales']
    
    st.dataframe(display_data, use_container_width=True)
    create_download_button(display_data, "correlation_data")
    
    # AI Insights
    data_summary = prepare_data_summary_for_ai(correlation_data, "correlation")
    display_ai_insights(data_summary, "correlation", api_key, model, temperature)

def handle_tab_content_with_correlation(tab_name, df_visits, df_universe, df_sales, specialties_options, products_options, reps_options, segmentation_func, primary_column, prefix, show_chart=False, chart_func=None, boundary_format="{:.2f}", api_key=None, model=None, temperature=0.3):
    """Enhanced handle_tab_content with correlation analysis and AI insights."""
    st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        selected_specialties = st.multiselect("Customer Speciality", specialties_options, key=f"{prefix}_specialties")
    with col2:
        selected_product = st.selectbox("Product", products_options, key=f"{prefix}_product")
        selected_product = None if selected_product == 'All' else selected_product
    with col3:
        selected_reps = st.multiselect("Rep Name", reps_options, key=f"{prefix}_reps")
    with col4:
        metric_type = st.radio("Metric for Market Share", ["Value", "Volume"], horizontal=True, key=f"{prefix}_metric")
    with col5:
        exclude_zeros = st.checkbox("Exclude Zeros from Segmentation", key=f"{prefix}_exclude_zeros", 
                                   help=f"When checked, excludes bricks with 0 {primary_column.lower()} from quintile calculations")

    brick_segments, quantiles, segment_labels, ms_column_name, ei_column_name, growth_column_name = segmentation_func(
        df_visits, df_universe, df_sales, selected_specialties, selected_product, metric_type, selected_reps, exclude_zeros
    )

    if brick_segments.empty:
        st.warning(f"No bricks with valid {tab_name} data found for the selected filters.")
        return

    segment_stats = calculate_segment_statistics(
        brick_segments, segment_labels, ms_column_name, ei_column_name, growth_column_name, primary_column
    )

    st.markdown("<h3>Segment Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(segment_stats)

    # Add correlation analysis based on the primary column
    correlation_title_map = {
        'Frequency': 'Frequency Correlation Analysis',
        'Ei': 'Evolution Index Correlation Analysis',
        'Coverage, %': 'Coverage Correlation Analysis'
    }
    
    correlation_title = correlation_title_map.get(primary_column, f"{primary_column} Correlation Analysis")
    
    st.markdown(f"<h3>{correlation_title}</h3>", unsafe_allow_html=True)
    correlations = calculate_correlations(segment_stats, primary_column)
    correlation_text = format_correlation_text(correlations, primary_column)
    
    # Display in an info box
    st.info(correlation_text)
    
    # Optional: Show all correlations in a more detailed format
    if correlations:
        with st.expander(f"View All Correlations with {primary_column}"):
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Metric', 'Correlation'])
            corr_df['Correlation'] = corr_df['Correlation'].round(3)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df, use_container_width=True)

    # Show chart after correlation analysis
    if show_chart and chart_func and primary_column == 'Frequency':
        # Fixed filtering logic for the chart display
        filtered_visits = df_visits.copy()
        if selected_specialties:
            filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(selected_specialties)]
        if selected_reps:
            filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]
        display_frequency_distribution_with_rep_table(brick_segments, filtered_visits)

    create_download_button(segment_stats, f"{prefix}_segment_statistics")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Brick Segmentation Table</h3>", unsafe_allow_html=True)
    st.dataframe(brick_segments)
    create_download_button(brick_segments, f"{prefix}_brick_segments")

    # Show segment boundaries with context about zero exclusion
    st.markdown("<h3>Segment Boundaries:</h3>", unsafe_allow_html=True)
    if exclude_zeros and quantiles:
        st.markdown("*Boundaries calculated excluding zero values:*")
        boundary_text = [f"- {segment_labels[i]}/{segment_labels[i+1]}: {boundary_format.format(q)}" for i, q in enumerate(quantiles)]
        st.markdown("\n".join(boundary_text))
    elif exclude_zeros and not quantiles:
        st.markdown("*No boundaries calculated - insufficient non-zero data for segmentation*")
    elif quantiles:
        st.markdown("*Boundaries calculated including zero values:*")
        boundary_text = [f"- {segment_labels[i]}/{segment_labels[i+1]}: {boundary_format.format(q)}" for i, q in enumerate(quantiles)]
        st.markdown("\n".join(boundary_text))
    else:
        st.markdown("*No boundaries calculated - insufficient data for segmentation*")
    
    # AI Insights
    context_type = f"{prefix}_segmentation"
    data_summary = prepare_data_summary_for_ai(brick_segments, context_type)
    display_ai_insights(data_summary, context_type, api_key, model, temperature)

def handle_tab_content(tab_name, df_visits, df_universe, df_sales, specialties_options, products_options, reps_options, segmentation_func, primary_column, prefix, show_chart=False, chart_func=None, boundary_format="{:.2f}", api_key=None, model=None, temperature=0.3):
    """Handle tab content for segmentation with AI insights."""
    st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_specialties = st.multiselect("Customer Speciality", specialties_options, key=f"{prefix}_specialties")
    with col2:
        selected_product = st.selectbox("Product", products_options, key=f"{prefix}_product")
        selected_product = None if selected_product == 'All' else selected_product
    with col3:
        selected_reps = st.multiselect("Rep Name", reps_options, key=f"{prefix}_reps")
    with col4:
        metric_type = st.radio("Metric for Market Share", ["Value", "Volume"], horizontal=True, key=f"{prefix}_metric")

    brick_segments, quantiles, segment_labels, ms_column_name, ei_column_name, growth_column_name = segmentation_func(
        df_visits, df_universe, df_sales, selected_specialties, selected_product, metric_type, selected_reps
    )

    if brick_segments.empty:
        st.warning(f"No bricks with valid {tab_name} data found for the selected filters.")
        return

    segment_stats = calculate_segment_statistics(
        brick_segments, segment_labels, ms_column_name, ei_column_name, growth_column_name, primary_column
    )

    st.markdown("<h3>Segment Statistics</h3>", unsafe_allow_html=True)
    if show_chart and chart_func:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(segment_stats)
        with col2:
            if primary_column == 'Frequency':
                # Fixed filtering logic for the chart display
                filtered_visits = df_visits.copy()
                if selected_specialties:
                    filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(selected_specialties)]
                if selected_reps:
                    filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]
                display_frequency_distribution_with_rep_table(brick_segments, filtered_visits)
            else:
                st.pyplot(chart_func(brick_segments))
    else:
        st.dataframe(segment_stats)

    create_download_button(segment_stats, f"{prefix}_segment_statistics")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Brick Segmentation Table</h3>", unsafe_allow_html=True)
    st.dataframe(brick_segments)
    create_download_button(brick_segments, f"{prefix}_brick_segments")

    st.markdown("<h3>Segment Boundaries:</h3>", unsafe_allow_html=True)
    boundary_text = [f"- {segment_labels[i]}/{segment_labels[i+1]}: {boundary_format.format(q)}" for i, q in enumerate(quantiles)]
    st.markdown("\n".join(boundary_text))
    
    # AI Insights
    context_type = f"{prefix}_segmentation"
    data_summary = prepare_data_summary_for_ai(brick_segments, context_type)
    display_ai_insights(data_summary, context_type, api_key, model, temperature)

@st.cache_data(ttl=3600)
def analyze_vacant_territories(visits_df, sales_df, universe_df, specialties=None, product=None, metric_type="Value", selected_reps=None):
    """Analyze vacant territories - had visits in year-1 but no visits in last 6 months of current year."""
    # Apply filters to visits
    filtered_visits = visits_df.copy()
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    if selected_reps:
        filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]
    
    # Convert Call Date to datetime
    filtered_visits['Call Date'] = pd.to_datetime(filtered_visits['Call Date'])
    
    # Get available years
    all_years = sorted(sales_df['Year'].unique())
    if len(all_years) < 2:
        return pd.DataFrame(), pd.DataFrame(), "Insufficient years of data (need at least 2 years)"
    
    latest_year = all_years[-1]
    previous_year = all_years[-2]
    
    # Get territories with visits in previous year
    prev_year_visits = filtered_visits[filtered_visits['Call Date'].dt.year == previous_year]
    territories_with_prev_visits = set(prev_year_visits['Geo brick ID'].unique())
    
    # Get territories with visits in current year
    current_year_visits = filtered_visits[filtered_visits['Call Date'].dt.year == latest_year]
    
    # Get last 6 months of current year
    current_year_end = pd.Timestamp(f"{latest_year}-12-31")
    six_months_ago = current_year_end - pd.DateOffset(months=6)
    
    # Get territories with visits in last 6 months of current year
    last_6_months_visits = current_year_visits[current_year_visits['Call Date'] >= six_months_ago]
    territories_with_recent_visits = set(last_6_months_visits['Geo brick ID'].unique())
    
    # Identify vacant territories
    vacant_territories = territories_with_prev_visits - territories_with_recent_visits
    
    if not vacant_territories:
        return pd.DataFrame(), pd.DataFrame(), f"No vacant territories found. All {len(territories_with_prev_visits)} territories from {previous_year} had visits in the last 6 months of {latest_year}."
    
    # Analyze performance for vacant territories
    vacant_list = list(vacant_territories)
    
    # Get market share data for vacant territories
    market_share_df, ms_column_name, ei_column_name, growth_column_name = calculate_market_share(
        sales_df, vacant_list, product, metric_type
    )
    
    # Filter sales data
    filtered_sales = sales_df[sales_df['Product'] != 'COMPETITOR']
    if product:
        filtered_sales = filtered_sales[filtered_sales['Product'] == product]
    
    # Calculate sales for latest and previous year for vacant territories
    latest_sales = filtered_sales[filtered_sales['Year'] == latest_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name=f'Sales_{latest_year}')
    previous_sales = filtered_sales[filtered_sales['Year'] == previous_year].groupby('Geo brick ID')[metric_type].sum().reset_index(name=f'Sales_{previous_year}')
    
    # Create vacant territories performance dataframe
    vacant_performance = pd.DataFrame({'Geo brick ID': vacant_list})
    vacant_performance = vacant_performance.merge(market_share_df, on='Geo brick ID', how='left')
    vacant_performance = vacant_performance.merge(latest_sales, on='Geo brick ID', how='left')
    vacant_performance = vacant_performance.merge(previous_sales, on='Geo brick ID', how='left')
    
    # Fill missing values
    vacant_performance = vacant_performance.fillna({
        ms_column_name: 0, ei_column_name: 0, growth_column_name: 0,
        f'Sales_{latest_year}': 0, f'Sales_{previous_year}': 0
    })
    
    # Calculate growth rate
    vacant_performance['Growth_Rate_%'] = ((vacant_performance[f'Sales_{latest_year}'] / vacant_performance[f'Sales_{previous_year}'].replace(0, np.nan)) - 1) * 100
    vacant_performance['Growth_Rate_%'] = vacant_performance['Growth_Rate_%'].fillna(0).replace([np.inf, -np.inf], 0).round(1)
    
    # Get customer counts for vacant territories
    vacant_customers = universe_df[universe_df['Geo brick ID'].isin(vacant_list)].groupby('Geo brick ID')['Customer ID'].nunique().reset_index(name='Total_Customers')
    vacant_performance = vacant_performance.merge(vacant_customers, on='Geo brick ID', how='left').fillna({'Total_Customers': 0})
    
    # Get visit counts for previous year
    prev_visits_count = prev_year_visits.groupby('Geo brick ID').size().reset_index(name=f'Visits_{previous_year}')
    vacant_performance = vacant_performance.merge(prev_visits_count, on='Geo brick ID', how='left').fillna({f'Visits_{previous_year}': 0})
    
    # Prepare summary statistics
    vacant_summary = {
        'Total_Vacant_Territories': len(vacant_territories),
        'Total_Territories_Prev_Year': len(territories_with_prev_visits),
        'Vacant_Percentage': round(len(vacant_territories) / len(territories_with_prev_visits) * 100, 1) if territories_with_prev_visits else 0,
        'Avg_Market_Share': round(vacant_performance[ms_column_name].mean(), 1),
        'Avg_EI': round(vacant_performance[ei_column_name].mean(), 0),
        'Avg_Growth_Rate': round(vacant_performance['Growth_Rate_%'].mean(), 1),
        'Total_Sales_Lost': round(vacant_performance[f'Sales_{latest_year}'].sum(), 0),
        'Total_Customers_Uncovered': vacant_performance['Total_Customers'].sum(),
        'Previous_Year': previous_year,
        'Current_Year': latest_year
    }
    
    # Clean up column names for display
    display_columns = [
        'Geo brick ID', ms_column_name, ei_column_name, 'Growth_Rate_%',
        f'Sales_{latest_year}', f'Sales_{previous_year}', 'Total_Customers', f'Visits_{previous_year}'
    ]
    vacant_display = vacant_performance[display_columns].copy()
    
    # Rename columns for better display
    column_renames = {
        ms_column_name: f'Market Share {latest_year} (%)',
        ei_column_name: 'Evolution Index',
        'Growth_Rate_%': 'Growth Rate (%)',
        f'Sales_{latest_year}': f'Sales {latest_year}',
        f'Sales_{previous_year}': f'Sales {previous_year}',
        f'Visits_{previous_year}': f'Visits {previous_year}'
    }
    vacant_display = vacant_display.rename(columns=column_renames)
    
    # Sort by market share descending
    vacant_display = vacant_display.sort_values(f'Market Share {latest_year} (%)', ascending=False)
    
    message = f"Found {len(vacant_territories)} vacant territories out of {len(territories_with_prev_visits)} territories that had visits in {previous_year}"
    
    return vacant_display, vacant_summary, message

@st.cache_data(ttl=3600)
def process_carryover_data(sales_df, universe_df, visits_df, specialties=None, product=None, metric_type="Value", selected_reps=None):
    """Process data for Carryover tab."""
    # Fixed filtering logic
    filtered_visits = visits_df.copy()
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    if selected_reps:
        filtered_visits = filtered_visits[filtered_visits['Rep name'].isin(selected_reps)]

    filtered_sales = sales_df[sales_df['Product'] != 'COMPETITOR']
    if product:
        filtered_sales = filtered_sales[filtered_sales['Product'] == product]

    unique_bricks = pd.DataFrame({'Geo brick ID': filtered_sales['Geo brick ID'].unique()})
    all_customers_per_brick = universe_df.groupby('Geo brick ID').size().reset_index(name='Number of customers')
    visits_per_brick = filtered_visits.groupby('Geo brick ID').size().reset_index(name='Visits')

    result = unique_bricks.merge(all_customers_per_brick, on='Geo brick ID', how='left').merge(
        visits_per_brick, on='Geo brick ID', how='left'
    ).fillna({'Number of customers': 0, 'Visits': 0}).astype({'Number of customers': int, 'Visits': int})

    result['Frequency'] = (result['Visits'] / result['Number of customers'].replace(0, np.nan)).fillna(0).round(2)

    years = sorted(filtered_sales['Year'].unique())
    latest_year = years[-1]
    previous_year = years[-2] if len(years) > 1 else None

    latest_sales = filtered_sales[filtered_sales['Year'] == latest_year].groupby('Geo brick ID')[metric_type].sum().reset_index(
        name=f'Sales {latest_year}'
    )
    result = result.merge(latest_sales, on='Geo brick ID', how='left').fillna({f'Sales {latest_year}': 0})

    year_ratio_col = f'Year {latest_year} / Year {previous_year}'
    if previous_year:
        previous_sales = filtered_sales[filtered_sales['Year'] == previous_year].groupby('Geo brick ID')[metric_type].sum().reset_index(
            name=f'Sales {previous_year}'
        )
        result = result.merge(previous_sales, on='Geo brick ID', how='left').fillna({f'Sales {previous_year}': 0})
        result[year_ratio_col] = (result[f'Sales {latest_year}'] / result[f'Sales {previous_year}'].replace(0, np.nan) * 100).fillna(0).round(2)
        # Handle inf values in year ratio
        result[year_ratio_col] = result[year_ratio_col].replace([np.inf, -np.inf], 0)

    bricks_with_visits = result[result['Visits'] > 0]
    frequency_min = bricks_with_visits['Frequency'].min() if not bricks_with_visits.empty else 0
    frequency_max = bricks_with_visits['Frequency'].max() if not bricks_with_visits.empty else 0
    frequency_threshold = frequency_min + (frequency_max - frequency_min) * 0.1 if frequency_max > frequency_min else 0

    result['Segment'] = np.where(result['Visits'] == 0, 'Filtered out',
                                np.where(result['Frequency'] <= frequency_threshold, 'Non significant activity', 'Significant activity'))

    agg_cols = {'Geo brick ID': 'count', 'Frequency': 'mean', f'Sales {latest_year}': 'sum'}
    if previous_year:
        agg_cols[f'Sales {previous_year}'] = 'sum'

    agg_data = result.groupby('Segment').agg(agg_cols).reset_index().rename(columns={'Geo brick ID': 'Number of bricks'})
    agg_data['Frequency'] = agg_data['Frequency'].round(2)

    if previous_year:
        agg_data[year_ratio_col] = (agg_data[f'Sales {latest_year}'] / agg_data[f'Sales {previous_year}'].replace(0, np.nan) * 100).round(2).fillna(0)
        # Handle inf values in aggregated year ratio
        agg_data[year_ratio_col] = agg_data[year_ratio_col].replace([np.inf, -np.inf], 0)

    return result, frequency_threshold, agg_data, frequency_min, frequency_max

@st.cache_data(ttl=3600)
def prepare_rep_clustering_data(visits_df, universe_df, sales_df, specialties=None, product=None, metric_type="Value"):
    """Prepare data for Rep clustering analysis."""
    # Apply filters to visits
    filtered_visits = visits_df.copy()
    if specialties:
        filtered_visits = filtered_visits[filtered_visits['Customer Speciality'].isin(specialties)]
    
    # Filter sales data
    filtered_sales = sales_df[sales_df['Product'] != 'COMPETITOR']
    if product:
        filtered_sales = filtered_sales[filtered_sales['Product'] == product]
    
    # Get all years
    all_years = sorted(sales_df['Year'].unique())
    if len(all_years) < 2:
        return pd.DataFrame(), "Insufficient years of data (need at least 2 years)"
    
    latest_year = all_years[-1]
    previous_year = all_years[-2]
    
    # Calculate metrics per rep
    rep_metrics = []
    
    for rep_id, rep_name in filtered_visits[['Rep ID', 'Rep name']].drop_duplicates().values:
        rep_visits = filtered_visits[filtered_visits['Rep ID'] == rep_id]
        rep_bricks = rep_visits['Geo brick ID'].unique()
        
        # Coverage: percentage of customers visited out of total customers in rep's bricks
        total_customers_in_bricks = universe_df[universe_df['Geo brick ID'].isin(rep_bricks)]['Customer ID'].nunique()
        covered_customers = rep_visits['Customer ID'].nunique()
        coverage = (covered_customers / total_customers_in_bricks * 100) if total_customers_in_bricks > 0 else 0
        
        # Frequency: average visits per customer
        total_visits = len(rep_visits)
        frequency = total_visits / covered_customers if covered_customers > 0 else 0
        
        # Market Share and Growth for rep's bricks
        latest_sales_data = filtered_sales[(filtered_sales['Year'] == latest_year) & 
                                         (filtered_sales['Geo brick ID'].isin(rep_bricks))]
        previous_sales_data = filtered_sales[(filtered_sales['Year'] == previous_year) & 
                                           (filtered_sales['Geo brick ID'].isin(rep_bricks))]
        
        # Calculate market share for rep's bricks
        ms_list = []
        for brick in rep_bricks:
            ms = calculate_single_year_ms(sales_df[sales_df['Year'] == latest_year], 
                                        brick, product, metric_type)
            ms_list.append(ms)
        
        market_share = np.mean(ms_list) if ms_list else 0
        
        # Calculate growth
        latest_sales = latest_sales_data[metric_type].sum()
        previous_sales = previous_sales_data[metric_type].sum()
        growth = ((latest_sales / previous_sales) - 1) * 100 if previous_sales > 0 else 0
        
        # Calculate Evolution Index (EI)
        latest_ms = market_share
        previous_ms_list = []
        for brick in rep_bricks:
            prev_ms = calculate_single_year_ms(sales_df[sales_df['Year'] == previous_year], 
                                             brick, product, metric_type)
            previous_ms_list.append(prev_ms)
        previous_ms = np.mean(previous_ms_list) if previous_ms_list else 0
        
        ei = (latest_ms / previous_ms * 100) if previous_ms > 0 else 100
        
        # Handle inf/nan values
        if not np.isfinite(growth):
            growth = 0
        if not np.isfinite(ei):
            ei = 100
        
        rep_metrics.append({
            'Rep ID': rep_id,
            'Rep Name': rep_name,
            'Coverage': round(coverage, 2),
            'Frequency': round(frequency, 2),
            'Total Visits': total_visits,
            'Market Share': round(market_share, 2),
            'Growth': round(growth, 2),
            'EI': round(ei, 2),
            'Covered Customers': covered_customers,
            'Total Customers': total_customers_in_bricks,
            'Num Bricks': len(rep_bricks)
        })
    
    return pd.DataFrame(rep_metrics), f"Analyzed {len(rep_metrics)} reps"

def perform_kmeans_clustering(rep_data, activity_weights={'Coverage': 0.3, 'Frequency': 0.4, 'Total Visits': 0.3},
                             performance_weights={'Market Share': 0.5, 'EI': 0.5}):
    """Perform K-means clustering with Activity and Performance indices."""
    # Create composite indices
    # Activity Index = weighted average of coverage, frequency, and visits
    activity_cols = list(activity_weights.keys())
    performance_cols = list(performance_weights.keys())
    
    # Standardize the features
    scaler = StandardScaler()
    
    # Prepare data for scaling
    activity_data = rep_data[activity_cols].values
    performance_data = rep_data[performance_cols].values
    
    # Scale the data
    activity_scaled = scaler.fit_transform(activity_data)
    performance_scaled = scaler.fit_transform(performance_data)
    
    # Create weighted composite scores
    activity_index = np.zeros(len(rep_data))
    for i, col in enumerate(activity_cols):
        activity_index += activity_scaled[:, i] * activity_weights[col]
    
    performance_index = np.zeros(len(rep_data))
    for i, col in enumerate(performance_cols):
        performance_index += performance_scaled[:, i] * performance_weights[col]
    
    # Normalize indices to 0-100 scale for interpretability
    activity_index_norm = (activity_index - activity_index.min()) / (activity_index.max() - activity_index.min()) * 100
    performance_index_norm = (performance_index - performance_index.min()) / (performance_index.max() - performance_index.min()) * 100
    
    # Add indices to dataframe
    rep_data['Activity Index'] = activity_index_norm.round(2)
    rep_data['Performance Index'] = performance_index_norm.round(2)
    
    # Prepare data for K-means (using standardized indices)
    X = np.column_stack([activity_index, performance_index])
    
    # Perform K-means with k=9
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, clusters)
    
    # Assign cluster labels based on centroids
    centroids = kmeans.cluster_centers_
    
    # Determine thresholds for Low/Medium/High (using 33rd and 67th percentiles)
    activity_33 = np.percentile(activity_index, 33)
    activity_67 = np.percentile(activity_index, 67)
    performance_33 = np.percentile(performance_index, 33)
    performance_67 = np.percentile(performance_index, 67)
    
    # Label clusters based on centroid positions - Strategic business-focused names
    cluster_labels = {}
    cluster_descriptions = {
        'Low-Low': ('Development Required', 'Low investment, low returns - need fundamental improvements'),
        'Low-Med': ('Efficiency Optimizers', 'Low investment, moderate returns - efficient but limited scale'),
        'Low-High': ('Strategic Gems', 'Low investment, high returns - exceptional ROI, scale opportunity'),
        'Med-Low': ('Tactical Review', 'Moderate investment, low returns - strategy adjustment needed'),
        'Med-Med': ('Core Performers', 'Moderate investment, moderate returns - stable foundation'),
        'Med-High': ('Growth Accelerators', 'Moderate investment, high returns - ready for expansion'),
        'High-Low': ('Resource Reallocation', 'High investment, low returns - critical efficiency gap'),
        'High-Med': ('Market Builders', 'High investment, moderate returns - building market presence'),
        'High-High': ('Market Leaders', 'High investment, high returns - sustain and replicate')
    }
    
    for i, centroid in enumerate(centroids):
        activity_level = 'Low' if centroid[0] < activity_33 else 'Med' if centroid[0] < activity_67 else 'High'
        performance_level = 'Low' if centroid[1] < performance_33 else 'Med' if centroid[1] < performance_67 else 'High'
        
        matrix_position = f"{activity_level}-{performance_level}"
        cluster_labels[i] = cluster_descriptions[matrix_position][0]
    
    # Assign cluster names to reps
    rep_data['Cluster'] = [cluster_labels[c] for c in clusters]
    rep_data['Matrix Position'] = clusters
    
    return rep_data, silhouette_avg, cluster_labels, activity_weights, performance_weights

def create_cluster_visualization(cluster_data, cluster_labels):
    """Create visualization for K-means clusters with Plotly for interactive hover."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Define colors for each cluster type
    cluster_colors = {
        'Development Required': '#DC143C',
        'Efficiency Optimizers': '#FF8C00',
        'Strategic Gems': '#FFD700',
        'Tactical Review': '#CD5C5C',
        'Core Performers': '#708090',
        'Growth Accelerators': '#32CD32',
        'Resource Reallocation': '#B22222',
        'Market Builders': '#4169E1',
        'Market Leaders': '#9370DB'
    }
    
    # Create subplots with Plotly
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('K-means Clustering: Activity vs Performance Matrix', 'Rep Distribution in 3×3 Matrix'),
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.15,
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # Plot 1: Interactive scatter plot with hover
    for cluster_name in cluster_data['Cluster'].unique():
        cluster_df = cluster_data[cluster_data['Cluster'] == cluster_name]
        
        # Create hover text
        hover_text = []
        for _, row in cluster_df.iterrows():
            text = f"<b>{row['Rep Name']}</b><br>" + \
                   f"Cluster: {cluster_name}<br>" + \
                   f"Activity Index: {row['Activity Index']:.1f}<br>" + \
                   f"Performance Index: {row['Performance Index']:.1f}<br>" + \
                   f"Coverage: {row['Coverage']:.1f}%<br>" + \
                   f"Frequency: {row['Frequency']:.1f}<br>" + \
                   f"Market Share: {row['Market Share']:.1f}%"
            hover_text.append(text)
        
        fig.add_trace(
            go.Scatter(
                x=cluster_df['Activity Index'],
                y=cluster_df['Performance Index'],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    size=12,
                    color=cluster_colors.get(cluster_name, '#000000'),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add grid lines for zones
    for val in [33, 67]:
        fig.add_vline(x=val, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Add zone labels
    fig.add_annotation(x=16, y=90, text="Low Activity", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xref="x", yref="y")
    fig.add_annotation(x=50, y=90, text="Med Activity", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xref="x", yref="y")
    fig.add_annotation(x=84, y=90, text="High Activity", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xref="x", yref="y")
    fig.add_annotation(x=5, y=16, text="Low<br>Perf", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xanchor='left', xref="x", yref="y")
    fig.add_annotation(x=5, y=50, text="Med<br>Perf", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xanchor='left', xref="x", yref="y")
    fig.add_annotation(x=5, y=84, text="High<br>Perf", showarrow=False, font=dict(size=8, color='gray'), 
                      opacity=0.7, xanchor='left', xref="x", yref="y")
    
    # Update scatter plot axes - make them equal length (square aspect ratio)
    fig.update_xaxes(title_text="Activity Index (0-100)", range=[-5, 105], row=1, col=1, 
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     constrain="domain")  # Constrain to domain instead of scaleanchor
    fig.update_yaxes(title_text="Performance Index (0-100)", range=[-5, 105], row=1, col=1,
                     showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                     scaleanchor="x", scaleratio=1, constrain="domain")  # Keep aspect ratio but constrain
    
    # Plot 2: Create heatmap matrix
    matrix_data = np.zeros((3, 3))
    
    # Count reps in each position
    for _, row in cluster_data.iterrows():
        activity_idx = 0 if row['Activity Index'] < 33 else 1 if row['Activity Index'] < 67 else 2
        performance_idx = 0 if row['Performance Index'] < 33 else 1 if row['Performance Index'] < 67 else 2
        matrix_data[performance_idx, activity_idx] += 1  # Changed to normal order for Y axis
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix_data,
            x=['Low', 'Medium', 'High'],
            y=['Low', 'Medium', 'High'],  # Changed to normal order
            colorscale='YlOrRd',
            showscale=False,
            text=matrix_data.astype(int),
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Update heatmap axes
    fig.update_xaxes(title_text="Activity Level", row=1, col=2, tickfont=dict(size=10))
    fig.update_yaxes(title_text="Performance Level", row=1, col=2, tickfont=dict(size=10))
    
    # Update overall layout
    fig.update_layout(
        height=500,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=8)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def handle_clusters_tab(df_visits, df_universe, df_sales, specialties_options, products_options, api_key, model, temperature):
    """Handle Clusters tab content."""
    st.markdown("<h3>Rep Clustering Analysis - K-means 3×3 Matrix Approach</h3>", unsafe_allow_html=True)
    st.markdown("This analysis uses K-means clustering to group sales representatives into a 3×3 matrix based on Activity and Performance indices.")
    st.markdown("**Hover over any point in the scatter plot to see the Rep name and details.**")
    
    # Filter options
    st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_specialties = st.multiselect("Customer Speciality", specialties_options, key="clusters_specialties")
    with col2:
        selected_product = st.selectbox("Product", products_options, key="clusters_product")
        selected_product = None if selected_product == 'All' else selected_product
    with col3:
        metric_type = st.radio("Metric for Sales", ["Value", "Volume"], horizontal=True, key="clusters_metric")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings - Adjust Index Weights"):
        st.markdown("**Activity Index Weights** (must sum to 1.0)")
        col1, col2, col3 = st.columns(3)
        with col1:
            coverage_weight = st.slider("Coverage Weight", 0.0, 1.0, 0.3, 0.1)
        with col2:
            frequency_weight = st.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.1)
        with col3:
            visits_weight = st.slider("Visits Weight", 0.0, 1.0, 0.3, 0.1)
        
        activity_sum = coverage_weight + frequency_weight + visits_weight
        if abs(activity_sum - 1.0) > 0.01:
            st.warning(f"Activity weights sum to {activity_sum:.2f}, should be 1.0")
        
        st.markdown("**Performance Index Weights** (must sum to 1.0)")
        col1, col2 = st.columns(2)
        with col1:
            ms_weight = st.slider("Market Share Weight", 0.0, 1.0, 0.5, 0.1)
        with col2:
            ei_weight = st.slider("Evolution Index Weight", 0.0, 1.0, 0.5, 0.1)
        
        performance_sum = ms_weight + ei_weight
        if abs(performance_sum - 1.0) > 0.01:
            st.warning(f"Performance weights sum to {performance_sum:.2f}, should be 1.0")
    
    # Prepare weights
    activity_weights = {
        'Coverage': coverage_weight,
        'Frequency': frequency_weight,
        'Total Visits': visits_weight
    }
    performance_weights = {
        'Market Share': ms_weight,
        'EI': ei_weight
    }
    
    # Prepare data
    rep_data, message = prepare_rep_clustering_data(
        df_visits, df_universe, df_sales, selected_specialties, selected_product, metric_type
    )
    
    if rep_data.empty:
        st.warning(f"No data available for clustering analysis. {message}")
        return
    
    st.info(message)
    
    # Perform K-means clustering
    clustered_data, silhouette_avg, cluster_labels, final_activity_weights, final_performance_weights = perform_kmeans_clustering(
        rep_data, activity_weights, performance_weights
    )
    
    # Display the 3×3 matrix concept
    with st.expander("📖 3×3 Matrix Clustering Concept"):
        st.markdown("""
        ### Activity-Performance Matrix
        
        <table>
        <tr>
            <th></th>
            <th><b>Low Activity</b></th>
            <th><b>Medium Activity</b></th>
            <th><b>High Activity</b></th>
        </tr>
        <tr>
            <td><b>Low Performance</b></td>
            <td>Development Required</td>
            <td>Tactical Review</td>
            <td>Resource Reallocation</td>
        </tr>
        <tr>
            <td><b>Medium Performance</b></td>
            <td>Efficiency Optimizers</td>
            <td>Core Performers</td>
            <td>Market Builders</td>
        </tr>
        <tr>
            <td><b>High Performance</b></td>
            <td>Strategic Gems</td>
            <td>Growth Accelerators</td>
            <td>Market Leaders</td>
        </tr>
        </table>
        
        **Activity Index** combines:
        - Coverage (% of customers visited)
        - Frequency (average visits per customer)
        - Total visits
        
        **Performance Index** combines:
        - Market Share
        - Evolution Index (EI)
        """, unsafe_allow_html=True)
    
    # Display cluster summary statistics
    st.markdown("<h3>Cluster Summary</h3>", unsafe_allow_html=True)
    
    cluster_summary = clustered_data.groupby('Cluster').agg({
        'Rep Name': 'count',
        'Activity Index': 'mean',
        'Performance Index': 'mean',
        'Coverage': 'mean',
        'Frequency': 'mean',
        'Market Share': 'mean',
        'EI': 'mean',
        'Total Visits': 'sum'
    }).round(2)
    
    cluster_summary.columns = ['Number of Reps', 'Avg Activity Index', 'Avg Performance Index',
                              'Avg Coverage %', 'Avg Frequency', 'Avg Market Share %', 'Avg EI', 'Total Visits']
    cluster_summary = cluster_summary.sort_values('Number of Reps', ascending=False)
    
    st.dataframe(cluster_summary)
    create_download_button(cluster_summary, "cluster_summary")
    
    # Display visualizations
    st.markdown("<h3>Cluster Visualizations</h3>", unsafe_allow_html=True)
    fig = create_cluster_visualization(clustered_data, cluster_labels)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed rep data
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Detailed Rep Clustering Data</h3>", unsafe_allow_html=True)
    
    # Allow filtering by cluster
    selected_clusters = st.multiselect("Filter by Cluster", 
                                     options=clustered_data['Cluster'].unique().tolist(),
                                     default=clustered_data['Cluster'].unique().tolist(),
                                     key="cluster_filter")
    
    filtered_rep_data = clustered_data[clustered_data['Cluster'].isin(selected_clusters)]
    
    # Sort by Activity and Performance indices
    display_data = filtered_rep_data.sort_values(['Activity Index', 'Performance Index'], ascending=[False, False])
    
    # Select columns to display
    display_columns = ['Rep Name', 'Cluster', 'Activity Index', 'Performance Index', 
                      'Coverage', 'Frequency', 'Total Visits', 'Market Share', 'EI', 'Growth']
    
    st.dataframe(display_data[display_columns], use_container_width=True)
    create_download_button(display_data, "rep_clustering_data")
    
    # AI Insights
    data_summary = prepare_data_summary_for_ai(clustered_data, "clusters")
    display_ai_insights(data_summary, "clusters", api_key, model, temperature)

def main():
    """Main application logic."""
    # Setup OpenAI configuration in sidebar
    api_key, model, temperature = setup_openai_sidebar()
    
    cols = st.columns([1, 3])
    with cols[0]:
        st.markdown('<h2>SFE Assistant</h2>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div style="display: flex; align-items: center;"><h3>Upload Excel File</h3><p style="margin-left: 10px;">with Sales, Universe, and Visits sheets</p></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"], label_visibility="collapsed")

    if uploaded_file:
        is_valid, message, dataframes = validate_excel_file(uploaded_file)
        if is_valid:
            st.success(message)
            tabs = st.tabs(["Frequency Segmentation", "Sales Evolution Segmentation", "Coverage Segmentation", 
                           "Response Curve", "Correlation", "Carryover", "Clusters"])

            unique_specialties = sorted(dataframes['Visits']['Customer Speciality'].unique())
            unique_products = ['All'] + sorted([p for p in dataframes['Sales']['Product'].unique() if p != 'COMPETITOR'])
            unique_reps = sorted(dataframes['Visits']['Rep name'].unique())

            with tabs[0]:
                # Use the enhanced handler for Frequency Segmentation tab
                handle_tab_content_with_correlation(
                    "Frequency", dataframes['Visits'], dataframes['Universe'], dataframes['Sales'],
                    unique_specialties, unique_products, unique_reps, segment_bricks_by_frequency,
                    'Frequency', "freq", True, frequency_distribution_chart, "{:.2f} frequency",
                    api_key, model, temperature
                )
            with tabs[1]:
                # Use the enhanced handler for Sales Evolution Segmentation tab with EI correlation
                handle_tab_content_with_correlation(
                    "Evolution Index", dataframes['Visits'], dataframes['Universe'], dataframes['Sales'],
                    unique_specialties, unique_products, unique_reps, segment_bricks_by_ei,
                    'Ei', "ei", False, None, "{:.0f} EI",
                    api_key, model, temperature
                )
            with tabs[2]:
                # Use the enhanced handler for Coverage Segmentation tab with Coverage correlation
                handle_tab_content_with_correlation(
                    "Coverage", dataframes['Visits'], dataframes['Universe'], dataframes['Sales'],
                    unique_specialties, unique_products, unique_reps, segment_bricks_by_coverage,
                    'Coverage, %', "cov", False, None, "{:.1f}% coverage",
                    api_key, model, temperature
                )
            with tabs[3]:
                handle_response_curve_tab(dataframes['Visits'], dataframes['Sales'], unique_specialties, unique_products, unique_reps, api_key, model, temperature)
            with tabs[4]:
                handle_correlation_tab(dataframes['Visits'], dataframes['Universe'], dataframes['Sales'], unique_specialties, unique_products, unique_reps, api_key, model, temperature)
            with tabs[5]:
                st.header("Carryover Analysis")
                st.markdown("<h3>Filter Options</h3>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    selected_specialties = st.multiselect("Customer Speciality", unique_specialties, key="carryover_specialties")
                with col2:
                    selected_product = st.selectbox("Product", unique_products, key="carryover_product")
                    selected_product = None if selected_product == 'All' else selected_product
                with col3:
                    selected_reps = st.multiselect("Rep Name", unique_reps, key="carryover_reps")
                with col4:
                    metric_type = st.radio("Metric for Sales", ["Value", "Volume"], horizontal=True, key="carryover_metric")

                carryover_data, frequency_threshold, agg_data, frequency_min, frequency_max = process_carryover_data(
                    dataframes['Sales'], dataframes['Universe'], dataframes['Visits'],
                    selected_specialties, selected_product, metric_type, selected_reps
                )

                st.markdown("<h3>Segment Information</h3>", unsafe_allow_html=True)
                info_col1, info_col2 = st.columns([1, 2])

                with info_col1:
                    st.markdown("### Segment Definitions")
                    st.markdown("- **Filtered out**: Bricks with 0 visits")
                    range_text = f"{frequency_min:.2f}-{frequency_max:.2f}"
                    st.markdown(f"- **Non significant activity**: Frequency ≤ {frequency_threshold:.2f} (lowest 10% of range {range_text})")
                    st.markdown(f"- **Significant activity**: Frequency > {frequency_threshold:.2f}")
                with info_col2:
                    st.markdown("### Aggregated Data")
                    st.dataframe(agg_data, use_container_width=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3>Carryover Table</h3>", unsafe_allow_html=True)
                st.dataframe(carryover_data, use_container_width=True)
                create_download_button(carryover_data, "carryover_data")
                
                # Add Vacant Territories Analysis
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2>Vacant Territories Analysis</h2>", unsafe_allow_html=True)
                st.markdown("**Vacant territories** are those that had visits in the previous year but have **6+ consecutive months without visits** in the current year.")
                
                # Analyze vacant territories
                vacant_display, vacant_summary, vacant_message = analyze_vacant_territories(
                    dataframes['Visits'], dataframes['Sales'], dataframes['Universe'],
                    selected_specialties, selected_product, metric_type, selected_reps
                )
                
                st.info(vacant_message)
                
                if not vacant_display.empty:
                    # Display note about analysis methodology
                    st.markdown("**📝 Analysis Note:** Performance metrics show full-year data, while vacant periods indicate specific 6+ month gaps without visits.")
                    
                    # Display summary metrics
                    st.markdown("<h3>Vacant Territories Summary</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Vacant Territories", vacant_summary['Total_Vacant_Territories'])
                        st.metric("Vacant Percentage", f"{vacant_summary['Vacant_Percentage']}%")
                    with col2:
                        st.metric("Avg Market Share", f"{vacant_summary['Avg_Market_Share']}%")
                        st.metric("Avg Evolution Index", vacant_summary['Avg_EI'])
                    with col3:
                        st.metric("Avg Growth Rate", f"{vacant_summary['Avg_Growth_Rate']}%")
                        st.metric("Total Customers Uncovered", vacant_summary['Total_Customers_Uncovered'])
                    with col4:
                        current_year = vacant_summary['Current_Year']
                        st.metric(f"Total Sales {current_year}", f"{vacant_summary['Total_Sales_Lost']:,.0f}")
                        prev_year = vacant_summary['Previous_Year']
                        st.metric("Analysis Period", f"{prev_year} → {current_year}")
                    
                    # Performance insights with updated logic
                    st.markdown("<h3>Performance Insights</h3>", unsafe_allow_html=True)
                    
                    # Calculate insights
                    high_ms_vacant = len(vacant_display[vacant_display[f'Market Share {current_year} (%)'] > vacant_summary['Avg_Market_Share']])
                    positive_growth_vacant = len(vacant_display[vacant_display['Growth Rate (%)'] > 0])
                    high_ei_vacant = len(vacant_display[vacant_display['Evolution Index'] > 100])
                    
                    insight_col1, insight_col2 = st.columns(2)
                    with insight_col1:
                        st.markdown("**High-Impact Vacant Territories:**")
                        st.write(f"• {high_ms_vacant} territories with above-average market share")
                        st.write(f"• {positive_growth_vacant} territories with positive growth despite gaps")
                        st.write(f"• {high_ei_vacant} territories with EI > 100 (market share improving)")
                        
                        avg_customers_per_territory = vacant_summary['Total_Customers_Uncovered'] / vacant_summary['Total_Vacant_Territories']
                        st.write(f"• Average {avg_customers_per_territory:.0f} customers per vacant territory")
                        
                        # Priority calculation
                        priority_territories = len(vacant_display[
                            (vacant_display[f'Market Share {current_year} (%)'] > vacant_summary['Avg_Market_Share']) &
                            (vacant_display['Growth Rate (%)'] > 0)
                        ])
                        st.write(f"• {priority_territories} high-priority territories for immediate attention")
                    
                    with insight_col2:
                        st.markdown("**Coverage Patterns:**")
                        
                        if vacant_summary['Total_Sales_Lost'] > 0:
                            avg_sales_per_territory = vacant_summary['Total_Sales_Lost'] / vacant_summary['Total_Vacant_Territories']
                            st.write(f"• Average {avg_sales_per_territory:,.0f} sales per vacant territory")
                        
                        total_potential_impact = vacant_summary['Total_Sales_Lost'] + (vacant_summary['Total_Customers_Uncovered'] * 1000)  # Assuming 1000 per customer potential
                        st.write(f"• Estimated potential impact: {total_potential_impact:,.0f}")
                    
                    # Display detailed vacant territories table
                    st.markdown("<h3>Vacant Territories Details</h3>", unsafe_allow_html=True)
                    st.dataframe(vacant_display, use_container_width=True)
                    create_download_button(vacant_display, "vacant_territories")
                    
                    # Performance distribution charts
                    if len(vacant_display) > 5:  # Only show charts if we have enough data
                        st.markdown("<h3>Vacant Territories Performance Distribution</h3>", unsafe_allow_html=True)
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # Market Share distribution
                            fig_ms, ax_ms = plt.subplots(figsize=(5, 3))
                            ax_ms.hist(vacant_display[f'Market Share {current_year} (%)'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                            ax_ms.axvline(vacant_summary['Avg_Market_Share'], color='red', linestyle='--', label=f'Average: {vacant_summary["Avg_Market_Share"]}%')
                            ax_ms.set_xlabel('Market Share (%)', fontsize=9)
                            ax_ms.set_ylabel('Number of Territories', fontsize=9)
                            ax_ms.set_title('Market Share Distribution - Vacant Territories', fontsize=10, fontweight='bold')
                            ax_ms.legend(fontsize=8)
                            ax_ms.tick_params(axis='both', which='major', labelsize=8)
                            ax_ms.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_ms)
                        
                        with chart_col2:
                            # Growth Rate distribution
                            fig_gr, ax_gr = plt.subplots(figsize=(5, 3))
                            ax_gr.hist(vacant_display['Growth Rate (%)'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
                            ax_gr.axvline(vacant_summary['Avg_Growth_Rate'], color='red', linestyle='--', label=f'Average: {vacant_summary["Avg_Growth_Rate"]}%')
                            ax_gr.axvline(0, color='orange', linestyle='-', alpha=0.7, label='Zero Growth')
                            ax_gr.set_xlabel('Growth Rate (%)', fontsize=9)
                            ax_gr.set_ylabel('Number of Territories', fontsize=9)
                            ax_gr.set_title('Growth Rate Distribution - Vacant Territories', fontsize=10, fontweight='bold')
                            ax_gr.legend(fontsize=8)
                            ax_gr.tick_params(axis='both', which='major', labelsize=8)
                            ax_gr.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig_gr)
                
                else:
                    st.success("✅ No vacant territories found - all territories from the previous year maintained regular visit coverage (no 6+ month gaps).")
                
                # AI Insights for Carryover Analysis
                if not carryover_data.empty:
                    carryover_summary = prepare_data_summary_for_ai(carryover_data, "carryover")
                    if not vacant_display.empty:
                        carryover_summary += f"\n\nVacant Territories Summary:\n{vacant_summary}"
                    display_ai_insights(carryover_summary, "carryover", api_key, model, temperature)
            with tabs[6]:
                handle_clusters_tab(dataframes['Visits'], dataframes['Universe'], dataframes['Sales'], 
                                  unique_specialties, unique_products, api_key, model, temperature)
        else:
            st.error(message)

if __name__ == "__main__":
    main()