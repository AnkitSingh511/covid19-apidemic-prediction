# ============================================================
#   COVID-19 EPIDEMIC DASHBOARD — TRACK C
#   CodeCure AI Hackathon
#   Run: python -m streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="COVID-19 Epidemic Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# FIX 2: GLOBAL CSS — visible metrics in dark mode
# ======================================
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    div[data-testid="metric-container"] label {
        color: #bbbbbb !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #aaaaaa !important;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #e74c3c;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================
# FILE PATHS
# ======================================
BASE = r"D:\JAVA\projects\project 4(IITB)"

# ======================================
# LOAD DATA (cached)
# ======================================
@st.cache_data
def load_owid():
    try:
        df = pd.read_excel(fr"{BASE}\final_owid_output.csv", engine='openpyxl')
    except Exception:
        df = pd.read_csv(fr"{BASE}\final_owid_output.csv",
                         encoding='latin1', on_bad_lines='skip')
    df.columns        = df.columns.str.strip().str.lower()
    df['date']        = pd.to_datetime(df['date'], errors='coerce')
    df['cases']       = pd.to_numeric(df['cases'],       errors='coerce').fillna(0)
    df['deaths']      = pd.to_numeric(df['deaths'],      errors='coerce').fillna(0)
    df['population']  = pd.to_numeric(df['population'],  errors='coerce').fillna(1)
    df['daily_cases'] = pd.to_numeric(df['daily_cases'], errors='coerce').fillna(0)
    df['country']     = df['country'].fillna("Unknown").astype(str)
    df = df.dropna(subset=['date'])
    df = df.sort_values(['country', 'date']).reset_index(drop=True)
    df['day']           = (df['date'] - df['date'].min()).dt.days
    df['cases_filled']  = df.groupby('country')['cases'].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0))
    df['daily_cases']   = df.groupby('country')['cases_filled'].diff().fillna(0).clip(lower=0)
    df['moving_avg_14'] = (df.groupby('country')['cases']
                             .rolling(14, min_periods=1).mean()
                             .reset_index(0, drop=True))
    df['growth_rate']   = df['daily_cases'] / (df['cases'] + 1)
    df['cases_per_100k']= (df['cases'] / df['population'].clip(lower=1)) * 100000
    df['death_rate']    = (df['deaths'] / (df['cases'] + 1)) * 100
    df = df.drop(columns=['cases_filled'], errors='ignore').fillna(0)
    EXCLUDE = ['European Union', 'High-income', 'Upper-middle', 'Low-income',
               'Lower-middle', 'World', 'income', 'International']
    df = df[~df['country'].str.contains('|'.join(EXCLUDE), case=False, na=False)]
    return df

@st.cache_data
def load_risk():
    try:
        return pd.read_csv(fr"{BASE}\global_risk_map_data.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv(fr"{BASE}\final_predictions.csv")
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except Exception:
        return pd.DataFrame()

COORDS = {
    'Afghanistan':(33.93,67.71),'Albania':(41.15,20.17),'Algeria':(28.03,1.66),
    'Argentina':(-38.42,-63.62),'Australia':(-25.27,133.78),'Austria':(47.52,14.55),
    'Bangladesh':(23.68,90.36),'Belgium':(50.50,4.47),'Bolivia':(-16.29,-63.59),
    'Brazil':(-14.24,-51.93),'Canada':(56.13,-106.35),'Chile':(-35.68,-71.54),
    'China':(35.86,104.20),'Colombia':(4.57,-74.30),'Croatia':(45.10,15.20),
    'Czech Republic':(49.82,15.47),'Denmark':(56.26,9.50),'Ecuador':(-1.83,-78.18),
    'Egypt':(26.82,30.80),'Ethiopia':(9.14,40.49),'Finland':(61.92,25.75),
    'France':(46.23,2.21),'Germany':(51.17,10.45),'Ghana':(7.95,-1.02),
    'Greece':(39.07,21.82),'Hungary':(47.16,19.50),'India':(20.59,78.96),
    'Indonesia':(-0.79,113.92),'Iran':(32.43,53.69),'Iraq':(33.22,43.68),
    'Ireland':(53.41,-8.24),'Israel':(31.05,34.85),'Italy':(41.87,12.57),
    'Japan':(36.20,138.25),'Jordan':(30.59,36.24),'Kazakhstan':(48.02,66.92),
    'Kenya':(-0.02,37.91),'Kuwait':(29.31,47.48),'Malaysia':(4.21,101.98),
    'Mexico':(23.63,-102.55),'Morocco':(31.79,-7.09),'Nepal':(28.39,84.12),
    'Netherlands':(52.13,5.29),'Nigeria':(9.08,8.68),'Norway':(60.47,8.47),
    'Pakistan':(30.38,69.35),'Peru':(-9.19,-75.02),'Philippines':(12.88,121.77),
    'Poland':(51.92,19.15),'Portugal':(39.40,-8.22),'Romania':(45.94,24.97),
    'Russia':(61.52,105.32),'Saudi Arabia':(23.89,45.08),
    'South Africa':(-30.56,22.94),'South Korea':(35.91,127.77),
    'Spain':(40.46,-3.75),'Sweden':(60.13,18.64),'Switzerland':(46.82,8.23),
    'Thailand':(15.87,100.99),'Tunisia':(33.89,9.54),'Turkey':(38.96,35.24),
    'Ukraine':(48.38,31.17),'United Arab Emirates':(23.42,53.85),
    'United Kingdom':(55.38,-3.44),'United States':(37.09,-95.71),
    'Uruguay':(-32.52,-55.77),'Venezuela':(6.42,-66.59),
    'Vietnam':(14.06,108.28),'Zimbabwe':(-19.02,29.15),
}

# ======================================
# LOAD DATA
# ======================================
with st.spinner("Loading data..."):
    df      = load_owid()
    risk_df = load_risk()
    pred_df = load_predictions()

countries = sorted(df['country'].unique().tolist())

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    # FIX 1: text logo instead of broken image
    st.markdown("""
    <div style='text-align:center;padding:10px 0'>
        <span style='font-size:42px'>🦠</span><br>
        <span style='font-size:18px;font-weight:700;color:#e74c3c'>COVID-19</span><br>
        <span style='font-size:12px;color:#aaa'>Epidemic Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**CodeCure AI Hackathon — Track C**")
    st.markdown("---")

    st.subheader("Filters")
    selected_countries = st.multiselect(
        "Select countries",
        options=countries,
        default=['United States', 'India', 'Brazil', 'France', 'Germany']
    )
    if not selected_countries:
        selected_countries = ['United States', 'India', 'Brazil']

    date_min   = df['date'].min().date()
    date_max   = df['date'].max().date()
    date_range = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date   = pd.Timestamp(date_range[1])
    else:
        start_date = pd.Timestamp(date_min)
        end_date   = pd.Timestamp(date_max)

    st.markdown("---")
    st.subheader("Forecast Settings")
    forecast_country = st.selectbox(
        "Country for forecast",
        options=countries,
        index=countries.index('United States') if 'United States' in countries else 0
    )
    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)

    st.markdown("---")
    st.subheader("Map Settings")
    show_heatmap = st.checkbox("Show heatmap layer", value=False)
    show_high    = st.checkbox("High risk markers",  value=True)
    show_medium  = st.checkbox("Medium risk markers",value=True)
    show_low     = st.checkbox("Low risk markers",   value=True)

    st.markdown("---")

    # FIX 3: Action buttons in sidebar
    st.subheader("Actions")
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("📊  Run Full Analysis", use_container_width=True):
        try:
            subprocess.Popen(
                ["python", fr"{BASE}\prediction.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            st.success("Analysis started in a new window!")
        except Exception as e:
            st.error(f"Could not start: {e}")

    if st.button("🗺️  Open Risk Map", use_container_width=True):
        import webbrowser
        webbrowser.open(fr"{BASE}\ml_risk_map.html")
        st.success("Map opened in browser!")

    st.markdown("---")
    st.caption("Data: OWID · JHU · WHO · 2020–2024")

# ======================================
# FILTER DATA
# ======================================
df_filtered = df[
    (df['country'].isin(selected_countries)) &
    (df['date'] >= start_date) &
    (df['date'] <= end_date)
]

# ======================================
# HEADER
# ======================================
st.title("🦠 COVID-19 Epidemic Spread Prediction")
st.markdown("**Track C — Epidemic Spread Prediction | CodeCure AI Hackathon**")
st.markdown("---")

# ======================================
# TABS
# ======================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "📈  Trend Analysis",
    "🔮  Forecast",
    "🗺️  Risk Map",
    "📋  Data Explorer"
])

# ======================================
# TAB 1: OVERVIEW
# ======================================
with tab1:
    st.subheader("Global Summary")
    latest = df.sort_values('date').groupby('country').last().reset_index()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Global Cases",  f"{int(latest['cases'].sum()):,}")
    with col2:
        st.metric("Total Global Deaths", f"{int(latest['deaths'].sum()):,}")
    with col3:
        st.metric("Avg Death Rate",      f"{latest['death_rate'].mean():.2f}%")
    with col4:
        st.metric("Countries Tracked",   f"{latest['country'].nunique()}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 10 Countries by Total Cases")
        top10 = latest.nlargest(10, 'cases')[['country','cases','deaths','death_rate']].copy()
        top10.columns = ['Country','Total Cases','Deaths','Death Rate (%)']
        top10['Total Cases']    = top10['Total Cases'].apply(lambda x: f"{int(x):,}")
        top10['Deaths']         = top10['Deaths'].apply(lambda x: f"{int(x):,}")
        top10['Death Rate (%)'] = top10['Death Rate (%)'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("Cases Distribution")
        top15    = latest.nlargest(15, 'cases')
        fig_pie  = px.pie(top15, values='cases', names='country', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=320,
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Global Cases Over Time — Selected Countries")
    if not df_filtered.empty:
        fig_ov = px.line(df_filtered, x='date', y='cases', color='country',
                         labels={'cases':'Total Cases','date':'Date'},
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_ov.update_layout(hovermode='x unified', height=380,
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)',
                             xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                             yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig_ov, use_container_width=True)
    else:
        st.warning("No data for selected filters.")

# ======================================
# TAB 2: TREND ANALYSIS
# ======================================
with tab2:
    st.subheader("Trend Analysis — Selected Countries")

    def make_chart(data, x, y, kind='line', label=None, title=None):
        kwargs = dict(x=x, y=y, color='country',
                      labels={y: label or y},
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig = px.line(data, **kwargs) if kind == 'line' else px.bar(data, barmode='group', **kwargs)
        fig.update_layout(height=320, hovermode='x unified',
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                          yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                          title=title)
        return fig

    if df_filtered.empty:
        st.warning("No data for selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Daily New Cases**")
            st.plotly_chart(make_chart(df_filtered,'date','daily_cases','bar',
                            'Daily Cases'), use_container_width=True)
        with c2:
            st.markdown("**14-Day Moving Average**")
            st.plotly_chart(make_chart(df_filtered,'date','moving_avg_14','line',
                            '14-Day Avg'), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Growth Rate**")
            st.plotly_chart(make_chart(df_filtered,'date','growth_rate','line',
                            'Growth Rate'), use_container_width=True)
        with c4:
            st.markdown("**Cases per 100k Population**")
            st.plotly_chart(make_chart(df_filtered,'date','cases_per_100k','line',
                            'Cases per 100k'), use_container_width=True)

        st.markdown("---")
        st.subheader("Death Rate Comparison")
        death_data = df_filtered.sort_values('date').groupby('country').last().reset_index()
        fig_dr = px.bar(death_data.sort_values('death_rate', ascending=False),
                        x='country', y='death_rate', color='country',
                        labels={'death_rate':'Death Rate (%)'},
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig_dr.update_layout(height=320, showlegend=False,
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)',
                             yaxis=dict(gridcolor='rgba(255,255,255,0.08)'))
        st.plotly_chart(fig_dr, use_container_width=True)

# ======================================
# TAB 3: FORECAST
# ======================================
with tab3:
    st.subheader(f"Prophet Forecast — {forecast_country}")
    st.markdown(f"Predicting the next **{forecast_days} days** of COVID-19 cases.")

    country_data = df[df['country'] == forecast_country].copy()

    if len(country_data) < 10:
        st.warning(f"Not enough data for {forecast_country}.")
    else:
        model_df = (country_data[['date','cases']]
                    .rename(columns={'date':'ds','cases':'y'})
                    .drop_duplicates(subset='ds').sort_values('ds'))
        model_df = model_df[model_df['y'] >= 0]

        with st.spinner(f"Training Prophet model for {forecast_country}..."):
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                            daily_seasonality=False, interval_width=0.95)
                m.fit(model_df)
                future   = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)
                forecast['yhat']       = forecast['yhat'].clip(lower=0)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

                merged_eval = model_df.merge(forecast[['ds','yhat']], on='ds', how='inner')
                mae = mean_absolute_error(merged_eval['y'], merged_eval['yhat'])

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Model MAE",         f"{mae:,.0f} cases")
                with c2: st.metric("Last Known Cases",  f"{int(model_df['y'].iloc[-1]):,}")
                with c3: st.metric(f"Predicted in {forecast_days}d",
                                   f"{int(forecast['yhat'].iloc[-1]):,}")

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=model_df['ds'], y=model_df['y'],
                    mode='lines', name='Actual',
                    line=dict(color='#3498db', width=2)))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines', name='Forecast',
                    line=dict(color='#f39c12', width=2, dash='dash')))
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(243,156,18,0.15)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% Confidence'))
                fig_fc.update_layout(
                    title=f"Prophet Forecast — {forecast_country}",
                    xaxis_title="Date", yaxis_title="Total Cases",
                    hovermode='x unified', height=420,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                    legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_fc, use_container_width=True)

                st.subheader("Predicted Values")
                future_only = forecast[forecast['ds'] > model_df['ds'].max()][
                    ['ds','yhat','yhat_lower','yhat_upper']].head(forecast_days).copy()
                future_only.columns = ['Date','Predicted','Lower','Upper']
                future_only['Date']      = future_only['Date'].dt.strftime('%Y-%m-%d')
                future_only['Predicted'] = future_only['Predicted'].apply(lambda x: f"{int(x):,}")
                future_only['Lower']     = future_only['Lower'].apply(lambda x: f"{int(x):,}")
                future_only['Upper']     = future_only['Upper'].apply(lambda x: f"{int(x):,}")
                st.dataframe(future_only, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Forecast failed: {e}")

    st.markdown("---")
    st.subheader("Hotspot Detection — Growth Rate Classification")
    if not df_filtered.empty:
        latest_f = df_filtered.sort_values('date').groupby('country').last().reset_index()
        latest_f['risk'] = latest_f['growth_rate'].apply(
            lambda g: "High" if g > 0.05 else ("Medium" if g > 0.02 else "Low"))

        c1, c2 = st.columns(2)
        with c1:
            rc = latest_f['risk'].value_counts().reset_index()
            rc.columns = ['Risk Level','Count']
            fig_rp = px.pie(rc, values='Count', names='Risk Level',
                            color='Risk Level',
                            color_discrete_map={'High':'#e74c3c',
                                                'Medium':'#f39c12','Low':'#27ae60'})
            fig_rp.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rp, use_container_width=True)
        with c2:
            st.dataframe(
                latest_f[['country','cases','growth_rate','risk']]
                .sort_values('growth_rate', ascending=False)
                .rename(columns={'country':'Country','cases':'Cases',
                                 'growth_rate':'Growth Rate','risk':'Risk'}),
                use_container_width=True, hide_index=True)

# ======================================
# TAB 4: RISK MAP
# ======================================
with tab4:
    st.subheader("Interactive Risk Map")

    # FIX 4: prevent brightness change when map loads
    st.markdown("""
    <style>
        iframe[title="streamlit_folium.st_folium"] {
            filter: none !important;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stApp { background-color: #0e1117 !important; }
    </style>
    """, unsafe_allow_html=True)

    c_map, c_info = st.columns([3, 1])

    with c_info:
        st.markdown("**Risk Summary**")
        if not risk_df.empty and 'risk' in risk_df.columns:
            rc = risk_df['risk'].value_counts()
            icons = {'High':'🔴','Medium':'🟠','Low':'🟢'}
            for r, c in rc.items():
                st.markdown(f"{icons.get(r,'⚪')} **{r}**: {c} countries")
        st.markdown("---")
        st.markdown("**Legend**")
        st.markdown("🔴 High risk")
        st.markdown("🟠 Medium risk")
        st.markdown("🟢 Low risk")
        st.markdown("---")
        st.caption("Circle size = predicted cases")

    with c_map:
        m = folium.Map(location=[20, 0], zoom_start=2,
                       tiles='CartoDB positron',
                       prefer_canvas=True)

        CMAP = {'High':'#e74c3c','Medium':'#f39c12','Low':'#27ae60'}

        if not risk_df.empty and 'predicted_cases' in risk_df.columns:
            plot_df = risk_df.copy()
            if 'lat' not in plot_df.columns:
                plot_df['lat'] = plot_df['country'].map(
                    lambda c: COORDS.get(c,(np.nan,np.nan))[0])
                plot_df['lon'] = plot_df['country'].map(
                    lambda c: COORDS.get(c,(np.nan,np.nan))[1])
            plot_df   = plot_df.dropna(subset=['lat','lon'])
            max_cases = max(plot_df['predicted_cases'].max(), 1)
            heat_data = []

            for _, row in plot_df.iterrows():
                try:
                    risk  = row.get('risk','Low')
                    if risk == 'High'   and not show_high:   continue
                    if risk == 'Medium' and not show_medium: continue
                    if risk == 'Low'    and not show_low:    continue

                    color = CMAP.get(risk,'gray')
                    vax   = f"{row['vax_rate']:.1f}%" if 'vax_rate' in row and pd.notna(row.get('vax_rate')) else "N/A"
                    pos   = f"{row['positivity_rate']:.2f}" if 'positivity_rate' in row and pd.notna(row.get('positivity_rate')) else "N/A"

                    popup_html = f"""
                    <div style='font-family:Arial;font-size:12px;min-width:180px'>
                        <b style='font-size:14px'>{row['country']}</b><br>
                        <hr style='margin:3px 0'>
                        <b>Current:</b> {int(row['current_cases']):,}<br>
                        <b>Predicted (14d):</b> {int(row['predicted_cases']):,}<br>
                        <b>Vaccinated:</b> {vax}<br>
                        <b>Positivity:</b> {pos}<br>
                        <b>Risk: <span style='color:{color}'>{risk}</span></b>
                    </div>"""

                    folium.CircleMarker(
                        location=[float(row['lat']), float(row['lon'])],
                        radius=max(row['predicted_cases'] / max_cases * 30, 4),
                        color=color, fill=True, fill_color=color,
                        fill_opacity=0.65, weight=1,
                        popup=folium.Popup(popup_html, max_width=220),
                        tooltip=f"{row['country']} — {risk} Risk"
                    ).add_to(m)

                    heat_data.append([float(row['lat']), float(row['lon']),
                                      float(row['predicted_cases'])])
                except Exception:
                    continue

            if show_heatmap and heat_data:
                HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(m)

        m.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;bottom:30px;left:30px;background:white;
            border:1px solid #ccc;border-radius:8px;padding:10px 14px;
            font-family:Arial;font-size:12px;z-index:9999;
            box-shadow:2px 2px 6px rgba(0,0,0,0.2)">
            <b>14-Day Risk</b><br>
            <span style='color:#e74c3c'>&#9679;</span> High<br>
            <span style='color:#f39c12'>&#9679;</span> Medium<br>
            <span style='color:#27ae60'>&#9679;</span> Low
        </div>"""))

        st_folium(m, width=None, height=500)

    st.markdown("---")
    st.subheader("Risk Choropleth Map")
    if not risk_df.empty and 'risk' in risk_df.columns:
        fig_ch = px.choropleth(
            risk_df, locations="country", locationmode="country names",
            color="risk",
            color_discrete_map={'High':'red','Medium':'orange','Low':'green'},
            hover_data=[c for c in ['current_cases','predicted_cases',
                                     'vax_rate','death_rate']
                        if c in risk_df.columns],
            title="Global COVID-19 14-Day Risk Forecast")
        fig_ch.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                             geo=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_ch, use_container_width=True)

# ======================================
# TAB 5: DATA EXPLORER
# ======================================
with tab5:
    st.subheader("Data Explorer")

    explore_country = st.selectbox(
        "Select country",
        options=countries,
        index=countries.index('United States') if 'United States' in countries else 0,
        key="explorer"
    )
    country_exp = df[df['country'] == explore_country].copy()

    if not country_exp.empty:
        lr = country_exp.sort_values('date').iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Cases",    f"{int(lr['cases']):,}")
        with c2: st.metric("Total Deaths",   f"{int(lr['deaths']):,}")
        with c3: st.metric("Death Rate",     f"{lr['death_rate']:.2f}%")
        with c4: st.metric("Cases per 100k", f"{lr['cases_per_100k']:.1f}")

        st.markdown("---")
        cl, cr = st.columns(2)
        with cl:
            metric_choice = st.selectbox(
                "Plot metric",
                ['cases','daily_cases','moving_avg_14',
                 'growth_rate','cases_per_100k','deaths'])
            fig_ex = px.line(country_exp, x='date', y=metric_choice,
                             title=f"{metric_choice.replace('_',' ').title()} — {explore_country}",
                             color_discrete_sequence=['#e74c3c'])
            fig_ex.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                                 yaxis=dict(gridcolor='rgba(255,255,255,0.08)'))
            st.plotly_chart(fig_ex, use_container_width=True)

        with cr:
            st.markdown(f"**Raw data — {explore_country}**")
            show_cols = [c for c in ['date','cases','daily_cases','deaths',
                                     'moving_avg_14','growth_rate','cases_per_100k']
                         if c in country_exp.columns]
            st.dataframe(
                country_exp[show_cols].sort_values('date', ascending=False)
                .head(50).reset_index(drop=True),
                use_container_width=True, height=350)

        st.markdown("---")
        csv = country_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {explore_country} data as CSV",
            data=csv,
            file_name=f"{explore_country.replace(' ','_')}_covid_data.csv",
            mime='text/csv',
            use_container_width=True
        )

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:13px;padding:8px'>
    COVID-19 Epidemic Spread Prediction Dashboard &nbsp;|&nbsp;
    CodeCure AI Hackathon — Track C &nbsp;|&nbsp;
    Data: OWID · JHU · WHO
</div>
""", unsafe_allow_html=True)
