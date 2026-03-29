# ============================================================
#   COVID-19 EPIDEMIC SPREAD PREDICTION — TRACK C
#   CodeCure AI Hackathon
#   Files: OWID + JHU + Vaccination + Testing + Location
# ============================================================

# ======================================
# STEP 1: IMPORTS
# ======================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

pio.renderers.default = 'browser'

# ======================================
# STEP 2: FILE PATHS
# NOTE: All files on disk are CSVs.
#       final_owid_output.csv is actually an Excel file — handled below.
#       cleaned_covid_data.csv is a true CSV.
#       The 3 files with spaces in names are Excel files — handled below.
# ======================================
BASE      = r"D:\JAVA\projects\project 4(IITB)"
PATH_OWID = fr"{BASE}\final_owid_output.csv"
PATH_JHU  = fr"{BASE}\cleaned_covid_data.csv"
PATH_VAC  = fr"{BASE}\final vactination data.csv"
PATH_TEST = fr"{BASE}\final testing data.csv"
PATH_LOC  = fr"{BASE}\final location data.csv"

print("=" * 60)
print("  COVID-19 EPIDEMIC SPREAD PREDICTION — TRACK C")
print("=" * 60)

# ======================================
# STEP 3: LOAD OWID DATA
# (file is Excel despite .csv extension)
# ======================================
print("\n[1/5] Loading OWID dataset...")
try:
    df = pd.read_excel(PATH_OWID, engine='openpyxl')
except Exception:
    df = pd.read_csv(PATH_OWID, encoding='latin1', on_bad_lines='skip')

df.columns        = df.columns.str.strip().str.lower()
df                = df.drop_duplicates()
df['date']        = pd.to_datetime(df['date'], errors='coerce')
df['cases']       = pd.to_numeric(df['cases'],       errors='coerce').fillna(0)
df['deaths']      = pd.to_numeric(df['deaths'],      errors='coerce').fillna(0) if 'deaths'     in df.columns else 0
df['population']  = pd.to_numeric(df['population'],  errors='coerce').fillna(1) if 'population' in df.columns else 1
df['daily_cases'] = pd.to_numeric(df['daily_cases'], errors='coerce').fillna(0) if 'daily_cases' in df.columns else 0
df = df.sort_values(['country', 'date']).reset_index(drop=True)
df['cases_filled'] = df.groupby('country')['cases'].transform(
    lambda x: x.replace(0, np.nan).ffill().fillna(0))
df['daily_cases'] = df.groupby('country')['cases_filled'].diff().fillna(0).clip(lower=0)
df = df.drop(columns=['cases_filled'])
df['country']     = df['country'].fillna("Unknown").astype(str)
df                = df.dropna(subset=['date'])
df                = df.sort_values(['country', 'date']).reset_index(drop=True)
df['day']         = (df['date'] - df['date'].min()).dt.days
print(f"    Rows: {len(df):,}  |  Countries: {df['country'].nunique()}")

# ======================================
# STEP 4: LOAD JHU DATA (lat/lon coords)
# (true CSV on disk)
# ======================================
print("[2/5] Loading JHU dataset (coordinates)...")
JHU_LOADED = False
jhu_coords = pd.DataFrame(columns=['country', 'lat', 'lon'])
try:
    jhu = pd.read_csv(PATH_JHU, encoding='latin1', on_bad_lines='skip')
    jhu.columns = jhu.columns.str.strip().str.lower()
    if 'long' in jhu.columns:
        jhu.rename(columns={'long': 'lon'}, inplace=True)
    jhu['date'] = pd.to_datetime(jhu['date'], errors='coerce')
    for c in ['cases', 'daily_cases', 'growth_rate', 'moving_avg_14', 'lat', 'lon']:
        if c in jhu.columns:
            jhu[c] = pd.to_numeric(jhu[c], errors='coerce')
    if 'lat' in jhu.columns and 'lon' in jhu.columns and 'country' in jhu.columns:
        jhu_coords = (jhu.dropna(subset=['lat', 'lon'])
                        .sort_values('date')
                        .groupby('country')[['lat', 'lon']]
                        .last().reset_index())
        JHU_LOADED = True
        print(f"    Rows: {len(jhu):,}  |  Countries with coords: {len(jhu_coords)}")
    else:
        print("    JHU loaded but missing lat/lon columns — using fallback coords")
except Exception as e:
    print(f"    JHU skipped: {e}")

# ======================================
# STEP 5: LOAD VACCINATION DATA
# (Excel file with packed CSV in one column)
# ======================================
print("[3/5] Loading vaccination data...")
vac_latest = pd.DataFrame(columns=['country'])
try:
    vac_raw = pd.read_excel(PATH_VAC, engine='openpyxl')
    # Data is packed: each row = one CSV string in 'country' column
    vac_col = vac_raw['country'].apply(lambda x: str(x).rsplit(',', 1)[0])
    vac_cols = [
        'country', 'date', 'total_vaccinations', 'people_vaccinated',
        'people_fully_vaccinated', 'total_boosters', 'daily_vaccinations_raw',
        'daily_vaccinations', 'daily_people_vaccinated',
        'daily_vaccinations_per_million', 'total_vaccinations_per_hundred',
        'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred', 'daily_vaccinations_per_hundred',
        'daily_people_vaccinated_per_hundred', 'share_doses_used',
        'new_people_vaccinated_smoothed'
    ]
    vac_df         = vac_col.str.split(',', expand=True)
    vac_df.columns = vac_cols[:vac_df.shape[1]]
    vac_df['date'] = pd.to_datetime(vac_df['date'], errors='coerce')
    for c in ['people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
                'total_boosters_per_hundred', 'daily_vaccinations',
                'total_vaccinations']:
        if c in vac_df.columns:
            vac_df[c] = pd.to_numeric(vac_df[c], errors='coerce')
    keep_v   = [c for c in ['people_vaccinated_per_hundred',
                            'people_fully_vaccinated_per_hundred',
                            'total_boosters_per_hundred',
                            'daily_vaccinations'] if c in vac_df.columns]
    vac_latest = (vac_df.sort_values('date')
                        .groupby('country')[keep_v]
                        .last().reset_index())
    print(f"    Countries: {vac_latest['country'].nunique()}")
except Exception as e:
    print(f"    Vaccination skipped: {e}")

# ======================================
# STEP 6: LOAD TESTING DATA
# (Excel file with all columns packed into header)
# ======================================
print("[4/5] Loading testing data...")
test_latest = pd.DataFrame(columns=['country'])
try:
    test_raw  = pd.read_excel(PATH_TEST, engine='openpyxl')
    col0      = test_raw.columns[0]
    test_cols = [c.strip() for c in col0.split(',')]
    test_df   = test_raw[col0].str.split(',', expand=True)
    test_df   = test_df.iloc[:, :len(test_cols)]
    test_df.columns = test_cols[:test_df.shape[1]]
    # Extract country from entity (format: "Country - tests performed")
    if 'entity' in test_df.columns:
        test_df['country'] = (test_df['entity']
                            .str.replace(r'\s*-\s*tests.*', '', regex=True)
                            .str.strip())
    test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')
    # Rename relevant columns
    col_map = {}
    for c in test_df.columns:
        cl = c.lower().strip()
        if 'short-term positive rate' in cl:         col_map[c] = 'positivity_rate'
        elif '7-day smoothed daily change' == cl:    col_map[c] = 'tests_7day_avg'
        elif 'cumulative total per thousand' == cl:  col_map[c] = 'tests_per_thousand'
    test_df.rename(columns=col_map, inplace=True)
    for c in ['positivity_rate', 'tests_7day_avg', 'tests_per_thousand']:
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors='coerce')

    # Cap positivity rate — values above 100 are raw counts not percentages
    if 'positivity_rate' in test_df.columns:
        test_df['positivity_rate'] = test_df['positivity_rate'].where(
            test_df['positivity_rate'] <= 100, np.nan)

    keep_t = [c for c in ['positivity_rate', 'tests_7day_avg', 'tests_per_thousand']
                if c in test_df.columns]
    test_latest = (test_df.sort_values('date')
                            .groupby('country')[keep_t]
                            .last().reset_index())
    print(f"    Countries: {test_latest['country'].nunique()}")
except Exception as e:
    print(f"    Testing skipped: {e}")

# ======================================
# STEP 7: LOAD LOCATION DATA
# (Excel file with all columns packed into header)
# ======================================
print("[5/5] Loading location data...")
loc_df = pd.DataFrame(columns=['country', 'iso_code'])
try:
    loc_raw = pd.read_excel(PATH_LOC, engine='openpyxl')
    col0l   = loc_raw.columns[0]
    # Extract only country and iso_code (first 2 fields)
    # vaccines column has commas inside so use regex to grab first 2 fields safely
    extracted = loc_raw[col0l].str.extract(r'^([^,]+),([^,]+)')
    extracted.columns = ['country', 'iso_code']
    loc_df = extracted.dropna(subset=['country'])
    loc_df['country']  = loc_df['country'].str.strip()
    loc_df['iso_code'] = loc_df['iso_code'].str.strip()
    print(f"    Countries: {len(loc_df)}")
except Exception as e:
    print(f"    Location skipped: {e}")

# ======================================
# STEP 8: FEATURE ENGINEERING
# ======================================
print("\n Engineering features...")
df['moving_avg_14']  = (df.groupby('country')['cases']
                        .rolling(14, min_periods=1).mean()
                        .reset_index(0, drop=True))
df['moving_avg_7']   = (df.groupby('country')['cases']
                        .rolling(7,  min_periods=1).mean()
                        .reset_index(0, drop=True))
df['growth_rate']    = df['daily_cases'] / (df['cases'] + 1)
df['cases_per_100k'] = (df['cases'] / df['population'].clip(lower=1)) * 100000
df['death_rate']     = (df['deaths'] / (df['cases'] + 1)) * 100
df                   = df.fillna(0)

# ======================================
# STEP 9: BUILD MERGED LATEST SNAPSHOT
# ======================================
print(" Merging all datasets...")
latest_df = df.sort_values('date').groupby('country').last().reset_index()
latest_df = latest_df.merge(vac_latest,  on='country', how='left')
latest_df = latest_df.merge(test_latest, on='country', how='left')
latest_df = latest_df.merge(loc_df,      on='country', how='left')
if JHU_LOADED:
    latest_df = latest_df.merge(jhu_coords[['country','lat','lon']],
                                on='country', how='left')
else:
    latest_df['lat'] = np.nan
    latest_df['lon'] = np.nan

# Fallback coordinates
COORDS = {
    'Afghanistan':(33.93,67.71),'Albania':(41.15,20.17),'Algeria':(28.03,1.66),
    'Argentina':(-38.42,-63.62),'Australia':(-25.27,133.78),'Austria':(47.52,14.55),
    'Bangladesh':(23.68,90.36),'Belgium':(50.50,4.47),'Bolivia':(-16.29,-63.59),
    'Brazil':(-14.24,-51.93),'Cambodia':(12.57,104.99),'Canada':(56.13,-106.35),
    'Chile':(-35.68,-71.54),'China':(35.86,104.20),'Colombia':(4.57,-74.30),
    'Croatia':(45.10,15.20),'Czech Republic':(49.82,15.47),'Denmark':(56.26,9.50),
    'Ecuador':(-1.83,-78.18),'Egypt':(26.82,30.80),'Ethiopia':(9.14,40.49),
    'Finland':(61.92,25.75),'France':(46.23,2.21),'Germany':(51.17,10.45),
    'Ghana':(7.95,-1.02),'Greece':(39.07,21.82),'Hungary':(47.16,19.50),
    'India':(20.59,78.96),'Indonesia':(-0.79,113.92),'Iran':(32.43,53.69),
    'Iraq':(33.22,43.68),'Ireland':(53.41,-8.24),'Israel':(31.05,34.85),
    'Italy':(41.87,12.57),'Japan':(36.20,138.25),'Jordan':(30.59,36.24),
    'Kazakhstan':(48.02,66.92),'Kenya':(-0.02,37.91),'Kuwait':(29.31,47.48),
    'Malaysia':(4.21,101.98),'Mexico':(23.63,-102.55),'Morocco':(31.79,-7.09),
    'Nepal':(28.39,84.12),'Netherlands':(52.13,5.29),'Nigeria':(9.08,8.68),
    'Norway':(60.47,8.47),'Pakistan':(30.38,69.35),'Peru':(-9.19,-75.02),
    'Philippines':(12.88,121.77),'Poland':(51.92,19.15),'Portugal':(39.40,-8.22),
    'Romania':(45.94,24.97),'Russia':(61.52,105.32),'Saudi Arabia':(23.89,45.08),
    'South Africa':(-30.56,22.94),'South Korea':(35.91,127.77),
    'Spain':(40.46,-3.75),'Sweden':(60.13,18.64),'Switzerland':(46.82,8.23),
    'Thailand':(15.87,100.99),'Tunisia':(33.89,9.54),'Turkey':(38.96,35.24),
    'Ukraine':(48.38,31.17),'United Arab Emirates':(23.42,53.85),
    'United Kingdom':(55.38,-3.44),'United States':(37.09,-95.71),
    'Uruguay':(-32.52,-55.77),'Venezuela':(6.42,-66.59),
    'Vietnam':(14.06,108.28),'Zimbabwe':(-19.02,29.15),
}
mask = latest_df['lat'].isna()
latest_df.loc[mask, 'lat'] = latest_df.loc[mask, 'country'].map(
    lambda c: COORDS.get(c, (np.nan, np.nan))[0])
latest_df.loc[mask, 'lon'] = latest_df.loc[mask, 'country'].map(
    lambda c: COORDS.get(c, (np.nan, np.nan))[1])

print(f" Merged: {latest_df.shape[0]} countries x {latest_df.shape[1]} columns")

# ======================================
# STEP 10: PER-COUNTRY LINEAR REGRESSION
#           14-day forecast + risk
# ======================================
print("\n Training per-country forecast models...")

def safe_val(row_extra, col):
    try:
        if len(row_extra) and col in row_extra.columns:
            v = row_extra[col].values[0]
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        pass
    return np.nan

results = []
for country, group in df.groupby('country'):
    group = group.sort_values('day')
    if len(group) < 5:
        continue
    row_extra = latest_df[latest_df['country'] == country]
    lat = safe_val(row_extra, 'lat')
    lon = safe_val(row_extra, 'lon')
    if pd.isna(lat) or pd.isna(lon):
        continue
    try:
        lr = LinearRegression()
        lr.fit(group[['day']].values, group['cases'].values)
        predicted = max(float(lr.predict([[group['day'].max() + 14]])[0]), 0)
    except Exception:
        predicted = float(group['cases'].iloc[-1])

    results.append({
        'country':         country,
        'iso_code':        safe_val(row_extra, 'iso_code') if 'iso_code' in (row_extra.columns if len(row_extra) else []) else '',
        'lat':             lat,
        'lon':             lon,
        'current_cases':   int(group['cases'].iloc[-1]),
        'predicted_cases': int(predicted),
        'cases_per_100k':  round(safe_val(row_extra, 'cases_per_100k') or 0, 2),
        'death_rate':      round(safe_val(row_extra, 'death_rate') or 0, 4),
        'vax_rate':        safe_val(row_extra, 'people_vaccinated_per_hundred'),
        'full_vax_rate':   safe_val(row_extra, 'people_fully_vaccinated_per_hundred'),
        'positivity_rate': safe_val(row_extra, 'positivity_rate'),
        'tests_7day_avg':  safe_val(row_extra, 'tests_7day_avg'),
        'growth_rate':     round(float(group['growth_rate'].iloc[-1]), 6),
    })

result_df = pd.DataFrame(results)

# Percentile-based risk (fair across all countries)
p66 = result_df['predicted_cases'].quantile(0.66)
p33 = result_df['predicted_cases'].quantile(0.33)

def map_risk(v):
    if v >= p66:   return "High"
    elif v >= p33: return "Medium"
    else:          return "Low"

result_df['risk'] = result_df['predicted_cases'].apply(map_risk)
print(f" Countries modelled: {len(result_df)}")
print(result_df[['country','current_cases','predicted_cases','risk']].head(8).to_string())
print("\n Risk distribution:\n", result_df['risk'].value_counts().to_string())

# ======================================
# STEP 11: PROPHET MODEL — TOP COUNTRY
# ======================================
country_name = df['country'].value_counts().index[0]
country_df   = df[df['country'] == country_name].copy()
print(f"\n Prophet model: {country_name}")

model_df = (country_df[['date', 'cases']]
            .rename(columns={'date': 'ds', 'cases': 'y'})
            .drop_duplicates(subset='ds')
            .sort_values('ds'))
model_df = model_df[model_df['y'] >= 0]

prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, interval_width=0.95)
prophet.fit(model_df)
future     = prophet.make_future_dataframe(periods=30)
forecast   = prophet.predict(future)
prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
prediction['yhat']       = prediction['yhat'].clip(lower=0)
prediction['yhat_lower'] = prediction['yhat_lower'].clip(lower=0)

merged_eval = model_df.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
mae  = mean_absolute_error(merged_eval['y'], merged_eval['yhat'])
rmse = np.sqrt(mean_squared_error(merged_eval['y'], merged_eval['yhat']))
print(f" MAE: {mae:,.0f}  |  RMSE: {rmse:,.0f}")
print("\n 30-day predictions (tail):")
print(prediction.tail(10).to_string(index=False))

# ======================================
# STEP 12: RANDOM FOREST FEATURE IMPORTANCE
# ======================================
print("\n Random Forest feature importance...")
rf_cols = ['cases_per_100k', 'death_rate', 'vax_rate',
            'full_vax_rate', 'positivity_rate', 'predicted_cases']
rf_avail = [c for c in rf_cols if c in result_df.columns]
rf_df    = result_df[rf_avail].dropna()
RF_DONE  = False
feat_imp = pd.Series(dtype=float)
if len(rf_df) > 20 and 'predicted_cases' in rf_df.columns:
    X_rf = rf_df.drop(columns=['predicted_cases'])
    y_rf = rf_df['predicted_cases']
    if X_rf.shape[1] > 0:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_rf, y_rf)
        feat_imp = pd.Series(rf.feature_importances_,
                            index=X_rf.columns).sort_values(ascending=False)
        print(feat_imp.to_string())
        RF_DONE = True
    else:
        print("  Not enough feature columns available")
else:
    print("  Not enough overlapping data for RF")

# ======================================
# STEP 13: PLOT 1 — CASES TREND TOP 5
# ======================================
print("\n Generating charts...")
EXCLUDE = ['European Union', 'High-income', 'Upper-middle', 'Low-income',
            'Lower-middle', 'World', 'Asia', 'Europe', 'Africa', 'Americas',
            'North America', 'South America', 'Oceania', 'International',
            'Monaco', 'income']
df_real = df[~df['country'].str.contains('|'.join(EXCLUDE), case=False, na=False)]
top5   = df_real['country'].value_counts().head(5).index.tolist()
df_top = df_real[df_real['country'].isin(top5)]

fig1 = px.line(df_top, x='date', y='cases', color='country',
                title="COVID-19 Cases Trend — Top 5 Countries",
                labels={'cases': 'Total Cases', 'date': 'Date'})
fig1.update_layout(hovermode='x unified')
fig1.show()

# ======================================
# STEP 14: PLOT 2 — PROPHET FORECAST
# ======================================
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=model_df['ds'], y=model_df['y'],
                            mode='lines', name='Actual',
                            line=dict(color='steelblue', width=2)))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'].clip(0),
                            mode='lines', name='Forecast',
                            line=dict(color='orange', width=2)))
fig2.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'].clip(0),
                forecast['yhat_lower'].clip(0)[::-1]]),
    fill='toself', fillcolor='rgba(255,165,0,0.15)',
    line=dict(color='rgba(0,0,0,0)'), name='95% Confidence'))
fig2.update_layout(title=f"Prophet 30-Day Forecast — {country_name}",
                    xaxis_title="Date", yaxis_title="Cases",
                    hovermode='x unified')
fig2.show()

# ======================================
# STEP 15: PLOT 3 — DAILY NEW CASES
# ======================================
fig3 = px.bar(df_top, x='date', y='daily_cases', color='country',
                barmode='group', title="Daily New Cases — Top 5 Countries",
                labels={'daily_cases': 'Daily Cases', 'date': 'Date'})
fig3.show()

# ======================================
# STEP 16: PLOT 4 — 14-DAY MOVING AVERAGE
# ======================================
fig4 = px.line(df_top, x='date', y='moving_avg_14', color='country',
                title="14-Day Moving Average — Top 5 Countries",
                labels={'moving_avg_14': '14-Day Avg Cases'})
fig4.show()

# ======================================
# STEP 17: PLOT 5 — VACCINATION vs CASES
# ======================================
vac_scatter = result_df.dropna(subset=['vax_rate', 'cases_per_100k'])
if len(vac_scatter) > 3:
    fig5 = px.scatter(
        vac_scatter, x='vax_rate', y='cases_per_100k',
        color='risk',
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
        size='current_cases', size_max=45,
        hover_name='country',
        hover_data={'positivity_rate': True, 'death_rate': True,
                    'vax_rate': False, 'cases_per_100k': False},
        title="Vaccination Rate vs Cases per 100k",
        labels={'vax_rate': '% Vaccinated', 'cases_per_100k': 'Cases per 100k'})
    fig5.show()

# ======================================
# STEP 18: PLOT 6 — FEATURE IMPORTANCE
# ======================================
if RF_DONE and len(feat_imp) > 0:
    fig6 = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h',
                title="What Drives Predicted Cases? — Random Forest",
                labels={'x': 'Importance Score', 'y': 'Feature'},
                color=feat_imp.values, color_continuous_scale='Reds')
    fig6.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig6.show()

# ======================================
# STEP 19: PLOT 7 — DEATH RATE vs VAX RATE
# ======================================
dr_plot = result_df.dropna(subset=['vax_rate', 'death_rate'])
dr_plot = dr_plot[dr_plot['death_rate'] > 0]
if len(dr_plot) > 5:
    fig7 = px.scatter(dr_plot, x='vax_rate', y='death_rate',
                        color='risk', trendline='ols',
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                        hover_name='country',
                        title="Vaccination Rate vs Death Rate (with trend line)",
                        labels={'vax_rate': '% Vaccinated', 'death_rate': 'Death Rate (%)'})
    fig7.show()

# ======================================
# STEP 20: PLOT 8 — GLOBAL CASES CHOROPLETH
# ======================================
hover_cols = {c: True for c in ['cases_per_100k',
            'people_vaccinated_per_hundred', 'positivity_rate']
            if c in latest_df.columns}
fig8 = px.choropleth(
    latest_df, locations="country", locationmode="country names",
    color="cases", color_continuous_scale="Reds",
    hover_data=hover_cols,
    title="Global COVID-19 Total Cases Map")
fig8.show()

# ======================================
# STEP 21: PLOT 9 — RISK CHOROPLETH
# ======================================
fig9 = px.choropleth(
    result_df, locations="country", locationmode="country names",
    color="risk",
    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
    hover_data=['current_cases', 'predicted_cases',
                'vax_rate', 'positivity_rate', 'death_rate'],
    title="Global COVID-19 Risk Map — 14-Day Forecast")
fig9.show()

# ======================================
# STEP 22: PLOT 10 — TOP 20 POSITIVITY RATES
# ======================================
if 'positivity_rate' in result_df.columns:
    pos_plot = result_df.dropna(subset=['positivity_rate'])
    pos_plot = pos_plot[pos_plot['positivity_rate'] > 0].nlargest(20, 'positivity_rate')
    if len(pos_plot) > 0:
        fig10 = px.bar(pos_plot, x='country', y='positivity_rate',
                        color='risk',
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                        title="Top 20 Countries by Test Positivity Rate",
                        labels={'positivity_rate': 'Positivity Rate'})
        fig10.update_layout(xaxis_tickangle=-45)
        fig10.show()

# ======================================
# STEP 23: FOLIUM INTERACTIVE RISK MAP
# ======================================
print("\n Building Folium interactive map...")
m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

high_grp   = folium.FeatureGroup(name="High Risk",   show=True)
medium_grp = folium.FeatureGroup(name="Medium Risk", show=True)
low_grp    = folium.FeatureGroup(name="Low Risk",    show=True)
heat_grp   = folium.FeatureGroup(name="Heatmap",     show=False)

CMAP = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}

def fmt(val, suffix='', dec=1):
    try:
        return f"{float(val):.{dec}f}{suffix}" if pd.notna(val) else "N/A"
    except Exception:
        return "N/A"

max_pred = result_df['predicted_cases'].max()
if max_pred == 0:
    max_pred = 1

for _, row in result_df.iterrows():
    try:
        risk_color = CMAP.get(row['risk'], 'gray')
        popup_html = f"""
        <div style='font-family:Arial;font-size:13px;min-width:210px'>
            <b style='font-size:15px'>{row['country']}</b><br>
            <hr style='margin:4px 0'>
            <b>Current cases:</b> {int(row['current_cases']):,}<br>
            <b>Predicted (14d):</b> {int(row['predicted_cases']):,}<br>
            <b>Cases per 100k:</b> {fmt(row['cases_per_100k'])}<br>
            <b>Death rate:</b> {fmt(row['death_rate'], '%', 2)}<br>
            <hr style='margin:4px 0'>
            <b>Vaccinated:</b> {fmt(row['vax_rate'], '%')}<br>
            <b>Fully vaccinated:</b> {fmt(row['full_vax_rate'], '%')}<br>
            <b>Positivity rate:</b> {fmt(row['positivity_rate'])}<br>
            <b>Tests 7-day avg:</b> {fmt(row['tests_7day_avg'], '', 0)}<br>
            <hr style='margin:4px 0'>
            <b>Risk: <span style='color:{risk_color}'>{row['risk']}</span></b>
        </div>"""
        radius = max(row['predicted_cases'] / max_pred * 35, 4)
        circle = folium.CircleMarker(
            location=[float(row['lat']), float(row['lon'])],
            radius=radius,
            color=risk_color, fill=True, fill_color=risk_color,
            fill_opacity=0.65, weight=1,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['country']} — {row['risk']} Risk"
        )
        if   row['risk'] == 'High':   circle.add_to(high_grp)
        elif row['risk'] == 'Medium': circle.add_to(medium_grp)
        else:                         circle.add_to(low_grp)
    except Exception:
        continue  # skip countries with bad coords

heat_data = []
for _, row in result_df.iterrows():
    try:
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            heat_data.append([float(row['lat']), float(row['lon']),
                                float(row['predicted_cases'])])
    except Exception:
        continue

if heat_data:
    HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(heat_grp)

for grp in [high_grp, medium_grp, low_grp, heat_grp]:
    grp.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;bottom:40px;left:40px;background:white;
    border:1px solid #ccc;border-radius:10px;padding:14px 18px;
    font-family:Arial;font-size:13px;z-index:9999;
    box-shadow:2px 2px 8px rgba(0,0,0,0.25);min-width:200px">
    <b style='font-size:14px'>14-Day Predicted Risk</b><br><br>
    <span style='color:#e74c3c;font-size:18px'>&#9679;</span> <b>High risk</b><br>
    <span style='color:#f39c12;font-size:18px'>&#9679;</span> <b>Medium risk</b><br>
    <span style='color:#27ae60;font-size:18px'>&#9679;</span> <b>Low risk</b><br>
    <hr style='margin:8px 0'>
    <small><b>Circle size</b> = relative predicted cases</small><br>
    <small><b>Click</b> for full country stats</small><br>
    <small><b>Toggle layers</b> top-right</small>
</div>"""))

map_path = fr"{BASE}\ml_risk_map.html"
m.save(map_path)
print(f" Map saved: {map_path}")

# ======================================
# STEP 24: HOTSPOT SUMMARY
# ======================================
print("\n" + "=" * 60)
print("  TOP 10 HIGH RISK COUNTRIES")
print("=" * 60)
high_risk = (result_df[result_df['risk'] == 'High']
            .sort_values('predicted_cases', ascending=False)
            .head(10))
display_cols = [c for c in ['country', 'current_cases', 'predicted_cases',
                            'cases_per_100k', 'vax_rate', 'positivity_rate',
                            'death_rate', 'risk'] if c in high_risk.columns]
print(high_risk[display_cols].to_string(index=False))

# ======================================
# STEP 25: SAVE ALL OUTPUTS
# ======================================
print("\n Saving outputs...")
try:
    prediction.to_csv(fr"{BASE}\final_predictions.csv",    index=False)
    country_df.to_csv(fr"{BASE}\final_risk.csv",           index=False)
    result_df.to_csv( fr"{BASE}\global_risk_map_data.csv", index=False)
    latest_df.to_csv( fr"{BASE}\merged_latest.csv",        index=False)
    print(" All CSVs saved.")
except Exception as e:
    print(f" Save error: {e}")

print("\n" + "=" * 60)
print("  OUTPUTS")
print("=" * 60)
print("  final_predictions.csv     Prophet 30-day forecast")
print(f"  final_risk.csv            Hotspot risk — {country_name}")
print("  global_risk_map_data.csv  Per-country ML risk (all 5 sources)")
print("  merged_latest.csv         Full merged dataset")
print("  ml_risk_map.html          Interactive Folium risk map")
print(f"\n  Prophet MAE  : {mae:,.0f}")
print(f"  Prophet RMSE : {rmse:,.0f}")
print("\n  FINAL PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
