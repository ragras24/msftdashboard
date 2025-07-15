import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="USD Strength Dashboard", layout="wide")

#---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    /* Metric box styling (unchanged) */
    [data-testid="stMetric"] {
        background-color: #1c1c1c;
        color: white;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Style for st.metric() boxes */
    [data-testid="stMetric"] {
        background-color: #000000; /* pure black */
        color: #d3d3d3;  /* light grey text */
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.05);
        text-align: center;
    }

    /* Ensure inner text like delta and value is also grey */
    [data-testid="stMetric"] div {
        color: #d3d3d3 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR INPUTS ----------
st.sidebar.image("microsoft_logo.png", use_container_width=True)
st.sidebar.title("⚙️ Model Configuration")

country_currency = st.sidebar.selectbox(
    "Country and Currency",
    options=["Japan - JPY", "Australia - AUD", "Canada - CAD", "Eurozone - EUR", "United Kingdom - GBP"]
)

if country_currency == "Japan - JPY":
    data = pd.read_csv(r"Monthly Data/Japan Monthly Data - Final.csv")
    currency_symbol = "¥"
    base_quote = "USD/JPY"
elif country_currency == "Australia - AUD":
    data = pd.read_csv(r"Monthly Data/Australia Monthly Data - Final.csv")
    currency_symbol = "A$"
    base_quote = "USD/AUD"

elif country_currency == "Canada - CAD":
    data = pd.read_csv(r"Monthly Data/Canada Monthly Data - Final.csv")
    currency_symbol = "C$"
    base_quote = "USD/CAD"

elif country_currency == "United Kingdom - GBP":
    data = pd.read_csv(r"Monthly Data/UK Monthly Data - Final.csv")
    currency_symbol = "£"
    base_quote = "USD/GBP"    

elif country_currency == "Eurozone - EUR":
    data = pd.read_csv(r"Monthly Data/EU Monthly Data - Final.csv")
    currency_symbol = "€"
    base_quote = "USD/EUR"

else:
    st.warning("Dataset for this country is not loaded yet.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
data = data.set_index("Date").sort_index()
data.index = data.index.date



#Dates Input
available_dates = data.index
start_input = st.sidebar.date_input("Start Date", value=available_dates.min(), min_value=available_dates.min(), max_value=available_dates.max())

end_input = st.sidebar.date_input("End Date", value=available_dates.max(), min_value=available_dates.min(), max_value=available_dates.max())

start_input = pd.to_datetime(start_input).date()
end_input = pd.to_datetime(end_input).date()



# Filter your data using the adjusted dates
filtered_data = data.loc[start_input:end_input]

na_variables = filtered_data.columns[filtered_data.isna().any()].to_list()

# Drop vraibles that have NA values (insufficient data)
if na_variables:
    st.warning(f"Dropping the following variables due to missing/insufficient data: {', '.join(na_variables)}")
    filtered_data = filtered_data.drop(columns = na_variables)

y_raw = filtered_data.iloc[:, 0]  # first column = fx_rate target
X_raw_all = filtered_data.iloc[:, 1:]  # all other columns as features



# # Variable descriptions
# var_descriptions = {
#     "Interest Rate Differential": "The difference between U.S. and Japan interest rates.",
#     "Inflation Rate": "CPI or core CPI inflation used as a monetary pressure proxy.",
#     "GDP Growth": "Quarterly or yearly change in GDP.",
#     "Money Supply": "Monetary base growth, indicating liquidity.",
#     "Trade Balance": "Exports minus imports, affecting currency demand.",
#     # Add all relevant variables here
# }


independent_vars = st.sidebar.multiselect(
    "Independent Variables",
    options=X_raw_all.columns.to_list(),
    default=X_raw_all.columns.to_list())

# # Display descriptions below
# with st.sidebar.expander("ℹ️ Variable Definitions"):
#     for var in independent_vars:
#         if var in var_descriptions:
#             st.markdown(f"**{var}**: {var_descriptions[var]}")


if len(independent_vars) == 0:
    st.error("Please select at least one independent variable.")
    st.stop()


# Subset predictors to selected variables
X_raw = X_raw_all[independent_vars]

# 🔍 Drop any predictors with 0 standard deviation (flat series)
X_std_check = X_raw.std()
zero_std_vars = X_std_check[X_std_check == 0].index.tolist()

if zero_std_vars:
    st.warning(f"These variables have no variation and were removed: {', '.join(zero_std_vars)}")
    X_raw = X_raw.drop(columns=zero_std_vars)
    independent_vars = [v for v in independent_vars if v not in zero_std_vars]


model_choice = st.sidebar.radio(
    "Select Model Type:",
    ["Scorecard", "Gradient Boosting"]
)

# ---------- HEADER ----------
st.title("USD Strength Dashboard")

# ---------- HISTORICAL CHART WITH PREDICTION ----------
st.subheader(f"📈 {base_quote} Exchange Rate - Historical & Predicted")

# Create the main chart first (we'll update it with prediction later)
def create_exchange_rate_chart(historical_data, predicted_value=None, prediction_date=None, actual_value=None):
    fig = go.Figure()
    
    # Historical data line
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name=f'Historical {base_quote}',
        line=dict(color="#42cfd4", width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Rate:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add predicted point if provided
    if predicted_value is not None and prediction_date is not None:
        # Add a connecting line from last historical point to prediction
        last_date = historical_data.index[-1]
        last_value = historical_data.iloc[-1]
        
        fig.add_trace(go.Scatter(
            x=[last_date, prediction_date],
            y=[last_value, predicted_value],
            mode='lines',
            name='Prediction Connection',
            line=dict(color='#f65d35', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add predicted point
        fig.add_trace(go.Scatter(
            x=[prediction_date],
            y=[predicted_value],
            mode='markers',
            name='Predicted Value',
            marker=dict(
                color='#ff7f0e',
                size=10,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Predicted Date:</b> %{x}<br><b>Predicted Rate:</b> %{y:.4f}<extra></extra>'
        ))

        if actual_value is not None:
            fig.add_trace(go.Scatter(
                x=[prediction_date],
                y=[actual_value],
                mode='markers',
                name='Actual Value',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Actual Date:</b> %{x}<br><b>Actual Rate:</b> %{y:.4f}<extra></extra>'
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Date",
        yaxis_title=f"Exchange Rate ({base_quote})",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Initially show just historical data
chart_placeholder = st.empty()
with chart_placeholder.container():
    fig = create_exchange_rate_chart(y_raw)
    st.plotly_chart(fig, use_container_width=True)

# ---------- MODEL OUTPUT ----------
st.subheader(f"📊 Results from: {model_choice}")

# --------------------- SCORECARD MODEL ---------------------
if model_choice == "Scorecard":
    # 1. Standardize X and y (based on selected variables)
    X_mean = X_raw.mean()
    X_std = X_raw.std()
    X = (X_raw - X_mean) / X_std

    y_mean = y_raw.mean()
    y_std = y_raw.std()
    y = (y_raw - y_mean) / y_std



    # 2. Fit regression with robust SE
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit(cov_type='HC3')
    r_squared = model.rsquared

    betas = model.params
    beta_no_const = betas.drop("const") if "const" in betas else betas
    abs_betas = np.abs(beta_no_const)
    abs_betas_sum = abs_betas.sum()
    normalized_weights = abs_betas / abs_betas_sum
    beta_signs = np.sign(beta_no_const)

    col_w, col_v, col_s = st.columns([0.95, 0.95, 1.1])


    # 1) Weight sliders (default = normalized weights)

    # Sort or keep as-is depending on context
    top_vars = independent_vars[:6]
    other_vars = independent_vars[6:]

    with col_w:
        st.markdown("#### Set Variable Importance")
        st.markdown("")
        weights = {}
        total_weight = 0
        for var in top_vars:
            default_val = round(normalized_weights[var], 3)
            weights[var] = st.slider(f"{var}", 0.0, 1.0, default_val, step=0.001, format ='%.3f' , key = f"weight_slider_{var.replace(' ', '_')}")
            total_weight += weights[var]

        if other_vars:
            with st.expander("Display More Variables"):
                for var in other_vars:
                    default_val = round(normalized_weights[var], 3)
                    weights[var] = st.slider(f"{var}", 0.0, 1.0, default_val, step=0.001, format ='%.3f' , key = f"weight_slider_{var.replace(' ', '_')}")
                    total_weight += weights[var]

        # if abs(total_weight - 1) > 0.001:
        #     st.error(f"⚠️ Weights must sum to 1. Current sum: {round(total_weight,3)}")
        #     st.stop()

    # 2) Predictor values input (raw scale)

    top_vars = independent_vars[:6]
    other_vars = independent_vars[6:]

    with col_v:
        st.markdown("#### Enter Predicted Values")
        st.markdown("")
        values = {}
        for var in top_vars:
            min_val = float(X_raw[var].min())
            max_val = float(X_raw[var].max())
            default_val = float(X_raw[var].iloc[-1])

            # buffer = (max_val - min_val) * 0.6  # 60% range buffer
            # input_min = min(min_val, default_val) - buffer
            # input_max = max(max_val, default_val) + buffer

            values[var] = st.number_input(f"{var}", value=default_val, step=0.01,key=f"predictor_input_{var.replace(' ', '_')}")
            st.markdown("")

        if other_vars:
            with st.expander("Display More Variables"):
                for var in other_vars:
                    min_val = float(X_raw[var].min())
                    max_val = float(X_raw[var].max())
                    default_val = float(X_raw[var].iloc[-1])

                    # buffer = (max_val - min_val) * 0.6  # 60% range buffer
                    # input_min = min(min_val, default_val) - buffer
                    # input_max = max(max_val, default_val) + buffer

                    values[var] = st.number_input(f"{var}", value=default_val, step=0.01,key=f"predictor_input_{var.replace(' ', '_')}")
                    st.markdown("")
        


    # 3) Score summary and prediction
    with col_s:
        st.markdown("#### 💡 Score Summary")

        # Convert weights back to betas (with sign and magnitude)
        betas_manual = {}
        for v in independent_vars:
            betas_manual[v] = beta_signs[v] * weights[v] * abs_betas_sum

        # Standardize user input values
        values_std = {}
        for v in independent_vars:
            values_std[v] = (values[v] - X_mean[v]) / X_std[v]

        # Predict standardized y
        y_pred_std = sum(betas_manual[v] * values_std[v] for v in independent_vars)

        # Convert predicted y back to original scale
        y_pred = y_pred_std * y_std + y_mean

        # Prepare scorecard dataframe
        scorecard_df = pd.DataFrame({
            "Variable": independent_vars,
            "Weight": [weights[v] for v in independent_vars],
            "Weighted Impact": [(betas_manual[v] * values_std[v])*y_std  for v in independent_vars]
        })

        scorecard_df["Direction"] = scorecard_df["Weighted Impact"].apply(lambda x: "↑" if x > 0 else "↓")

        total_score = scorecard_df["Weighted Impact"].sum()

        # Display predicted FX rate
        st.metric("Predicted FX Rate", f"{currency_symbol}{round(y_pred, 4)}")
        st.info(f"**Model R²:** {r_squared:.4f}")

        # FX Strength Indicator
        if y_pred > (y_raw.iloc[-1]):
            st.success("The USD is Expected to Grow Stronger 📈")
        elif y_pred < (y_raw.iloc[-1]):
            st.error("The USD is Expected to Become Weaker 📉")
        else:
            st.info("The USD is Expected to Remain Stable ⚖️")

        st.dataframe(scorecard_df.set_index("Variable"))

        # Update the chart with the prediction
        # Use next business day as prediction date
        last_date = y_raw.index[-1]
        prediction_date = last_date + pd.Timedelta(days=1)

        # Try to get actual FX value from full data (not filtered)
        try:
            # Get the first available date in full data on or after prediction_date
            future_date = min(d for d in data.index if d > prediction_date)
            actual_fx = data.loc[future_date, data.columns[0]]
            show_actual = True
        except (ValueError, KeyError):
            actual_fx = None
            show_actual = False
        
        # Update the chart with prediction
        with chart_placeholder.container():
            fig_updated = create_exchange_rate_chart(y_raw, y_pred, prediction_date, actual_value=actual_fx if show_actual else None)
            st.plotly_chart(fig_updated, use_container_width=True)

