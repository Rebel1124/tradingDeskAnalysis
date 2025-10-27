# Import Libraries

import pandas as pd
# import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def varMetrics(df, percentile=0.005):
    r = pd.to_numeric(df['Returns'], errors='coerce').dropna()
    VaR = r.quantile(percentile)
    expectedShortfall = r[r < VaR].mean()          # only the Returns column
    avgLoss = r[r < 0].mean()                      # mean of negative returns (will be negative)
    stdDev = r.std(ddof=1)
    return VaR, expectedShortfall, avgLoss, stdDev


def varTable(VaR, expectedShortfall, avgLoss, stdDev):

    palette = px.colors.qualitative.Set3

    colours = [palette[2], "white", palette[2], "white"]

    headerColor = palette[4]

    head = ['<b>VaR<b>', '<b>ES<b>', '<b>Avg Loss<b>', '<b>Std Dev<b>']

    vals = ['{:.2%}'.format(VaR), '{:.2%}'.format(expectedShortfall),
            '{:.2%}'.format(avgLoss), '{:.2%}'.format(stdDev)]

    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4],
        columnwidth = [30,30,30,30],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['center']*4),
        cells=dict(values=vals,
                fill_color=colours,
                line_color='darkslategray',
                font=dict(color='black'),
                align=['center']*4))
    ])   

    fig.update_layout(title=f'Daily VaR Analysis', height=100, width=600, margin=dict(l=2, r=20, b=0,t=45))
    
    return fig


def analysisTable(metric, bps):

    palette = px.colors.qualitative.Set3

    uncollateralized = metric
    collateralized = uncollateralized * 0.85
    incrementalRisk = uncollateralized - collateralized
    avgYield = bps
    requiredTrades = abs(incrementalRisk/avgYield)

    colours = [palette[1], "white", palette[1], "white", palette[1]]

    headerColor = palette[5]

    head = ['<b>Uncollateralized<b>', '<b>Collateralized<b>', '<b>Incremental Risk<b>', '<b>Avg Yield<b>', '<b>Required Trades<b>']

    vals = ['{:.2%}'.format(uncollateralized), '{:.2%}'.format(collateralized),
            '{:.2%}'.format(incrementalRisk), '{:.2%}'.format(avgYield), '{:,.0f}'.format(requiredTrades)]

    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4, 5],
        columnwidth = [25,25,25,25, 25],
        
        header=dict(values=head,
                    fill_color=headerColor,
                    line_color='darkslategray',
                    font=dict(color='black'),
                    align=['center']*5),
        cells=dict(values=vals,
                fill_color=colours,
                line_color='darkslategray',
                font=dict(color='black'),
                align=['center']*5))
    ])   

    fig.update_layout(title=f'Incremental Risk Analysis', height=100, width=600, margin=dict(l=2, r=20, b=2,t=45))
    
    return fig, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades


def clean_data(df):


    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    
    # Clean and convert numeric columns
    for col in df.columns[1:]:
        df[col] = (
            df[col].astype(str)
                   .str.replace('%', '', regex=False)
                   .str.replace(',', '', regex=False)
        )
        df[col] = df[col].replace({'K': '*1e3', 'M': '*1e6'}, regex=True)
        df[col] = df[col].map(lambda x: eval(x) if isinstance(x, str) and any(c.isdigit() for c in x) else None)
    
    return df


# --- 1 Plot the price as a line graph ---
def plot_price(df):
    fig = px.line(
        df, x='Date', y='Price',
        title='USDT/ZAR Price Over Time',
        labels={'Price': 'Price (ZAR)', 'Date': 'Date'},
        template='plotly_dark'
    )
    fig.update_traces(line=dict(width=2, color='#00CC96'))
    fig.update_layout(title_x=0.5)
    return fig


# --- 2 Plot returns as a scatter plot ---
def plot_returns_scatter(df):

    fig = px.scatter(
        df, x='Date', y='Returns',
        title='Daily Returns Scatter Plot',
        labels={'Returns': 'Daily Returns (%)', 'Date': 'Date'},
        template='plotly_dark'
    )
    fig.update_traces(marker=dict(size=6, color='#EF553B', opacity=0.7))
    fig.update_layout(title_x=0.5)
    return fig


# --- 3 Plot histogram of returns ---
def plot_returns_distribution(df):

    # Create subplot with histogram and box plot
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df['Returns'],
        nbinsx=40,
        name='Returns',
        marker_color='#636EFA',
        opacity=0.7
    ))

    fig.update_layout(
        title='Historgram of Daily Returns',
        xaxis_title='Daily Returns (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        title_x=0.5,
        barmode='overlay'
    )

    return fig

# --- 4 Plot box plot of returns ---
def plot_returns_boxPlot(df):

    fig = go.Figure()

    # Box plot
    fig.add_trace(go.Box(
        x=df['Returns'],
        name='Box Plot',
        marker_color='#EF553B',
        boxmean='sd'
    ))

    fig.update_layout(
        title='Historgram of Daily Returns',
        xaxis_title='Daily Returns (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        title_x=0.5,
    )

    return fig



def date_summary(df):
    """
    Extracts the oldest and newest dates (as dd-mmm-yyyy)
    and counts total rows in the dataframe.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    oldest_date = df['Date'].min().strftime('%d-%b-%Y')
    newest_date = df['Date'].max().strftime('%d-%b-%Y')
    num_rows = len(df)
    
    return oldest_date, newest_date, num_rows

zarCurrency = ['BTC_ZAR', 'USDT_ZAR', 'USD_ZAR']
ngnCurrency = ['BTC_NGN', 'USD_NGN']


######## Streamlit App #########################

st.title('Trading Desk Collateral Risk Analysis')

tab1, tab2 = st.tabs(['ZAR', 'NGN'])

with tab1:
    st.header('ZAR Risk Analysis')
    cur = st.selectbox("Currency", zarCurrency)
    df_raw = pd.read_csv(f"{cur}_Historical_Data.csv")
    df_raw = df_raw.drop(columns=['Vol.'])
    df = clean_data(df_raw)
    df = df.sort_values('Date')
    df['Returns'] = df['Price'].pct_change()
    df = df.dropna(subset=['Returns'])
    old, new, num = date_summary(df)
    # st.dataframe(df)

    col1, col2 = st.columns([1,1])
    col3, col4 = st.columns([1,1])
    col5, col6 = st.columns([1,1])
    col7, col8 = st.columns([1,1])

    price_graph = plot_price(df)
    col1.plotly_chart(price_graph)
    returns_graph = plot_returns_scatter(df)
    col2.plotly_chart(returns_graph)
    hist = plot_returns_distribution(df)
    col3.plotly_chart(hist)
    box = plot_returns_boxPlot(df)
    col4.plotly_chart(box)
    VaR, expectedShortfall, avgLoss, stdDev = varMetrics(df, percentile=0.005)
    table = varTable(VaR, expectedShortfall, avgLoss, stdDev)
    col5.plotly_chart(table)
    analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(VaR, 0.0015)
    metric = col5.radio("ZAR Risk Metric", ['VaR', 'ES', 'Avg Loss'], horizontal=True)
    if cur == 'USDT_ZAR':
        basis=0.002
    elif cur == 'USD_ZAR':
        basis=0.002
    else:
        basis=0.004
    if (metric == 'VaR'):
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(VaR, basis)
    elif (metric == 'ES'):
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(expectedShortfall, basis)
    else:
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(avgLoss, basis)
    col5.plotly_chart(analysis)

    col6.subheader('Setup')

    col6.markdown(f"""
    - Between **{old}** and **{new}**, there were **{num}** daily observations.  
    - From this a time series of daily prices changes were computed and a returns histogram produced.
    - Using the frequency distribution of returns the 99.5% percentle loss or Value At Risk (VaR) was determined.
    - The results shown in our findings below makes two assumptions:
        - Trade Size for each pair is the same and
        - Yield earned on USDT/ZAR is 20bps, BTC/ZAR 40bps and 15bps each is earned for USDT/NGN and BTC/NGN. 
        Since there were no trades done for BTC/NGN we assumed the yield would be the same as USDT/NGN.
    - Our findings are as follows:
    """)

    col6.subheader('Findings')

    col6.markdown(f""" 
    - The **{metric}** of an uncollateralized trade for **{cur}** was computed to be **{uncollateralized:.2%}**.  
    - For a collateralized position, the risk would reduce to **{collateralized:.2%}**.  
    - Given the average yield of **{bps:.2%}**, the desk would need an additional **{requiredTrades:,.0f}** trades  
    to cover the **{incrementalRisk:.2%}** incremental risk.
    """)


    show = col5.toggle("Show ZAR DataFrame")
    if show:
        col5.dataframe(df)

with tab2:
    st.header('NGN Risk Analysis')
    cur = st.selectbox("Currency", ngnCurrency)
    df_raw = pd.read_csv(f"{cur}_Historical_Data.csv")
    df_raw = df_raw.drop(columns=['Vol.'])
    df = clean_data(df_raw)
    df = df.sort_values('Date')
    df['Returns'] = df['Price'].pct_change()
    df = df.dropna(subset=['Returns'])
    old, new, num = date_summary(df)
    
    col1, col2 = st.columns([1,1])
    col3, col4 = st.columns([1,1])
    col5, col6 = st.columns([1,1])
    col7, col8 = st.columns([1,1])

    price_graph = plot_price(df)
    col1.plotly_chart(price_graph)
    returns_graph = plot_returns_scatter(df)
    col2.plotly_chart(returns_graph)
    hist = plot_returns_distribution(df)
    col3.plotly_chart(hist)
    box = plot_returns_boxPlot(df)
    col4.plotly_chart(box)
    VaR, expectedShortfall, avgLoss, stdDev = varMetrics(df, percentile=0.005)
    table = varTable(VaR, expectedShortfall, avgLoss, stdDev)
    col5.plotly_chart(table)
    metric = col5.radio("NGN Risk Metric", ['VaR', 'ES', 'Avg Loss'], horizontal=True)
    if (metric == 'VaR'):
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(VaR, 0.0015)
    elif (metric == 'ES'):
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(expectedShortfall, 0.0015)
    else:
        analysis, collateralized, uncollateralized, incrementalRisk, bps, requiredTrades = analysisTable(avgLoss, 0.0015)

    col5.plotly_chart(analysis)


    col6.subheader('Setup')

    col6.markdown(f"""
    - Between **{old}** and **{new}**, there were **{num}** daily observations.  
    - From this a time series of daily prices changes were computed and a returns histogram produced.
    - Using the frequency distribution of returns the 99.5% percentle loss or Value At Risk (VaR) was determined.
    - The results shown in our findings below makes two assumptions:
        - Trade Size for each pair is the same and
        - Yield earned on USDT/ZAR is 20bps, BTC/ZAR 40bps and 15bps each is earned for USDT/NGN and BTC/NGN. 
        Since there were no trades done for BTC/NGN we assumed the yield would be the same as USDT/NGN.
    - Our findings are as follows:
    """)


    col6.subheader('Findings')

    col6.markdown(f"""
    - The **{metric}** of an uncollateralized trade for **{cur}** was computed to be **{uncollateralized:.2%}**.  
    - For a collateralized position, the risk would reduce to **{collateralized:.2%}**.  
    - Given the average yield of **{bps:.2%}**, the desk would need an additional **{requiredTrades:,.0f}** trades  
    to cover the **{incrementalRisk:.2%}** incremental risk.
    """)

    show = col5.toggle("Show NGN DataFrame")
    if show:
        col5.dataframe(df)


