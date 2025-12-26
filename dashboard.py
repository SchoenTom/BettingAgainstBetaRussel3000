"""
dashboard.py - Interactive Streamlit dashboard for BAB strategy analysis

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from config import OUTPUT_DIR, DATA_DIR, NUM_DECILES

st.set_page_config(
    page_title="BAB Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load all required data."""
    files = {
        'monthly_perf': os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv'),
        'bab_portfolio': os.path.join(OUTPUT_DIR, 'bab_portfolio.csv'),
        'decile_returns': os.path.join(OUTPUT_DIR, 'decile_returns.csv'),
        'summary': os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv'),
    }

    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            if key == 'summary':
                data[key] = pd.read_csv(path)
            else:
                data[key] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            data[key] = None

    return data


def check_data_exists():
    """Check if required files exist."""
    required = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    return os.path.exists(required)


def format_pct(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"


def create_equity_curve(perf, log_scale=True):
    """Create cumulative returns chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf['BAB_Cumulative'],
        mode='lines', name='BAB Strategy',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf['Benchmark_Cumulative'],
        mode='lines', name='S&P 500',
        line=dict(color='#ff7f0e', width=2)
    ))

    fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Cumulative Returns: BAB vs S&P 500',
        xaxis_title='Date',
        yaxis_title='Growth of $1',
        yaxis_type='log' if log_scale else 'linear',
        hovermode='x unified',
        height=500
    )

    return fig


def create_drawdown_chart(perf):
    """Create drawdown chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf['BAB_Drawdown'] * 100,
        fill='tozeroy', name='BAB',
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf['Benchmark_Drawdown'] * 100,
        fill='tozeroy', name='S&P 500',
        line=dict(color='#ff7f0e'),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))

    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400
    )

    return fig


def create_rolling_sharpe(perf):
    """Create rolling Sharpe chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf.index, y=perf['Rolling_12M_BAB_Sharpe'],
        mode='lines', name='BAB',
        line=dict(color='#1f77b4', width=1.5)
    ))

    if 'Rolling_12M_Benchmark_Sharpe' in perf.columns:
        fig.add_trace(go.Scatter(
            x=perf.index, y=perf['Rolling_12M_Benchmark_Sharpe'],
            mode='lines', name='S&P 500',
            line=dict(color='#ff7f0e', width=1.5)
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)

    fig.update_layout(
        title='Rolling 12-Month Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        height=400
    )

    return fig


def create_beta_spread_chart(bab_portfolio):
    """Create beta spread chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bab_portfolio.index, y=bab_portfolio['Beta_Spread'],
        mode='lines', name='Beta Spread',
        line=dict(color='#9467bd', width=1.5)
    ))

    avg = bab_portfolio['Beta_Spread'].mean()
    fig.add_hline(y=avg, line_dash="dot", line_color="darkred",
                  annotation_text=f"Avg: {avg:.2f}")

    fig.update_layout(
        title='Beta Spread (High - Low)',
        xaxis_title='Date',
        yaxis_title='Beta Spread',
        height=400
    )

    return fig


def create_decile_returns_chart(decile_returns):
    """Create decile returns bar chart."""
    avg_returns = []
    for d in range(1, NUM_DECILES + 1):
        col = f'D{d}_Return'
        if col in decile_returns.columns:
            avg_returns.append(decile_returns[col].mean() * 100)
        else:
            avg_returns.append(0)

    colors = px.colors.diverging.RdYlGn_r[:NUM_DECILES]

    fig = go.Figure(data=[
        go.Bar(
            x=[f'D{i}' for i in range(1, NUM_DECILES + 1)],
            y=avg_returns,
            marker_color=colors,
            text=[f'{v:.2f}%' for v in avg_returns],
            textposition='outside'
        )
    ])

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title='Average Monthly Returns by Beta Decile',
        xaxis_title='Decile (D1=Low Beta, D10=High Beta)',
        yaxis_title='Avg Monthly Return (%)',
        height=400,
        showlegend=False
    )

    return fig


def create_yearly_returns(perf):
    """Create yearly returns chart."""
    yearly_bab = (1 + perf['BAB_Return']).resample('YE').prod() - 1
    yearly_bench = (1 + perf['Benchmark_Return']).resample('YE').prod() - 1

    years = [str(y) for y in yearly_bab.index.year]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='BAB', x=years, y=yearly_bab.values * 100,
        marker_color='#1f77b4'
    ))

    fig.add_trace(go.Bar(
        name='S&P 500', x=years, y=yearly_bench.values * 100,
        marker_color='#ff7f0e'
    ))

    fig.add_hline(y=0, line_color="black", line_width=0.5)

    fig.update_layout(
        title='Annual Returns Comparison',
        xaxis_title='Year',
        yaxis_title='Annual Return (%)',
        barmode='group',
        height=400
    )

    return fig


def display_metrics(summary):
    """Display key metrics cards."""
    if summary is None or len(summary) == 0:
        return

    s = summary.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Annualized Return", format_pct(s.get('Annualized_Return', 0)))

    with col2:
        st.metric("Sharpe Ratio", f"{s.get('Sharpe_Ratio', 0):.3f}")

    with col3:
        st.metric("Max Drawdown", format_pct(s.get('Max_Drawdown', 0)))

    with col4:
        st.metric("Win Rate", format_pct(s.get('Win_Rate', 0)))


def main():
    """Main dashboard."""
    st.title("📈 Betting-Against-Beta (BAB) Strategy")
    st.markdown("---")

    if not check_data_exists():
        st.error("""
        **Data files not found!**

        Run the pipeline first:
        ```
        python main.py
        ```
        """)
        return

    data = load_data()
    perf = data.get('monthly_perf')
    bab_portfolio = data.get('bab_portfolio')
    decile_returns = data.get('decile_returns')
    summary = data.get('summary')

    if perf is None:
        st.error("Performance data not loaded.")
        return

    # Sidebar
    st.sidebar.header("Controls")

    min_date = perf.index.min().to_pydatetime()
    max_date = perf.index.max().to_pydatetime()

    start = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    mask = (perf.index >= pd.Timestamp(start)) & (perf.index <= pd.Timestamp(end))
    filtered_perf = perf.loc[mask].copy()
    filtered_perf['BAB_Cumulative'] = (1 + filtered_perf['BAB_Return']).cumprod()
    filtered_perf['Benchmark_Cumulative'] = (1 + filtered_perf['Benchmark_Return']).cumprod()

    log_scale = st.sidebar.checkbox("Log Scale", value=True)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Period: {start} to {end}")

    # Metrics
    st.header("Key Metrics")
    display_metrics(summary)

    # Equity Curve
    st.header("Cumulative Returns")
    st.plotly_chart(create_equity_curve(filtered_perf, log_scale), use_container_width=True)

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rolling Sharpe")
        st.plotly_chart(create_rolling_sharpe(filtered_perf), use_container_width=True)

    with col2:
        st.subheader("Beta Spread")
        if bab_portfolio is not None:
            filtered_bab = bab_portfolio.loc[bab_portfolio.index.isin(filtered_perf.index)]
            st.plotly_chart(create_beta_spread_chart(filtered_bab), use_container_width=True)

    # Drawdowns
    st.header("Drawdown Analysis")
    st.plotly_chart(create_drawdown_chart(filtered_perf), use_container_width=True)

    # Decile returns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Decile Returns")
        if decile_returns is not None:
            filtered_decile = decile_returns.loc[decile_returns.index.isin(filtered_perf.index)]
            st.plotly_chart(create_decile_returns_chart(filtered_decile), use_container_width=True)

    with col2:
        st.subheader("Annual Returns")
        st.plotly_chart(create_yearly_returns(filtered_perf), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("BAB Strategy Dashboard | Frazzini & Pedersen (2014)")


if __name__ == '__main__':
    main()
