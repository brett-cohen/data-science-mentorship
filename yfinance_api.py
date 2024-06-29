import yfinance
import plotly.graph_objs as go

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'INTC']

data = yfinance.download(TICKERS, start='2024-01-01', end='2024-6-01')['Adj Close']

fig = go.Figure()

for ticker in TICKERS:
    fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))

fig.update_layout(
    title='Tech Stock Prices YTD',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    legend_title='Stocks',
    template='plotly_dark'
)

fig.show()
