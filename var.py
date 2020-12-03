import numpy as np
import pandas as pd
from scipy.stats import binom

# https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC daily historical S&P 500 table
df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)

df['pct_change'] = df['Close'].pct_change()

p = 0.01  # var level
nyears = 3

window = round(5 / 7 * 365.25 * nyears)
df['hist_var'] = df['pct_change'].rolling(window).quantile(p).shift(1)
# `shift` will push the roll one day into the future, so we can just look at a single row and see both the end-of-day PNL and start-of-day var prediction. See:
# ```
# pd.DataFrame(
#     dict(
#         foo=df['Close'].iloc[-10:],
#         bar=df['Close'].rolling(3).max().shift(1).iloc[-10:],
#     ))
# ```

df = df.dropna()
df['hist_break'] = df['pct_change'] < df.hist_var
nbreaks = sum(df['hist_break'])
nbreaks_prob = binom.pmf(nbreaks, len(df), p)


def breaks_spacing(breaks: pd.Series, n: int, p: float):
    tmp = breaks.rolling(n).sum()
    tmp = tmp[breaks]  # narrow to just entries with breaks
    # 1 is from the break at the start of the window
    breaks_within_n_of_break = sum(tmp > 1)
    nbreaks = sum(breaks)
    return binom.pmf(breaks_within_n_of_break, (n - 1) * nbreaks, p)


print(df)
print(
    dict(
        nbreaks=nbreaks,
        nbreaks_prob=nbreaks_prob,
        day2=breaks_spacing(df['hist_break'], 2, p),
        day10=breaks_spacing(df['hist_break'], 10, p),
    ))
