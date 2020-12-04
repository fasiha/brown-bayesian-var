import numpy as np
import pandas as pd
from scipy.stats import binom, ttest_ind, norm

# https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC daily historical S&P 500 table
df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)

# df = df.iloc[:(19922 + 784 + 30)] # Try to replicate Brown's study exactly

df['pnl'] = df['Close'].pct_change()
# N.B.! sign change seems to be needed to match Brown? But that'll fit the wrong side of the distrubtion, right?

p = 0.01  # var level
nyears = 3

window = round(5 / 7 * 365.25 * nyears)
std_scalar = abs(norm.ppf(p))
df['scaled_std'] = df.pnl.rolling(window).std().shift(1) * std_scalar

df['hist_var'] = df.pnl.rolling(window).quantile(p).shift(1)
# `shift` will push the roll one day into the future, so we can just look at a single row and see both the end-of-day PNL and start-of-day var prediction. See:
# ```
# pd.DataFrame(
#     dict(
#         foo=df['Close'].iloc[-10:],
#         bar=df['Close'].rolling(3).max().shift(1).iloc[-10:],
#     ))
# ```

df = df.dropna()
df['hist_break'] = df.pnl < df.hist_var
hist_nbreaks = sum(df['hist_break'])
hist_nbreaks_prob = binom.pmf(hist_nbreaks, len(df), p)


def breaks_spacing(breaks: pd.Series, n: int, p: float):
    tmp = breaks.rolling(n).sum()
    tmp = tmp[breaks]  # narrow to just entries with breaks
    # 1 is from the break at the start of the window
    breaks_within_n_of_break = sum(tmp > 1)
    nbreaks = sum(breaks)
    return binom.pmf(breaks_within_n_of_break, (n - 1) * nbreaks, p)


def breaks_levels(breaks: pd.Series, var: pd.Series):
    return dict(test=ttest_ind(var[breaks], var[~breaks], equal_var=False),
                onbreak=var[breaks].mean(),
                notbreak=var[~breaks].mean(),
                all=var.mean())


def add_b_var(df):
    "Add Brown Bayesian VaR column to dataframe"
    # Init b-var as the NEGATIVE scaled standard deviations
    # This only matters for the first day
    df['b_var'] = df.scaled_std.copy() * -1

    # find the numeric index of b_var so we can overwrite it
    b_var_idx = list(df.columns).index('b_var')

    # loop over each adjacent pair of rows, except the very last row
    for i in range(len(df) - 1):
        yesterday = df.iloc[i]
        today = df.iloc[i + 1]
        b_break = yesterday.pnl < yesterday.b_var
        if b_break:
            df.iloc[i + 1, b_var_idx] = 2 * yesterday.b_var
        else:
            df.iloc[
                i + 1,
                b_var_idx] = 0.94 * yesterday.b_var + 0.06 * -today.scaled_std
            # Recall that `b.scaled_std` is the trailing scaled std as of this morning
            # We're not cheating by using that here: it includes only yesterday's close
    return df


df = add_b_var(df)
df['b_break'] = df.pnl < df.b_var
b_nbreaks = sum(df['b_break'])
b_nbreaks_prob = binom.pmf(b_nbreaks, len(df), p)

print(df)
summary = dict(
    nbreaks=hist_nbreaks,
    nbreaks_prob=hist_nbreaks_prob,
    hist_day2=breaks_spacing(df['hist_break'], 2, p),
    hist_day10=breaks_spacing(df['hist_break'], 10, p),
    hist_levels=breaks_levels(df.hist_break, df.hist_var),
    b_nbreaks=b_nbreaks,
    b_nbreaks_prob=b_nbreaks_prob,
    b_day2=breaks_spacing(df.b_break, 2, p),
    b_day10=breaks_spacing(df.b_break, 10, p),
    b_levels=breaks_levels(df.b_break, df.b_var),
)
for k in summary:
    print(f"{k}: {summary[k]}")

import pylab as plt
plt.ion()
plt.figure()
df.b_var.plot(marker='.')
(-1 * df.scaled_std).plot()
df.pnl.plot()
plt.grid()

df.to_json('df.json', orient='index', date_format='iso')
