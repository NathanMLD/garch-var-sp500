import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis, norm, t, chi2
from arch import arch_model
import scipy.optimize
from scipy.special import xlogy

# Fixed dates and random seed -> reproducible results
START = '2005-01-01'
END = '2026-09-30'
SPLIT = '2015-01-01'
np.random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # work in the script's folder
os.makedirs('figures', exist_ok=True)

price = yf.download('^GSPC', start=START, end=END, progress=False).Close.squeeze()
Returns = np.log(price / price.shift(1)).dropna()*100
Squared_Returns = Returns**2

# ------------------------------------------------------------------
# 1. Data and descriptive statistics
# ------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(price, color='blue')
ax1.set_title('S&P 500')
ax1.set_ylabel('Index level')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(Returns, color='red')
ax2.set_title('Daily log returns')
ax2.set_ylabel('Return (%)')
ax2.grid(True, linestyle='--', alpha=0.7)

ax3.plot(Squared_Returns, color='red')
ax3.set_title('Squared returns')
ax3.set_ylabel('Return² (%²)')
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/01_returns.png', dpi=150, bbox_inches='tight')
plt.show()

average_return = np.mean(Returns)
print("Mean return (%):", average_return)

Standard_deviation = np.std(Returns)
print("Standard deviation (%):", Standard_deviation)

Skewness = skew(Returns)
print("Skewness:", Skewness)

iKurtosis = kurtosis(Returns)
print("Excess kurtosis:", iKurtosis)

shocks = Returns - average_return

# ------------------------------------------------------------------
# 2. GARCH(1,1) by hand
# ------------------------------------------------------------------
# Fixed parameters for now
w = 0.02
a = 0.10
b = 0.88

def garch_filter(shocks, w, a, b):
    sigma2 = [0] * len(shocks)
    sigma2[0] = np.var(shocks)
    for i in range(1, len(sigma2)):
        sigma2[i] = w + a*shocks.iloc[i-1]**2 + b*sigma2[i-1]
    sigma2 = pd.Series(sigma2, index=shocks.index)
    return sigma2

def gjr_filter(shocks, w, a, b, g):
    sigma2 = [0] * len(shocks)
    sigma2[0] = np.var(shocks)
    for i in range(1, len(sigma2)):
        if shocks.iloc[i-1] < 0:
            sigma2[i] = w + (a+g)*shocks.iloc[i-1]**2 + b*sigma2[i-1]
        else:
            sigma2[i] = w + a*shocks.iloc[i-1]**2 + b*sigma2[i-1]
    sigma2 = pd.Series(sigma2, index=shocks.index)
    return sigma2

sigma2 = garch_filter(shocks, w, a, b)

plt.figure(figsize=(10, 6))
plt.plot(Returns.abs(), color='grey', linewidth=1.5, label='|Returns|')
plt.plot(np.sqrt(sigma2), color='blue', linewidth=1.5, label='GARCH(1,1) volatility')
plt.title('GARCH(1,1) volatility vs absolute returns')
plt.xlabel('Year')
plt.ylabel('Daily volatility (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/02_garch_volatility.png', dpi=150, bbox_inches='tight')
plt.show()

LongTemVar = w/(1-a-b)
print("Long-term variance:", LongTemVar)
print("Long-term annualised volatility (model, %):", np.sqrt(LongTemVar)*np.sqrt(252))
print("Annualised volatility (data, %):", Standard_deviation*np.sqrt(252))

HLS = np.log(0.5)/np.log(a+b)
print("Half-life of a shock (days):", HLS)

# ------------------------------------------------------------------
# 3. Maximum likelihood estimation
# ------------------------------------------------------------------
# Hypothesis 1: (0.02 ; 0.10 ; 0.88) -> 7276.21
# Hypothesis 2: (0.5 ; 0.10 ; 0)     -> 9179.00
# Hypothesis 3: (0.02 ; 0.05 ; 0.94) -> 7431.74

parameters = [0.02, 0.05, 0.94]
def neg_log_likelihood(params, shocks):
    sigma3 = garch_filter(shocks, params[0], params[1], params[2])
    lt = -1/2 * (np.log(2*np.pi) + np.log(sigma3) + (shocks**2)/sigma3)
    return -np.sum(lt)

print("Negative log-likelihood (hypothesis 3):", neg_log_likelihood(parameters, shocks))

x0 = [0.02, 0.1, 0.88]

def stationarity_constraint(params):
    return 1-params[1]-params[2]

res_scipy = scipy.optimize.minimize(neg_log_likelihood, x0, args=(shocks,), method='SLSQP',
                                    bounds=[(1e-6, 1), (0, 1), (0, 1)],
                                    constraints={'type': 'ineq', 'fun': stationarity_constraint})

print("Estimated parameters (omega, alpha, beta):", res_scipy.x)
print("Negative log-likelihood:", res_scipy.fun)
print("Optimisation success:", res_scipy.success)

NewHLS = np.log(0.5)/np.log(res_scipy.x[1]+res_scipy.x[2])
print("Half-life of a shock (days):", NewHLS)

NewLongTemVar = res_scipy.x[0]/(1-res_scipy.x[1]-res_scipy.x[2])
print("Long-term variance:", NewLongTemVar)
print("Long-term annualised volatility (%):", np.sqrt(NewLongTemVar)*np.sqrt(252))

# Check with the arch library
res_arch = arch_model(Returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
print(res_arch.summary())

# ------------------------------------------------------------------
# 4. Comparison of 6 models (full sample)
# ------------------------------------------------------------------

spec = [{'name': 'GJR-t',    'params': {'vol': 'GARCH',  'o': 1, 'dist': 't'}},
        {'name': 'GARCH-t',  'params': {'vol': 'GARCH',          'dist': 't'}},
        {'name': 'EGARCH-t', 'params': {'vol': 'EGARCH', 'o': 1, 'dist': 't'}},
        {'name': 'GJR-n',    'params': {'vol': 'GARCH',  'o': 1, 'dist': 'normal'}},
        {'name': 'GARCH-n',  'params': {'vol': 'GARCH',          'dist': 'normal'}},
        {'name': 'EGARCH-n', 'params': {'vol': 'EGARCH', 'o': 1, 'dist': 'normal'}}]

resultats = []
fits = {}
for s in spec:
    res_fit = arch_model(Returns, **s['params']).fit(disp='off')
    fits[s['name']] = res_fit
    resultats.append({"Model": s['name'], "LogLik": res_fit.loglikelihood, "AIC": res_fit.aic,
                      "BIC": res_fit.bic, "Parameters": len(res_fit.params)})

tableau = pd.DataFrame(resultats).sort_values('AIC')
print(tableau)

plt.figure(figsize=(10, 6))
plt.plot(fits['GJR-t'].conditional_volatility['2020-02':'2020-05'], linewidth=1.5, label='GJR-t')
plt.plot(fits['GARCH-n'].conditional_volatility['2020-02':'2020-05'], linewidth=1.5, label='GARCH-n')
plt.plot(Returns.abs()['2020-02':'2020-05'], color='grey', linewidth=1.5, label='|Returns|')
plt.title('COVID-19 crash: GARCH-n vs GJR-t')
plt.xlabel('Date')
plt.ylabel('Daily volatility (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/03_covid_garch_vs_gjr.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 5. Out-of-sample forecasts (estimation 2005-2014, test 2015 onwards)
# ------------------------------------------------------------------
res_oos_garch = arch_model(Returns, vol='GARCH', p=1, q=1, dist='normal').fit(last_obs=SPLIT, disp='off')
res_oos_egarch = arch_model(Returns, vol='EGARCH', p=1, o=1, q=1, dist='t').fit(last_obs=SPLIT, disp='off')

fc_garch = res_oos_garch.forecast(horizon=1, start=SPLIT, reindex=False)
fc_egarch = res_oos_egarch.forecast(horizon=1, start=SPLIT, reindex=False)

# The value at date t is the forecast FOR t+1 -> shift by one day
prev_garch = fc_garch.variance['h.1'].shift(1)
prev_egarch = fc_egarch.variance['h.1'].shift(1)

hist20 = Returns.rolling(20).var().shift(1)

sigma_ewma = garch_filter(shocks, 0, 0.06, 0.94)   # already aligned: uses t-1 only

plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(prev_egarch['2020-02':'2020-05']), linewidth=1.5, label='EGARCH-t')
plt.plot(np.sqrt(prev_garch['2020-02':'2020-05']), linewidth=1.5, label='GARCH-n')
plt.plot(np.sqrt(sigma_ewma['2020-02':'2020-05']), linewidth=1.5, label='EWMA')
plt.plot(np.sqrt(hist20['2020-02':'2020-05']), linewidth=1.5, label='Hist-20')
plt.plot(Returns.abs()['2020-02':'2020-05'], color='grey', linewidth=1.5, label='|Returns|')
plt.title('Out-of-sample volatility forecasts during the COVID-19 crash')
plt.xlabel('Date')
plt.ylabel('Forecast daily volatility (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('figures/04_oos_forecasts_covid.png', dpi=150, bbox_inches='tight')
plt.show()

df = pd.DataFrame({
    'r2': Returns**2,
    'GARCH-n': prev_garch,
    'EGARCH-t': prev_egarch,
    'Hist-20': hist20,
    'EWMA': sigma_ewma,
}).dropna()
df = df[SPLIT:]

print("Alignment check (crash of 16 March 2020):")
print(df['2020-03-13':'2020-03-18'])

modeles = ['GARCH-n', 'EGARCH-t', 'Hist-20', 'EWMA']
pertes = []

for m in modeles:
    mse = ((df['r2'] - df[m])**2).mean()
    qlike = (np.log(df[m]) + (df['r2']/df[m])).mean()
    ratio = (df['r2']/df[m]).mean()
    pertes.append({'Model': m, 'MSE': mse, 'QLIKE': qlike, 'Ratio': ratio})

tableau_pertes = pd.DataFrame(pertes).sort_values('QLIKE')
print(tableau_pertes)

# ------------------------------------------------------------------
# 6. VaR and Expected Shortfall
# ------------------------------------------------------------------
q_001 = norm.ppf(0.01)
q_0025 = norm.ppf(0.025)
q_005 = norm.ppf(0.05)

nu = res_oos_egarch.params['nu']

q_t_raw = t.ppf(0.01, nu)                         # raw Student-t quantile
q_standard = q_t_raw*(np.sqrt((nu-2)/nu))         # standardised Student-t (variance 1)
print("Student-t quantile at 1% (raw / standardised):", q_t_raw, q_standard)

z = np.random.standard_t(nu, size=1_000_000) * np.sqrt((nu - 2) / nu)
print("Simulated std (should be ~1):", z.std())
print("Simulated 1% quantile:", np.quantile(z, 0.01))

quantiles_des_modeles = {
    'GARCH-n': q_001,
    'EGARCH-t': q_standard,
    'EWMA': q_001,
    'Hist-20': q_001
}

for m in modeles:
    df[f'VaR_99_{m}'] = np.sqrt(df[m]) * quantiles_des_modeles[m]

df['Returns'] = Returns

for m in modeles:
    df[f'Violation_{m}'] = df['Returns'] < df[f'VaR_99_{m}']

results_backtest = []
sum_days = len(df)
for m in modeles:
    nb_violations = df[f'Violation_{m}'].sum()
    taux = (nb_violations / sum_days) * 100
    results_backtest.append({
        'Model': m,
        'Days': sum_days,
        'Violations': nb_violations,
        'Expected violations': round(sum_days * 0.01, 1),
        'Violation rate (%)': round(taux, 2),
        'Expected rate (%)': 1.00
    })

tableau_backtest = pd.DataFrame(results_backtest)
print(tableau_backtest.to_string(index=False))

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Returns'], color='grey', alpha=0.5, label='Returns')
plt.plot(df.index, df['VaR_99_GARCH-n'], color='blue', linewidth=1.5, label='99% VaR GARCH-n')
plt.plot(df.index, df['VaR_99_EGARCH-t'], color='orange', linewidth=1.5, label='99% VaR EGARCH-t')

violations_egarch = df[df['Violation_EGARCH-t']]
plt.scatter(violations_egarch.index, violations_egarch['Returns'], color='red', label='EGARCH-t violations', zorder=5)

plt.title('99% VaR backtest - out-of-sample period (2015 onwards)')
plt.xlabel('Year')
plt.ylabel('Daily return (%)')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('figures/05_var_backtest.png', dpi=150, bbox_inches='tight')
plt.show()

# Expected Shortfall 97.5%
q_t_raw25 = t.ppf(0.025, nu)
q_standard25 = q_t_raw25*(np.sqrt((nu-2)/nu))
print("Student-t quantile at 2.5% (raw / standardised):", q_t_raw25, q_standard25)

quantiles_des_modeles25 = {
    'GARCH-n': q_0025,
    'EGARCH-t': q_standard25,
    'EWMA': q_0025,
    'Hist-20': q_0025
}

for m in modeles:
    df[f'VaR_975_{m}'] = np.sqrt(df[m]) * quantiles_des_modeles25[m]

for m in modeles:
    taux_975 = (df['Returns'] < df[f'VaR_975_{m}']).mean() * 100
    print(f"{m}: 97.5% VaR violation rate = {taux_975:.2f} % (expected: 2.5 %)")

z_norm = np.random.standard_normal(1_000_000)
es_z_norm = z_norm[z_norm < np.quantile(z_norm, 0.025)].mean()
es_z_t = z[z < np.quantile(z, 0.025)].mean()
print("ES of z: normal =", es_z_norm, " Student-t =", es_z_t)

df['ES_975_GARCH-n'] = np.sqrt(df['GARCH-n']) * es_z_norm
df['ES_975_EGARCH-t'] = np.sqrt(df['EGARCH-t']) * es_z_t

for m in ['GARCH-n', 'EGARCH-t']:
    viol = df['Returns'] < df[f'VaR_975_{m}']
    print(f"{m}: actual mean loss = {df.loc[viol, 'Returns'].mean():.3f}  |  mean forecast ES = {df.loc[viol, f'ES_975_{m}'].mean():.3f}")

# ------------------------------------------------------------------
# 7. Backtesting: Kupiec and Christoffersen tests
# ------------------------------------------------------------------
def kupiec(violations, p):
    T = len(violations)
    N = violations.sum()
    p_hat = N / T
    lnL_0 = (T-N)*np.log(1-p) + xlogy(N, p)
    lnL_1 = (T-N)*np.log(1-p_hat) + xlogy(N, p_hat)
    LR = -2 * (lnL_0 - lnL_1)
    p_value = 1 - chi2.cdf(LR, 1)
    return LR, p_value


def christoffersen(violations):
    I = violations.astype(int)
    prev = I.shift(1)
    I, prev = I[1:], prev[1:]

    n00 = ((prev == 0) & (I == 0)).sum()
    n01 = ((prev == 0) & (I == 1)).sum()
    n10 = ((prev == 1) & (I == 0)).sum()
    n11 = ((prev == 1) & (I == 1)).sum()

    pi01 = n01/(n01+n00)
    pi11 = n11/(n11+n10)
    pi = (n01+n11)/(n00+n01+n10+n11)

    lnL_0 = xlogy(n00+n10, 1-pi) + xlogy(n01+n11, pi)
    lnL_1 = xlogy(n00, 1-pi01) + xlogy(n01, pi01) + xlogy(n10, 1-pi11) + xlogy(n11, pi11)
    LR = -2 * (lnL_0 - lnL_1)
    p_value = 1 - chi2.cdf(LR, 1)
    return LR, p_value, pi01, pi11

results_test_hypothesis = []
for m in modeles:
    LR_uc, p_uc = kupiec(df[f'Violation_{m}'], 0.01)
    LR_ind, p_ind, pi01, pi11 = christoffersen(df[f'Violation_{m}'])
    LR_cc = LR_uc + LR_ind
    p_cc = 1 - chi2.cdf(LR_cc, 2)
    results_test_hypothesis.append({
        'Model': m,
        'Violations': df[f'Violation_{m}'].sum(),
        'Rate': df[f'Violation_{m}'].mean(),
        'LR_uc': LR_uc,
        'p_uc': p_uc,
        'pi01': pi01,
        'pi11': pi11,
        'LR_ind': LR_ind,
        'p_ind': p_ind,
        'LR_cc': LR_cc,
        'p_cc': p_cc,
        'Verdict': 'Accepted' if p_cc >= 0.05 else 'Rejected',
    })
tableau_test_hypothesis = pd.DataFrame(results_test_hypothesis).sort_values('LR_cc')
print(tableau_test_hypothesis.to_string(index=False))

# Conclusion: none of these models passes the conditional coverage test at the 5% level
# (p-value < 0.05). The best one is the EGARCH with Student-t innovations, but it still
# underestimates extreme losses: 48 violations of the 99% VaR versus 29.5 expected (2952 x 1%).
# To improve these models, we could estimate their parameters on a larger sample
# that includes more extreme events.
