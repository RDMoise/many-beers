import numpy as np
np.random.seed(271828)
from os import system, environ
from pickle import dump, load, HIGHEST_PROTOCOL
import argparse
import pandas as pd
from pprint import pprint
from glob import glob
from plottools import *
import datetime as dt
from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt
plt.switch_backend('agg') # when on batch
import matplotlib
from mplhep import histplot, style
style.use(style.LHCb2)
import matplotlib.dates as mdates

df = pd.read_csv("1202beers.csv", delimiter=';', encoding='utf-8')
plotList = open("plotList.md", "w")

def saveAndListPlot(plotname, description='test', url='many-beers/blob/main/'):
    plt.savefig(f"{plotname}")
    plotList.write(f"[{description}](https://nbviewer.org/github/RDMoise/{url}{plotname})\n\n")

###############################################################################
# histogram of ABVs
###############################################################################
hABV = np.histogram(df.ABV, np.linspace(0,17,18))
fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0)
histplot(hABV, color='#f1bf4b', histtype="fill", alpha=.5)
histplot(hABV, color='#f1bf4b', histtype="step")
plt.xlabel('ABV [%]')
plt.ylabel("Beers / 1%")
applyUniformFont(ax,24)
saveAndListPlot("histABV.pdf", "Histogram of ABVs")
plt.close()


###############################################################################
# histogram of countries
###############################################################################
dg = df.Country.value_counts()
bins = np.linspace(0,len(dg),len(dg)+1)
fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0)
histplot(dg, bins, color='#f1bf4b', histtype="fill", alpha=.5)
histplot(dg, bins, color='#f1bf4b', histtype="step")
ax.set_xticks(.5*(bins[1:]+bins[:-1]))
ax.minorticks_off()
plt.xlabel('Country code')
plt.ylabel("Beers / country")
applyUniformFont(ax,20)
ax.set_xticklabels(dg.index, fontsize=10, rotation=45)
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False,right=False)
saveAndListPlot("histCountries.pdf", "Number of beers from each country")
ax.set_yscale('log')
saveAndListPlot("histCountries_log.pdf", "Number of beers from each country (log scale)")
plt.close()


###############################################################################
# ABV per country
###############################################################################
dGrp = df.groupby("Country")
means = dGrp.ABV.mean()
eoms = dGrp.ABV.std() / dGrp.ABV.count()
dh = pd.concat([means,eoms], keys=['MeanABV','ErrorOnABV'], axis=1)
dh = dh.sort_values('MeanABV',ascending=False)
fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0.01)
plt.errorbar(.5*(bins[1:]+bins[:-1]), dh.MeanABV, dh.ErrorOnABV, False, ls='', marker='.', markersize=7.5, elinewidth=1, capthick=1, capsize=2.5, color='#f1bf4b')
ax.set_xticks(.5*(bins[1:]+bins[:-1]))
ax.minorticks_off()
plt.xlabel('Country code')
plt.ylabel("Mean ABV / country")
applyUniformFont(ax,20)
ax.set_xticklabels(dh.index, fontsize=10, rotation=45)
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False,right=False)
saveAndListPlot("countryABVs.pdf", "Mean ABV per country")
plt.close()


###############################################################################
# time evolution
###############################################################################
df['Date'] = pd.to_datetime(df['Date'], errors='coerce',format='%d/%m/%Y')
day0 = np.mean(df.Date).date()
dDates = df[pd.notna(df['Date'])][['Number','Date','ABV']]
dDates['usableDate'] = [(x.date() - dt.date(2018,1,1)).days for x in dDates['Date']]
dDates = dDates.query('usableDate > 0')
dDates = dDates[pd.notna(dDates['Number'])]
dDates['x'] = [(x.date() - day0).days for x in dDates['Date']]
# dDates['costheta'] = dDates['x'] * 2 / (dDates.x.max() - dDates.x.min()) # should also scale y if we wanna please them mathematicians

# fit a 2nd order Chebychev polynomial
def cheby2(x, c0=0, c1=1, c2=0):
    return c0 + c1 * x + c2 * (2*x*x - 1)
# fit a 4thd order Chebychev polynomial
def cheby4(x, c0=0, c1=1, c2=0, c3=0, c4=0):
    return cheby2(x,c0,c1,c2) + \
    c3 * (4*x*x*x - 3*x) + \
    c4 * (8*x*x*x*x - 8*x*x - 1)

minimiser = Minuit(LeastSquares(np.array(dDates.x), np.array(dDates.Number), np.ones_like(dDates.Number), cheby4), 
    c0 = 600, 
    c1 = .5,
    c2 = 1e-5,
    c3 = 1e-5,
    c4 = 1e-5,
    )
result = minimiser.migrad()
param_hesse = result.hesse()
param_errors = result.errors
print(result)

fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {minimiser.fval:1.1e} / {len(dDates) - minimiser.nfit}",
]
for p, v, e in zip(minimiser.parameters, minimiser.values, minimiser.errors):
    if not minimiser.fixed[p]:
        fit_info.append(f"{p} = ${v:1.1e} \\pm {e:1.1e}$")

fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0)
plt.xlabel('Year')
plt.ylabel('Total beers')
plt.plot(np.sort(dDates.Date), cheby4(np.sort(dDates.x), *minimiser.values), color='#751D1D', lw=2.5, zorder=-99, label='Fit')
plt.scatter(dDates.Date, dDates.Number, color='#f1bf4b', s=10, label='Data')
plotOrderedLegend([1,0])
ax.set_xlim([dt.date(2018,1,1), dt.date(2024,1,1)])
applyUniformFont(ax,24)
plt.text(.05,.75, "\n".join(fit_info), transform=ax.transAxes, fontsize=18, ha='left', va='top')
saveAndListPlot("growth.pdf", "Time evolution of the collection, with polynomial fit")
plt.close()


###############################################################################
# time evolution of mean ABV
###############################################################################
def rollingMeanVariance(l, mu=0, v=0, i0=0):
    mus, vs = [], []
    n = len(l)
    for i in range(n):
        delta = l[i] - mu
        mu += delta / (i+1+i0)
        mus.append(mu)
        v += (l[i] - mu) * delta
        vs.append(v / (i+1+i0))
    return mus, vs

dh = df[pd.notna(df['ABV'])][['Number','Date','ABV']]
# start with the mean before dating
mBefore = [pd.isna(x) or (x.date() - dt.date(2018,1,1)).days < 0 for x in dh['Date']]
listBefore = np.sort(dh[mBefore]['ABV'])
mus, vs = rollingMeanVariance(listBefore)
mu0, v0 = mus[-1], vs[-1]
nBefore = len(dh[mBefore])
# Continue with the entries that have a usable date
mAfter = [pd.notna(x) and (x.date() - dt.date(2018,1,1)).days > 0 for x in dh['Date']]
dAfter = dh[mAfter].sort_values('Date').reset_index()
listAfter = dAfter.ABV
nAfter = len(listAfter)
mus, vs = rollingMeanVariance(listAfter, mu0, v0*nBefore, nBefore)
mus.insert(0,mu0)
vs.insert(0,v0)
len(mus), nAfter
errs = []
for i in range(len(mus)):
    errs.append(vs[i]**.5 / (i+nBefore))

fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0)
plt.xlabel('Year')
plt.ylabel('Average ABV')
# plt.errorbar([dAfter.Date[0]] + list(dAfter.Date), mus, errs, color='#f1bf4b', ls='',marker='.', markersize=.5, elinewidth=1, capthick=1, capsize=.5, lw=.5) 
ax.fill_between([dAfter.Date[0]] + list(dAfter.Date), np.array(mus)+errs, np.array(mus)-errs, color='#f1bf4b', lw=0)
p1 = ax.plot([dAfter.Date[0]] + list(dAfter.Date), mus, color='#751D1D', lw=.25, ls='--', label='bla')
ax.set_xlim([dt.date(2018,1,1), dt.date(2024,1,1)])
ax.set_ylim([5.5,6.1])
# p2 = ax.fill(np.NaN, np.NaN, color='#f1bf4b', lw=0)
# plt.legend([(p2[0],p1[0]),], ['Bla'])
applyUniformFont(ax,24)
saveAndListPlot("growthABV.pdf", "Time evolution of mean ABV")
plt.close()


###############################################################################
# bye
###############################################################################
plotList.close()