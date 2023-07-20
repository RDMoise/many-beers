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

import matplotlib.pyplot as plt
plt.switch_backend('agg') # when on batch
import matplotlib
from mplhep import histplot, style
style.use(style.LHCb2)
import matplotlib.dates as mdates

df = pd.read_csv("1202beers.csv", delimiter=';', encoding='utf-8')

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
plt.savefig("histABV.pdf")
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
plt.savefig("histCountries.pdf")
ax.set_yscale('log')
plt.savefig("histCountries_log.pdf")
plt.close()


###############################################################################
# time evolution
###############################################################################
df['Date'] = pd.to_datetime(df['Date'], errors='coerce',format='%d/%m/%Y')
fig, ax = plt.subplots(figsize=(16*.6,9*.6))
plt.tight_layout()
plt.margins(x=0)
plt.scatter(df.Date, df.Number, color='#f1bf4b', s=20)
plt.xlabel('Year')
plt.ylabel('Total beers')
ax.set_xlim([dt.date(2018,1,1), dt.date(2024,1,1)])
applyUniformFont(ax,24)
plt.savefig("growth.pdf")
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
plt.ylabel("Average ABV / country")
applyUniformFont(ax,20)
ax.set_xticklabels(dh.index, fontsize=10, rotation=45)
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False,right=False)
plt.savefig("countryABVs.pdf")
plt.close()