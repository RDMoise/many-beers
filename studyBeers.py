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
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import *
matplotlib.colormaps.register(name="beer", cmap=LinearSegmentedColormap.from_list("beer", colors=[niceColour('beeryellow'), niceColour('beerbrown')]))
import seaborn as sns

# df = pd.read_csv("1202beers.csv", delimiter=';', encoding='utf-8')
# df = pd.read_csv("1230beers.csv", delimiter=',', encoding='utf-8')
# df = pd.read_csv("1247beers.csv", delimiter=';', encoding='utf-8')
# df = pd.read_csv("1298beers.csv", delimiter=';', encoding='utf-8')
# df = pd.read_csv("1322beers.csv", delimiter=';', encoding='utf-8')
df = pd.read_csv("1407beers.csv", delimiter=';', encoding='utf-8')
plotList = open("plotList.md", "w")

def saveAndListPlot(plotname, description='test', url='many-beers/blob/main/'):
    plt.savefig(f"{plotname}")
    plotList.write(f"[{description}](https://nbviewer.org/github/RDMoise/{url}{plotname})\n\n")

###############################################################################
# histogram of ABVs
###############################################################################
hABV = np.histogram(df.ABV, np.linspace(-.5,18.5,20))
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
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
binc = .5*(bins[1:]+bins[:-1])
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0)
histplot(dg, bins, color='#f1bf4b', histtype="fill", alpha=.5)
histplot(dg, bins, color='#f1bf4b', histtype="step")
ax.set_xticks(binc)
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
eoms[np.isnan(eoms)]=0
dh = pd.concat([means,eoms], keys=['MeanABV','ErrorOnABV'], axis=1)
dh = dh.sort_values('MeanABV',ascending=False)
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0.01)
plt.errorbar(binc, dh.MeanABV, dh.ErrorOnABV, False, ls='', marker='.', markersize=7.5, elinewidth=1, capthick=1, capsize=2.5, color='#f1bf4b', label='Data')
ax.set_xticks(binc)
ax.minorticks_off()
plt.xlabel('Country code')
plt.ylabel("Mean ABV / country")
applyUniformFont(ax,20)
ax.set_xticklabels(dh.index, fontsize=10, rotation=45)
ax.set_ylim([3,13])
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False,right=False)

binningFine = np.linspace(bins[0], bins[-1], 10001)

# fit an exponential or a power-law
# def expo(x, N=1, l=0): return N*np.exp(-l*(x-1))
# def pwr(x, N=1, n=0): return N*x**(-n)
# minimiser = Minuit(LeastSquares(binc[dh.ErrorOnABV>0], dh[dh.ErrorOnABV>0].MeanABV, dh[dh.ErrorOnABV>0].ErrorOnABV, 
#                                 # expo), N=means[0], l=1)
#                                 pwr), N=means[0], n=-1e-8)
# result = minimiser.migrad()
# param_hesse = result.hesse()
# param_errors = result.errors
# print(result)

# plt.plot(binningFine, expo(binningFine, *minimiser.values), c=niceColour("beerbrown"), 
#          label="Best fit power law\n"+roundedLatex('Exponent', -minimiser.values['n'], minimiser.errors['n']))
#          # label=f"${{\\rm Exponent}}=-{minimiser.values['l']:.2f}\pm{minimiser.errors['l']:.2f}$")
# plotOrderedLegend([1,0], loc=1)

saveAndListPlot("countryABVs.pdf", "Mean ABV per country")
plt.close()


###############################################################################
# time evolution
###############################################################################
dates = [dt.datetime(2018,3,27,0,0), dt.datetime(2019,9,26,0,0), dt.datetime(2020,12,8,0,0), dt.datetime(2021,6,27,0,0), dt.datetime(2022,1,27,0,0), dt.datetime(2025,1,1,0,0)]
locations = ['CERN', 'London', 'RO', 'London', 'DENL']
def plotLandmarkDates(y, zorder=-99):
    for i in range(len(locations)):
        plt.axvline(x=dates[i], c='#9C4431', ls='--', zorder=zorder) # F1DD59 # EED892
        plt.text(dates[i]+dt.timedelta(days=15), y, locations[i], c='w', va='top', zorder=zorder, path_effects=[pe.Stroke(linewidth=.5, foreground='#9C4431')])

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
# fit a 4th order Chebychev polynomial
def cheby4(x, c0=0, c1=1, c2=0, c3=0, c4=0):
    return cheby2(x,c0,c1,c2) + \
    c3 * (4*x*x*x - 3*x) + \
    c4 * (8*x*x*x*x - 8*x*x - 1)
# fit a short-scale and a long-scale wave
# def coscos(x, A=0, T=0):#, B=0, t=0):
#     return A*np.cos(x/T)# + B*np.cos(x/t)

minimiser = Minuit(LeastSquares(np.array(dDates.x), np.array(dDates.Number), np.ones_like(dDates.Number), 
    cheby4), c0 = 600, c1 = .5, c2 = 1e-5, c3 = 1e-5, c4 = 1e-5,)
    # coscos), A=500, T=5)
result = minimiser.migrad()
param_hesse = result.hesse()
param_errors = result.errors
print(result)

fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {minimiser.fval:1.1e} / {len(dDates) - minimiser.nfit}",
]
for p, v, e in zip(minimiser.parameters, minimiser.values, minimiser.errors):
    if not minimiser.fixed[p]:
        fit_info.append(roundedLatex(p, v, e, scientific=bool(v < .01)))
        # pdgrounded = pdgRound(v, e)
        # fit_info.append(f"{p} = ${pdgrounded[0]:1.1e} \\pm {e:1.1e}$")

fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0)
plt.xlabel('Year')
plt.ylabel('Total beers')
plt.plot(np.sort(dDates.Date), cheby4(np.sort(dDates.x), *minimiser.values), color='#751D1D', lw=2.5, zorder=-99, label='Fit')
plt.scatter(dDates.Date, dDates.Number, color='#f1bf4b', s=10, label='Data')
plotOrderedLegend([1,0])
ax.set_xlim([dt.date(2018,1,1), dt.date(2024,1,1)])
applyUniformFont(ax,24)
plotLandmarkDates(1250)
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

errs = [] # error on the mean
for i in range(len(mus)):
    errs.append((vs[i]/(i+nBefore))**.5)

fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0)
plt.xlabel('Year')
plt.ylabel('Mean ABV [%]')
# plt.errorbar([dAfter.Date[0]] + list(dAfter.Date), mus, errs, color='#f1bf4b', ls='',marker='.', markersize=.5, elinewidth=1, capthick=1, capsize=.5, lw=.5) 
ax.fill_between([dAfter.Date[0]] + list(dAfter.Date), np.array(mus)+errs, np.array(mus)-errs, color='#f1bf4b', lw=0)
p1 = ax.plot([dAfter.Date[0]] + list(dAfter.Date), mus, color='#751d1d', lw=2.5, label='bla')
muABV, errABV = mus[-1], errs[-1]
plt.text(np.max(dAfter.Date), mus[-1], f"$({muABV:.2f}\pm{errABV:.2f})\\%$", color='#751d1d', va='center', rotation=-90, fontsize=14)
ax.set_xlim([dt.date(2018,1,1), dt.date(2025,1,1)])
ax.set_ylim([5.4, 6.2])

# Impact of country of residence
plotLandmarkDates(6.15)
applyUniformFont(ax,24)
saveAndListPlot("growthABV.pdf", "Time evolution of mean ABV")
plt.close()


###############################################################################
# Country - Location matrix
###############################################################################
mAfter = [pd.notna(x) and (x.date() - dt.date(2018,1,1)).days > 0 for x in df['Date']]
d = df[mAfter]
d['Timestamp'] = [x.timestamp() for x in d['Date']]
timestamps = [x.timestamp() for x in dates]
locations = ['CERN', 'London', 'RO', 'London', 'DENL']
d['Location'] = pd.cut(d['Timestamp'], bins=timestamps, labels=locations, ordered=False)
counts = d['Country'].value_counts()
d = d[d['Country'].isin(counts.index)].sort_values(by='Country', key=lambda x: x.map(counts), ascending=False)

xtab = pd.crosstab(d.Location, d.Country)
# sort by most popular countries and chronological order of locations
xtab = xtab[xtab.sum().sort_values(ascending=False).index].reindex(['CERN','London','RO','DENL'][::-1])
fig, ax = plt.subplots(figsize=(20*.66,9*.66), layout='constrained')
# plt.tight_layout()
plt.margins(x=0)
im = ax.imshow(xtab, aspect='auto', norm=LogNorm(), rasterized=True, cmap='beer')
locations = xtab.index.tolist()
countries = xtab.columns.tolist()

# Set axis labels and tick positions
plt.yticks(range(len(locations)), locations)
applyUniformFont(ax,16)
plt.xticks(range(len(countries)), countries, rotation=45, fontsize=10)
plt.minorticks_off()
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False, right=False, left=False)

cbar = fig.colorbar(im, shrink=1., pad=.01)
cbar.ax.tick_params(axis='both', pad=1)
applyUniformFont(cbar.ax,16)
saveAndListPlot("matrixCouLoc.pdf", "Matrix of beer origin vs. place of residence at the time")
plt.close()

###############################################################################
# Days between milestones (beer number x00)
###############################################################################
d100 = df[df.Number%100==0].sort_values(by='Number').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=50/(max(d100.Number)-min(d100.Number)+100))
plt.xlabel('Milestone')
plt.ylabel('Days from previous milestone')
plt.scatter(np.arange(min(d100.Number)+100, max(d100.Number)+1, 100), [x.days for x in d100['Date'].diff()][1:], color=niceColour('beeryellow'))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_ylim([0,300])
applyUniformFont(ax, 24)
saveAndListPlot("milestoneDelta.pdf", "Time between every hundreth beer")
plt.close()


###############################################################################
# How many beers were drunk in a given day
###############################################################################
dGrp = df.groupby('Date')
bdrunk = dGrp.size()
hdrunk = bdrunk.value_counts()
errs = np.sqrt(hdrunk.values)
binningFine = np.linspace(.5, max(hdrunk.index)+.5, 100*max(hdrunk.index)+1)

# fit an exponential or a power-law
# def expo(x, N=1, l=0): return N*np.exp(-l*(x-1))
def pwr(x, N=1, n=0): return N*x**(-n)

minimiser = Minuit(LeastSquares(hdrunk.index, hdrunk.values, errs, 
                                # expo), N=hdrunk[0][0], l=1)
                                pwr), N=hdrunk.values[0], n=1)
result = minimiser.migrad()
param_hesse = result.hesse()
param_errors = result.errors
print(result)

fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0)
ax.set_xlabel("Beers drunk in a day")
ax.set_ylabel("Number of occurrences")
plt.plot(binningFine, pwr(binningFine, *minimiser.values), c=niceColour("beerbrown"), 
         label="Best fit power law\n"+roundedLatex('Exponent', -minimiser.values['n'], minimiser.errors['n']))
         # f"${{\\rm Exponent}}=-{minimiser.values['n']:.2f}\pm{minimiser.errors['n']:.2f}$")
plt.errorbar(hdrunk.index, hdrunk.values, errs, c=niceColour("beeryellow"), ls='', marker='.', markersize=10., capsize=2.5, elinewidth=1, label='Data')
ax.set_xticks(np.linspace(1, max(hdrunk.index), max(hdrunk.index)))

plotOrderedLegend([1,0], loc=1)

ax.set_yscale('log')
applyUniformFont(ax,24)
saveAndListPlot("beerMultiplicity.pdf", "Histogram of beers drunk on the same date")
plt.close()


###############################################################################
# Histogram of volumes
###############################################################################
dGrp = df.groupby('Vol')
x, y = dGrp.size().index, dGrp.size().values
binw = .5*min(x[1:]-x[:-1])
bins = np.concatenate([x-binw, x+binw])
bins.sort()
values = np.zeros_like(bins)[1:]
values[np.digitize(x,bins)-1]=y
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
# plt.margins(x=50/(max(x)-min(x)+100))
plt.margins(x=0)
plt.xlabel('Volume [mL]')
plt.ylabel('Counts')
# plt.scatter(x, y, color=niceColour('beeryellow'))
plotBorderedHist((values, bins), color=niceColour('beeryellow'))
plt.xticks([100, 250, 330, 355, 440, 500, 660, 750], [100, 250, 330, 355, 440, 500, 660, 750], rotation=45)
plt.minorticks_off()
ax.set_yscale('log')
plt.tick_params(axis='both', top=False, right=False, left=False)
applyUniformFont(ax, 24)
saveAndListPlot("histVolumes.pdf", "Distribution of bottle/can sizes")
plt.close()


###############################################################################
# Volumes and mean ABVs
###############################################################################
dGrp = df.groupby('Vol')
dg = df.Vol.value_counts()
means = dGrp.ABV.mean()
eoms = dGrp.ABV.std() / dGrp.ABV.count()
# get rid of some of the outliers
means, eoms = means[2:], eoms[2:]
bins = np.linspace(0,len(means),len(means)+1)
fig, ax = plt.subplots(figsize=(16*.66,9*.66))
plt.tight_layout()
plt.margins(x=0.01)
plt.axhline(muABV, lw=1, c=niceColour('beerbrown'))
plt.axhline(muABV-errABV, lw=1, c=niceColour('beerbrown'), ls='--')
plt.axhline(muABV+errABV, lw=1, c=niceColour('beerbrown'), ls='--')
plt.errorbar(.5*(bins[1:]+bins[:-1]), means, eoms, False, ls='', marker='.', markersize=10, elinewidth=1, capthick=1, capsize=0, color='#f1bf4b')
ax.set_xticks(.5*(bins[1:]+bins[:-1]))
ax.minorticks_off()
plt.xlabel('Volume [mL]')
plt.ylabel("Mean ABV")
applyUniformFont(ax,20)
ax.set_xticklabels(means.index, rotation=45)
ax.set_ylim([4,8])
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False,right=False)
saveAndListPlot("volumeABVs.pdf", "Mean ABV per bottle/can size")
plt.close()

###############################################################################
# Country - Volume matrix
###############################################################################
d = df
xtab = pd.crosstab(d.Vol, d.Country)
# sort by most popular countries 
xtab = xtab[xtab.sum().sort_values(ascending=False).index][::-1]
fig, ax = plt.subplots(figsize=(20*.66,9*.66), layout='constrained')
plt.margins(x=0)
ax.set_xlabel('Country')
ax.set_ylabel('Volume [mL]')

# im = ax.imshow(xtab, aspect='auto', norm=LogNorm(), rasterized=True, cmap='beer')
sns.heatmap(xtab, mask=xtab==0, cmap='beer', square=True, vmin=0, linewidths=0, cbar_kws=dict(shrink=.75, pad=.01))
volumes = xtab.index.tolist()
countries = xtab.columns.tolist()

# Set axis labels and tick positions
# plt.yticks(range(len(volumes)), volumes)
applyUniformFont(ax,16)
# plt.xticks(range(len(countries)), countries, rotation=45, fontsize=10)
plt.xticks(np.arange(.5,len(countries),1), countries, rotation=45, fontsize=10)
plt.minorticks_off()
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False, right=False, left=False, bottom=False)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(axis='both', pad=1, labelsize=16)
cbar.ax.minorticks_off()
# cbar = fig.colorbar(im, shrink=1., pad=.01)
# cbar.ax.tick_params(axis='both', pad=1)
# applyUniformFont(cbar.ax,16)
saveAndListPlot("matrixCouVol.pdf", "Matrix of beer origin vs. bottle size")
plt.close()

# zoom-in of most popular countries and volumes
xtab = xtab.reindex([750, 660, 500, 355, 330, 300]).T[:10].T
# normalise countries
h = [x/sum(x) for x in xtab.values]
volumes = xtab.index.tolist()
countries = xtab.columns.tolist()
fig, ax = plt.subplots(figsize=(20*.66,9*.66), layout='constrained')
plt.margins(x=0)
ax.set_xlabel('Country')
ax.set_ylabel('Volume [mL]')

im = ax.imshow(h, aspect='auto', rasterized=True, cmap='beer', norm=LogNorm())
# im = ax.pcolormesh(h, cmap='beer', cmin=.01, rasterized=True)

# Set axis labels and tick positions
plt.yticks(range(len(volumes)), volumes)
applyUniformFont(ax,16)
plt.xticks(range(len(countries)), countries, rotation=45)
plt.minorticks_off()
plt.tick_params(axis='x', pad=1)
plt.tick_params(axis='both', top=False, right=False, left=False)

cbar = fig.colorbar(im, shrink=1., pad=.01)
cbar.ax.tick_params(axis='both', pad=1)
applyUniformFont(cbar.ax,16)
saveAndListPlot("matrixCouVolZoom.pdf", "Matrix of most popular beer origins vs. bottle sizes")
plt.close()

###############################################################################
# bye
###############################################################################
plotList.close()