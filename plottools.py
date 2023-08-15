from os import environ, system
import subprocess
import numpy as np
import hist as hist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.switch_backend('agg') # when on batch
import matplotlib
from mplhep import histplot, style
style.use(style.LHCb2)
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches

ROOTDIR = environ.get('ROOTDIR')
TUPLEDIR = environ.get('TUPLEDIR')
HPCWORK = environ.get('HPCWORK')

def niceColour(colourname):
    colours = {
        "RKorange":     (0.992, 0.6823, 0.3804), # f1b16e
        "RKlblue":      (119./255, 162./255, 251./255),
        "RKred":        (0.8431, 0.098, 0.1098), # D7191C
        "RKdblue":      (0.1725, 0.4824, 0.7137),
        "RKyellow":     (1., 1., 191/255.),
        "RKgreen":      (136./255,221./255,157./255),
        "RKdpurple":    (99./255,99./255,151./255),
        "RKlpurple":    (151./255,151./255,252./255),
        "RKturquoise":  (143./255, 222./255, 234./255),
        "seqGreen":     ['#f7fcfd','#e5f5f9','#ccece6','#99d8c9','#66c2a4','#41ae76','#238b45','#006d2c','#00441b'][2:],
        "seqBlue":      ['#fff7fb','#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858'],
        "seqPurple":    ['#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#54278f','#3f007d'],
        "seqOrange":    ['#ffffe5','#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404','#662506'],
        "cbrDark2_3":   ['#1b9e77','#d95f02','#7570b3'], # green orange purple
        "qualSet2":     ['#66c2a5','#fc8d62','#8da0cb','#a6d854','#e5c494','#b3b3b3'],
        "myrtle":       '#32746d', # greenish
        "claret":       '#710627', # dark red
        "emerald":      '#099a4f', 
        "airblue":      '#26547c', 
        "oniblue":      '#287095', 
        "onicyan":      '#7dc6c2',
        "oniturquoise": '#15a1b0',
        "onidred":      '#c24477',
        "onilred":      '#d682a6',
        "onilgrey":     '#89abb7',
        "onidgrey":     '#71939d',
        "oniyellow":    '#ebc892',
        "onidpurple":   '#393674',
        "onilpurple":   '#666da3',
        "LHCbdblue":    '#0057a7',
        "LHCblblue":    '#d2eefa',
        "LHCbred":      '#de2f29',
        "dgreenblue":   '#2e8f9e',
        "alexagreen":   '#348a82',
        "beeryellow":   '#f1bf4b',
        "beerbrown":    '#751d1d',
        # "qualSet2":     ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'],
    }
    return colours[colourname]


def saveplot(name):
    '''save matplotlib figure and prepress it (lossless compression, font embed)'''
    plt.savefig(f"{name}_temp.pdf")
    subprocess.run(f"gs -o {name} -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite {name}_temp.pdf".split(" "), stdout=subprocess.DEVNULL)
    system(f"rm -f {name}_temp.pdf")


class PlotOptions:
    def __init__(self, name, variable, binning, xlabel='', ylabel='', logx=False, logy=False, legendloc='best', figsize=(16*.66, 9*.66)):
        self.name = name
        self.variable = variable
        self.binning = binning
        self.binw = (binning[-1] - binning[0]) / (len(binning) - 1)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.logx = logx
        self.logy = logy
        self.legendloc = legendloc
        self.figsize = figsize

    def initPlot(self, density=False):
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()
        plt.margins(x=0)
        plt.xlabel(self.xlabel)
        if density: plt.ylabel(self.ylabel_density())
        else: plt.ylabel(self.ylabel)
        if self.logx: ax.set_xscale('log')
        if self.logy: ax.set_yscale('log')
        return fig, ax

    def ylabel_density(self): return rf"$1/N$ d$N/$d{self.xlabel}"
    def ylabel_candidates(self, unit=''): return f"Candidates / {self.binw}{unit}"


class pdgRound:
    '''Given a value and an error, round and format them according to the PDG rules for significant digits
       Source: https://github.com/gerbaudo/python-scripts/blob/master/various/pdgRounding.py'''
    def __init__(self, value, error):
        self.value = value
        self.error = error
    # def threeDigits(self):
        '''extract the three most significant digits and return them as an int'''
        self.threeDigits = int(("%.2e"%float(self.error)).split('e')[0].replace('.','').replace('+','').replace('-',''))
    # def nSignificantDigits(self):
        assert self.threeDigits<1000,"three digits (%d) cannot be larger than 10^3"%self.threeDigits
        if self.threeDigits<101: self.nSignificantDigits = 2 # not sure
        elif self.threeDigits<356: self.nSignificantDigits = 2
        elif self.threeDigits<950: self.nSignificantDigits = 1
        else: self.nSignificantDigits = 2
        self.extraRound = 1 if self.threeDigits>=950 else 0
    def frexp10(self, number):
        '''convert to mantissa+exp representation (same as frex, but in base 10)'''
        numberStr = ("%e"%float(number)).split('e')
        return float(numberStr[0]), int(numberStr[1])
    # def nDigitsValue(self, expVal, expErr, nDigitsErr):
    #     '''compute the number of digits we want for the value, assuming we keep nDigitsErr for the error'''
    #     return expVal-expErr+nDigitsErr
    def formatNumber(self, number, nDigits, extraRound=0):
        '''Format the value; extraRound is meant for the special case of threeDigits>950'''
        exponent = self.frexp10(number)[1]
        roundAt = nDigits-1-exponent - extraRound
        nDec = roundAt if exponent<nDigits else 0
        nDec = max([nDec, 0])
        return ('%.'+str(nDec)+'f')%round(number,roundAt)
    def print(self, scientific=False):
        expVal, expErr = self.frexp10(self.value)[1], self.frexp10(self.error)[1]
        norm = expVal if scientific else 0.
        return (self.formatNumber(self.value * 10**(-norm), self.nSignificantDigits + expVal - expErr, self.extraRound),
                self.formatNumber(self.error * 10**(-norm), self.nSignificantDigits, self.extraRound), norm)


def roundedLatex(name, value, error, scientific=False):
    '''Convert the output of pdgRound to a nicely-displayable piece of latex'''
    val, err, norm = pdgRound(value, error).print(scientific)
    if scientific: return f"{name}$=({val}\pm{err})\\times10^{{{norm}}}$"
    return f"{name}$={val}\pm{err}$"


def plotWithPulls(plotname, hists, styles, pulls, kwargs,
    xlabel, ylabel="A.U.", 
    kwargsPulls=[{'color': niceColour('onilred'), 'histtype': 'fill', 'alpha': 1}],
    legendloc='best', legendsize=24, legendtitle=None,
    directory=f'{ROOTDIR}plots/', height_ratios=[4,1]):
    fig = plt.figure(figsize=[10,7.5])
    gs = matplotlib.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=height_ratios)

    ax = plt.subplot(gs[0])
    plt.tight_layout()
    plt.margins(x=0)
    for i in range(len(hists)):
        if styles[i] == 'histplot':         histplot(*hists[i], **kwargs[i])
        elif styles[i] == 'histplotclean':  hideEmptyBins(hists[i], **kwargs[i])
        elif styles[i] == 'pltplot':        plt.plot(*hists[i], **kwargs[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legendloc, fontsize=legendsize, title=legendtitle).get_title().set_fontsize(legendsize)

    ax1 = plt.subplot(gs[1])
    plt.tight_layout()
    plt.margins(x=0)
    plt.axhline(y=0 , color='black', linewidth=2, alpha=1.) 
    plt.axhline(y=-3, color='black', linestyle='--', linewidth=1, alpha=1.) 
    plt.axhline(y=3, color='black', linestyle='--', linewidth=1, alpha=1.) 
    if type(pulls) == tuple: Hpulls = histplot(pulls, **kwargsPulls[0]) # for backwards compatibility
    else:
        for i in range(len(pulls)): Hpulls = histplot(*pulls[i], **kwargsPulls[i])
    plt.ylim([-5,5])
    plt.tight_layout()
    plt.margins(x=0)
    ax1.set_xticklabels([])
    plt.ylabel("Pulls")
    applyUniformFont(ax, 32)
    applyUniformFont(ax1, 32)
    if plotname != None: plt.savefig(f"{directory}{plotname}.pdf")

    return fig, gs


def makeWeightedPlot2D(plotname, dataX, dataY, weights, bins, xlabel, ylabel="A.U.", label='', directory=f'{ROOTDIR}plots/', cmin=1e-8, cmap='cividis'):
    fig, ax = plt.subplots(figsize=(9, 8))
    h, xedges, yedges, im = ax.hist2d(
                            dataX, dataY,
                            bins,
                            weights=weights,
                            label=label,
                            cmin=cmin, 
                            cmap=cmap, 
                            rasterized=True
                            )
    # plt.legend()
    plt.colorbar(im)
    # fig.colorbar(h[3], ax=ax)
    plt.text(.05,.9, label, transform=ax.transAxes, fontsize=24)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.margins(x=0)
    if plotname != None: plt.savefig(f"{directory}{plotname}.pdf")
    # plt.close()

    return h, ax, xedges, yedges, im


def makeWeightedPlot2DLog(plotname, dataX, dataY, weights, bins, xlabel, ylabel="A.U.", label='', directory=f'{ROOTDIR}plots/', norm=LogNorm(), cmin=1e-8, cmap='cividis'):
    fig, ax = plt.subplots(figsize=(9, 8)) # , layout='constrained'
    h, xedges, yedges, im = ax.hist2d(
                            dataX, dataY,
                            bins,
                            weights=weights,
                            label=label,
                            cmin=cmin, 
                            norm=LogNorm(),
                            cmap=cmap, 
                            rasterized=True
                            )
    # plt.legend()
    plt.colorbar(im)
    # fig.colorbar(h[3], ax=ax)
    plt.text(.05,.9, label, transform=ax.transAxes, fontsize=24)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.margins(x=0)
    if plotname != None: plt.savefig(f"{directory}{plotname}.pdf")
    # plt.close()

    return h, ax, xedges, yedges, im


def plotNormalisedSlicesLog(dataX, dataY, binsX, binsY, xlabel=None, ylabel=None, figsize=(16*.6,9*.6)):
    h0 = hist.Hist(hist.axis.Variable(binsX), 
                   hist.axis.Variable(binsY))
    h0.fill(dataX, dataY)
    h = np.array([x/sum(x) for x in h0.values()]) # normalise X slices to unity
    h[np.isnan(h)] = 0

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    plt.margins(x=0)
    ax.set_xlim([binsX[0], binsX[-1]])
    ax.set_ylim([binsY[0], binsY[-1]])
    im = ax.pcolormesh(*h0.axes.edges.T, h.T, cmap="cividis", norm=LogNorm(), rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cumulative_sum = [np.cumsum(x) for x in h]
    split_index = [np.argmax(s >= np.sum(x)/2) for x,s in zip(h,cumulative_sum)]
    medians = [binsY[i] for i in split_index]

    return h, h0, ax, im, medians


def makeKiwiPlot(plotname, hist, bins, errs=False, ylims=None, ring=None, title=None):
    fig = plt.gcf()
    img = plt.imread(f"{ROOTDIR}plots/velomodule.png")
    ax_img = fig.add_axes([0, 0, 1, 1], label='ax_img')
    ax_img.imshow(img)
    ax_img.axis('off')
    ax = fig.add_axes([0.008,0.035,.95,.95], projection='polar', label='ax')
    ax.patch.set_alpha(0)
    if ring != None:
        ax.plot(np.linspace(0, 2*np.pi,100), np.ones(100)*ring, linewidth=5)
    for i in range(len(bins)-1):
        plt.plot(np.linspace(bins[i], bins[i+1], 20)[1:-1], hist[i]*np.ones(20)[1:-1], color='black', linewidth=7.5)
    bars = ax.bar(bins[:-1], hist, yerr=errs, align='edge', width=np.diff(bins), fill=False, linewidth=0, alpha=.2, error_kw={'linewidth': 7.5})
    if ylims != None: ax.set_ylim(ylims)
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    if title!=None: plt.xlabel(title, loc='center')
    applyUniformFont(ax, 40)
    ax.set_theta_offset(np.pi)
    ax.spines['polar'].set_visible(False)

    plt.gcf().canvas.draw()
    angles = [0, 45, 0, -45, 0, 45, 0, 315]
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        labeltext = label.get_text() 
        if '180' in labeltext: labeltext = '    '+labeltext
        if len(labeltext) == 2: labeltext = labeltext+'  '
        lab = ax.text(x,y,labeltext,
                      transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va(), fontsize=40)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])
    plt.savefig(plotname)

    return fig, ax


def applyUniformFont(ax, fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                  ax.get_xticklabels() + ax.get_yticklabels() +
                  ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True)):
        item.set_fontsize(fontsize)


def plotOrderedLegend(order=None, handles=None, labels=None, fontsize=20, loc='best', title=None, titlesize=20):
    if handles== None or labels== None: handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=fontsize, loc=loc, title=title)
    leg.get_title().set_fontsize(titlesize)
    return leg


def getSubplotExtent(ax, hpad=0, vpad=0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    # items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + hpad, 1.0 + vpad)


# def hideEmptyBins(H, kwargs={'color': 'black', 'linestyle': '', 'marker': '.', 'markersize': 10., 'elinewidth': 1}):
def hideEmptyBins(H, color='black', linestyle='', marker='.', markersize=10., capsize=2.5, elinewidth=1, label=None, binwnorm=None, density=False):
    '''hide the error bars of bins with 0 entries'''
    h = histplot(H, yerr=True, density=density, histtype="errorbar", alpha=0, binwnorm=binwnorm)[0]
    binning = [x for x in h.errorbar][0].get_xdata()
    binvals = [x for x in h.errorbar][0].get_ydata()
    binlos = [x for x in h.errorbar][1][0].get_ydata()
    binhis = [x for x in h.errorbar][1][1].get_ydata()
    for i in range(len(binvals)):
        if binvals[i] <= 0:
            binlos[i] = binvals[i]
            binhis[i] = binvals[i]
    plt.errorbar(binning, binvals, yerr=(binvals-binlos, binhis-binvals), color=color, linestyle=linestyle, marker=marker, markersize=markersize, capsize=capsize, elinewidth=elinewidth, label=label)
    return h


def plotRectangles(H, ec=niceColour('onidgrey'), fc=niceColour('onilgrey'), lw=2.5, zorder=-99, binwnorm=None, density=False, label=None):
    '''plot a histogram as coloured rectangles'''
    h = histplot(H, yerr=True, density=density, binwnorm=binwnorm, histtype="errorbar", alpha=0)[0]
    binning = [x for x in h.errorbar][0].get_xdata()
    binvals = [x for x in h.errorbar][0].get_ydata()
    binlos = [x for x in h.errorbar][1][0].get_ydata()
    binhis = [x for x in h.errorbar][1][1].get_ydata()
    nBins = len(H[0])
    for i in range(nBins):
        rectangle = plt.Rectangle((H[1][i], binlos[i]), H[1][i+1]-H[1][i], binhis[i]-binlos[i], lw=lw, fc=fc, ec=ec, zorder=zorder, label=label if not i else None)
        plt.gca().add_patch(rectangle)
        # plt.gca().add_patch(plt.Rectangle((H[1][i], binvals[i]), H[1][i+1]-H[1][i], 1., fc='black', ec='black'))
    return h

def plotBorderedHist(h, handles=[], labels=[], color=niceColour('oniblue'), alpha=.3, label=None, kwargs={}, binwnorm=None, density=False):
    histplot(h, color=color, **kwargs, density=density, binwnorm=binwnorm, histtype="fill", alpha=alpha, label=label)  
    histplot(h, color=color, **kwargs, density=density, binwnorm=binwnorm, histtype="step") 
    pFill = mpatches.Patch(fc=color, **kwargs, alpha=alpha, label=label)
    pStep = mpatches.Patch(ec=color, **kwargs, color='none')
    handles.append((pStep, pFill))
    labels.append(label)
    return handles, labels


def getRatioBinning(binvar, binchoice='', logscale=True):
    '''Standard setup for helium ratio as a function of kinematics: bins for several variables (and variations) alongside axis labels'''
    key = binvar+binchoice
    # logspace base can be omitted if using log10
    binsPlot = {
        'PT':               np.concatenate((np.logspace(np.log10(.6), np.log10(6), 41), [6.5, 7., 10.])),
        'PTwide':           np.concatenate((np.logspace(np.log10(.5), np.log10(1), 3), np.logspace(np.log10(1), np.log10(2), 7)[1:], np.logspace(np.log10(2), np.log10(3), 4)[1:], [10.])),
        'PTITonly':         np.concatenate((np.logspace(np.log10(.5), np.log10(4), 31), [5., 10.])),
        'PT1bin':           np.array([.5, 10.]),
        'P':                np.logspace(np.log10(5), np.log10(100), 52),
        'PITonly':          np.concatenate((np.logspace(np.log10(3), np.log10(6), 2), np.logspace(np.log10(6), np.log10(100), 25)[1:])),
        'P1bin':            np.array([3, 1e5]),
        'ProbNNe':          np.logspace(-8,-1,36),
        'ProbNNeITonly':    np.logspace(-8,-1,15),
        'ProbNNpi':         np.logspace(-2,0,41),
        'IPCHI2_OWNPV':     np.logspace(-5, 2, 10),
        'IPCHI2_OWNPVfull': [1e-5, 1e-2, .1, 1, 2.5, 5, 10, 20, 1e5],
        'P_ETABin1':        np.logspace(np.log10(3), np.log10(30), 11),
        'P_ETABin2':        np.logspace(np.log10(3), np.log10(60), 11),
        'P_ETABin3':        np.logspace(np.log10(5), np.log10(100), 11),
        'P_ETABin4':        np.logspace(np.log10(8), np.log10(100), 11),
        'P_ETABin5':        np.logspace(np.log10(15), np.log10(100), 11),
        'P_ETABin6':        np.logspace(np.log10(25), np.log10(100), 11),
        # 'PT':               np.concatenate((np.logspace(np.log10(.5), np.log10(6), 41), [6.5, 7., 10.])),
        # 'P':                np.concatenate(([3], np.logspace(np.log10(4), np.log10(5), 4), np.logspace(np.log10(5), np.log10(100), 52)[1:])),
    } if logscale else \
    {
        'ETA1bin':          np.array([2., 5]),
        'ETA':              np.concatenate((np.linspace(2., 4.7 , 57), [5.])),
        'ETAITonly':        np.concatenate(([2], np.linspace(2.5, 4.7 , 29), [5.])),
        'ETAbeampipe':      np.array([2., 2.5, 3., 3.5, 3.9, 4.2, 4.4, 5]),
        'PHI':              np.array([-180., -160., -145, -125, -110, -70, -55, -35, -15., 0., 15., 35, 55, 70, 110, 125., 145., 160., 180.]),
        'lnIPCHI2':         np.linspace(-5, 4, 19),
        'lnIPCHI2syst':     np.concatenate(([-10], np.linspace(0, 5, 26))),
        'lnIPCHI2syst2':    np.array([-10, 0, 1., 1.5, 2., 2.5, 3., 4., 5.]),
        'lnIPCHI2syst3':    np.linspace(0, 5, 6),
        'IPCHI2_OWNPVfull': np.linspace(0, 1e4, 10),
        # 'Rapidity':         np.concatenate(([0.9], np.linspace(1.1,4.3,65), [4.6])),
        'Rapidity':         np.concatenate((np.linspace(1.3,4.3,62), [4.6])),
        'RapidityITonly':   np.concatenate(([0.9], np.linspace(1.5,4.3,29), [4.6])),
        'Rapidity1bin':     np.array([.9, 4.6]),
        'LLD_TT':           np.linspace(-12, 6, 36),
        'LLD_VELO':         np.linspace(-12, 6, 36),
        'ProbNNecoarse':    np.array([0,.1,.5,.9,1.]),
        'ProbNNpicoarse':   np.array([0,.1,.5,.9,1.]),
        'ProbNNe2bins':     np.array([0,.1,1.]),
        'ProbNNpi2bins':    np.array([0,.5,1.]),
        'TRACK_GhostProb':  np.array([0,.15,.5,1]),
        'nTracks':          np.linspace(0,1000,51),
        'PIDe':             np.concatenate(([-12], np.linspace(-10,1,23), [3, 10.])),
    }
    if logscale:
        binsPlot['PTalt'] = np.concatenate((binsPlot['PT'][:1],[(binsPlot['PT'][i]*binsPlot['PT'][i+1])**.5 for i in range(1,len(binsPlot['PT'])-2)],binsPlot['PT'][-1:]))
        binsPlot['Palt'] = np.concatenate((binsPlot['P'][:1],[(binsPlot['P'][i]*binsPlot['P'][i+1])**.5 for i in range(1,len(binsPlot['P'])-2)],binsPlot['P'][-1:]))
    else:
        binsPlot['ETAalt'] = np.concatenate((binsPlot['ETA'][:1],[(binsPlot['ETA'][i]+binsPlot['ETA'][i+1])*.5 for i in range(1,len(binsPlot['ETA'])-2)],binsPlot['ETA'][-1:]))
        binsPlot['Rapidityalt'] = np.concatenate((binsPlot['Rapidity'][:1],[(binsPlot['Rapidity'][i]+binsPlot['Rapidity'][i+1])*.5 for i in range(1,len(binsPlot['Rapidity'])-2)],binsPlot['Rapidity'][-1:]))
    bins = {
        'PT':               1e3*binsPlot[key],
        'P':                1e3*binsPlot[key],
        'IPCHI2_OWNPV':     binsPlot[key],
        'ProbNNe':          binsPlot[key],
        'ProbNNpi':         binsPlot[key],
    } if logscale else \
    {
        'PT':               1e3*binsPlot[key],
        'P':                1e3*binsPlot[key],
        'ETA':              binsPlot[key],
        'PHI':              np.pi/180*binsPlot[key],
        'lnIPCHI2':         binsPlot[key],
        'IPCHI2_OWNPV':     np.linspace(0, 1e4, 10),
        'Rapidity':         binsPlot[key],
        'LLD_TT':           binsPlot[key],
        'LLD_VELO':         binsPlot[key],
        'ProbNNe':          binsPlot[key],
        'ProbNNpi':         binsPlot[key],
        'TRACK_GhostProb':  binsPlot[key],
        'nTracks':          binsPlot[key],
        'PIDe':             binsPlot[key],
    }
    labelVar = {
        'PT':               r'$p_{\rm T}\,/\,1\,{\rm GeV}$',
        'P':                r'$p\,/\,1\,{\rm GeV}$',
        'ETA':              r'$\eta$',
        'PHI':              r'$\phi$ [deg]',
        'IPCHI2_OWNPV':     r'$\chi^2_{\rm IP}$',
        'lnIPCHI2':         r'$\ln \chi^2_{\rm IP}$',
        'Rapidity':         r'$y$',
        'LLD_TT':           r'$\Lambda_{\rm LD}^{\rm TT}$',
        'LLD_IT':           r'$\Lambda_{\rm LD}^{\rm IT}$',
        'LLD_VELO':         r'$\Lambda_{\rm LD}^{\rm VELO}$',
        'ProbNNe':          r'${\rm ProbNNe}$',
        'ProbNNpi':         r'${\rm ProbNNpi}$',
        'TRACK_GhostProb':  r'${\rm ProbGhost}_{\rm track}$',
        'nTracks':          r'${\rm nTracks}$',
        'PIDe':             r'${\rm DLL}_{e}$',
    }

    return binsPlot[key], bins[binvar], labelVar[binvar]

def adjustAxisTicks(ax, var):
    majorTicks = {
        'P':        [5, 10, 100],
        'PGeV':     [5, 10, 100],
        'PT':       [.6, 1, 10],
        'PTGeV':    [.6, 1, 10],
        'ETA':      [2,3,4,5],
        'Rapidity': [1,2,3,4,5],
    }[var]
    minorTicks = {
        'P':        [20,30,40,50],
        'PGeV':     [20,30,40,50],
        'PT':       [2,3,4,5],
        'PTGeV':    [2,3,4,5],
        'ETA':      None,
        'Rapidity': None,
    }[var]

    ax.set_xticks(majorTicks, minor=True)
    ax.set_xticklabels(majorTicks, minor=True)
    if not minorTicks == None:
        ax.set_xticks(minorTicks, minor=True)
        ax.set_xticklabels(minorTicks, minor=True)