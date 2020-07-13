'''
Copyleft May 11, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from __future__ import print_function

import matplotlib as mpl
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import UTILS.Util as utl
import UTILS.Hyperoxia as htl
from UTILS import *
def setStyle(style="darkgrid", lw=2, fontscale=1, fontsize=10):
    sns.axes_style(style)
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': fontsize});
    mpl.rc('text', usetex=True)
    sns.set_context(font_scale=fontscale, rc={"lines.linewidth": lw})


class PLOS:
    max_width = 7.5
    min_width = 2.6
    max_height = 8.75
    dpi = 300
    extention = 'tiff'

    @staticmethod
    def get_figsize(width=None, height=None):
        if width is not None:
            width = min(width, PLOS.max_width)
            return (width, 2. / 3 * width)
        else:
            return (6, 4)


def get_axis_limits(ax, upper=True):
    return ax.get_xlim()[(0, 1)[upper]], ax.get_ylim()[(0, 1)[upper]]


def annotate(comment, loc=1, fontsize=26, xpad=0.05, ypad=0.05, ax=None, axtoplot=None):
    """
    Args:
        comment: text
    """
    if ax is None: ax = plt.gca()
    if axtoplot is None: axtoplot = ax
    xrang = getAxRange(ax, 0)
    yrang = getAxRange(ax, 1)
    xy = get_axis_limits(ax, upper=False)[0] + xpad * xrang, get_axis_limits(ax)[1] - ypad * yrang
    axtoplot.annotate(comment, xy=xy, xycoords='data', size=fontsize, horizontalalignment='left',
                       verticalalignment='top')


def getAxRange(ax, axi=0):
    return get_axis_limits(ax, upper=True)[axi] - get_axis_limits(ax, upper=False)[axi]

def getColorMap(n):
    colors = ['darkblue', 'r', 'green', 'darkviolet', 'k', 'darkorange', 'olive', 'darkgrey', 'chocolate', 'rosybrown',
              'gold', 'aqua']
    if n == 1: return [colors[0]]
    if n <= len(colors):
        return colors[:n]
    return [mpl.cm.jet(1. * i / n) for i in range(n)]


def getMarker(n, addDashed=True):
    markers = np.array(['o', '^', 's',  'D', 'd', 'h', '*', 'p','v', '3',  'H', '8','<','2', '4'])[:n]# '<', '>'
    if addDashed: markers = map(lambda x: '--' + x, markers)
    return markers

# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':30}) ;
# mpl.rc('text', usetex=True)

def addGlobalPOSIndex(df,chroms):
    if df is not None:
        df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
        df.set_index('gpos', inplace=True);
        df.sort_index(inplace=True)


def GenomeChromosomewise(df, candSNPs=None, genes=None, axes=None,outliers=None):
    markerSize = 6
    fontsize = 6
    chrsize = df.reset_index().groupby('CHROM').POS.max()
    if axes is None:
        if chrsize.shape[0]>1:
            _, axes = plt.subplots(int(np.ceil(chrsize.shape[0] / 2.)), 2, sharey= True, dpi=200, figsize=(12, 6));
            ax = axes.reshape(-1)
        else:
            ax = [plt.subplots(1,1, sharey=True, dpi=200, figsize=(10, 6))[1]]

    for j, (chrom, a) in enumerate(df.groupby(level=0)):
        if candSNPs is not None:
            try:
                candSNPs.loc[chrom]
                for pos in candSNPs.loc[chrom].index.values:
                    ax[j].axvline(pos, color='r', linewidth=0.5, alpha=0.5)
                    ax[j].annotate(
                        '{:.0f},{:.2f}'.format(candSNPs['rank'].loc[(chrom, pos)], candSNPs.nu0.loc[(chrom, pos)]),
                        xy=(pos, a.max()), xytext=(pos, a.max()), fontsize=fontsize - 2)

            except:
                pass

        if genes is not None:
            try:
                X = genes.loc[chrom]
                if len(genes.loc[chrom].shape) == 1:
                    X = pd.DataFrame(X).T
                for _, row in X.iterrows():
                    ax[j].fill_between([row.start, row.end], a.min(), a.max(), color='r')
                    ax[j].annotate(row['name'], xy=(row.start, a.max()), xytext=(row.start, a.max()),
                                   fontsize=fontsize - 2)


            except:
                pass

        ax[j].scatter(a.loc[chrom].index, a.loc[chrom], s=markerSize, alpha=0.8, edgecolors='none')

        if outliers is not None:
            try:
                ax[j].scatter(outliers.loc[chrom].index, outliers.loc[chrom], s=markerSize, c='r', alpha=0.8, edgecolors='none')
            except:
                pass

        setSize(ax[j], fontsize)
        # ax[j].set_xlim([-1000, chrsize[chrom] + 1000])
        # ax[j].set_title(chrom, fontsize=fontsize+2)
        ax[j].set_xlabel(chrom , fontsize=fontsize + 6)
        # annotate(chrom, ax=ax[j],fontsize=fontsize+4)
        ax[j].locator_params(axis='x', nbins=10)
    plt.tight_layout(pad=0.1)
    plt.gcf().subplots_adjust(bottom=0.1)


def Manhattan(data, columns=None, names=None, fname=None, colors=['black', 'gray'], markerSize=20, ylim=None, show=True,
              std_th=None, top_k=None, cutoff=None, common=None, Outliers=None, shade=None, fig=None, ticksize=16,
              sortedAlready=False,lw=1,axes=None,shareY=False,color=None,CHROMLen=None,alpha=0.4,shade2=None):
    def reset_index(x):
        if x is None: return None
        if 'CHROM' not in x.columns.values:
            return x.reset_index()
        else:
            return x
    if type(data) == pd.Series:
        DF = pd.DataFrame(data)
    else:
        DF = data

    if columns is None: columns=DF.columns
    if names is None:names=columns

    df = reset_index(DF)
    Outliers = reset_index(Outliers)
    if not sortedAlready: df = df.sort_index()
    if not show:
        plt.ioff()
    from itertools import cycle
    def plotOne(b, d, name, chroms,common,shade,shade2,ax):
        a = b.dropna()
        c = d.loc[a.index]
        if ax is None:
            ax=plt.gca()
        def plotShade(shade,c):
            for _ ,  row in shade.iterrows():
                if shareY:
                    MAX = DF.replace({np.inf: None}).max().max()
                    MIN = DF.replace({-np.inf: None}).min().min()
                else:
                    MAX = a.replace({np.inf: None}).max()
                    MIN = a.replace({-np.inf: None}).min()
                ax.fill_between([row.gstart, row.gend], MIN,MAX, color=c, alpha=alpha)
                if 'name' in row.index:
                    if row['name'] == 1: row.gstart -=  1e6
                    if row['name']== 8: row.gstart=row.gend+1e6
                    if row['name'] == 'LR2.1': row.gstart -=  2e6
                    if row['name'] == 'LR2.2': row.gstart += 1e6
                    xy=(row.gstart, (MAX*1.1))
                    try:shadename=row['name']
                    except:shadename=row['gene']
                    ax.text(xy[0],xy[1],shadename,fontsize=ticksize+2,rotation=0,ha= 'center', va= 'bottom')
        if shade is not None: plotShade(shade,c='b')
        if shade2 is not None: plotShade(shade2,c='r')

                    # ax.annotate('   '+shadename,
                    #             # bbox=dict(boxstyle='round,pad=1.2', fc='yellow', alpha=0.3),
                    #             xy=xy, xytext=xy, xycoords='data',horizontalalignment='center',fontsize=ticksize,rotation=90,verticalalignment='bottom')

        ax.scatter(a.index, a, s=markerSize, c=c, alpha=0.8, edgecolors='none')

        outliers=None
        if Outliers is not None:
            outliers=Outliers[name].dropna()
        if cutoff is not None:
            outliers = a[a >= cutoff[name]]
        elif top_k is not None:
            outliers = a.sort_values(ascending=False).iloc[:top_k]
        elif std_th is not None:
            outliers = a[a > a.mean() + std_th * a.std()]
        if outliers is not None:
            if len(outliers):
                ax.scatter(outliers.index, outliers, s=markerSize, c='r', alpha=0.8, edgecolors='none')
                # ax.axhline(outliers.min(), color='k', ls='--',lw=lw)


        if common is not None:
            for ii in common.index: plt.axvline(ii,c='g',alpha=0.5)

        ax.axis('tight');
        if CHROMLen is not None:
            ax.set_xlim(0, CHROMLen.sum());
        else:
            ax.set_xlim(max(0,a.index[0]-10000), a.index[-1]);
        setSize(ax,ticksize)
        ax.set_ylabel(name, fontsize=ticksize * 1.5)
        if chroms.shape[0]>1:
            plt.xticks([x for x in chroms.mid], [str(x) for x in chroms.index], rotation=-90, fontsize=ticksize * 1.5)
        # plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.locator_params(axis='y', nbins=4)
        mpl.rc('ytick', labelsize=ticksize)
        if ylim is not None:    plt.ylim(ymin=ylim)
    chroms = pd.DataFrame(df.groupby('CHROM').POS.apply(lambda x:x.max()-x.min()).rename('len').loc[df.reset_index().CHROM.unique()] + 1000)
    chroms = pd.DataFrame(df.groupby('CHROM').POS.apply(lambda x:x.max()).rename('len').loc[df.reset_index().CHROM.unique()] + 1000)
    if CHROMLen is not None:
        chroms=pd.DataFrame(CHROMLen)
    chroms['offset'] = np.append([0], chroms.len.cumsum().iloc[:-1].values)
    chroms['color'] = [c for (_, c) in zip(range(chroms.shape[0]), cycle(colors))]
    if color is not None: chroms['color']=color
    chroms['start']=df.groupby('CHROM').POS.min()
    if CHROMLen is not None:
        chroms['start']=0

    chroms['mid'] = [x + y / 2 for x, y in zip(chroms.offset+chroms.start, chroms.len)]
    chroms['mid'] = [x + y / 2 for x, y in zip(chroms.offset+chroms.start, chroms.len)]
    df['color'] = chroms.color.loc[df.CHROM].values
    df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
    df['color'] = chroms.color.loc[df.CHROM].values
    df.set_index('gpos', inplace=True);
    def fff(shade):
        shade['gstart'] = shade.start  #
        shade['gend'] = shade.end  #
        if chroms.shape[0] > 1:
            shade['gstart'] += chroms.offset.loc[shade.CHROM].values
            shade['gend'] += + chroms.offset.loc[shade.CHROM].values
        if 'name' in shade.columns:
            shade.sort_values('gstart', ascending=False, inplace=True)
            shade['ID'] = range(1, shade.shape[0] + 1)
        return shade
    if shade is not None: shade=fff(shade)
    if shade2 is not None: shade2 = fff(shade2)
    addGlobalPOSIndex(common, chroms);
    addGlobalPOSIndex(Outliers, chroms)
    if fig is None and axes is None:
        fig,axes=plt.subplots(columns.size, 1, sharex=True,sharey=shareY,figsize=(20, columns.size * 4));
        if columns.size==1:
            axes=[axes]
    elif axes is None:
        axes=fig.axes

    for i in range(columns.size):
        if not i:
            sh=shade
        else:
            if shade is not None and 'name' in shade.columns:
                sh= shade.drop('name', 1)
        plotOne(df[columns[i]], df.color, names[i], chroms,common, sh,shade2,axes[i])
    # plt.setp(plt.gca().get_xticklabels(), visible=True)
    xlabel='Chromosome'
    if chroms.shape[0]==1:
        xlabel+=' {}'.format(chroms.index[0])
    axes[-1].set_xlabel(xlabel, size=ticksize * 1.5)

    plt.gcf().subplots_adjust(bottom=0.2,hspace=0.05)
    if fname is not None:
        print ('saving ', fname)
        plt.savefig(fname)
    if not show:
        plt.ion()

    return fig




def TimeSeries(data, methodColumn=None, ax=None, fname=None, color='r', ci=1,shade=[0,50],samplingTimes=None):
    """
    Args:
        data: a dataframe containing mu and st fields,
        methodColumn: when method column is given, it plots together
        ax:
        fname:
    Returns:
    """
    if ax is None: fig=plt.figure(figsize=(12,4), dpi=200)
    if methodColumn is None:
        dfs=[('aa',data)]
    else:
        dfs=data.groupby(methodColumn)
    for name,df in dfs:
        if 'color' in df.columns:color=df.color.unique()[0]
        df.mu.plot(linewidth=1, color=color, label=name, ax=ax)
        # plt.gca().fill_between(df.index,  (df.mu+df.st).apply(lambda x:min(x,1)), (df.mu-df.st).apply(lambda x:max(x,0)), color=color, alpha=0.25)
        ax.fill_between(df.index.values.astype(int), (df.mu + ci * df.st), (df.mu - ci * df.st), color=color,
                        alpha=0.25)
    if shade is not None:
        ax.axvspan(shade[0], shade[1], alpha=0.25, color='black')
        ax.set_xticks(np.append([50], plt.xticks()[0]))
    if samplingTimes is  not None:
        for t in samplingTimes:ax.axvline(t,color='k',ls='--',lw=1,alpha=0.35)

    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':26}) ;
    # mpl.rc('text', usetex=True)
    if fname is not None:
        plt.savefig(fname)



def QQPval(a,z,nq=20, s=40, alpha=0.8, fname=None):
    """pplt.QQPval(exp10(logpa),exp10(logpz))"""
    def getQuantilesLog2():
        q=[1]
        for i in range(nq):q+=[q[-1]/2.]
        q=pd.Series(q,index=q).iloc[1:]
        return q
    q=getQuantilesLog2()
    qq=pd.DataFrame(q.apply(lambda x: [abs((x)),z.quantile(x),a.quantile(x)]).sort_index().tolist(),index=q,columns=['expected','null','data']).applymap(lambda x: -np.log10(x))
    plt.figure(figsize=(8,6),dpi=200)
    qq.plot.scatter(x='expected',y='null',color='k',s=s,alpha=alpha,ax=plt.gca())
    qq.plot.scatter(x='expected',y='data',ax=plt.gca(),s=s,alpha=alpha,color='r',lw = 0);
    plt.ylim([-1, plt.ylim()[1]]);
    xmax = plt.xlim()[1]
    plt.plot([0, xmax], [0, xmax],ls='--', c="k",alpha=0.3)
    plt.xlim([0,xmax])
    plt.xlabel('Expected -log$_{10}$($p$-value)');
    plt.ylabel('Observed -log$_{10}$($p$-value)')
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':26}) ;
    mpl.rc('text', usetex=True)
    if fname is not None: plt.savefig(fname)


def plotSiteReal(site, ax=None, fontsize=8, legend=False, title=None):
    if ax is None:
        dpi = 300
        _, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi, sharex=True, sharey=True)
        sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.2})
    pos = site.name
    site = site.sort_index().groupby(level=[0, 1]).apply(lambda x: (x.iloc[0], x.iloc[1]))
    df = site.apply(lambda x: pd.Series(np.random.binomial(x[1], x[0] / x[1], 10000)) / x[1]).T
    df = df.stack(['REP', 'GEN']).reset_index(['REP', 'GEN'])
    idx = pd.Series(range(site.index.get_level_values('GEN').unique().shape[0]), index=np.sort(site.index.get_level_values('GEN').unique()))
    ax = sns.boxplot(data=df, x='GEN', y=0, hue='REP', width=0.3, ax=ax);
    for i, mybox in enumerate(ax.artists):
        # Change the appearance of that box
        c = mybox.get_facecolor()
        mybox.set_facecolor('None')
        mybox.set_edgecolor(c)
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(c)
            line.set_mfc('None')
    for nnn,(_, xxx) in enumerate(site.apply(lambda x: x[0] / x[1]).unstack('REP').iteritems()):
        # print idx.loc[xxx.dropna().index] + (nnn - 1) * (0.1)
        try:
            pd.Series(xxx.dropna().values, index=idx.loc[xxx.dropna().index] + (nnn - 1) * (0.1)).plot(style='-o',
                                                                                                       color=
                                                                                                       sns.color_palette()[
                                                                                                           nnn], ax=ax,
                                                                                                       markersize=3,
                                                                                                       grid=False,
                                                                                                       linewidth=0.5)
        except:
            pass
    handles, labels = ax.get_legend_handles_labels();
    ax.set_xlim([ax.get_xlim()[0] - ax.get_xlim()[1] * 0.03, ax.get_xlim()[1] + ax.get_xlim()[1] * 0.03])
    ax.set_ylim([ax.get_ylim()[0] - ax.get_ylim()[1] * 0.03, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.03])
    if legend:
        ax.legend(handles[3:], map(lambda x: 'Replicate {}'.format(int(x) + 1), labels[3:]), loc='best', title='',
                  fontsize=fontsize - 2)
    else:
        ax.legend_.remove()
    ax.set_ylabel('')
    ax.set_xlabel('Generation')
    setSize(ax, fontsize=fontsize - 2)
    if title is not None:
        ax.set_title('{}:{}'.format(pos[0], pos[1]), fontsize=fontsize)
    ax.xaxis.grid(True, linewidth=6)

def getNameColorMarker(df):
    f = lambda x: x.method.replace('HMM', r'$\mathcal{H}$').replace('MarkovChain', r'$\mathcal{M}')
    # + '$,\pi=$' + str(int(x.q * 100))
    # f = lambda x: x.method.replace('HMM', r'$\mathcal{H}$')
    cols = ['method']
    if 'q' in df.index.names:
        cols = ['q'] + cols
    names = df.unstack('S').reset_index()[cols].drop_duplicates()
    names['name'] = names.apply(f, axis=1)
    names = names.set_index(cols).sort_index(level='method')
    names['marker'] = getMarker(names.shape[0])
    names['color'] = getColorMap(names.shape[0])
    return names


def plotOnePower(df, info, axes, legendSubplot=-1, fontsize=7, markersize=5, ylabel='Hard', panel=list('ABC')):
    for j, (name, dff) in enumerate(df.groupby(level='coverage')):
        dff = dff.unstack('S')
        dff = dff.sortlevel(['method'], ascending=True)
        names = info.loc[dff.reset_index('coverage').index]
        dff.index = names.name

        dff.T.plot(ax=axes[j], legend=False, color=names.color.tolist(), style=names.marker.tolist(),
                   markersize=markersize)
        axes[j].axhline(y=5, color='k');
        setTicks(dff)
        if j == legendSubplot:
            handles, labels = axes[j].get_legend_handles_labels()
            axes[j].legend(handles[::-1], labels[::-1], loc='center left', fontsize=fontsize)
        if name == np.inf:
            name = r'$\infty$'
        else:
            name = '{:.0f}'.format(name)
        if ylabel == 'Hard': axes[j].set_title(r'$\lambda=$' + name, fontsize=fontsize)
        axes[j].set_xlabel(r'$s$')
        axes[j].set_ylabel(r'Power ({} Sweep)'.format(ylabel))
        setSize(axes[j], fontsize=fontsize)


def setSize(ax, fontsize=5):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    try:
        for item in ([ax.zaxis.label] + ax.get_zticklabels()):
            item.set_fontsize(fontsize)
    except:
        pass


def setLegendSize(ax, fontsize=5):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={'size': fontsize}, loc='best')

def setTicks(df):
    plt.xticks(df.columns.values);
    plt.xlim([0.018, 0.105]);
    plt.ylim([-2.0, 105]);
    plt.yticks(np.sort(np.append(np.arange(20, 101, 20), [5])))
    plt.xlabel('')


def savefig(name, dpi,path=PATH.paperFigures,extensions=['pdf','tiff']):
    import os
    for e in extensions:
        os.system('rm -f '+ path+ name + '.'+e)
        plt.savefig(path+ name + '.'+e, dpi=dpi)


def plotQuantile(df, kde):
    from UTILS import Util as utl
    quantiles = np.sort(np.append(np.linspace(0.0, 1, 1000)[:-1], np.linspace(0.999, 1, 10)))
    qq = pd.concat([utl.getQantilePvalues(df.COMALE, kde, quantiles=quantiles),
                    utl.getQantilePvalues(df.COMALENC, kde, quantiles=quantiles)], axis=1);
    qq.columns = ['data', 'null'];
    QQPval(qq, fname=utl.paperFiguresPath + 'qq.pdf')


def plotGeneTrack(a,ax=None,ntracks=4,minx=None,maxx=None,genesToColor=None):
    d=0.01
    if ax is None:
        plt.figure();    ax=plt.gca()
    for i ,(n,row) in enumerate(a.set_index('name')[['start', 'end']].iterrows()):
        y=(i%ntracks) *d
        if minx is not None:
            row['start']=max(minx,row['start'])
        if maxx is not None:
            row['end'] = min(maxx, row['end'])
        c=None;alpha=1
        if genesToColor is not None:
            alpha=0.75
            c='k'
            if n in genesToColor:
                c='r'
        ax.plot([row['start'],row['end']],[y,y],lw=5,label=n,c=c,alpha=alpha)
        # xy=row.min()-5000,y+d/3
        xy = row.min() , y + d / 4
        text = ax.annotate(n, xy=xy,    horizontalalignment='left', verticalalignment='bottom', size=5)
        # text=ax.annotate(n, xy=xy, xytext=xy,horizontalalignment='left',verticalalignment='bottom',size=5)
    ax.legend()
    ax.set_ylim([-d/3,(ntracks-1)*d+2*d -d])
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    # ax.legend(frameon=True, loc='upper left', bbox_to_anchor=(-0.0, 1.2), ncol=5);
    try:ax.legend_.remove()
    except:pass
    ax.set_ylabel('Genes \n')
    # ax.xaxis.label.set_size(2)

def overlap(I, a,start=None,end=None):
    if start is not None:return I[(I.start <= end) & (I.end >= start)]
    return I[(I.start <= a.index[-1]) & (I.end >= a.index[0])]


def plotTracks(a,i=None,dosharey=True,marker='o',ls='',alpha=0.3,CHROM=None,xmin=None,xmax=None,
               ymax=None,ymin=None,hline=None,dmel=False,ntracks=-1,dpi=200,figWidth=7,colors=None,
               cjet=None,markersize=4,CMAP=None,genesToColor=None,oneColorCMAP=False,plotColorBar=True,
               hue=None,ax=None,DROP_CR_Genes=False,subsample=None,fontsize=12,fig=None,alpha_shade=0.2):
    if subsample is not None:
        if subsample<a.shape[0] and subsample>0:
            np.random.seed(0)
            a=a.sample(subsample).sort_index()
    GENECOORDS=None
    if ntracks>0:
        if CHROM is None:
            if 'CHROM' in a.index.names:
                CHROM = a.index.get_level_values('CHROM').unique()[0]
                a = a.loc[CHROM]
            else:
                CHROM = i.CHROM
        if dmel is not None:
            f=utl.home+'storage/Data/Dmelanogaster/geneCoordinates/dmel{}.df'.format(dmel)
            GENECOORDS=pd.read_pickle(f).set_index('CHROM').loc[[CHROM]]
            GENECOORDS['len']=GENECOORDS.end - GENECOORDS.start
        else:
            from Scripts.KyrgysHAPH import Util as kutl
            GENECOORDS=kutl.GENECOORS.loc[[CHROM]]
    if len(a.index.names)==1: a.index.name='POS'
    if len(a.shape)==1: a=pd.DataFrame(a)
    def sharey():
        ma = max(map(lambda ax: ax.get_ylim()[1], plt.gcf().axes))
        mi = min(map(lambda ax: ax.get_ylim()[0], plt.gcf().axes))

        for ax in plt.gcf().axes[:-1]: ax.set_ylim([mi, ma])
        if ntracks<0:plt.gcf().axes[-1].set_ylim([mi, ma])
    n=a.shape[1]
    if ntracks>0:n+=1
    if ax is None and fig is None:
        fig, ax = plt.subplots(n, 1, sharex=True,figsize=(figWidth,n*1.5),dpi=dpi)

    if ntracks>0:
        axgene=ax[-1]
    if n>1:ax=ax.reshape(-1)
    else: ax=[ax]
    # print  a.shape
    for ii in range(a.shape[1]):
        color='darkblue'
        if colors is not None: color=colors[ii]
        if CMAP is not None:
            col=a.columns[ii]

            df=a[[col]].join(CMAP.rename('c'),how='inner').reset_index()
            # cmap=sns.cubehelix_palette("coolwarm", as_cmap=True)
            # cmap=sns.dark_palette("purple",input="xkcd", as_cmap=True,reverse=True,light=1)

            # cmap=sns.diverging_palette(240, 10, as_cmap=True, center = "dark")
            # cmap=sns.color_palette("coolwarm", as_cmap=True)
            # cmap=sns.choose_colorbrewer_palette("coolwarm", as_cmap=True)
            if oneColorCMAP:
                cmap = sns.cubehelix_palette(dark=0, light=.85, as_cmap=True)
            else:
                cmap = 'jet'
            dfc=df[['POS',col,'c']].dropna()
            sc=ax[ii].scatter(dfc.POS,dfc[col], c=dfc.c.values,cmap=cmap,alpha=alpha,s=markersize)
            if plotColorBar:
                if fig is None:
                    fig=plt.gcf()
                cbaxes = fig.add_axes([0.905, 0.537, 0.02, 0.412]) #[ lower left corner X, Y, width, height]
                tl=np.linspace(0,1,5)
                cb = plt.colorbar(sc, cax=cbaxes, ticks=tl, orientation='vertical')
                cb.ax.set_yticklabels(["%.2f" % x for x in tl],size=8)
                cb.set_label(label='Final Frequency', size=10)
                dosharey=False
        else:
            if cjet is not None:
                for jj,_ in a.iterrows():
                    a.loc[[jj]].iloc[:, ii].plot(legend=False, ax=ax[ii], marker=marker, ls=ls,  alpha=alpha, c=cjet.loc[jj],markersize=4)
            elif hue is not None:
                for name in hue.groupby(0).size().sort_values().index[::-1]:
                    # if name!=2:continue
                    jj = hue[hue[0] == name]['index'].sort_values().tolist()
                    color=hue[hue[0] == name].c.iloc[0]
                    a.iloc[:, ii].loc[jj].plot(legend=False, ax=ax[ii], marker=marker, ls=ls, c=color, alpha=alpha,markersize=markersize)

            else:
                a.iloc[:, ii].plot(legend=False, ax=ax[ii],marker=marker,ls=ls,c=color,alpha=alpha,markersize=markersize)

        if hline is not None:ax[ii].axhline(hline, c='r', alpha=0.6)
        ax[ii].set_ylabel(a.columns[ii])
        setSize(ax[ii],fontsize)
    # plotGeneTrack(overlap(kutl.GENECOORS.loc[[i.CHROM]], 0,xmin, xmax), ax[-1], minx=a.index.min(), maxx=a.index.max())
    if xmin is None:
        xmin, xmax=a.index[0],a.index[-1]
    if ymax is not None:
        axylim=(ax,ax[:-1])[ntracks>0]
        for axi in axylim:
            if ymin is not None:
                axi.set_ylim([ymin, ymax])
            else:axi.set_ylim([axi.get_ylim()[0],ymax])
    if ntracks>0:
        if DROP_CR_Genes:
            GENECOORDS=GENECOORDS[GENECOORDS.name.apply(lambda x: x[:2] != 'CR').values]
        plotGeneTrack(overlap(GENECOORDS, a, xmin, xmax), axgene, minx=xmin, maxx=xmax,ntracks=ntracks,genesToColor=genesToColor)

    if dosharey:sharey()

    if i is not None:
        for ii in range(a.shape[1]):
            ax[ii].fill_between([i.start, i.end], ax[ii].get_ylim()[0], ax[ii].get_ylim()[1], color = 'k', alpha = alpha_shade)
        # ax[-1].fill_between([i.start, i.end], ax[ii].get_ylim()[0], ax[ii].get_ylim()[1], color='k', alpha=alpha_shade)
    if CHROM is not None:
        jj=-1
        if plotColorBar: jj-=1
        ax[jj].set_xlabel('Chromosome {}'.format(CHROM))
    # plt.tight_layout()
    plt.gcf().subplots_adjust(top=0.95, left=0.09)
    if CMAP is not None:
        plt.gcf().subplots_adjust(hspace=0.0)
    return GENECOORDS



def plotTracksList(a,i,marker=None, ls='-',shade=True, alpha=0.3,height=1.5):
    from Scripts.KyrgysHAPH import Util as kutl
    n=len(a)+1
    fig, ax = plt.subplots(n, 1, sharex=True,figsize=(10,n*height),dpi=200)
    for ii in range(len(a)):
        if marker is None:a.iloc[ ii].plot(legend=True, ax=ax[ii])
        else:
            color = getColorMap(a.iloc[ ii].shape[1])
            for jj in range(a.iloc[ ii].shape[1]):
                a.iloc[ii].iloc[:,jj].plot( ax=ax[ii], marker=marker, ls=ls, color=color[jj],alpha=alpha, markersize=4, label=a.iloc[ii].columns[jj])
        if shade:
            ax[ii].fill_between([i.start, i.end], ax[ii].get_ylim()[0], ax[ii].get_ylim()[1], color='k', alpha=0.2)
        ax[ii].set_ylabel(a.index[ii])
    plotGeneTrack(overlap(kutl.GENECOORS.loc[[i.CHROM]], None,start=i.start,end=i.end), ax[-1],minx=a.iloc[0].index.min(),maxx=a.iloc[0].index.max())
    if shade:
        ax[-1].fill_between([i.start, i.end], ax[-1].get_ylim()[0], ax[-1].get_ylim()[1], color='k', alpha=0.1)
    plt.xlabel('Chromosome {}'.format(i.CHROM))
    plt.gcf().subplots_adjust(top=0.95, left=0.09,hspace=0.03)

def plotDAF(i,pairs,AA=False,pad=500000,lite=0.2,lites=None,delta=True,ABS=False,pos=None,compact=False):
    try:
        if delta:
            ylim = [-1.05, 1.05]
            diff=lambda x: x.iloc[:,0]-x.iloc[:,1]
            # for pair in pairs:print utl.loadFreqs(pair, i,pad=pad,AA=AA)
            a= utl.quickMergeGenome(map(lambda pair: diff(utl.loadFreqs(pair, i,pad=pad,AA=AA)).rename(pair[0]+'\nvs\n'+pair[1]),pairs)).loc[i.CHROM]
        else:
            pairs=list(set(pairs))
            print(pairs)
            ylim = [-.05, 1.05]
            a = utl.loadFreqs(list(set(pairs)), i, pad=pad, AA=AA).loc[i.CHROM]
        xmin,xmax=[utl.BED.expand(i, pad).start, utl.BED.expand(i, pad).end]

        if a is None:
            print('AA',i.CHROM,i.start,i.end,AA)
            return

        if lite>0:
            a = a[a.iloc[:, 0] >= lite]
        if lites is not None:
            for ii , lite in enumerate(lites):
                a = a[a.iloc[:, ii].abs() >= lite]

        if ABS is True:
            a=a.abs();
            ylim[0] +=1
        a=a[a.iloc[:,0].abs()>0]
        # a = a.apply(lambda x: x.dropna().rolling(50, center=True).mean())
        if pos is not None: a=a.loc[pos]
        if compact:
            plotTracksList(pd.Series([a]), i, ls='', marker='o', shade=False);
            plt.gcf().axes[0].set_ylabel('Derived Allele Freq.');
            plt.gcf().axes[0].legend(borderaxespad=0.4, fontsize=12, frameon=True, shadow=not True)
        else:
            plotTracks(a, i, xmin=xmin, xmax=xmax)
        plt.xlim(xmin,xmax)
        if ylim is not None:
            for x in plt.gcf().axes[:-1]:x.set_ylim(ylim)
        return a
    except:pass

def plotDAFpops(i,pop=None,HA=False,additionalPairs=[],against=None,pairs=None,lite=0,AA=False,delta=True,ABS=False,pos=None,pad=500000,compact=False):

    if pairs is  None:
        if not HA:
            if against is None:
                # pairs=[[pop, 'JPT'], [pop, 'SAS'], [pop, 'AMR'], [pop, 'EUR'], [pop, 'YRI']]
                pairs = [[pop, 'EAS'], [pop, 'SAS'], [pop, 'EUR']]
            else:
                pairs=[[pop,x] for x in against]
        else:
            pairs = [[pop, 'TIB'], [pop, 'AND'], [pop, 'ETH'], [pop, 'BGI'],[pop, 'EAS'],[pop, 'AFR'],[pop, 'EUR']]

    pairs =  additionalPairs+ pairs
    if not delta: pairs=[pop]+ [pair[1] for pair in pairs]
    else:
        pairs=[ pair for pair in pairs if pair[0]!=pair[1]]
    return plotDAF(i,pairs,lite=lite,  AA= AA, delta=delta,ABS=ABS,pos=pos,pad=pad,compact=compact)


def plotPBS(i,pops,additionalPairs=[],lite=0,AA=False,delta=True,pad=500000,pos=None):
    try:
        ie=utl.BED.expand(i,pad=pad)
        a=utl.pbsi(ie,pops)
        a = a[a.iloc[:, 0] > lite]
        a = a[a.iloc[:, -1] > 0]

        # print utl.BED.expand(i, pad).start
        xmin, xmax = [utl.BED.expand(i, pad).start, utl.BED.expand(i, pad).end]
        if pos is not None: a = a.loc[pos]
        plotTracks(a, i,xmax=xmax,xmin=xmin)
        plt.xlim(xmin,xmax)
        for x in plt.gcf().axes[:-2]:x.set_ylim([-0.05,1.05])
        plt.gcf().axes[-2].set_ylim([0,1])
        # plt.figure()
        # b = pd.concat([a.PBS], keys=[i.CHROM])
        # b.index.names=['CHROM','POS']
        # Manhattan(utl.scanGenome(b))
        # plt.figure()
        # Manhattan(utl.scanGenome(b),np.sum)
        # plt.show()
    except:pass






class Trajectory:
    @staticmethod
    def Fly(xx,reps=[1],title='',pop='H',ax=None,hue=None,color=None,sumFreqCutoff=0,subsample=-1,titles=None,
            foldOn=None,fname=None,alpha=None,suptitle=None,logscale=True,fontsize=14,ticksSize=10,noticks=[7, 15]):

        if len(xx.columns.names)<2:
            xx=htl.aug(xx)
        x=xx[xx.sort_index(1)[pop].loc[:,pd.IndexSlice[:,reps]].sum(1)>sumFreqCutoff]
        if subsample>0 and subsample<x.shape[0]:
            ii=np.random.choice(x.shape[0], subsample, replace=False)
            x=x.iloc[ii]
            if hue is not None: hue=hue.iloc[ii]
        if color is not None:
            hue = pd.Series([color for _ in range(x.shape[0])], index=x.index.tolist(),name='c').reset_index()
            hue[0] = True
        if ax is not None:
            try:
                len(ax)
                axes=ax
            except:
                axes = [ax]
        else:
            fig, axes = plt.subplots(1, len(reps), sharey=True, figsize=(3*len(reps)+2, 3), dpi=100)
            if len(reps)==1:axes=[axes]
        for i,rep in enumerate(reps):
            if not i:
                if title!='':title='('+title+')'
            ax=axes[i]
            xrep=x.xs(rep, 1, 2)[pop]
            if len(noticks):
                for t in noticks:
                    if t in xrep.columns:
                        xrep=xrep.drop(t,1)
            Trajectory.FlyRep(xrep, suff=' Rep. {}'.format(rep),ax=ax,hue=hue,title=title,foldOn=foldOn,alpha=alpha,logscale=logscale)

        if fname is not None:plt.savefig(fname)
        for ax in axes:
            setSize(ax,ticksSize)
            if logscale:
                ax.set_xlim(99,300)
        if titles is not None:
            for ax,t in zip(axes,titles):
                ax.set_title(t,fontsize=fontsize)
        axes[0].set_ylabel('Allele Frequency', fontsize=fontsize)
        axes[(0,1)[len(axes)==3]].set_xlabel('Generation', fontsize=fontsize)
        if len(reps)==3:
            plt.gcf().tight_layout(pad=0.1)
        if suptitle:
            # plt.suptitle(suptitle,fontsize=14)
            # plt.gcf().tight_layout(pad=0.1, rect=[0.0, 0, 0.9, .9])
            axt=axes[-1].twinx()
            axt.set_ylabel(suptitle,fontsize=fontsize)
            axt.set_yticks([])
            # plt.setp(axes[-1].get_yticklabels(), visible=False)
        # axes[-1].set_yticks([])

    @staticmethod
    def FlyRep(zz, suff='', ax=None, hue=None, title='', hueDenovo=False, foldOn=None,alpha=None,logscale=True):
        if not zz.shape[0]: return
        if alpha is None: alpha = 0.12

        def one(y, ax, pref, hue,alpha):
            x = y.copy(True)
            if not (foldOn is None):
                x = x.T
                if foldOn>0:
                    fold = x[foldOn] < 0.5
                else:
                    fold = x[-foldOn] > 0.5
                x.loc[fold, :] = 1 - x.loc[fold, :]
                x = x.T
            g = list(map(str, x.index.values.astype(int)))

            if logscale:x.index = x.index.values + 100
            x=x.rename({280:290})
            #         print x
            title = pref + suff
            # title = ''

            if hue is None:
                x.plot(legend=False, c='k', alpha=alpha, ax=ax, title=title)
            else:
                for name in hue.groupby(0).size().sort_values().index[::-1]:

                    group = hue[hue[0] == name]
                    xx = x.loc[:, group['index']]
                    # print xx
                    lab = str(group[0].iloc[0])
                    try:
                        lab += ' {}'.format(xx.shape[1])
                    except:
                        pass
                    c = group['c'].iloc[0]
                    # alpha = (0.2, 1)[name == 'BA']
                    for iii, (_, y) in enumerate(xx.T.iterrows()):
                        if not iii:
                            y.plot(legend=False, c=c, label=lab, alpha=alpha, ax=ax, title=title)
                        else:
                            y.plot(legend=False, c=c, label='_nolegend_', alpha=alpha, ax=ax, title=title)
                    # plt.tight_layout(rect=[0, 0, 0.5, 1])
                    # ax.legend(bbox_to_anchor=(1., 0.9))

                # ax.legend()
            if logscale:
                ax.set_xscale("log", basex=2);
                ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                # print(x.index.tolist())
                ax.set_xticks(x.index)
                ax.set_xticklabels(g)


            # ax.tick_params(axis='both', which='both', bottom='on', top='on', labelbottom='on', right='on', left='on',
            #                labelleft='on', labelright='on')

        if ax is None: fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3), dpi=100)
        one(zz.T, ax,  title + '', hue,alpha)
        ax.set_title(ax.get_title() + ' ({} SNPs {})'.format(zz.shape[0], ''))

    @staticmethod
    def clusters(a,seq,i=None,reps=[1,2,3],ss=300,pop=['H'],allpops=False):
        if allpops: pop=list('HCL')
        if len(a.columns.names)<2:
            a=htl.aug(a)
            reps=[1];pop=['H']
        if i is not None:
            seq=map(lambda x: utl.mask(x, i), seq)
        SNPS=[]
        def snps(V):
            v=V.copy(True)
            v.iloc[1:-1] = v.iloc[1:-1].replace({False: None});
            v = v.dropna();
            v = v.astype(int)
            return v
        for p in pop:
            axes = plt.subplots(len(seq), len(reps), figsize=(3.5*len(reps)+2, len(seq)*3), sharex=True, sharey=True)[1]
            if len(seq) ==1: axes=[axes]
            for xx, ax in zip(seq, axes):
                if i is not None or 'CHROM' in a.index.names:
                    aa=a.loc[pd.IndexSlice[i.CHROM, utl.TI(xx), :]]
                else:
                    aa=a.loc[utl.TI(xx)]
                SNPS+=[ snps(xx)]
                Trajectory.Fly(aa, pop=p, subsample=ss,reps=reps, ax=ax);
            if len(pop)>1:
                plt.suptitle(p)

    @staticmethod
    def Win(j, X, ax, rep=1, subsample=-1, verbose=False, foldOn=None):
        x = utl.mask(X, j.loc[j.name], full=True)
        if not x.shape[0]: return
        ss = lambda x: x
        if subsample > 0: ss = lambda x: x.iloc[np.random.choice(x.shape[0], min(x.shape[0], subsample), replace=False)]
        x = ss(x)
        Trajectory.Fly(x, reps=[rep], ax=ax, foldOn=foldOn)

        ax.set_title('({})  {}.   n={}'.format(j.name + 1, Trajectory.title(j.iloc[0]), x.shape[0]), fontsize=12)
        if verbose: print('{},'.format(j.name),)

    @staticmethod
    def title(istr):
        CHROM, start = istr.split('-')[0].split(':');
        end = istr.split('-')[1]
        return '{}:{:,d}K-{:,d}k'.format(CHROM, int(start) / 1000, int(end) / 1000)
    @staticmethod
    def slidingWindowRep(batch, X, rep, subsample=-1, verbose=False, foldOn=None, name='',shape=None):
        if shape is None:
            rows, cols = 10, 10
            rows = batch.shape[0] / 10
        else:
            rows,cols=shape

        ax = plt.subplots(rows, cols, dpi=100, figsize=(4*cols, 3 * rows), sharex=True, sharey=True)[1].reshape(-1)
        batch.groupby(level=0).apply(
            lambda i: Trajectory.Win(i, X, ax[i.name], subsample=subsample, rep=rep, verbose=verbose, foldOn=foldOn));

        title = 'Interval {}. {}:{}K-{}K Rep {}. '.format(name, batch.iloc[0].split('-')[0].split(':')[0],
                                                            int(batch.iloc[0].split('-')[0].split(':')[1]) / 1000,
                                                            int(batch.iloc[-1].split('-')[1]) / 1000, rep)
        title += ('ALT allele', 'MAF at Gen {}'.format(foldOn))[foldOn != None]
        plt.suptitle(title, fontsize=12, y=1.08)

    @staticmethod
    def slidingWindowReps(batch, X, subsample=-1, verbose=False, foldOn=None):
        for i in batch:
            Trajectory.Fly(utl.mask(X,i),reps=[1,2,3],foldOn=foldOn)
            plt.suptitle(Trajectory.title(i), fontsize=16, y=1.04)


    @staticmethod
    def getBatch(i = 'chr2R:7700000-7730000',N=100,step = 10000):
        return pd.Series(np.arange(0, step * N, step)).apply(lambda x: utl.BED.shift(i, x))


def FlySFSRep(a,title=None,subplots=True,includeOther=True):

    if includeOther:
        b = utl.renameColumns(a.H, 'H', pre=True)
        b=pd.concat([b,a.C[180].rename('C180'),a.L[180].rename('L180')],1)
        fs = (10, 6)
    else:
        b = utl.renameColumns(a.H, 'F', pre=True)
        b=b.drop('F7',1)
        fs = (6, 3)
    # b=b.round(1).apply(lambda x: x.value_counts()).fillna(0)
    b=b.apply(utl.sfs).fillna(0)
    b.plot.bar(width=0.9,subplots=subplots,layout=(-1, 3), figsize=fs,sharex=True,sharey=True,color='k',alpha=0.75);
    plt.gcf().axes[0].set_ylabel('Num. of SNPs')
    plt.gcf().axes[3].set_ylabel('Num. of SNPs')
    plt.gcf().axes[4].set_xlabel('Frequency Bin')
    for ax in plt.gcf().axes: ax.set_title('')
    # if title!=None:
        # plt.suptitle(title)
        # annotate(title,loc=0,ax=plt.gcf().axes[0])
    # plt.gcf().tight_layout(pad=0.2, rect=[0.0, 0.2, 1, 1])
    plt.subplots_adjust(left=.0, bottom=.0, right=1, top=.9,wspace=0.05, hspace=0.05)
    # plt.tight_layout()

def FlySFS(a,reps=[1,2,3]):
    for rep in reps:
        FlySFSRep(a.xs(rep,1,2),includeOther=False)
        plt.suptitle('Rep. {}'.format( rep),fontsize=14)
#

def getColorBlindHex():
    return sns.color_palette('colorblind').as_hex()

def visualizeRealFeatures(X,hue=None,transform=None):
    # pplt.visualizeRealFeatures(X.join(a.CHD),hue='CHD')
    XX=X.copy(True)
    cols=utl.TI(X.apply(lambda x: x.unique().size)>2).tolist()
    if hue is not None: cols+=[hue]
    XX=XX[list(set(cols))]

    scale = lambda x: x / (x.max() - x.min())
    if transform=='scale':
        XX = scale(XX)
        XX=XX-XX.min()
    elif transform == 'z':
        XX=XX.apply(utl.pval.zscore)
    ax = plt.subplots(3, 1, figsize=(XX.shape[1]/1.5, 8), dpi=150, sharex= True)[1]
    if hue is not None:
        XX=XX.melt(id_vars=[hue])
        palette = {0: 'gray', 1: 'r'}
        color=None
    else:
        XX=XX.melt()
        palette=None
        color='gray'

    palette=None
    sns.stripplot(data=XX, x='variable', y='value', jitter=0.15, ax=ax[0], alpha=0.2, hue=hue,palette=palette)
    try:
        sns.violinplot(data=XX, x='variable', y='value', ax=ax[1], hue=hue, palette=palette, color=color, split=True)
    except:
        sns.violinplot(data=XX, x='variable', y='value', ax=ax[1], hue=hue, palette=palette, color=color, split=False)
    sns.boxplot(data=XX, x='variable', y='value', ax=ax[2])
    # ax[1].set_xticks(ax[1].get_xticks(), rotation='vertical')
    plt.xticks(rotation=90)
    ax[0].set_xlabel('');ax[1].set_xlabel('')
    try:ax[1].legend_.remove()
    except:pass


def visualizeOneRealFeat(data,x,y):
    ax = plt.subplots(1, 3, sharey=True, dpi=100, figsize=(12, 3.5))[1]
    sns.stripplot(data=data, x=x, y=y, jitter=0.1, alpha=0.1, ax=ax[0])
    sns.boxplot(data=data, x=x, y=y, ax=ax[1])
    sns.violinplot(data=data, x=x, y=y, ax=ax[2])


def visualizeCatFeatures(X,hue):
    XX = X.copy(True)
    cols = utl.TI(XX.apply(lambda x: x.unique().size) < 10).tolist()
    XX = XX[cols].astype(int)
    norm = lambda x: x / x.sum()
    r=[]
    rows=max(1,int(np.ceil(len(cols)/3.)))
    ax=plt.subplots(rows,3,dpi=150,figsize=(8,rows*2))[1].reshape(-1)
    j=0
    for c in XX.columns:
        if c==hue:continue
        norm(pd.crosstab(XX[hue], XX[c])).plot.bar(ax=ax[j])
        leg=ax[j].legend(loc='best', prop={'size': 6})
        leg.set_title(c, prop={'size': 6})
        j+=1

    ax = plt.subplots(rows, 3, dpi=150, figsize=(8, rows * 2))[1].reshape(-1)
    j = 0
    for c in XX.columns:
        if c == hue: continue
        norm(pd.crosstab(XX[hue], XX[c])).T.plot.bar(ax=ax[j])
        leg = ax[j].legend(loc='best', prop={'size': 6})
        leg.set_title(hue, prop={'size': 6})
        j += 1
def visualizeFeatures(X,hue=None):
    visualizeCatFeatures(X, hue)
    visualizeRealFeatures(X, hue)

def puComment(fig, comment):
    if comment is not None:
        fig.text(.05, .05, 'Comment: ' + comment, fontsize=26, color='red')




