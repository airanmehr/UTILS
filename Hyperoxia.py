import numpy as np
import pandas as pd

pd.options.display.max_rows = 40;
pd.options.display.expand_frame_repr = False
import UTILS.Util as utl
import UTILS.Plots as pplt
import pylab as plt
import seaborn as sns
path=utl.home+'storage/Data/Dmelanogaster/OxidativeStress/'

CHROMS=['2L','2R','3L','3R','X','4']
pops={'C':'Control','H':'Hyperoxia','L':'Hypoxia'}
taus=['1-31', '7-61',  '31-61', '61-114', '114-180']
def X(D=False,C=False):
    a=rename(pd.read_pickle('/home/arya/fly/all/RC/all.df'))#.stack([0, 1, 2])
    a=a[a.H.xs('D', 1, 2).min(1)>9]
    a=a.loc[CHROMS].loc[:, pd.IndexSlice[:, [1, 4, 7, 12, 17, 31, 61, 114, 180]]].dropna()
    if D: return a.xs('D', 1, 3)
    if C: return a.xs('C', 1, 3)
    a = (a.xs('C', 1, 3) / a.xs('D', 1, 3)).round(2)
    return a.dropna()

def rename(c):
    def one(x):
        if 'final' in x:
            gen = {1: 1, 2: 7, 3: 12, 4: 15, 5: 31, 6: 61, 7: 114}
            x = x[1:].split('_')
            return 'H', gen[int(x[0])], int(x[1].split('.')[0])
        if 'Clean' in x:
            x = x.split('_')[1:]
            return x[0][0], 180, int(x[0][-1])
        ash=utl.execute('cat /home/arya/fly/F4-17/SraRunTable.tsv | cut -f7,9').iloc[1:].set_index(0)[1]
        return ash.apply(lambda x: x[1:]).apply(lambda x: (x[-2].replace('H','L'),int(x[:-2]),int(x[-1])    )).loc[x]
    if len(c.columns)==1:
        c.columns = pd.MultiIndex.from_tuples(map(one, c.columns), names=['POP', 'GEN', 'REP'])
    else:
        cols= [x+(y,)  for x,y in zip(map(one, c.columns.get_level_values(0)), c.columns.get_level_values(1))]
        c.columns=pd.MultiIndex.from_tuples(cols, names=['POP', 'GEN', 'REP','READ'])
    return c.sort_index(1)

def fixcols(a):
    gmap={7:1,12:7,31:12,61:31,114:61,180:114}
    a.columns=pd.Series(a.columns).apply(lambda x: '{}-{}'.format( gmap[int(x.split('-')[0])],x.split('-')[-1])).tolist()
    return a

def PCA(x):
    X=utl.pcaX(x.dropna().T,2)
    c=pd.DataFrame(map(lambda x: list(x)[:2],X.index)).drop_duplicates().set_index([0,1]).sort_index()
    marker=pd.Series(pplt.getMarker(c.index.levels[1].size,False),index=c.index.levels[1])
    for xx in marker.index:    c.loc[pd.IndexSlice[:,xx],'m']=marker.loc[xx]
    c.loc['L', 'c'] = 'darkblue'
    c.loc['H', 'c'] = 'r'
    c.loc['C', 'c'] = 'g'
    fig=plt.figure(dpi=150);ax=plt.gca()
    for i in c.index:
        if i[1] =='': continue
        X.sort_index().loc[i].plot.scatter(x=0,y=1,c=c.loc[i].c,label='{}.{}'.format(i[0],i[1]),ax=ax,s=70,alpha=0.6,marker=c.loc[i].m)
    plt.xlabel('PC1');plt.ylabel('PC2');
    plt.title('Genomewide PCA (H:Hyperoxia, C:Control, L:Hypoxia) of Flies');
    plt.gcf().axes[0].legend(frameon=True, bbox_to_anchor=(1,1),ncol=1);


def getFixationCutoffs():
    steps = pd.Series(0, taus).groupby(level=0).apply(
        lambda x: int(x.name.split('-')[1]) - int(x.name.split('-')[0]) - 1)
    import CLEAR.Libs.Markov as mkv
    def getFreqCutoff(tau):
        T = mkv.Markov.computePower(mkv.Markov.computeTransition(0, 100, 50), tau)
        p = T[.95].cumsum() / T[.95].cumsum()[1]
        return p[p > 0.01].index[0]

    return steps.apply(getFreqCutoff).sort_values()


def getHueEpistatis(z,t):
    t1=z.columns[-1]
    i0=(z[t]<0.11)
    import seaborn as sns
    cm=sns.color_palette("colorblind", 6)
    colors = ['k']+[cm[2]]+[cm[1]]
    hue=pd.concat([i0,~i0&(z[t1]<0.5),~i0&(z[t1]>0.5)],1,keys=[0,1,2]).apply(lambda x: x.idxmax(),1).rename(0).reset_index().rename(columns={'POS':'index'})
    hue['c']=hue.apply(lambda x: colors[x[0]],1)
    return hue

def oversample(x,L=1e5,start=0):
    np.random.seed(0)
    z=pd.concat([x,x]).sample(frac=1)
    z.index=sorted(np.random.choice(int(L), z.shape[0], replace=False)+start)
    z=z[~z.index.duplicated()]
    z.index=map(int,z.index)
    return z


def plotHaplotypes(x,hue=None,track=True,t=130,lite=True,traj=True , km=None,ss=-1,distinguishHapsInTracks=False,CHROM=None,recPos=None,ntracks=6,clean=True,fold=True,freqAt=200,maxGen=200):
    freqAt = x.columns[pd.Series(x.columns - freqAt).abs().idxmin()]
    try:
        t=x.columns[x.columns>=t][0]
    except:
        t=x.columns[x.shape[1]//2]
        print 'Warining: t=x.columns[x.columns>=t][0]'
    xf=x.copy(True).fillna(0)
    i=xf[t]>0.5
    xf[i]=1-xf[i]
    haps = utl.kmeans(xf[t], km)
    h12 = haps.unique()
    if clean and km==2:
        cf = np.mean(haps.value_counts().index)
        drop =utl.TI((haps == min(h12)) & (xf[t] > cf - 0.05)).tolist()+ utl.TI((haps == max(h12)) & (xf[t] < cf + 0.05)).tolist()
        xf = xf.drop(drop)
        haps = haps.drop(drop)

    if hue is None and  not (km is  None):
        if km >1:
            t2=x.columns[x.columns>t][0]


        splitted=0
        if xf[haps==h12[0]][t2].mean()>xf[haps==h12[1]][t2].mean():
            splitted=1
        sp=haps == h12[splitted]
        ff = lambda (x, c, k): pd.DataFrame(([(y, c, k) for y in utl.TI(x)]), columns=['index', 'c', 0])
        cm = sns.color_palette("colorblind", 6)
        hue=pd.concat(map(ff,[(~sp,'k',0),((sp) &( xf[t2]>0.5),cm[1],1),((sp) &( xf[t2]<0.5),cm[2],2)]))
    # else:


        # hue=getHueEpistatis(xf,t)


    REALDATA=CHROM is not None
    if not REALDATA:ax=plt.subplots(1,2,figsize=(8,3),dpi=120)[1]

    if traj:

        if REALDATA:
            pplt.Trajectory.Fly(xf, hue=hue, subsample=ss,)
            plt.gca().set_title('')
        else:
            xx=xf
            if not fold:
                xx=x.loc[:,x.columns<=freqAt]
                xx.loc[:,maxGen+1]=np.nan
            pplt.Trajectory.Fly(xx, logscale=False, hue=hue, subsample=ss, ax=ax[0])


    if distinguishHapsInTracks:
        jj=(xf[t]>0.1) & (xf.iloc[:,-1]<0.1)
        xf[jj]-=0.01
    if lite:
        j=(x[[t,x.columns[-1]]].sum(1)-1).abs()<0.9
    else:
        j=haps.fillna(0)>-np.inf

    xf.index.name='POS'
    if track:
        if REALDATA:
            if hue is not None:
                pplt.plotTracks(haps[j], ntracks=ntracks,dmel=5,CHROM=CHROM, markersize=8, ymin=-0.07, ymax=1.03, hue=hue, alpha=0.3,genesToColor=[]);
            else:
                pplt.plotTracks(haps[j], ntracks=ntracks, dmel=5, CHROM=CHROM, markersize=8, ymin=-0.07, ymax=1.03, alpha=0.3, genesToColor=[]);
            plt.gcf().axes[0].set_ylabel('Frequency\nat Gen. 114')
            plt.tight_layout(pad=0)
        else:
            if hue is not None:
                if fold:
                    pplt.plotTracks(haps[j], ntracks=-1, markersize=8, ymin=-0.07, ymax=1.03, hue=hue, alpha=0.3, ax=ax[1]);
                else:

                    pplt.plotTracks(x[freqAt], ntracks=-1, markersize=8, ymin=-0.07, ymax=1.03, hue=hue, alpha=0.3,ax=ax[1]);
            else:
                pplt.plotTracks(haps[j], ntracks=-1, markersize=8, ymin=-0.07, ymax=1.03,colors='k', alpha=0.3, ax=ax[1]);


            ax[1].set_ylabel('');    ax[1].set_yticks([])
            ax[0].set_title('')
            ax[1].set_xlabel('Position')
            if fold:
                ax2 = ax[1].twinx()
                ax2.set_ylabel('Frequency at Gen. 150')
                map(lambda x: pplt.setSize(x,12),list(ax)+[ax2])
            else:
                map(lambda x: pplt.setSize(x, 12), list(ax) )
            ax[1].set_xlim([xf.index.min()-2000,xf.index.max()+2000])
            plt.tight_layout(pad=0.1)
    if recPos:
        plt.gcf().axes[0].axvline(recPos, c='r', alpha=0.4)
        plt.gcf().axes[1].axvline(recPos, c='r', alpha=0.4)
