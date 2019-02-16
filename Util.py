'''
Copyleft Dec 17, 2015 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numba
import pandas as pd;
import numpy as np;


from subprocess import Popen, PIPE, STDOUT

np.set_printoptions(linewidth=140, precision=5, suppress=True)
import os

try:
    import readline

except:
    pass

def mkdir(path):os.system('mkdir -p {}'.format(path))
parentdir=lambda path:os.path.abspath(os.path.join(path, os.pardir))
home = os.path.expanduser('~') + '/'
UKBB_PATH=home+'/processed/genetics/imputed/hg38/QCed/'

class PATH:
    def __init__(self,home):
        self.paper = home + 'workspace/timeseries_paper/'
        self.data = home + 'storage/Data/'
        self.scan = self.data + 'Human/scan/'
        self.Dmel = self.data + 'Dmelanogaster/'
        self.OKG = self.data + 'Human/20130502/ALL/'
        self.paperFigures = paperPath + 'figures/'
        self.plot = home + 'out/plots/';
        self.out = home + 'out/';
        self.simout = self.data + 'SimulationOutFiles/'
        self.stdout = outpath + 'std/'
        mkdir(self.out)
        mkdir(self.simout)
        mkdir(self.plot)
        mkdir(self.stdout)


paperPath = home + 'workspace/timeseries_paper/'
dataPath=home+'storage/Data/'
scanPath=dataPath + 'Human/scan/'
dataPathDmel=dataPath+'Dmelanogaster/'
dataPath1000GP=dataPath+'Human/20130502/ALL/'
paperFiguresPath = paperPath + 'figures/'
plotpath=home+'out/plots/';
outpath=home+'out/';
simoutpath=dataPath+'SimulationOutFiles/'
stdoutpath = outpath + 'std/'
mkdir(outpath)
mkdir(simoutpath)
mkdir(outpath)
mkdir(plotpath)
mkdir(stdoutpath)

comaleName = r'\sc{Clear}'
def googleDocURL(name,url):return '=HYPERLINK("{}","{}")'.format(url,name)
import scipy.stats as st


def FoldOn(y,foldOn):
    x = y.copy(True)
    if not (foldOn is None):
        fold = x[foldOn] < 0.5
        x.loc[fold, :] = 1 - x.loc[fold, :]
    return x

class ReadCount:
    @staticmethod
    def freq(X,fold=False):
            x=X.C/X.D
            if fold:x[x>0.5]=1-x[x>0.5]
            return x

def ipca(a,n_components=3,batch_size=None):
    from sklearn.decomposition import  IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X = ipca.fit_transform(a)
    return pd.DataFrame(X,index=a.index)

def pca(a,n=2):
    if a.shape[0]==a.shape[1]:
        l,v=np.linalg.eig(a)
        return pd.DataFrame(v[:,:n],index=a.index).applymap(lambda x: x.real)
    else:
        from sklearn.decomposition import PCA
        return pd.DataFrame(PCA(n_components=n).fit(a).transform(a), index=a.index)

class pval:
    @staticmethod
    def KS(a):
        """returns neg-log-pval of Kolgomorov and Smirnov test"""
        if not a.shape[0]: return 0
        try:
            return np.round(abs(np.log10(sc.stats.ks_2samp(a.iloc[:, 0], a.iloc[:, 1])[1])), 1)
        except:
            return 0
    @staticmethod
    def getFDR(a, T=[0.05, 0.01, 0.0025, 0.001, 0.0001]):
        b = pd.DataFrame([(t, a.size * t, a[a >= -np.log10(t)].size) for t in T], columns=['t', 'mt', 'discoveries']);
        b['fdr'] = b.mt / b.discoveries
        return b

    @staticmethod
    def getPvalKDE(x, kde=None):
        if kde is None:   kde = getDensity(x)
        pval = x.apply(lambda y: kde.integrate_box_1d(y, np.inf))
        return -pval.apply(np.log10).sort_index()

    @staticmethod
    def getQuantiles(X, quantiles):
        return X.quantile(quantiles, interpolation='nearest')

    @staticmethod
    def OR(cc):
        """Nx 2 dataframe
        First column is a binary label
        Second column is a categorical var
        """
        odds=lambda x: x.iloc[1]/x.iloc[0]
        return  odds(pval.crosstab(cc))
    @staticmethod
    def getQantilePvalues(X, kde=None, quantiles=np.arange(0, 1.001, 0.01)):
        if kde is None:   kde = getDensity(X)
        return pval.getPvalKDE(pval.getQuantiles(X, quantiles=quantiles), kde)

    @staticmethod
    def MW(yp, yn):
        import scipy as sc
        return -np.log10(sc.stats.mannwhitneyu(yp, yn, use_continuity=True)[1]).round(2)
    @staticmethod
    def crosstab(cc):
        return pd.crosstab(cc.iloc[:,0], cc.iloc[:,1])
    @staticmethod
    def qval(p,concat=False):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        qvalue = importr("qvalue")
        pvals = ro.FloatVector(p.values)
        rcode = 'qobj <-qvalue(p=%s,  lambda=seq(0.05, 0.95, 1))' % (pvals.r_repr())
        res = ro.r(rcode)
        q= pd.Series(list(ro.r('qobj$qvalue')), index=p.index)
        if concat:q=pd.concat([p,q],1,keys=['pval','qval'])
        return q

    #c=utl.scan.Genome(a,f=lambda x: utl.chi2SampleMeanPval(x,1))
    @staticmethod
    def zscore(x): return ((x-x.mean())/x.std()).astype(float)
    @staticmethod
    def zscoreChr(x): return x.groupby(level='CHROM').apply(pval.zscore).astype(float)
    @staticmethod
    def chi2(x,df=1):return - np.log10(np.exp(st.chi2.logsf(x,df)))


    @staticmethod
    def chi2SampleMean(x,df,dftot=None):
        if not x.size: return None
        df=df*x.size
        mu=x.mean()
        if dftot is not None: df=dftot
        return -np.log10(np.exp(st.gamma.logsf(mu,df/2.,scale=2./x.size)))

    @staticmethod
    def zpval(z):return -pd.Series(1-st.norm.cdf(pval.zscore(z).values)+ 1e-16,index=z.index).apply(np.log)
    @staticmethod
    def zgenome(x): return -pd.Series(1-st.norm.cdf(pval.zscoreChr(x).values)+ 1e-16,index=x.index).apply(np.log)

    # @staticmethod
    # zgenome2tail= lambda x: -pd.Series(1-st.norm.cdf(pval.zscoreChr(x).abs().values)+ 1e-16,index=x.index).apply(np.log)

    @staticmethod
    def z2tail(x):  return -pd.Series(1 - st.norm.cdf(pval.zscore(x).abs().values) + 1e-16, index=x.index).apply(np.log)

    @staticmethod
    def gammachi2Test(x,df):return -st.chi2.logsf(x,df), -st.gamma.logsf(x,df/2.,scale=2.),-st.gamma.logsf(x/df,df/2.,scale=2./df)
    @staticmethod
    def fisher(A):
        import rpy2.robjects as robjects
        if isinstance(A,pd.DataFrame):a=A.values
        else:a=A
        if a.shape[0]==2:
            r='fisher.test(rbind(c({},{}),c({},{})), alternative="less")$p.value'
            return robjects.r(r.format(a[0, 0], a[0, 1], a[1, 0], a[1, 1]))[0]
        elif a.shape[0]==3:
            r='fisher.test(rbind(c({},{}),c({},{}),c({},{})), alternative="less")$p.value'
            return robjects.r(r.format(a[0, 0], a[0, 1], a[1, 0], a[1, 1], a[2, 0], a[2, 1]))[0]

    @staticmethod
    def chi2ContingencyDF(A):
        a=A.dropna()
        try:
            return pval.chi2Contingency(pval.crosstab(a),True)
        except:
            pass
    @staticmethod
    def chi2ContingencyDFApply(A,ycol):
        cols=A.drop(ycol,1).columns
        return pd.Series(cols,index=cols).apply(lambda x: pval.chi2ContingencyDF(A[[ycol,x]]) )



    @staticmethod
    def chi2Contingency(A,log=False):
        import scipy as sc
        if isinstance(A,pd.DataFrame):a=A.values
        else:a=A
        p= sc.stats.chi2_contingency(a, correction=False)[1]
        if log: p=np.round(abs(np.log10(p)),2)
        return p

    @staticmethod
    def empirical(A,Z,positiveStatistic=True):#Z is null scores
        if positiveStatistic:
            a=A[A>0].sort_values()
            z=Z[Z>0].sort_values().values
        else:
            a=A.sort_values()
            z=Z.sort_values().values
        p=np.zeros(a.size)
        j=0
        N=z.size
        for i in range(a.size):
            while j<N:
                if a.iloc[i] <= z[j]:
                    p[i]=N-j +1
                    break
                else:
                    j+=1
            if j==N: p[i]=1
        return -pd.concat([pd.Series(p,index=a.index).sort_index()/(Z.size+1),A[A==0]+1]).sort_index().apply(np.log10)

    def CMH(x, num_rep=3):
        import rpy2.robjects as robjects
        r = robjects.r
        response_robj = robjects.IntVector(x.reshape(-1))
        dim_robj = robjects.IntVector([2, 2, num_rep])
        response_rar = robjects.r['array'](response_robj, dim=dim_robj)
        testres = r['mantelhaen.test'](response_rar);
        pvalue = testres[2][0];
        return pvalue

    def CMHcd(cd, DisCoverage=True, eps=1e-20, negLog10=True, damp=1):
        name = 'CMH ' + '-'.join(cd.columns.get_level_values('GEN').unique().values.astype(str))
        a = cd + damp
        num_rep = cd.shape[1] / (2 * cd.columns.get_level_values('GEN').unique().size)
        if DisCoverage:
            a.loc[:, pd.IndexSlice[:, :, 'D']] = (a.xs('D', level=2, axis=1) - a.xs('C', level=2, axis=1)).values
        a = a.apply(lambda x: pval.CMH(x.values.reshape(num_rep, 2, 2)), axis=1).rename(name) + eps
        if negLog10: a = -a.apply(np.log10)
        return a

    def getPvalFisher(AllGenes, putativeList, myList):
        cont = pd.DataFrame(pval.getContingencyTable(AllGenes=AllGenes, putativeList=putativeList, myList=myList));
        pval = -np.log10(1 - pval.fisher(cont.values))
        return pval, cont

    def getContingencyTable(AllGenes, putativeList, myList):
        """
                         |COMALE|Other |
        -------------------------------------
        Putative(Knouwn) |  a   |  b   |  A
        Other            |  c   |  d   |  B
        -------------------------------------
                         |  C   |  D   |  N
        """
        N = AllGenes.size;
        A = putativeList.size;
        C = myList.size
        a = np.intersect1d(putativeList, myList).size
        b = A - a
        c = C - a
        d = (N - C) - b
        df = pd.DataFrame(np.array([[a, b], [c, d]]), index=['Putative', 'Other'], columns=['myList', 'Other'])
        return df

def DataframetolaTexTable(DF, alignment=None, fname=None,shade=False):
    """
    Args:
        df: pandas dataframe
        alignment: python list of allignment of columns; default is ['c',..]; use ['c', 'p{4in}', 'c', 'c'] for wrapping
        fname: path to save latex table

    Returns:
        object:
    Returns: latex table
    """
    df=DF.copy(True)
    if alignment is None: alignment = list('c' * (df.shape[1]))
    sh=('',r'\rowcolor{Gray} ')[shade]
    df.iloc[:,0]=df.iloc[:,0].astype(str)
    df.iloc[::2,0]=sh + df.iloc[::2,0]
    csv = r'\centering \begin{tabular}{' + '|'.join(alignment) + '}\n' + df.to_csv(sep='\t', index=False).replace('\t',
                                                                                                                  '\t&').replace(
        '\n', '\\\\\n').replace('\\\\\n', '\\\\\hline\n', 1) + r'\end{tabular}'
    csv = csv.replace("%", '\%').replace('inf',r'$\infty$')
    if fname is not None:
        with open(fname, 'w') as f:  print >> f, csv
    return csv

def files(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]




def batch(iterable, n=10000000):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# def getDensity(X,width='silverman'):# not width can be string {scott or silverman} or positive real
def getDensity(X,width='scott'):# not width can be string {scott or silverman} or positive real
    from scipy.stats import gaussian_kde
    return gaussian_kde(X, bw_method=width)

@numba.vectorize
def vectorizedLog(x):
    return float(np.log(x))
def numbaLog(df):
    return  pd.DataFrame(vectorizedLog(df.values),columns=df.index,index=df.index).astype(float)

@numba.vectorize
def vectorizedExp(x):
    return float(np.exp(x))
def numbaExp(df):
    return  pd.DataFrame(vectorizedExp(df.values),columns=df.index,index=df.index).astype(float)

class EE:
    @staticmethod
    def fx(x, s=0.0, h=0.5):
        Z=(1 + s) * x ** 2 + 2 * (1 + h * s) * x * (1 - x) + (1 - x) ** 2
        if Z>0:
            return ((1 + s) * x ** 2 + (1 + h * s) * x * (1 - x)) / (Z)
        else:
            return 0

    @staticmethod
    def sig(x): return 1. / (1 + np.exp(-x))

    @staticmethod
    def logit(p): return np.log(p) - np.log(1 - p)


    # def logit_(p): return T.log(p) - T.log(1 - p)


    # def sig_(x): return 1. / (1 + T.exp(-x))

    @staticmethod
    def Nu(s, t, nu0, theta, n=2000): return EE.Z(EE.sig(t * s / 2 + EE.logit(nu0)), n, theta)

    @staticmethod
    def forward(x0=0.005,h=0.5,s=1,t=150):
        def f(x,h=0.5,s=1): return ((1+s)*x*x + (1+h*s)*x*(1-x) )/((1+s)*x*x + 2*(1+h*s)*x*(1-x)  +(1-x)**2)
        x=[x0]
        for i in range(t):
            x+=[f(x[-1],h,s)]
        return pd.Series(x)

    floatX = 'float64'

    @staticmethod
    def Z(nu, n, theta): return theta * (
    nu * ((nu + 1) / 2. - 1. / ((1 - nu) * n + 1)) + (1 - nu) * ((n + 1.) / (2 * n) - 1. / ((1 - nu) * n + 1)))


def floorto(x, base=50000):
    return int(base * np.floor(float(x)/base))

def roundto(x, base=50000):
    return int(base * np.round(float(x)/base))

def ceilto(x, base=50000):
    return int(base * np.ceil(float(x)/base))


class scan:
    @staticmethod
    def cdf(x):
        import pylab as plt
        ax=plt.subplots(1,2,figsize=(8,3),dpi=1)[1]
        # sns.distplot(x,ax=ax[0])
        CDF(x).plot(label='CDF',lw=4,c='k',alpha=0.75, ax=ax[1]);
        c='darkblue'
        ax[1].axvline(x.quantile(0.5),c=c,alpha=0.5,label='Median={}'.format(x.quantile(0.5)));
        ax[1].axvline(x.quantile(0.95),c=c,ls='--',alpha=0.5,label='Q95     ={}'.format(x.quantile(0.95)));
        ax[1].axvline(x.quantile(0.99), c=c,ls='-.', alpha=0.5, label='Q99     ={}'.format(x.quantile(0.99)));
        ax[1].legend();
    @staticmethod
    def topK(x, k=2000):
        return x.sort_values(ascending=False).iloc[:k]
    @staticmethod
    def idf(a, winSize=50000, names=None):
        if names == None: names = [a.name, 'n']
        x=scan.Genome(a.dropna(), f={names[0]: np.mean, names[1]: len}, winSize=winSize)
        x.columns=[0,'n']
        return x

    @staticmethod
    def Genome(genome, f=lambda x: x.mean(), uf=None,winSize=50000, step=None, nsteps=5, minSize=None):
        """
        Args:
            genome: scans genome, a series which CHROM and POS are its indices
            windowSize:
            step:
            f: is a SCALAR function or dict of SCALAR fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
            Only good for scanning a series with dictionary of scalar fucntions
            uf: is a universal function which returns a dataframe e.g. uf=lambda x: pd.DataFrame(np.random.rand(2,3))
            good for scanning a dataframe (which each column to be scanned) with a scalar or  univesal fucntions
        Returns:
        """
        if len(genome.shape)>1:
            return genome.apply(lambda x: scan.Genome(x,f=f,uf=uf,winSize=winSize,step=step,nsteps=nsteps))

        if step is None:step=winSize/nsteps
        df = genome.groupby(level='CHROM').apply(lambda ch: scan.Chromosome(ch.loc[ch.name],f,uf,winSize,step))
        if minSize is not None:
            n=scan.Genome(genome, f=lambda x: x.size, winSize=winSize, step=step, minSize=None)
            if f==np.sum:
                df=df.loc[TI(n>=minSize)]
            else:
                df=df[n>=minSize]
        return df

    @staticmethod
    def Chromosome(x,f=np.mean,uf=None,winSize=50000,step=10000):
        """
        Args:
            chrom: dataframe containing chromosome, positions are index and the index name should be set
            windowSize: winsize
            step: steps in sliding widnow
            f: is a SCALAR function or dict of SCALAR fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
            uf: is a universal function which returns a dataframe e.g. uf=lambda x: pd.DataFrame(np.random.rand(2,3))
        Returns:
        """
        # print 'Chromosome',x.name
        if x.index[-1] - x.index[0] < winSize:
            f=(f,uf)[uf is not None]
            i= roundto(((x.index[-1] + x.index[0]) / 2.),10000)+5000
            z=pd.DataFrame([f(x)], index=[i])
            z.index.name='POS'
            return z

        POS=x.index.get_level_values('POS')
        res=[]
        # Bins=np.arange(max(0,roundto(POS.min()-winSize,base=step)), roundto(POS.max(),base=step),winSize)
        Bins = np.arange(0, roundto(POS.max(), base=step), winSize)
        for i in range(int(winSize/step)):
            bins=i*step +Bins
            windows=pd.cut( POS, bins,labels=(bins[:-1] + winSize/2).astype(int))
            if uf is None:
                tmp=x.groupby(windows).agg(f)
                tmp.index=tmp.index.astype(int);
                tmp.index.name='POS'

            else:
                tmp=x.groupby(windows).apply(uf)
                tmp=tmp.reset_index()
                tmp.iloc[:,0]=tmp.iloc[:,0].astype(int)
                tmp.columns=['POS']+tmp.columns[1:].tolist()
                tmp= tmp.set_index(tmp.columns[:-1].tolist()).iloc[:,0]
            res+=[tmp]
        df=pd.concat(res).sort_index().dropna()
        # if minSize is not None:
        #     df[df.COUNT < minSize] = None
        #     df = df.loc[:, df.columns != 'COUNT'].dropna()
        return df

    @staticmethod
    def scanGenomeSNP(genome, f=np.mean, winSize=300,skipFromFirst=0,step=None):
        if step is None:step=int(winSize/5)
        return  genome.groupby(level=0).apply(lambda x: scan.ChromosomeSNP(x.iloc[skipFromFirst:],f,winSize,step))

    @staticmethod
    def scanChromosomeSNP(x,f,winSize,step):
        """
        Args:
            chrom: dataframe containing chromosome, positions are index and the index name should be set
            windowSize: winsize
            step: steps in sliding widnow
            f: is a function or dict of fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
        Returns:
        """
        BinsStart=pd.Series(np.arange(0, roundto(x.size,base=step),winSize),name='start')
        def createBins(i):
            bins=pd.DataFrame(i*step +BinsStart)
            bins['end'] = bins.start+ winSize
            bins.index=((bins.start+bins.end)/2).astype(int)
            return bins
        bins=pd.concat(map(createBins,range(int(winSize/step)))).sort_index()
        bins[bins>x.size]=None
        bins=bins.dropna().astype(int)
        bins=bins.apply(lambda bin: f(x.iloc[range(bin.start,bin.end)]),axis=1)
        bins.index=x.index[bins.index]
        if bins.shape[0]:return bins.loc[x.name]

    @staticmethod
    def smooth(a, winsize, normalize=True):
        if normalize:
            f = lambda x: x / x.sum()
        else:
            f = lambda x: x
        return scan.scan3way(f(a), winsize, np.mean)

    @staticmethod
    def threeWay(a, winsize, f):
        return pd.concat([a.rolling(window=winsize).apply(f),
                          a.rolling(window=winsize, center=True).apply(f),
                          a.iloc[::-1].rolling(window=winsize).apply(f).iloc[::-1]],
                         axis=1)

    @staticmethod
    def scan3way(a, winsize, f):
        return scan.threeWay(a, winsize, f).apply(lambda x: np.mean(x), axis=1)

    @staticmethod
    def scan2wayLeft(a, winsize, f):
        """Moving average with left ellements and centered"""
        X = scan.threeWay(a, winsize, f)
        x = X[[0, 1]].mean(1)
        x[x.isnull] = x[2]
        return x

    @staticmethod
    def scan2wayRight(a, winsize, f):
        """Moving average with left ellements and centered"""
        return scan.threeWay(a, winsize, f).iloc[:, 1:].apply(lambda x: np.mean(x), axis=1)

    @staticmethod
    def plotBestFly(windowStat, X,  pad=30000, i=None, mann=True, foldOn=None,rep=None):
        # i0 = (x.sum(1) > 0.05) & (x.sum(1) < 6.95)
        if rep is None: x=X
        else: x=X.xs(rep,1,1)
        if i is None:
            i = BED.intervali(windowStat.dropna().sort_values().index[-1], pad);
        import UTILS.Plots as pplt
        pplt.Trajectory.Fly(mask(x, i), subsample=2000, reps=[1, 2, 3], foldOn=foldOn);
        # plt.title('Rep {}, {} '.format(rep, utl.BED.strMbp(i)));plt.show()
        if mann: pplt.Manhattan(windowStat, top_k=1)
        return BED.str(i)


class BED:
    @staticmethod
    def shift(i, offset):
        if isinstance(i,str):
            j = BED.intervals(i)
        else:
            j=i.copy(True)
        j.end += offset
        j.start += offset
        return BED.str(j)

    @staticmethod
    def interval(CHROM,start,end=None):
        if end is None: end=start
        i=pd.Series( {'CHROM':CHROM,'start':start,'end':end})
        i['len']=i.end-i.start
        i['istr']=BED.str(i)
        return i
    @staticmethod
    def intervali(i,expand=0):
        CHROM,POS=i
        i= BED.interval(CHROM,POS)
        if expand: i=BED.expand(i,expand)
        return i
    @staticmethod
    def intervals(i):
        if i[:3]=='chr': i=i[3:]
        CHROM,start,end=INT(i.split(':')[0]),INT(i.split(':')[1].split('-')[0]),INT(i.split(':')[1].split('-')[1])
        return BED.interval(CHROM,start,end)

    @staticmethod
    def expand(i, pad=500000,left=None,right=None):
        pad=int(pad)
        x = i.copy(True)
        if left is not None: pad=left
        x.start = x.start - pad;
        if right is not None: pad = right
        x.end += pad;
        x.start = max(0, x.start)
        x['len']=x.end-x.start
        return x

    @staticmethod
    def str(i):
        return '{}:{}-{}'.format(INT(i.CHROM),INT(i.start),INT(i.end))

    @staticmethod
    def strMbp(i,short=False):
        s= 'chr{}:{:.2f}-{:.2f}Mb'.format(INT(i.CHROM), (i.start/1e6), (i.end/1e6))
        if short: s=s.replace('chr','').replace('Mb','')
        return s

    @staticmethod
    def drop_duplicates(file_a,file_b,outfile=None):
        a=pd.concat([pd.read_csv(file_a,sep='\t',header=None),pd.read_csv(file_b,sep='\t',header=None)]).drop_duplicates().sort_values([0,1])
        if outfile is None: return a
        a.to_csv(outfile,index=None,header=None,sep='\t')

    @staticmethod
    def save(dff, fname=None, intervalName=None, intervalScore=None, fhandle=None):
        if 'start' not in dff.columns:
            df = dff.copy(True)
            df['end'] = df.POS
            df.rename(columns={'POS': 'start'}, inplace=True)
        else:
            df = dff
        cols = ['start', 'end']
        if intervalName is not None:
            cols += [intervalName]
            if intervalScore is not None:
                cols += [intervalScore]
        bed = df.reset_index().sort_values(['CHROM', 'start']).set_index('CHROM')[cols]
        if fhandle is not None:
            fhandle.write( bed.to_csv(header=False, sep='\t'))
            fhandle.flush()
        else:
            bed = bed.to_csv(header=False, sep='\t', path_or_buf=fname)
            return bed
    @staticmethod
    def getIntervals(regions, padding=0,agg='max',ann=None,expandedIntervalGenes=False,dropi=True):
        """WARNING: i columns correspond to the iloc of the input that is soted by CHROM,start,end
        To Avoid Bugs in downstream analysis sort the input by position so that i field make sense all the time.
        """
        regions=regions.sort_index() #important
        def get_interval(df, padding, merge=False):
            df = df.sort_index()
            df = pd.DataFrame([df.values, df.index.get_level_values('POS').values - padding,
                               df.index.get_level_values('POS').values + padding], index=['score', 'start', 'end']).T
            df.start = df.start.apply(lambda x: (x, 0)[x < 0])
            df['len'] = df.end - df.start
            return df.set_index('start')
        if len(regions.shape)==1:
            df = regions.groupby(level=0).apply(lambda x: get_interval(x, padding)).reset_index().set_index('CHROM')
        elif 'start' not in regions.columns:
            df = regions.groupby(level=0).apply(lambda x: get_interval(x, padding)).reset_index().set_index('CHROM')
        else:
            df=regions.copy(True)
            df.start-=padding;df.end+=padding
            df.loc[df.start<0,'start']=0
        if 'score' not in df: df['score']=1
        df=df.reset_index().sort_values(['CHROM','start'])
        df[['CHROM','start','end']]=df[['CHROM','start','end']].applymap(INT).sort_values(['CHROM','start'])
        csv = df[['CHROM','start', 'end']]
        csv['name']=range(csv.shape[0])
        csv=csv.to_csv(header= False, sep='\t',index=False)
        cmd=[home+'miniconda2/bin/bedtools', 'merge'] #+'-c 4 -o distinct'.split()
        csv = Popen(cmd , stdout=PIPE, stdin=PIPE, stderr=STDOUT).communicate(input=csv)[0]
        df = pd.DataFrame(map(lambda x: x.split(), csv.split('\n'))).dropna()
        df.columns=['CHROM', 'start', 'end']

        df.CHROM=df.CHROM.apply(INT)
        df=df.dropna().set_index('CHROM').applymap(INT)
        df['len'] = df.end - df.start
        df=df.reset_index()
        df['istr']=df.apply(BED.str,1)
        if ann is not None:
            if not expandedIntervalGenes:
                DF=ann.loc[regions.index]
                if 'genes' in DF.columns:
                    # print(DF)
                    # x=df.reset_index().i.apply(lambda x: pd.Series(str(x).split(';')).astype(int)).stack().astype(int).reset_index(level=1,drop=True)
                    # y=x.groupby(level=0).apply(lambda xx: pd.DataFrame(DF.loc[regions.sort_index().iloc[xx].index].genes.dropna().tolist()).stack().unique()).rename('genes')
                    #
                    # df=df.reset_index().join(y)
                    df['genes']=df.apply(lambda x: getGeneList( mask(ann,x.istr).genes),1)
            else:
                df=df.reset_index()
                df['genes']=None
                for idx in df.index:
                    df.set_value(idx,'genes',getGeneList(mask(ann, interval=df.loc[idx]).genes))
        # df=df.reset_index()
        # df['stri']=df.apply(lambda x: 'chr'+BED.str(x),1)
        # if dropi: df=df.drop('i',1)
        return df

    @staticmethod
    def getIntervals_v2_17(regions, padding=0, agg='max', ann=None, expandedIntervalGenes=False, dropi=True):
        """WARNING: i columns correspond to the iloc of the input that is soted by CHROM,start,end
        To Avoid Bugs in downstream analysis sort the input by position so that i field make sense all the time.
        """
        regions = regions.sort_index()  # important

        def get_interval(df, padding, merge=False):
            df = df.sort_index()
            df = pd.DataFrame([df.values, df.index.get_level_values('POS').values - padding,
                               df.index.get_level_values('POS').values + padding], index=['score', 'start', 'end']).T
            df.start = df.start.apply(lambda x: (x, 0)[x < 0])
            df['len'] = df.end - df.start
            return df.set_index('start')

        if len(regions.shape) == 1:
            df = regions.groupby(level=0).apply(lambda x: get_interval(x, padding)).reset_index().set_index('CHROM')
        elif 'start' not in regions.columns:
            df = regions.groupby(level=0).apply(lambda x: get_interval(x, padding)).reset_index().set_index('CHROM')
        else:
            df = regions.copy(True)
            df.start -= padding;
            df.end += padding
            df.loc[df.start < 0, 'start'] = 0
        df = df.reset_index().sort_values(['CHROM', 'start']).set_index('CHROM')
        df['name'] = range(df.shape[0])
        df.score = (df.score * 1000).round()
        df[['start', 'end', 'name', 'score']] = df[['start', 'end', 'name', 'score']].applymap(int)
        csv = df[['start', 'end', 'name', 'score']].to_csv(header=False, sep='\t')
        csv = \
            Popen([home + 'miniconda2/bin/bedtools', 'merge', '-nms', '-scores', agg, '-i'], stdout=PIPE, stdin=PIPE,
                  stderr=STDOUT).communicate(input=csv)[
                0]
        df = pd.DataFrame(map(lambda x: x.split(), csv.split('\n'))).dropna()
        df.columns = ['CHROM', 'start', 'end', 'i', 'score']
        df.CHROM = df.CHROM.apply(INT)
        df = df.dropna().set_index('CHROM').applymap(INT)
        df.score /= 1000
        df['len'] = df.end - df.start
        if ann is not None:
            if not expandedIntervalGenes:
                DF = ann.loc[regions.index]
                if 'genes' in DF.columns:
                    x = df.reset_index().i.apply(lambda x: pd.Series(str(x).split(';')).astype(int)).stack().astype(
                        int).reset_index(level=1, drop=True)
                    y = x.groupby(level=0).apply(lambda xx: pd.DataFrame(
                        DF.loc[regions.sort_index().iloc[xx].index].genes.dropna().tolist()).stack().unique()).rename(
                        'genes')

                    df = df.reset_index().join(y)
            else:
                df = df.reset_index()
                df['genes'] = None
                for idx in df.index:
                    df.set_value(idx, 'genes', getGeneList(mask(ann, interval=df.loc[idx]).genes))
        df = df.reset_index()
        df['stri'] = df.apply(lambda x: 'chr' + BED.str(x), 1)
        if dropi: df = df.drop('i', 1)
        return df



    @staticmethod
    def intersection(dfa, dfb, dfa_interval_name='Gene_ID',dfb_interval_name='len'):
        dfb.start = dfb.start.astype(int)
        dfb.end = dfb.end.astype(int)
        import tempfile
        with tempfile.NamedTemporaryFile()as f1, tempfile.NamedTemporaryFile() as f2:
            if 'POS' in dfa.index.names:
                BED.save(dfa.reset_index()[['CHROM', 'POS', dfa_interval_name]].drop_duplicates(),intervalName=dfa_interval_name, fhandle=f1)
            else:
                BED.save(dfa, intervalName=dfa_interval_name, fhandle=f1)
            BED.save(dfb, intervalName=dfb_interval_name, fhandle=f2)
            csv = Popen(['bedtools', 'intersect', '-wb', '-wa', '-a', f1.name, '-b', f2.name], stdout=PIPE, stdin=PIPE,
                        stderr=STDOUT).communicate()[0]
        df = pd.DataFrame(map(lambda x: x.split(), csv.split('\n')),).dropna()
        try:
            df=df.iloc[:,[0,1,2,3,7]]
            df.columns=['CHROM', 'start', 'end', dfa_interval_name, dfb_interval_name]
            df=df.set_index('CHROM')
            df[['start', 'end']] = df[['start', 'end']].astype(int)
            return df
        except:
            return None

    @staticmethod
    def saveBEDGraph(a,name,fout_name,color='255,0,0',browser_pos='chrX:1-1000',chromLen=None,URL=None):
        if len(a.shape)==1:
            b=a.dropna().rename('score').reset_index()
        else:
            b=a.dropna().reset_index()
        if 'start' not in b.columns:
            b['start']=b.POS-1;b['end']=b.POS
        if not b.CHROM.astype(str).apply(lambda x:'chr' in x).sum():
            b.CHROM=b.CHROM.apply(lambda x: 'chr{}'.format(x))
        if URL is not None:
            b['UCSC']=None
            for i,row in b.iterrows():
                b.loc[i,'UCSC'] ='=HYPERLINK("{}","GB")'.format(URL.format(row.CHROM,int(row.start),int(row.end)))
            b.to_csv(fout_name+'.tsv',sep='\t',index=None)
        with open(fout_name,'w') as fout:
            print >> fout,"browser position",browser_pos
            print >> fout,"browser hide all"
            if type(b.score.iloc[0])==str :
                print >> fout,'track name="{}" color={} '.format(name,color)
            else:
                print >> fout,'track type=bedGraph name="{}" autoScale=off  visibility=full color={} viewLimits={}:{} priority=20'.format(name,color,min(0,b.score.min()), np.ceil(b.score.max()*10)/10)
            print >>fout,b[['CHROM','start','end','score']].to_csv(sep=' ',index=None,header=None)
            fout.flush()
        import subprocess
        subprocess.call('rm {}'.format(fout_name+'.gz'),shell=True)
        subprocess.call('bgzip {}'.format(fout_name),shell=True)


    @staticmethod
    def saveBEDGraphDF(DF,fout_path,color='255,0,0',colors=None,browser_pos='chrX:10000-12000',ver=None,winSize=None,viewlim=None):
        df=DF.reset_index()
        tracks=DF.columns
        colormap={'r':'255,0,0','g':'0,255,0','b':'0,0,255','k':'0,0,0'}
        if winSize is None:
            df['start']=df.POS-1;df['end']=df.POS
            fout_path+='.snps.bedgraph'
        else:
            df['start']=df.POS-winSize/5;df['end']=df.POS+winSize/5-1
            fout_path+='.intervals.bedgraph'
        if ver is not None:
            chromLen=getChromLen(ver)
            df=df.groupby('CHROM').apply(lambda x:x[(x.end<chromLen[x.name]).values & (x.start>=0) ])
        df.set_index(['CHROM','POS']).groupby(level=0).apply(lambda x:x.end)
        df.CHROM=df.CHROM.apply(lambda x: 'chr{}'.format(x))
        with open(fout_path,'w') as fout:
            print >> fout,"browser position",browser_pos
            print >> fout,"browser hide all"
            for i,track in enumerate(tracks):
                if colors is not None:color=colormap[colors[i]]
                if viewlim is None:
                    vl=min(0, df[track].min()), np.ceil(df[track].max() * 10) / 10
                else:
                    vl=viewlim[i]
                    if vl is None:vl=min(0, df[track].min()), np.ceil(df[track].max() * 10) / 10
                print >> fout,'track type=bedGraph name="{}" autoScale=off  visibility=dense color={} viewLimits={}:{} priority=20'.format(track,color,vl[0],vl[1])
                print >>fout,df[['CHROM','start','end',track]].dropna().to_csv(sep=' ',index=None,header=None)
            fout.flush()
        import subprocess
        subprocess.call('rm -f {}'.format(fout_path+'.gz'),shell=True)
        subprocess.call('bgzip {}'.format(fout_path),shell=True)

    @staticmethod
    def xmap_bed(Interval=None,variants=None,hgFrom=19, hgTo=38,removeXPchromSNPs=True,keepOnlyPos=False,chainPath=home+'storage/Data/Human/CrossMap-0.2.5/chains'):
        """
        Args:
            hgFrom: (int) assembly version eg: 19
            hgTo: int) assembly version eg: 38
            interval: dataframe with CHROM, start, end
        Returns:
            out: dataframe with CHROM, start, end
        """
        if variants is not None:
            Interval=variants.reset_index();
            Interval['start']=Interval.POS;Interval['end']=Interval.POS
        if keepOnlyPos:
            interval=Interval[['CHROM','start','end']]
        else:
            interval=Interval.copy(True)
        hasChr=False
        # print interval
        if 'chr' in str(interval.CHROM.iloc[0]): hasChr=True
        if not interval.CHROM.astype(str).apply(lambda x:'chr' in x).sum() and hgFrom !=37:
            interval.CHROM='chr'+interval.CHROM.apply(convertToIntStr)
        interval.start=interval.start.astype(int)
        interval.end=interval.end.astype(int)
        hgFrom=('hg{}'.format(hgFrom),'GRCh37')[hgFrom==37]
        hgTo=('Hg{}'.format(hgTo),'GRCh37')[hgTo==37]
        chainfile = "{}/{}To{}.over.chain.gz".format(chainPath,hgFrom, hgTo)
        in_file=home+'xmap.in.tmp'
        out_file=home+'xmap.out.tmp'
        import subprocess
        with open(in_file ,'w') as f1:
            BED.save(interval.reset_index()[['CHROM','start','end','index']], fhandle=f1,intervalName='index')
        cmd = "/home/arya/miniconda2/bin/CrossMap.py bed  {} {} {}".format(chainfile, in_file, out_file)
        # print cmd
        subprocess.call(cmd,shell=True)
        maped=pd.DataFrame(map(lambda x: x.split(), open(out_file).readlines()),columns=['CHROM','start','end','ID']).dropna()
        maped.ID=maped.ID.astype('int')
        maped=maped.set_index('ID').sort_index()
        maped=pd.concat([interval,maped],1,keys=[hgFrom,hgTo])
        # print maped
        def ff(x):
            try:return x[3:]
            except:return x
        fff=(lambda x:x[3:],lambda x:x)['chr' not in str(maped[hgFrom].CHROM.iloc[0])]
        maped[(hgFrom,'CHROM')]=maped[(hgFrom,'CHROM')].apply(lambda x: INT(fff(x)))
        fff=(lambda x:x[3:],lambda x:x)['chr' not in str(maped[hgTo].CHROM.iloc[0])]
        maped[(hgTo,'CHROM')]=maped[(hgTo,'CHROM')].apply(lambda x: INT(ff(x)))
        maped.sort_values([(hgFrom,'CHROM'),(hgFrom,'start')])
        maped=maped.set_index((hgFrom,'CHROM'))
        maped.index.name='CHROM'
        if removeXPchromSNPs:maped=maped[maped.index==maped[(hgTo,'CHROM')]]
        os.remove(in_file)
        os.remove(out_file)
        os.remove(out_file+'.unmap')
        return maped

def createAnnotation(vcf ,db='BDGP5.75',computeSNPEFF=True,ud=0,snpeff_args=''):
    #snps=loadSNPID()
    import subprocess
    fname=vcf.replace('.vcf','.SNPEFF.vcf').replace('.gz','')
    fname=vcf+'.SNPEFF.vcf'
    assert fname!=vcf
    if computeSNPEFF:
        cmd='java -Xmx4g -jar ~/bin/snpEff/snpEff.jar {} -ud {} -s snpeff.html {} {} | cut -f1-8 > {}'.format(snpeff_args,ud,db,vcf,fname)
        # print cmd
        subprocess.call(cmd,shell=True)
        # print 'SNPEFF is Done'
    import vcf
    def saveAnnDataframe(fname,x='ANN'):
        # print(x), fname
        ffields = lambda x: x.strip().replace("'", '').replace('"', '').replace(' >', '')
        vcf_reader = vcf.Reader(open(fname, 'r'))
        csv=fname.replace('SNPEFF.vcf',x+'.csv')
        with open(csv,'w') as fout:
            print >>fout,'\t'.join(['CHROM','POS','REF','ID']+map(ffields,vcf_reader.infos[x].desc.split(':')[1].split('|')))
            for rec in  vcf_reader:
                if x in rec.INFO:
                    for line in map(lambda y:('\t'.join(map(str,[INT(rec.CHROM),rec.POS,rec.REF,rec.ID]+y))),map(lambda ann: ann.split('|') ,rec.INFO[x])):
                        # print line
                        if x=='LOF':
                            line=line.replace('(','').replace(')','')
                        print >>fout, line
        uscols=[range(10),range(6)][x=='LOF']
        df = pd.read_csv(csv, sep='\t', usecols=uscols).set_index(['CHROM', 'POS']).apply(lambda x: x.astype('category'))
        df.to_pickle(csv.replace('.csv','.df'))
        try:
            df=df[['Annotation', 'Annotation_Impact', 'Gene_Name', 'Feature_Type']]
            df.to_pickle(csv.replace('.csv','.sdf'))
            gz.save(df, csv.replace('.csv', '.s.gz'))
        except:
            pass
    saveAnnDataframe(fname,'ANN')
    saveAnnDataframe(fname,'LOF')

def burden():
    f= UKBB_PATH+ 'chr21.vcf.gz.ann.gz.LOF.df'
    a=pd.read_pickle(f)
    a['REF_ID']=a.ID.apply(lambda x: x.split('_')[2])
    a['ALT_ID'] = a.ID.apply(lambda x: x.split('_')[3])
    a


def localOutliers(a, q=0.99,winSize = 2e6):
    def f(xx):
        window = int(winSize / 10000)
        th =scan.scan3way(xx,window,f=lambda x: pd.Series(x).quantile(q))
        return xx[xx >= th].loc[xx.name]
    return a.loc[a.groupby(level=0).apply(f).index]

def renameColumns(DF,suffix,pre=True):
    df=DF.copy(True)
    if pre:
        df.columns=map(lambda x:'{}{}'.format(suffix,x),df.columns)
    else:
        df.columns=map(lambda x:'{}{}'.format(x,suffix),df.columns)
    return df

def makeCategory(df,field):
    try:
        df[field]=df[field].astype('category')
    except:
        pass
    return df

import numbers
def isNumber(x):
    return isinstance(x, numbers.Number)
def convertToIntStr(x):
    if isNumber(x):
        return '{:.0f}'.format(x)
    else:
        return x


class  Enrichment:
    @staticmethod
    def GOEA(bg,study,assoc=None,alpha=0.05,propagate=False):
        """
        Args:
            bg: list, Background Gene set (possibly the polymorphic genes in the experiment), eg: ['FBgn111','FBgn112',...]
            study: list, a  subset of bg, eg: ['FBgn111']
            assoc: series which index is gene id and value is a set of go terms, eg.
            pd.Series([{GO:0005615, GO:0007566}],index=[ 'FBgn111'])
            alpha:significance level
            propagate: if propagate counts in GO hierarchy
        Returns:

        """
        print('bg={} stydy={}'.format(len(bg),len(study)))
        from goatools.go_enrichment import GOEnrichmentStudy
        from goatools.obo_parser import GODag
        if assoc is None:
            assoc=Enrichment.loadAssociations()
        obodag = GODag(dataPath+"GO/go-basic.obo")
        goea= GOEnrichmentStudy(bg,assoc.to_dict(),obodag,propagate_counts = propagate,alpha = alpha, methods = ['fdr_bh'])
        goea_results_all = goea.run_study(study)
        goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < alpha]
        import tempfile
        # print goea_results_sig
        try:
            with tempfile.NamedTemporaryFile()as f:
                goea.wr_tsv(f.name, goea_results_sig)
                df=pd.read_csv(f.name,sep='\t')
            return df
        except:
            print('No Association found!')

    @staticmethod
    def loadAssociations(species='fly'):
        taxid={'fly':7227, 'human':9606,'mouse':10090,'rat':10116}
        from goatools.associations import read_ncbi_gene2go
        aa=pd.Series(read_ncbi_gene2go(dataPath+"GO/gene2go", taxids=[taxid[species]]))
        if species == 'fly':
            bb=pd.read_pickle(dataPath+'GO/fly.mygene.df')
            bb.index=map(int,bb.index)
            aa=bb.join(aa.rename('GO')).set_index("FLYBASE")['GO']
        return aa

    @staticmethod
    def getGeneName(geneIDs=None,species='human'):
        try:
            return pd.read_pickle(dataPath+'GO/{}.mygene.symbol.df'.format(species))
        except:
            import mygene
            names=mygene.MyGeneInfo().querymany(geneIDs, scopes="entrezgene,flybase",  species=species, as_dataframe=True,fields='all')
            names.to_pickle(dataPath+'GO/{}.mygene.df'.format(species))
            return names

    @staticmethod
    def GOtablPrint(a):
        return a.join(a.study_items.apply(lambda x: ', '.join(getGeneName().loc[x.split(', ')].tolist())).rename('genes')).drop(
            ['enrichment', '# GO', 'ratio_in_study', 'p_uncorrected', 'ratio_in_pop', 'study_items'], axis=1).sort_values(
            ['NS', 'p_fdr_bh']).set_index('NS').rename(columns={'study_count': 'count'})

    @staticmethod
    def loadGowinda(path='/home/arya/out/real/gowinda/', fname='cands.final.out.snp.tsv'):
        gowinda = pd.read_csv(path + fname, sep='\t', header=None)[[0, 4, 5, 6, 7, 8, 9]]
        gowinda.columns = ['GO ID', '-log($p$-value)', 'Hits', 'Num of Genes', 'Total Genes', 'GO Term', 'Genes']
        return gowinda

    @staticmethod
    def saveGowinda(cands, all, path=outpath + 'real/gowinda/', fname='cands.final.txt'):
        cands.sort_index().reset_index().drop_duplicates().dropna().to_csv(path + fname, sep='\t', header=None,
                                                                           index=False)
        all.sort_index().reset_index().drop_duplicates().dropna().to_csv(path + 'allsnps.txt', sep='\t', header=None,
                                                                         index=False)


class Dmel:
    @staticmethod
    def loadDGRP(i):
        cmd = '/home/arya/bin/bcftools/bcftools view -r {}:{}-{} /pedigree2/projects/arya/Data/Dmelanogaster/DGRP/dgrp2.vcf.gz | grep -v "##"'.format(
            i.CHROM, i.start, i.end)
        a = execute(cmd, header=0)
        a.iloc[:, 9:] = a.iloc[:, 9:].replace({'./.': '-1', '0/0': 0, '1/1': 1}).astype(int)
        a.rename(columns={'#CHROM': 'CHROM'})
        return a.set_index(['CHROM', 'POS'])
    @staticmethod
    def geneAssociations(relation=False):
        """
        Args:
            relation: if True returns 1 to N relation otherwise returns 1 to a set relation
        Returns:
        """
        # ftp://ftp.flybase.net/releases/current/precomputed_files/go/gene_association.fb.gz
        a=pd.read_csv(dataPathDmel+'gene_association.fb',sep='\t',comment='!',header=None).set_index(1)[4].rename('GO')
        a.index.name=None
        if relation:
            return a
        else:
            return a.groupby(level=0).apply(lambda x: set(x.tolist()))

    @staticmethod
    def geneCoordinates(assembly=5,allOrganisms=False):
        # ftp://ftp.flybase.net/releases/FB2014_03/precomputed_files/genes/gene_map_table_fb_2014_03.tsv.gz
        # ftp://ftp.flybase.net/releases/FB2016_04/precomputed_files/genes/gene_map_table_fb_2016_04.tsv.gz
        fname=dataPathDmel+('gene_map_table_fb_2016_04.tsv','gene_map_table_fb_2014_03.tsv')[assembly==5]
        import re
        try:
            return pd.read_pickle(fname+'.{}.df'.format(('dmel','all')[allOrganisms]))
        except:
            col=4;# sequence_loc column
            a=pd.read_csv(fname, sep='\t', comment='#',header=None,na_values='',keep_default_na=False).dropna(subset=[col])
            df=pd.DataFrame([re.split('\..|:|\(',r[col])[:-1]+r[:2].tolist()+['\\' not in r[0]] for _,r in a.iterrows()])
            df.columns=['CHROM','start','end','name','FBgn','dmel']
            df[['start', 'end']] = df[['start', 'end']].astype(int)
            df = df.sort_values(['CHROM', 'start'])
            df.index=range(df.shape[0])
            df.to_pickle(fname+'.all.df')
            df=df[df.dmel]
            df.index=range(df.shape[0])
            df.to_pickle(fname+'.dmel.df')
            return df[df.dmel]

    @staticmethod
    def getEuChromatin(scores):
        def filter(x):
            try:
                return scores.loc[x.name][(x.loc[x.name].start <= scores.loc[x.name].index.values) & (
                            x.loc[x.name].end >= scores.loc[x.name].index.values)]
            except:
                pass

        return Dmel.getEuChromatinCoordinates().groupby(level=0).apply(filter)

    @staticmethod
    def getEuChromatinCoordinates():
        """http://petrov.stanford.edu/cgi-bin/recombination-rates_updateR5.pl"""
        a = pd.Series(
            """X : 1.22 .. 21.21
            2L : 0.53 .. 18.87
            2R : 1.87 .. 20.86
            3L : 0.75 .. 19.02
            3R : 2.58 .. 27.44 """.split('\n'))
        return pd.DataFrame(a.apply(lambda x: [x.split(':')[0]] + x.split(':')[1].split('..')).tolist(),
                            columns=['CHROM', 'start', 'end']).applymap(str.strip).set_index('CHROM').astype(
            float) * 1e6

    @staticmethod
    def getChromLen(ver=5):
        return pd.read_csv(home + 'storage/Data/Dmelanogaster/refrence/dmel{}.fasta.fai'.format(ver), sep='\t',
                           header=None).replace({'dmel_mitochondrion_genome': 'M'}).rename(
            columns={0: 'CHROM'}).set_index(['CHROM'])[1].rename("length")

    loadSNPID = lambda: pd.read_csv('/home/arya/storage/Data/Dmelanogaster/dm5.vcf', sep='\t', usecols=range(5),
                                    header=None, comment='#', names=['CHROM', 'POS', 'ID', 'REF', 'ALT']).set_index(
        ['CHROM', 'POS'])

def to_hdf5(filename, df, metadf=None, **kwargs):
    store = pd.HDFStore(filename)
    store.put('data', df)
    if metadf is not None:
        store.put('meta', metadf)
    store.get_storer('data').attrs.metadata = kwargs
    store.close()

def read_hdf5(filename):
    with pd.HDFStore(filename) as store:
        data = store['data']
        metadf=None
        try:
            metadf = store['meta']
        except:
            pass
        metadata = store.get_storer('data').attrs.metadata
    return data, metadf, metadata

class SynchronizedFile:
    @staticmethod
    def processSyncFileLine(x,dialellic=True):
        z = x.apply(lambda xx: pd.Series(xx.split(':'), index=['A', 'T', 'C', 'G', 'N', 'del'])).astype(float).iloc[:, :4]
        ref = x.name[-1]
        alt = z.sum().sort_values()[-2:]
        alt = alt[(alt.index != ref)].index[0]
        if dialellic:   ## Alternate allele is everthing except reference
            return pd.concat([z[ref].astype(int).rename('C'), (z.sum(1)).rename('D')], axis=1).stack()
        else:           ## Alternate allele is the allele with the most reads
            return pd.concat([z[ref].astype(int).rename('C'), (z[ref] + z[alt]).rename('D')], axis=1).stack()

    @staticmethod
    def load(fname = './sample_data/popoolation2/F37.sync'):
        # print 'loading',fname
        cols=pd.read_csv(fname+'.pops', sep='\t', header=None, comment='#').iloc[0].apply(lambda x: map(int,x.split(','))).tolist()
        data=pd.read_csv(fname, sep='\t', header=None).set_index(range(3))
        data.columns=pd.MultiIndex.from_tuples(cols)
        data.index.names= ['CHROM', 'POS', 'REF']
        data=data.sort_index().reorder_levels([1,0],axis=1).sort_index(axis=1)
        data=data.apply(SynchronizedFile.processSyncFileLine,axis=1)
        data.columns.names=['REP','GEN','READ']
        data=SynchronizedFile.changeCtoAlternateAndDampZeroReads(data)
        data.index=data.index.droplevel('REF')
        return data

    @staticmethod
    def changeCtoAlternateAndDampZeroReads(a):
        C = a.xs('C', level=2, axis=1).sort_index().sort_index(axis=1)
        D = a.xs('D', level=2, axis=1).sort_index().sort_index(axis=1)
        C = D - C
        if (D == 0).sum().sum():
            C[D == 0] += 1
            D[D == 0] += 2
        C.columns = pd.MultiIndex.from_tuples([x + ('C',) for x in C.columns], names=C.columns.names + ['READ'])
        D.columns = pd.MultiIndex.from_tuples([x + ('D',) for x in D.columns], names=D.columns.names + ['READ'])
        return pd.concat([C, D], axis=1).sort_index(axis=1).sort_index()

def INT(x):
    try: return int(x)
    except: return x

def intIndex(df):
    names=df.index.names
    df=df.reset_index()
    df[names]=df[names].applymap(INT)
    return df.set_index(names).sort_index()

def uniqIndex(df,keep=False,subset=['CHROM','POS']): #keep can be first,last,None
    names=df.index.names
    if subset is None: subset=names
    return df.reset_index().drop_duplicates(subset=subset,keep=keep).set_index(names).sort_index()


def mask(genome,interval=None,keep=True,CHROM=None,start=None,end=None,pad=0,returnIndex=False,full=False):
    if isinstance(interval,str): interval=BED.intervals(interval)
    if interval is not None: CHROM, start, end = interval.CHROM, interval.start, interval.end
    start-=pad;end+=pad
    if not keep:
        return genome[~mask(genome,interval=interval,returnIndex=True)]

    if returnIndex:
        return (genome.index.get_level_values('CHROM')==CHROM) & (genome.index.get_level_values('POS')>=start)&(genome.index.get_level_values('POS')<=end)
    else:
        try:
            tmp=genome.loc[CHROM]
        except:
            # print( 'Warning, CHROM does not exist in the index!')
            tmp=genome
            # return None
        tmp=tmp[(tmp.index.get_level_values('POS')>=start)&(tmp.index.get_level_values('POS')<=end)]
        if full:
            tmp=pd.concat([tmp],keys=[INT(CHROM)])
            tmp.index.names=['CHROM','POS']
        return tmp

def maskChr(a,i):
    return a[(a.index.get_level_values('POS') >= i.start) & (a.index.get_level_values('POS') <= i.end)]


def getRegionPrameter(CHROM,start,end):
    if start is not None and end is not None:CHROM='{}:{}-{}'.format(CHROM,start,end)
    elif start is None and end is not None:CHROM='{}:-{}'.format(CHROM,end)
    elif start is not None and end is None :CHROM='{}:{}-'.format(CHROM,start)
    return CHROM


class VCF:
    @staticmethod
    def loadCHROMLenCDF(PMF=False):
        a=VCF.loadCHROMLen()
        if PMF:
            return (a/a.sum()).round(2)
        return (a.cumsum()/a.sum()).round(2)
    @staticmethod
    def loadCHROMLen(assembly=19,CHROM=None,all=False,autosomal=False):
        if assembly is None:
            return pd.concat([VCF.loadCHROMLen(19), VCF.loadCHROMLen(38)], 1, keys=[19, 38])
        a=pd.read_csv(home + 'storage/Data/Human/ref/hg{}.chrom.sizes'.format(assembly), sep='\t', header=None).applymap(
            lambda x: INT(str(x).replace('chr', ''))).set_index(0)[1]
        if CHROM is not None: a=a.loc[CHROM]
        if not all: a=a.loc[range(1,23)+list('XYM')]
        a.index.name='CHROM'
        if autosomal:
            a=a.loc[range(1,23)]
        return a.rename('len')

    @staticmethod
    def AllPops():
        p = home + 'Kyrgyz/info/kyrgyz.panel'
        return ['1KG']+list(set(VCF.pops(p) + VCF.pops() + VCF.superPops(p) + VCF.superPops()))

    @staticmethod
    def All1KGPops():
        p = '/home/arya/storage/Data/Human/1000GP/info/panel'
        return ['1KG'] + list(set(VCF.pops(p) + VCF.superPops(p) ))

    @staticmethod
    def IDs(P, panel=home + 'POP/HAT/panel', color=None, name=None, maxn=1e6):
        return pd.concat([VCF.ID(p=p,panel=panel,color=color,name=name,maxn=maxn) for p in P])

    @staticmethod
    def IDfly():
        z = pd.read_csv('/home/arya/fly/all/RC/all.folded.gz.col').iloc[1:, 0]
        z.index = pd.MultiIndex.from_tuples(z.apply(lambda x: tuple(map(INT,x.split('.')))), names=['POP', 'GEN', 'REP'])
        return z.sort_index()

    @staticmethod
    def ID(p,panel=home + 'POP/HAT/panel',color=None,name=None,maxn=1e6):
        a = VCF.loadPanel(panel)
        try:a=pd.concat([a, VCF.loadPanel(home + 'Kyrgyz/info/kyrgyz.panel')])
        except: pass
        if p=='1KG':
            x=a.set_index('super_pop').loc[['AFR','EUR','EAS','SAS','AMR']]
        else:
            try:
                x = a.set_index('pop').loc[p]
            except:
                x = a.set_index('super_pop').loc[p]
        x= list(set(x['sample'].tolist()))
        x=pd.Series(x,index=[(name,p)[name is None]] *len(x))
        if color is not None:
            x=x.rename('ID').reset_index().rename(columns={'index':'pop'})
            x['color']=color
        maxn = min(x.shape[0],int(maxn))
        x=x.iloc[:maxn].astype(str)
        x.index.name='pop'
        return x.rename('ID')

    @staticmethod
    def pops(panel=home + 'POP/HAT/panel'):
        return list(VCF.loadPanel(panel)['pop'].unique())
    @staticmethod
    def superPops(panel=home + 'POP/HAT/panel'):
        return list(VCF.loadPanel(panel)['super_pop'].unique())

    @staticmethod
    def getN(panel=home+'/storage/Data/Human/1000GP/info/panel'):
        pan=VCF.loadPanel(panel)
        return pd.concat([pan.groupby('pop').size(),pan.groupby('super_pop').size(),pd.Series({'ALL':pan.shape[0]})])
    @staticmethod
    def getField(fname,field='POS'):
        fields={'CHROM':1,'POS':2,'ID':3}
        cmd="zgrep -v '#' {} | cut -f{}".format(fname,fields[field])
        return pd.Series(Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].strip().split('\n')).astype(int)

    @staticmethod
    def header(fname):
        cmd="zgrep -w '^#CHROM' -m1 {}".format(fname)
        return Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].split('\n')[0].split()
    @staticmethod
    def headerSamples(fname):
        return map(INT,VCF.header(fname)[9:])

    @staticmethod
    def loadPanel(fname=home + 'POP/HAT/panel'):
        return  pd.read_table(fname,sep='\t').dropna(axis=1)

    @staticmethod
    def loadPanels():
        panels = pd.Series({'KGZ': '/home/arya/storage/Data/Human/Kyrgyz/info/kyrgyz.panel',
                           'ALL':  '/home/arya/storage/Data/Human/1000GP/info/panel'})
        load = lambda x: VCF.loadPanel(x).set_index('sample')[['super_pop', 'pop']]
        return pd.concat(map(load, panels.tolist()))

    @staticmethod
    def getDataframeColumns(fin,panel=None,haploid=False):
        def f(x):
            try:return tuple(panel.loc[x].tolist())
            except:return ('NAs','NAp')
        cols=[]
        if panel is not None:
            load=lambda x: VCF.loadPanel(x).set_index('sample')[['super_pop','pop']]
            if isinstance(panel,str): panel=[panel]
            else: panel=panel.tolist()
            panel= pd.concat(map(load,panel))
            try:
                ids=VCF.headerSamples(fin)
                for x in ids:
                    if haploid:
                        cols += [f(x) + (x, 'A')]
                    else:
                        cols += [f(x) + (x, 'A'), f(x) + (x, 'B')]
                cols = pd.MultiIndex.from_tuples(cols, names=['SPOP', 'POP', 'ID', 'HAP'])
            except:
                panel['HAP']='A'
                cols= panel.reset_index().rename(columns={'super_pop':'SPOP','pop':'POP','sample':'ID'}).set_index(['SPOP', 'POP', 'ID', 'HAP']).index
        else:
            for x in VCF.headerSamples(fin):
                cols+=[( x,'A'),(x,'B')]
            cols=pd.MultiIndex.from_tuples(cols,names=[ 'ID','HAP'])
        return cols

    @staticmethod
    def getDataframe(CHROM,start=None,end=None,
                     fin=dataPath1000GP+'ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz',
                     bcftools="/home/arya/bin/bcftools/bcftools",
                     panel=dataPath1000GP+'integrated_call_samples_v3.20130502.ALL.panel',haploid=False,dropDots=True,
                     gtfile=False,pop=None,freq=True
                     ):
        reg=getRegionPrameter(CHROM,start,end)

        fin=fin.format(CHROM)
        if freq:

            df= gz.loadFreqGT(f=fin, istr=reg,pop=pop)#.set_index(range(5))
        else:
            cmd="{} filter {} -i \"N_ALT=1 & TYPE='snp'\" -r {} | {} annotate -x INFO,FORMAT,FILTER,QUAL,FORMAT | grep -v '#' | tr '|' '\\t'|  tr '/' '\\t' | cut -f1-5,10-".format(bcftools,fin,reg,bcftools)
            #cmd="{} filter {} -i \"N_ALT=1 & TYPE='snp'\" -r {} | {} annotate -x INFO,FORMAT,FILTER,QUAL,FORMAT | grep -v '#' | cut -f1-5,10-".format(bcftools,fin,reg,bcftools)
            csv=Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].split('\n')
            df = pd.DataFrame(map(lambda x: x.split('\t'),csv)).dropna().set_index(range(5))#.astype(int)


        df.index.names=['CHROM','POS', 'ID', 'REF', 'ALT']

        if freq:
            df=df.rename(pop)
        else:
            df.columns=VCF.getDataframeColumns(fin,panel,haploid)
        dropDots=False
        # if dropDots:df[df=='.']=None;
        # else:df=df.replace({'.':0})

        if not freq:
            if haploid:df=df.replace({'0/0':'0','1/1':'1','0/1':'1'})
            try:df=df.astype(int)
            except:df=df.astype(float)

        return df

    @staticmethod
    def computeFreqs(CHROM,start=None,end=None,
                     fin=dataPath1000GP+'ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz',
                     panel=dataPath1000GP+'integrated_call_samples_v3.20130502.ALL.panel',
                     verbose=0,hap=False,genotype=False,haploid=False,gtfile=False,pop=None):


        try:
            if verbose:
                import sys
                print( 'chr{}:{:.1f}-{:.1f}'.format(CHROM,start/1e6,end/1e6)); sys.stdout.flush()
            a=VCF.getDataframe(CHROM,int(start),int(end),fin=fin,panel=panel,haploid=haploid,gtfile=gtfile,pop=pop)
            if pop is not None: return a
            if panel is None:
                return a
            if isinstance(panel,str):
                panel=pd.Series({'HA':panel})
            if panel.size==1:
                a=pd.concat([a],1,keys=panel.index)
            else:
                a=a.T.sort_index()
                a['DS']='ALL';a.loc[['Sick','Healthy'],'DS']='KGZ'
                a=a.set_index('DS',append=True).reorder_levels([4,0,1,2,3]).sort_index().T
            if hap: return a
            elif genotype:
                # print 'aaaa'
                return a.groupby(level=[0,1,2,3],axis=1).sum()
            else:   #compute AF
                if panel is not None:
                    return pd.concat([a.groupby(level=0,axis=1).mean(),a.groupby(level=1,axis=1).mean(),a.groupby(level=[1,2],axis=1).mean()],1)
                else:
                    return a.mean(1).rename('ALL')

        except :
            # print None
            return None

    @staticmethod
    def len(CHROM,ref=19):
        if ref==19:
            a=pd.read_csv(home+'storage/Data/Human/ref/hg19.chrom.sizes',sep='\t',header=None).set_index(0)[1]
            return a.loc['chr'+str(CHROM)]
    @staticmethod
    def batch(CHROM,winSize=1e6,ref=19):
        winSize=int(winSize)
        L=VCF.len(CHROM,ref)
        a=pd.DataFrame(range(0, ceilto(L, winSize), winSize),columns=['start'])
        a['end']=a.start+winSize-1
        a['CHROM']=CHROM
        return a
    @staticmethod
    def computeFreqsChromosome(CHROM,fin,panel,verbose=0,winSize=500000,haplotype=False,genotype=False,save=False,haploid=False,nProc=1,gtfile=False):
        print ("""
        :param CHROM: {}
        :param fin:  {}
        :param panel:  {}
        :param verbose: {}
        :param winSize: {}
        :param haplotype: {}
        :param genotype: {}
        :param save: {}
        :param haploid: {}
        :param nProc: {}
         ***************************************
        """.format(CHROM,fin,panel,verbose,winSize,haplotype,genotype,save,haploid,nProc))
        CHROM=INT(CHROM)
        import vcf
        try:
            L=vcf.Reader(open(fin.format(CHROM), 'r')).contigs['chr{}'.format(CHROM)].length
        except:
            try:
                L=vcf.Reader(open(fin.format(CHROM), 'r')).contigs[str(CHROM)].length
                assert L!=None
            except:
                cmd='zgrep -v "#" {} | cut -f2 | tail -n1'.format(fin.format(CHROM))
                L= int(Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].strip())
        print( 'Converting Chrom {}. ({}, {} Mbp Long)'.format(CHROM,L,int(L/1e6)))
        #a=[VCF.computeFreqs(CHROM,start,end=start+winSize-1,fin=fin,panel=panel,hap=haplotype,genotype=genotype,haploid=haploid,verbose=verbose) for start in xrange(0,ceilto(L,winSize),winSize)]
        args=map(lambda start: (CHROM,start,fin,panel,haplotype,genotype,haploid,verbose,winSize,gtfile), range(0,ceilto(L,winSize),winSize))
        from multiprocessing import Pool
        a=Pool(nProc).map(computeFreqsHelper,args)
        a = intIndex(uniqIndex(pd.concat([x  for x in a if x is not None]),subset=['CHROM','POS']))
        if save:
            if haplotype: suff='.hap'
            elif genotype: suff='.gt'
            else: suff=''
            a.to_pickle(fin.format(CHROM).replace('.vcf.gz','{}.df'.format(suff)))
        return  a


    @staticmethod
    def createGeneticMap(VCFin, chrom,gmpath=dataPath+'Human/map/GRCh37/plink.chr{}.GRCh37.map',recompute=False):
        if os.path.exists(VCFin+'.map') and not recompute:
            print('map file exist!')
            return
        print ('Computing Genetic Map for ', VCFin)
        gm = pd.read_csv(gmpath.format(chrom), sep='\t', header=None,names=['CHROM','ID','GMAP','POS'])
        df = pd.DataFrame(VCF.getField(VCFin).rename('POS'))
        df['GMAP'] = np.interp(df['POS'].tolist(), gm['POS'].tolist(),gm['GMAP'].tolist())
        df['CHROM']=chrom
        df['ID']='.'
        df[['CHROM','ID','GMAP','POS']].to_csv(VCFin+'.map',sep='\t',header=None,index=None)

    @staticmethod
    def subset(VCFin, pop,panel,chrom,fileSamples=None,recompute=False):
        # print pop
        bcf='/home/arya/bin/bcftools/bcftools'
        assert len(pop)
        if pop=='ALL' or pop is None:return VCFin
        fileVCF=VCFin.replace('.vcf.gz','.{}.vcf.gz'.format(pop))
        if os.path.exists(fileVCF) and not recompute:
            print ('vcf exits!')
            return fileVCF
        print ('Creating a vcf.gz file for individuals of {} population'.format(pop))
        if fileSamples is None:
            fileSamples='{}.{}.chr{}'.format(panel,pop,chrom)
            os.system('grep {} {} | cut -f1 >{}'.format(pop,panel,fileSamples))
        cmd="{} view -S {} {} | {} filter -i \"N_ALT=1 & TYPE='snp'\" -O z -o {}".format(bcf,fileSamples,VCFin,bcf,fileVCF)
        os.system(cmd)
        return fileVCF

    @staticmethod
    def loadDP(fname):
        a= pd.read_csv(fname,sep='\t',na_values='.').set_index(['CHROM','POS'])
        a.columns=pd.MultiIndex.from_tuples(map(lambda x:(int(x.split('R')[1].split('F')[0]),int(x.split('F')[1])),a.columns))
        return  a

    @staticmethod
    def loadCD(vcfgz,vcftools='~/bin/vcftools_0.1.13/bin/vcftools'):
        """
            vcfgz: vcf file where samples are in the format of RXXFXXX
        """
        vcf=os.path.basename(vcfgz)
        path=vcfgz.split(vcf)[0]
        os.system('cd {0} && {1} --gzvcf {2} --extract-FORMAT-info DP && {1} --gzvcf {2} --extract-FORMAT-info AD'.format(path,vcftools,vcf))
        fname='out.{}.FORMAT'
        a=map(lambda x: VCF.loadDP(path +fname.format(x)) ,['AD','DP'])
        a=pd.concat(a,keys=['C','D'],axis=1).reorder_levels([1,2,0],1).sort_index(1)
        a.columns.names=['REP','GEN','READ']
        return a

def MAF(y,t=None):
    x = y.copy(True)
    if t is not None:
        x[x[t] > 0.5] = 1 - x[x[t] > 0.5]
    else:
        x[x>0.5]=1-x[x>0.5]
    return x
def polymorphixDF(a,MAF=1e-15):
    if len(a.shape)==1:
        a=pd.DataFrame(a)
    return a[polymorphix(a.abs().mean(1),MAF,True)]
def polymorphix(x, MAF=1e-9,index=False):
    I=(x>=MAF)&(x<=1-MAF)
    if index: return I
    return x[I]
def polymorphic(data, minAF=1e-9,mincoverage=10,index=True):
    def poly(x):return (x>=minAF)&(x<=1-minAF)
    C,D=data.xs('C',level='READ',axis=1),data.xs('D',level='READ',axis=1)
    I=(C.sum(1)/D.sum(1)).apply(lambda x:poly(x)) & ((D>=mincoverage).mean(1)==1)
    if index:
        return I
    return data[I]
def loadGenes(Intervals=True):
    a=pd.read_csv(dataPath+'Human/WNG_1000GP_Phase3/gene_info.csv')[['chrom','pop','gene','POS_hg19']].rename(columns={'chrom':'CHROM','POS_hg19':'POS'})
    a.CHROM=a.CHROM.apply(lambda x: INT(x[3:]))
    a=a.set_index('pop')
    if Intervals:
        a['start']=a.POS-2e6
        a['end']=a.POS+2e6
        a['name']=a.gene
    return a

def normalizeIHS(a,field=None):
    if field is None:
        field=a.columns[-1]
    m=a.set_index('x')[field].groupby(level=0).mean().sort_index()
    s=a.set_index('x')[field].groupby(level=0).std().sort_index()
    return (a[field]-m.loc[a.x].values)/s.loc[a.x].values


def loadGap(assempbly=19,pad=0):
    gap = pd.read_csv(dataPath + 'Human/gaps/hg{}.gap'.format(assempbly), sep='\t')[
        ['chrom', 'chromStart', 'chromEnd']].rename(
        columns={'chrom': 'CHROM', 'chromStart': 'start', 'chromEnd': 'end'}).reset_index()
    gap.start -= pad;
    gap.end += pad;
    gap.loc[gap.start < 0, 'start'] = 0
    gap.CHROM = gap.CHROM.apply(lambda x: INT(x[3:]));
    gap=gap.set_index('CHROM')
    return gap
def loadFst(fname):
    a=pd.read_csv(fname,sep='\t');
    a.CHROM=a.CHROM.replace({'22':22});a=a.set_index(['CHROM','POS']).sort_index().iloc[:,0].rename('Fst')
    a=a.loc[range(1,23)]
    return a[a>0]
def computeFreqsHelper(args):
    CHROM,start,fin,panel,hap,genotype,haploid,verbose,winSize,gtfile=args
    end=start+winSize-1
    return VCF.computeFreqs(CHROM=CHROM,start=start,end=end,fin=fin,panel=panel,verbose=verbose,hap=hap,genotype=genotype,haploid=haploid,gtfile=gtfile)

def lite1d(a,q=0.9,cutoff=None):
    return a[a>a.quantile(q)]

def execute(cmd,returnDF=True,verbose= False, sep='\t',header=None,escapechar=None):
    if verbose:print( cmd)
    cmd= [cmd]
    with open(os.devnull, 'w') as FNULL:
        if not returnDF: return Popen(cmd, stdout=PIPE, stdin=FNULL, stderr=FNULL, shell=True).communicate()
        return pd.read_csv(StringIO(Popen(cmd, stdout=PIPE, stdin=FNULL, stderr=FNULL,shell=True) .communicate()[0]),sep=sep, header=header,escapechar=escapechar)


        # a=Popen([cmd], stdout=PIPE, stdin=FNULL, stderr=FNULL,shell=True) .communicate()[0]
    # if returnDF: return pd.read_csv(StringIO(a), sep='\t',header=None)
def gzLoadHelper(args):
    f,p,x=args
    return gz.loadFreqChrom(f=f, p=p, x=x)
import traceback
class gz:
    @staticmethod
    def CPRA(chrom,f=home+'storage/Data/Human/HLI/GT/bim/CPRA/all.gz',keepCHROM=False):
        cut=(' | cut -f2-','')[keepCHROM]
        a=execute('{}/bin/tabix {} {}'.format(home,f,chrom)+cut)
        a.columns=['POS','REF','ALT']
        return a.set_index('POS')

    @staticmethod
    def loadFly(i, pos=None):
        z = gz.load(i=i, f='/home/arya/fly/all/RC/all.folded.gz')
        z.columns = pd.MultiIndex.from_tuples(map(lambda x: tuple(map(INT, x.split('.'))), z.columns),
                                              names=['POP', 'GEN', 'REP'])
        z = z.loc[i.CHROM]
        if pos is not None:
            z = z.loc[pos]
        return z
    @staticmethod
    def loadAA(f, i, code='linear'):
        a = gz.load(f, i, dropIDREFALT=False)
        cols = ['REF', 'ALT', 'ID']
        aa = a.reset_index(cols)[cols].join(gz.load(f, i, AA=True))
        a = a.reset_index(cols, drop=True)
        a = a[(aa.REF == aa.AA) | (aa.ALT == aa.AA)]
        I = (aa.ALT == aa.AA)

        def fix(a, I, code):
            if code == 'linear': k = 2
            if code == 'freq': k = 1
            a.loc[TI(I)] = k - a.loc[TI(I)]
            return a
        return fix(a, I, code)
    @staticmethod
    def load(f='/home/arya/POP/HA/GT/chr{}.vcf.gz',i=None,istr=None,index=True,dropIDREFALT=True,indvs=None,pop=None,AA=False,CHROMS=None,pad=None):
        if pad is not None:i=BED.expand(i, pad)
        if CHROMS is not None:return pd.concat(map(lambda x: gz.load(f.format(x)),CHROMS)).sort_index()
        if i is not None:
            try:f=f.format(i.CHROM)
            except:pass
            istr='{}:{}-{}'.format(i.CHROM,i.start,i.end)
        # if istr is not None:
        #     xx=istr.split(':')
            # i=pd.Series({'CHROM': xx[0], 'start':xx[1].split('-')[0], 'end':xx[1].split('-')[1]}).apply(INT)
        if AA: f+='.aa.gz'
        if pop is not None:indvs=VCF.ID(pop)

        try:
            cols = pd.read_csv(f + '.col', header=None)[0]
            if indvs is not None:
                if isinstance(indvs,pd.Series):indvs=indvs.tolist()
                try:
                    colsi= (cols.reset_index().set_index(0).iloc[:, 0].loc[['CHROM','POS','ID','REF','ALT']+indvs]).astype(int).tolist()
                except:
                    colsi = (cols.reset_index().set_index(0).iloc[:, 0].loc[['CHROM', 'POS'] + list(indvs)]).astype(int).tolist()
                cols=cols.iloc[colsi]
            else:cols=cols
        except:
            pass


        try:
            if istr is not None:   cmd='/home/arya/bin/tabix {} {}'.format(f,istr)
            else:               cmd='zcat {} '.format(f)
            if indvs is not None:cmd += ' | cut -f' + ','.join(map(lambda x: str(x+1),colsi))

            a=execute(cmd)
        except:
            # print 'No SNPs in '+istr
            return None
        try:
            try:
                a.columns=cols.sort_index().tolist() ### this is very important, cut,sortys by index
            except:
                a.columns = ['ID','REF','ALT'] + cols.sort_index().tolist()
            if dropIDREFALT:
                if 'ID' in a.columns:
                    a=a.drop(['ID','REF','ALT'],axis=1)
        except:
            pass
        if index:
            if a.shape[1]==3:
                name=0
                if AA:name='AA'
                if 'CHROM' in a.columns:
                    a.CHROM=a.CHROM.apply(INT)
                    a=(a.set_index(['CHROM', 'POS'])).iloc[:,0].rename(name)
                else:
                    a[0] = a[0].apply(INT)
                    a = a.set_index([0, 1]).iloc[:, 0].rename(name)
                    a.index.names = ['CHROM', 'POS']
            else:
                a.CHROM = a.CHROM.apply(INT)
                if 'ID' in a.columns:a = a.set_index(['CHROM', 'POS','ID','REF','ALT'])
                else:a=(a.set_index(['CHROM','POS']))

        if len(a.shape)==1 and indvs is not None: a=a.rename(indvs[0])

        return a

    @staticmethod
    def loadFreqChrom(p, x, f =None):
        fs=['/home/arya/POP/KGZU/GT/AF.gz', '/home/arya/POP/KGZU+ALL/GT/AF.gz', '/home/arya/POP/HAT/GT/AF.gz']
        try:
            for f in fs:
                a = polymorphixDF(pd.DataFrame(gz.load(f=f, indvs=p, istr=x)))
                if  a.shape[0]: break
        except: #single population
            a = polymorphixDF(pd.DataFrame(gz.load(f=f.replace('/HAT/', '/{}/'.format(p)), indvs=p, istr=x)))
        if a.shape[1]==1:a=a.iloc[:,0]
        return a.dropna()
    @staticmethod
    def loadFreqGenome(pop, f='/home/arya/POP/KGZU+ALL/GT/AF.gz', daf=False, nProc=1):
        if daf: f=f.replace('/AF.','/DAF.')
        p=pop
        if isinstance(pop,str):p=[pop]
        CHROMS=map(str,range(1,23))
        if nProc==1:
            return pd.concat(map(lambda x: gz.loadFreqChrom(f=f, p=p, x=x), CHROMS))
        else:
            from multiprocessing import Pool
            pool=Pool(nProc)
            args=map( lambda x: (f,p,x), CHROMS)
            a=pd.concat(pool.map(gzLoadHelper,args))
            pool.terminate()
            return a


    @staticmethod
    def loadFreqGT(i=None, f='/home/arya/POP/HA/GT/chr{}.vcf.gz', istr=None, pop=None, AA=False):
        """
        Loads freq from .gz which is GT file and there should be an n file associatged with it for header
        :param i:
        :param f:
        :return:
        """
        try:
            if AA:
                a = (gz.load(i=i, f=f, istr=istr,dropIDREFALT=False,pop=pop).mean(1)/2).rename(pop)
                a=pd.concat([a.reset_index(['ID','REF','ALT']),gz.load(i=i, f=f, istr=istr,AA=True)],1)
                a = a[(a.AA == a.REF) | (a.AA == a.ALT)]
                I = a.ALT == a.AA
                a=a[pop]
                a[I]=1-a[I]
            else:
                a=gz.load(i=i, f=f, istr=istr, pop=pop,dropIDREFALT=False)
                freq=lambda x: x.mean()/2#(x.mean() / 2).rename(pop)
                nomissing=lambda x: x[x>=0]
                a = a.apply(lambda x: freq(nomissing(x)),1)
            return a
        except:
            return None



    @staticmethod
    def code(A,coding='linear'):
        """
        :param coding: can be
        linear: GT={0,1,2}
        dominant: GT={0,1}
        recessive: GT={0,1}
        het: GT={0,1}
        """
        a=A.copy(True)
        if coding=='linear':
            pass
        elif coding=='dominant':
            a[a>0]=1
        elif coding == 'recessive':
            a[a <= 1] = 0
            a[a > 1] = 1
        elif coding == 'het':
            a[a > 1] = 0
        return a
    @staticmethod
    def GT(vcf,coding='linear'):
        """
        :param vcf: path to vcf file
        :param coding: can be
                linear: GT={0,1,2}hq
                dominant: GT={0,1}
                recessive: GT={0,1}
                het: GT={0,1}
        :return:
        """
        from subprocess import Popen, PIPE
        sh='/home/arya/workspace/bio/Scripts/Bash/VCF/createGTSTDOUT.sh'
        sh2='/home/arya/workspace/bio/Scripts/Bash/VCF/sampleNames.sh'
        from StringIO import  StringIO
        with open(os.devnull, 'w') as FNULL:
            a= pd.read_csv(StringIO(Popen([sh, vcf], stdout=PIPE, stdin=FNULL, stderr=FNULL).communicate()[0]), sep='\t', header=None).set_index([0, 1])
            try:
                cols = pd.read_csv(StringIO(Popen([sh2, vcf], stdout=PIPE, stdin=FNULL, stderr=FNULL).communicate()[0]), sep='\t',header=None)[0].tolist()
                a.columns = cols
            except:
                pass
        a.index.names=['CHROM','POS']

        return gz.code(a,coding)

    @staticmethod
    def save(df,f,index=True):
        import uuid
        os.system('mkdir -p '+home+'storage/tmp/')
        tmp=home+'storage/tmp/'+str(uuid.uuid4())
        df.to_csv(tmp,sep='\t',header=None)
        if isinstance(df,pd.DataFrame):pd.Series(df.reset_index().columns).to_csv(f+'.col',sep='\t',index=False)
        os.system(home + 'bin/bgzip -c {0} > {1} &&  rm -f {0}'.format(tmp,f))
        if index:os.system(home + 'bin/tabix -p vcf {} '.format(f))

def MultiIndex(df):
    return pd.MultiIndex.from_tuples(df.apply(lambda x: tuple(x),1).tolist(),names=df.columns)

def tmpFileName(x=None):
    if x is None:
        import uuid
        tmpPath = home + 'storage/tmp'
        os.system('mkdir -p ' + tmpPath)
        return '/home/arya/storage/tmp/' + str(uuid.uuid4())
    else:
        os.system('rm -f {}*'.format(x))

def IBS(i,path='/home/arya/POP/HA/', onlyRefPopSNPs=False):
    bcf = '/home/arya/bin/bcftools/bcftools'
    plink = '/home/arya/bin/plink'
    uid = tmpFileName()
    VCF = path + 'chr{}.vcf.gz'.format(i.CHROM)
    vcf = uid + '.vcf.gz'
    ibs = uid
    if onlyRefPopSNPs:
        freqs = gz.Freq(i)
        if freqs is not None:
            pos = uid + '.vcf'
            polymorphix(freqs.KGZ).reset_index().iloc[:, :2].to_csv(pos, sep='\t', header=None, index=False)
            execute('{} view -R {} {} | {} -T {} -Oz -o {}'.format(bcf, BED.str(i), VCF, bcf, pos, vcf),False)
    else:
        execute('{} view -r {} {} -Oz -o {}'.format(bcf, BED.str(i), VCF, vcf),False)
    from subprocess import Popen, PIPE, STDOUT, call
    cmd = '{} --vcf {} --cluster --matrix --out {}'.format(plink, vcf, ibs)
    execute(cmd,False)
    # with open(os.devnull, 'w') as FNULL:

    #     call(cmd.split(), stdout=FNULL, stderr=FNULL)
    names = pd.read_csv(ibs + '.mibs.id', sep='\t', header=None)[0].values
    c = pd.read_csv(ibs + '.mibs', sep=' ', header=None).T.dropna().T
    c.index = names;
    c.columns = names
    tmpFileName(uid)
    return augmentIndex(c,path=path)


def augmentIndex(c,axes=[0,1],path='/home/arya/POP/HA/'):
    pop = VCF.loadPanel(path+'panel').iloc[:, :-1].set_index('sample')
    one=lambda i: MultiIndex(pop.loc[i].reset_index().rename(columns={'index': 'sample','super_pop':'sup'})[['sup','pop','sample']])
    if 0 in axes:c.index= one(map(str,c.index))
    if 1 in axes:c.columns = one(c.columns)
    return c.sort_index(axis=0).sort_index(axis=1)

def slice(A,pops=None,axes=[0,1],maxn = int(1e6),I=None):
    sort = lambda x: x.sort_index(0).sort_index(1)
    m=sort(A)
    f=lambda x: x.reset_index([0, 1], drop=True)
    if I is not None:
        if axes == [0]:
            return f(m).loc[I.astype(str)]
        return f(f(m).loc[I.astype(str)].T).T[I.astype(str)]
    I = []
    for p in pops:I += VCF.ID(p,maxn=maxn).astype(str).tolist()
    if axes==[0]:
        return sort(m.loc[pd.IndexSlice[:, :, I], :])
    return sort(m.loc[pd.IndexSlice[:, :, I], pd.IndexSlice[:, :, I]])

def rankLogQ(a,positiveTail=True):
    return (a.dropna().rank(ascending=not positiveTail)/a.dropna().size).apply(np.log10).abs()

def rankPercentile(a,measure=100,n=None):
    if n is not None:
        rq=(a.rank() + (n - a.shape[0])) / n
    else:
        rq=a.rank() / a.shape[0]
    return (rq * measure).apply(np.ceil).astype(int)


def significantLog(a,q=0.05):
    return a[a>abs(np.log10(q))]
def nxScan(a,w=50,step=5,name=None,minn=0,f=np.mean):
    if name is None:
        if a.name is not None:name=a.name
        else:name='stat'
    # print name
    x=scan.Genome(a, f={name: f, 'n': len},winSize=w*1000,nsteps=step)
    return x[x.n > minn]

def ihsScan(a,minn=0,topQuantile=0.05):
    # print a
    # if a[a >= a.quantile(0.999)].value_counts().size>1:
    return nxScan(a[a>=a.quantile(0.95)],f=np.mean, minn=minn)
    # else:return nxScan(a[a>=a.quantile(0.995)],f=np.mean, minn=minn)
    return nxScan(significantLog(rankLogQ(a,positiveTail=positiveTail),q=topQuantile),minn=minn)



def UCSC(i):
    return 'https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&lastVirtModeType=default&lastVirtModeExtraState=&virtModeType=default&virtMode=0&nonVirtPosition=&position=chr11%3A14125000-14225000&hgsid=604049611_XYcrNVjMubaqAEYhMU3mOa75y1nA'

def saveSingletons(CHROM,f='/home/arya/POP/HA/GT/chr{}.df'):
    f=f.format(CHROM)
    a=pd.read_pickle(f)

    super=lambda X: X.loc[:,map(lambda x: not isinstance(x,tuple), X.columns)]
    a=super(a.iloc[:,1:])

    b=a>0
    a[b.sum(1)==1].reset_index()[['CHROM','POS']].to_csv(f.replace('.df','.singletonPos.vcf'),sep='\t',index=False,header=False)

def triPopColor(pops):
    try:return {pops[0]: 'b', pops[1]: 'r', pops[2]: 'g'}
    except: return {pops[0]: 'b', pops[1]: 'r'}

def saveBegelePosAsia(chrom):
    a=pd.read_pickle('/home/arya/POP/KGZU+ALL/chr{}.df'.format(chrom))[['SAS','EAS']].reset_index(['ID','REF','ALT'],drop=True).loc[chrom]
    b=pd.read_csv('/home/arya/storage/Data/Human/Beagle/hg19/chr{}.pos.vcf'.format(22),sep='\t',header=None)[1]
    a=a.loc[b]
    a=a[((a==0).sum(1)<2)&((a==1).sum(1)<2)].reset_index()
    a['CHROM']=chrom
    a[['CHROM','POS']].to_csv('/home/arya/POP/ASIA/chr{}.pos.vcf'.format(chrom),sep='\t',header=None,index=False)


dedup=lambda x: x[~x.index.duplicated()]

def loadPiarPop(f,pop,popxp,negate=False):
    load=pd.read_pickle
    if f[-3:]=='.gz':load=gz.load
    try:return load(f.format(pop, popxp))
    except:
        alpha=(1,-1)[negate]
        return load(f.format(popxp, pop))*alpha

def pbsi(i,pops):
    def load(p1, p2):
        import os
        fname = '/home/arya/scan/Fst/{}.{}.gz'
        if os.path.isfile(fname.format(p1, p2)):
            a = gz.load(f=fname.format(p1, p2), i=i)
        else:
            a = gz.load(f=fname.format(p2, p1), i=i)
        return a.dropna().rename(p1 + '\nvs\n' + p2).loc[i.CHROM]

    HS = load(pops[0], pops[1])
    HL = load(pops[0], pops[2])
    SL = load(pops[1], pops[2])
    dedup= lambda x: x[~x.index.duplicated()]
    a = pd.concat(map(dedup,[HS, HL, SL]), 1)
    a[a < 0] = 0
    a=(1-a[a.iloc[:,0]>0].dropna()).apply(np.log).abs()
    ff = lambda x: (x.iloc[:, 0] + x.iloc[:, 1] - x.iloc[:, 2]).rename('PBS')  # .rolling(50,center=True).mean()

    a=pd.concat([a,ff(a)],1)
    # a = pd.concat([a, ff((1 - a[a < 1]).apply(np.log).dropna().abs()).round(3)], 1)
    return a

def pbs(pop,popxp,outgroup,hudson=False):
    n=None
    try:
        exit()
        return loadPiarPop('/home/arya/scan/PBS/{}.{}.PBS'+('','Hudson')[hudson]+'.gz',pop,popxp)
    except:
        if hudson:
            path=scanPath + 'SFS/{}.{}.df'
            HS = FstHudson(loadPiarPop(path, pop, popxp).xs('pi', 0, 2), pop).rename('HS').dropna()
            HL =FstHudson(loadPiarPop(path, pop, outgroup).xs('pi', 0, 2), pop).rename('HL').dropna()
            SL =FstHudson(loadPiarPop(path,  popxp, outgroup).xs('pi', 0, 2), popxp).rename('SL').dropna()
            n=loadPiarPop(path, pop, popxp).xs('m', 0, 2)[pop].rename('n').astype(int)
        else:
            HS = loadPiarPop('/home/arya/scan/Fst/{}.{}.gz',pop,popxp).rename('HS').dropna()
            HL = loadPiarPop('/home/arya/scan/Fst/{}.{}.gz',pop,outgroup).rename('HL').dropna()
            SL = loadPiarPop('/home/arya/scan/Fst/{}.{}.gz',popxp,outgroup).rename('SL').dropna()
        def pos(x):
            x[x<=0]=0
            return x
        a = pos(quickMergeGenome([HS,HL,SL])).fillna(0)
        # print 'Merging is done',a.shape
        # a = (1 - a[a <1]).apply(np.log).apply(lambda x: x.fillna(x.min()-1)).abs().round(2)
        a = (1 - a[a < 1]).apply(np.log).dropna().abs().round(3)
        a=(a.HS + a.HL - a.SL).rename('PBS'+('','Hudson')[hudson])
        if n is not None: a=pd.concat([a,n],1).dropna()
        gz.save(a,'/home/arya/scan/PBS/{}.{}.PBS{}.gz'.format(pop,popxp,('','Hudson')[hudson]))
        return a

def loadHLIMetrics(path='~/storage/Data/Human/Tibet/HLI'):
    def picardMetrics():
        a=execute("find {} | grep _collect_wgs_metrics_output.txt".format(path))[0]
        i=a.apply(lambda x: os.path.basename(x).split('_')[0]).rename('ID')
        a=a.apply(lambda x: execute("head -n8 {} | grep -v '#'".format(x)).T.set_index(0)[1])
        a.index=i
        return a
    def freemix():
        a = execute("find {} | grep selfSM".format(path))[0]
        a=a.apply(lambda x: execute('cat {}'.format(x)).T.set_index(0)[1]).set_index('#SEQ_ID')
        a.index.name='ID'
        return a
    return pd.concat([picardMetrics(),freemix()],1)

def getGeneList( x):  return pd.DataFrame(x.tolist()).stack().unique().tolist()
def removeChr(a):
    IDX='CHROM' in a.index.names
    if IDX:a=a.reset_index()
    a.CHROM=a.CHROM.apply(lambda x: INT(str(x).replace('chr','')))
    if IDX:a=a.set_index(['CHROM','POS'])
    return a


def loadWNG(pop,padding=2000000):
    def get_interval(a, padding=padding):
        b = a.reset_index()
        b['start'] = b.POS - padding
        b['end'] = b.POS + padding
        return b
    # a=pd.read_csv(dataPath + 'Human/scan/gene-info.csv').set_index(['pop'])
    if pop in ['Healthy', 'No-HAPH', 'KGZ']: pop = 'KGZ'
    a=pd.read_pickle(dataPath + 'Human/scan/WNG.df').set_index('pop').loc[[pop]]
    a.start-=padding;a.start[a.start<0]=0
    a.end += padding
    return a

def mergeResults(path='/home/arya/POP/HAT/CEU+CHB/CEU/',f='chr{}.xpehh.CEU.CHB.gz',out='xpehh.CEU.CHB',CHROMS=range(1,23),outpath=None):
    if outpath==None: outpath=path
    os.system('rm -f ' + path + out)
    for c in CHROMS:os.system('zcat {} >> {}'.format( path + f.format(c),outpath+out))
    os.system('bgzip -c {0} > {1} && tabix -p vcf {1} && rm -f {0}'.format(outpath+out,outpath+out+'.gz'))



def scanXPSFS(pops=['CEU','CHB'],nProc=8):
    from itertools import product
    from multiprocessing import Pool
    try:
        exit()
        return loadPiarPop(scanPath + 'SFS/{}.{}.df',pops[0], pops[1])
    except:
        fname = scanPath + 'SFS/{}.{}.df'.format(pops[0], pops[1])
        CHROMS=range(1,23)
        pool = Pool(nProc)
        a=pd.concat(pool.map(scanXPSFSChr,product([pops],CHROMS))).sort_index()
        pool.terminate()
        a.to_pickle(fname)
        return a

def scanXPSFSChr(args):
    pops, CHROM=args
    import UTILS.Estimate as est

    df = gz.loadFreqChrom(pops, str(CHROM))
    N=pd.concat(map(lambda x: pd.Series({x:len(VCF.ID(x))}),pops))*2
    w=N/N.sum()
    df=df.join(df.dot(w).rename('all'))
    N['all']=N.sum()
    N = (1 / df[df > 0].min()).astype(int)
    removeFixedSites = False;
    winSize = 5e4
    f = lambda x: pd.DataFrame(scan.Genome(x[x.name],
                                              uf=lambda X: est.Estimate.getEstimate(X.dropna(), n=N[x.name], bins=20,
                                                                                    removeFixedSites=removeFixedSites,
                                                                                    normalizeTajimaD=False),
                                              winSize=int(winSize)))
    a=df.groupby(level=0, axis=1).apply(f).T.reset_index(level=0, drop=True).T
    n = df[(df > 0) & (df < 1)].apply(lambda x: scan.Genome(x.dropna(), len))
    n['stat'] = 'n'
    a = pd.concat([n.set_index('stat', append=True), a]).sort_index()
    return a

def FstHudson(a,pop):
    b=a.join(((a['all'] * 4 - a.loc[:,a.columns!='all'].sum(1)) / 2).rename('between'))
    c=(b.between- b[pop])/b.between
    c[(b.between<0) | (c<0)]=0
    return c.dropna()

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel Matrix"""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


from scipy.signal import convolve2d
import numpy as np
import scipy.stats as st


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    return kernel


def dens(b, filters=[5, 10, 20, 20]):
    conv = lambda x, k: pd.DataFrame(convolve2d(x.fillna(0), gkern(k, 1), mode='same'), index=x.index,
                                     columns=x.columns).replace({0: np.nan})
    x = b.copy(True)
    for k in filters:
        x = conv(x, k)
    return x



def outlier2D(a,k=200):
    from scipy.signal import convolve2d
    def dens(b, filters=[5, 10, 20]):
        conv = lambda x, k: pd.DataFrame(convolve2d(x.fillna(0), gkern(k, 1), mode='same'), index=x.index,
                                         columns=x.columns).replace({0: np.nan})
        x = b.copy(True)
        for k in filters:
            x = conv(x, k)
        return x
    top = a.set_index('RQ').loc[a.RQ.max()]['LRQP'].unique()
    b = a[['n', 'LRQP']].astype(str)
    b = (b.n + '.' + b['LRQP']).value_counts()
    b.index = pd.MultiIndex.from_tuples(map(lambda x: tuple(map(int, x.split('.'))), b.index.tolist()))
    b = b.unstack(0).sort_index(ascending=False)
    c = pd.concat([b.stack().rename('n'), dens(b)[~b.isnull()].loc[top].stack().rename('prob')],
                  1).dropna().sort_values(
        'prob')

    # o = c[c.n.cumsum() <= a.shape[0] / 1000] # top .1 %ile #
    try:
        o=pd.concat([c[c.n.cumsum()<=k],c[c.n.cumsum()>k].iloc[[0]]])
    except:
        # print c
        o=c[c.n.cumsum()<=k]
    o = a.reset_index().set_index(['LRQP', 'n']).loc[o.index].reset_index().set_index(['CHROM', 'POS'])
    return o
def damp(x , a=1):
    return x.apply(lambda y: y+a*np.sign(y))

def describe(a,cond=True):
    if cond:
        f = lambda x: describe(x,False )
        x=a
        return pd.concat([f(x),f(x[x>0]),f(-x),f(x[x>x.median()]),f(x[x>=x.quantile(0.95)]),f(x[x>=x.quantile(0.99)])],1,keys=['all','pos','median','q95','q99'])
    else:
        desc = a.describe()
        desc['q9'] = a.quantile(0.9)
        desc['q95'] = a.quantile(0.95)
        desc['q99'] = a.quantile(0.99)
        desc['q999'] = a.quantile(0.999)
        desc['q9999'] = a.quantile(0.9999)
        desc['q99999'] = a.quantile(0.99999)

        return desc

def describe2Tail(a):
    return pd.concat([describe(a),describe(-a)],1,keys=['+','-'])

def savePermille(fname,n=None):
    a=gz.load(fname)
    a=a.loc[:, (a.iloc[0].apply(type) != str).tolist()]
    gz.save(a.apply(lambda x: rankPercentile(x, 1e3,n)),fname.replace('.gz','.permille.gz'))
    gz.save(a.apply(lambda x: rankPercentile(x, 1e6, n)), fname.replace('.gz', '.ppm.gz'))

def savePermilleTruncated(pops,f='/home/arya/scan/Fisher/{}.{}.gz'):
    try:
        n=pd.read_pickle('/home/arya/Kyrgyz/scan/{}.{}.AF.desc.snp.df'.format(pops[0],pops[1]))['+']['all']['count']
        # print pops[:2],"{:,}".format(int(n))
        savePermille(f.format(pops[0],pops[1]),n)
    except:
        print('Error', pops)
def TI(a):
    return a.replace({False:None}).dropna().index
def loadAprioriList(i=0,x=None):
    a=pd.read_pickle('/home/arya/Kyrgyz/aprioriList/all.df')
    if x is not None:
        print('Loading {} List'.format(x))
        a = a.loc[x]
    else:
        print('Loading {} List'.format(a.index.levels[0][i]))
        a = a.loc[a.index.levels[0][i]]
    a=a.set_index('Category').loc['Protein Coding'].sort_values('Relevance score',ascending=False)
    return a.drop(['Gifts', 'GC Id'],1).reset_index(drop=True).rename(columns={'Relevance score':'GCscore'})





split = lambda x, i: [x[:i], x[i:]]
import random
def randomPartitionSet(seq,n):
    return split(random.sample(seq, len(seq)), n)


def loadUKBB_ANN(CHROM=None,path=UKBB_PATH):
    if CHROM is None:
        return pd.concat(map(lambda x: loadUKBB_ANN(x,path),range(1,23)+['X']))
    return pd.read_pickle(path+'chr{}.vcf.gz.ANN.sdf'.format(CHROM))

def intervalsSNPEff(CHROM=None,path=None,fname=None):
    # print CHROM
    if fname is not None:
        g = pd.read_pickle(fname)
    else:
        g=pd.read_pickle(path+'chr{}.ANN.sdf'.format(CHROM)).loc[[CHROM]]

    g=g.reset_index().set_index('Feature_Type').sort_index().xs('transcript').reset_index(drop=True).set_index(['CHROM','POS']).sort_index()

    a= scan.Genome(g.Gene_Name.iloc[:], f=lambda x: [x.unique()]).apply(lambda x: list(x[0])).rename('genes')
    b= scan.Genome(g.Annotation.iloc[:], f=lambda x: [x.value_counts()]).apply(lambda x: x[0])
    c= scan.Genome(g.Annotation_Impact.iloc[:], f=lambda x: [x.value_counts()]).apply(lambda x: x[0])
    if fname is not None:
        pd.concat([a, b, c], 1).to_pickle(fname+'.idf')
    else:
        pd.concat([a,b,c],1).to_pickle(path+'chr{}.ANN.idf'.format(CHROM))

def CDF(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index().cumsum()
        return x/x.iloc[-1]
    except:
        x = a.value_counts().cumsum()
        return x / x.iloc[-1]

def CDFCounts(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index().cumsum()
        return x
    except:
        x = a.value_counts().cumsum()
        return x

def CDFPDF(a,round2=2):
    return pd.concat([CDF(a,round2),PMF(a,round2),PMFCounts(a,round2),CDFCounts(a,round2)],1,keys=['CMF','PMF','Mass','CummulativeMass'])


def PMF(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index()
        return x/x.sum()
    except:
        # print 'Categorical'
        x = a.value_counts()
        return x / x.sum()

def PMFCounts(a,round2=2):
    try:
        return a.round(round2).value_counts().sort_index()
    except:
        # print 'Categorical'
        return a.value_counts()


def loadFreqs(pops,i,AA=False,pad=0):
    f=('/home/arya/POP/KGZU+ALL/GT/AF.gz','/home/arya/POP/KGZU+ALL/GT/DAF.gz')[AA]
    # f = ('/home/arya/POP/HAT/GT/AF.gz', '/home/arya/POP/HAT/GT/DAF.gz')[AA]
    try:
        if '.' in  pops[0]:
            x=loadHealthyFreqs(pops[0].split('.')[0],BED.expand(i, pad))
            x = polymorphixDF(pd.concat([x,gz.load(i=BED.expand(i, pad), f=f, indvs=pops[1:])],1))
        else:
            x=polymorphixDF(gz.load(i=BED.expand(i, pad), f=f, indvs=pops))
        return x
    except:
        pass

def loadHealthyFreqs(pop,i,AA=False,pad=0):
    try:return (gz.load('/home/arya/POP/HAT/GT/chr{}.vcf.gz', i, indvs=healthyIndvs(pop)).mean(1) / 2).rename(pop+'.HLT')
    except:
        pass

def healthyIndvs(pop):
    return VCF.loadPanel().set_index(['super_pop', 'pop']).sort_index().loc[(pop, 'HLT'), 'sample'].tolist()

def scanAndPlot(a,f,k=100,ann=None,winSize=50000):
    b= scan.Genome(a, f,winSize=winSize)
    import UTILS.Plots as pplt
    pplt.Manhattan(b, top_k=k)
    # if winSize=500
    I = BED.getIntervals(b.sort_values().iloc[-k:], padding=50000, ann=ann, expandedIntervalGenes=True)
    import pylab as plt
    plt.suptitle(intervalStats(I))
    return I
def intervalStats(I):
    return '{} genes in {} intervals of total {}Kb length '.format( len(getGeneList(I.genes)), I.shape[0], I.len.sum()/1000)


class skit:
    def __init__(self,i,pops,path = '/home/arya/POP/KGZU+ALL/GT/'):
        self.gt = skit.getGT(i, pops, path)
        self.p1, self.p2, self.p3 = skit.getIndvs(pops)

    @staticmethod
    def getGT(i,pops,path):
        freq = gz.load(path+'AF.gz', i, indvs=pops).loc[i.CHROM]
        freq = freq[(freq[pops[0]] - freq[pops[1]]).round(2).abs() > 0.0]
        gt = gz.load(path+'chr{}.vcf.gz', i, indvs=VCF.IDs(pops)).loc[i.CHROM].loc[freq.index].dropna()
        return gt
    @staticmethod
    def getIndvs(pops):
        p1 = VCF.ID(pops[0]).tolist()
        p2 = VCF.ID(pops[1]).tolist()
        p3=None
        if len(pops)>2:p3 = VCF.ID(pops[2]).tolist()
        return p1,p2,p3
    @staticmethod
    def __PBS(gt, p1, p2, p3):
        T=lambda x: abs(np.log(1-x))
        return (T(skit.__FstWin(gt, p1, p2)) +T(skit.__FstWin(gt, p1, p3)) - T(skit.__FstWin(gt, p2, p3)))/2.

    @staticmethod
    def __FstWin(gt, p1, p2, weir=True):
        import allel
        gt = gt[p1 + p2]
        id = pd.Series(range(gt.shape[1]), index=gt.columns)
        subpops = [id.loc[p1].tolist(), id.loc[p2].tolist()]
        m = {0: [0, 0], 1: [0, 1], 2: [1, 1]}
        g = np.array(gt.applymap(lambda x: m[x]).values.tolist())
        n = g.shape[0]
        if weir:
            f = allel.stats.windowed_weir_cockerham_fst
            a = f(np.arange(n), g, subpops, size=n, start=0, stop=n - 1)[0][0]
        else:
            f = allel.stats.windowed_hudson_fst
            g = allel.GenotypeArray(g)
            ac1 = g.count_alleles(subpop=subpops[0])
            ac2 = g.count_alleles(subpop=subpops[1])
            a = f(np.arange(n), ac1, ac2, size=n, start=0, stop=n - 1)[0][0]
        if a<0: a=0
        return a

    def PBS(self):
        return skit.__PBS(self.gt,self.p1,self.p2,self.p3)

    def FstWin(self):
        return skit.__FstWin(self.gt, self.p1, self.p2)

    def computeEmpiricalPvalPBS(self,k=1000):
        obs = self.PBS()
        p12 = self.p1 + self.p2
        def _null():
            p1, p2 = randomPartitionSet(p12, len(self.p2))
            a=skit.__PBS(self.gt,p1,p2 , self.p3)
            # print a
            return a
        null=pd.Series([_null() for _ in range(100)])
        # return null
        if k>100:
            if (pd.Series(null) >= obs).mean() <= 0.1:
                null = pd.Series([_null() for _ in range(k-100)]).append(null)
        return (null >= obs).mean()

def computeEmpiricalPvalPBSHelper(args):
    i,pops=args
    return skit(i, pops).computeEmpiricalPvalPBS(k=100)

def loadGTEx(i,tissue='Lung',f='/home/arya/GTEx/PAH/tabix/chr{}.{}.allpairs.txt.gz'):
    import mygene
    mg = mygene.MyGeneInfo()
    a=gz.load(f=f.format(i.CHROM,tissue),i=i)
    g=a.gene.drop_duplicates().tolist()
    def query(x):
        try: return mg.query(x.split('.')[0], scopes='ensembl.gene')['hits'][0]['symbol']
        except: pass
    g=pd.Series(map(query,g),index=g)
    return a.replace({'gene':g.to_dict()}).dropna()

def sfs(x,rangeIndex=False,strIndex=False,norm=False):
    y=  pd.cut( x.values,bins=np.arange(0,1.01,0.1),include_lowest=True).value_counts()
    if strIndex:
        f=lambda x: '{:.1f}'.format(x.right)
    else:
        f = lambda x: np.round(x.right,1)
    if not rangeIndex:
        y=pd.Series(y.values, index=map(f, y.index.values))
    if norm: y=y/y.sum()
    return y



def loadPlinkGTgz(f,header=True,dtype=None):
    """
    :param f: recode A created
    :return:
    """
    if header:
        a = pd.read_csv(f, sep='\t', index_col=0,dtype=dtype)
        a.columns = map(lambda x: '_'.join(x.split('_')[:-1]), a.columns)
    else:
        a = pd.read_csv(f, sep='\t', index_col=0, header=None,dtype=dtype)
    return a


def loadPlinkFreq(x,pop,AC=False,ID=False,name=False,level=False,CHROM=False,flat=False,splitID=False,path=dataPath+'Human/1KG/hg38/bed/pgen/refFromFa/CPRA/AF/'):
    if pop=='UKBB':
        path=UKBB_PATH+'AF/'
    ac = ['', ',7'][AC]
    id = ['', ',3'][ID|splitID]
    cols = ['POS', 'ID','REF','ALT', 'freq','AC']
    if not AC: cols=np.delete(cols, -1)
    if not len(id): cols = np.delete(cols, 1)
    f=path+'{}.gz'.format(pop)
    if x =='.':
        a=execute('zcat {}  |cut -f2{},4,5,6{}'.format(f, id, ac))
    else:
        a = execute('{}bin/tabix {} {}  |cut -f2{},4,5,6{}'.format(home, f, x,id, ac))
    a.columns=cols
    a.freq = a.freq.astype(float)
    a.POS=a.POS.astype(int)
    a=a.set_index('POS')
    if splitID:
        a[1] = a.ID.apply(lambda x: x.split('_')[2:])
        a['REF_ID'] = a[1].apply(lambda x: x[0])
        a['ALT_ID'] = a[1].apply(lambda x: x[1])
        a=a.drop(1,1)
        if (not ID) and (splitID):
            a = a.drop('ID', 1)
    if name: a=a.rename(columns={'freq':pop})
    if level: a=pd.concat([a],1,keys=[pop])
    if flat: a=renameColumns(a,'_'+pop,False)
    if CHROM:
        if x=='.':
            a['CHROM'] = a.ID.apply(lambda x: INT(x.split('_')[0]))
            a.index.name='POS'
            a=a.reset_index().set_index(['CHROM','POS'])
        else:
            a = pd.concat([a], keys=[x])
        a.index.names=['CHROM','POS']
    return a


class GENOME:
    @staticmethod
    def after(x, pos=1e7):
        if len(x.index.names)==1:
            x.index.name='POS'
        if pos > 0:
            return x[x.index.get_level_values('POS') > pos]
        else:
            return x[x.index.get_level_values('POS') < abs(pos)]

    def __init__(self,assembly=38,dmel=False,faPath='{}storage/Data/Human/ref/'.format(home)):
        import pysam
        self.assembly=assembly
        organism=('hg','dmel')[dmel]
        self.name='{}{}'.format(organism,self.assembly)
        GENOMEFA = '{}{}.fa'.format(faPath,self.name)
        self.g = pysam.Fastafile(GENOMEFA)
    def chrom(self, a,CHROM):
        return pd.DataFrame(a.groupby(level=0).apply(lambda x: self.base(CHROM,x.name)))#.loc[CHROM]#.rename(self.name)

    def base(self,CHROM,POS):
        return self.g.fetch('chr{}'.format(CHROM), POS - 1, POS).upper()
    def genome(self,a,join=True):
        b=a.groupby(level=0).apply(lambda x: self.chrom(a.loc[x.name], x.name))[0].rename(self.name)
        if join:
            b=quickJoinGenome(pd.DataFrame(b),a,CHROMS=a.index.get_level_values(0).unique().tolist())
        b.index.names = ["CHROM", "POS"]
        return b

    @staticmethod
    def mergeCHROM(a, verbose=False, keys=None):
        """
        :param a: list of series each of which is a chromosome
        :return:
        """
        a = [x for x in a if x is not None]
        if not len(a): return None
        CHROM = a[0].index[0][0]
        if verbose: print(CHROM)
        b = pd.concat([pd.concat([dedup(x.loc[CHROM]) for x in a], 1, keys=keys)], keys=[CHROM])
        b.index.names = ['CHROM', 'POS']
        return b

    @staticmethod
    def merge(a, CHROMS=range(1, 23), keys=None):
        if CHROMS is None: CHROMS = a[0].index.levels[0]
        def xs(x, c):
            try:
                return x.loc[[c]]
            except:
                pass

        a = [GENOME.mergeCHROM([xs(x, c) for x in a], keys=keys) for c in CHROMS]
        a = [x for x in a if x is not None]
        return pd.concat(a)

    @staticmethod
    def joinCHROM(a, b, how, verbose=False):
        """
        :param a: list of series each of which is a chromosome
        :return:
        """
        if a is None: return
        if a.shape[0] == 0: return
        CHROM = a.index[0][0]
        return pd.concat([a.loc[CHROM].join(b.loc[CHROM], how=how)], keys=[CHROM])

    @staticmethod
    def join(a, b, CHROMS=range(1, 23), how='inner'):
        if CHROMS is None:  CHROMS = a.index.get_level_values(0).unique().tolist()
        if how == 'inner':    CHROMS = b.index.get_level_values(0).unique().tolist()

        def xs(x, c):
            try:
                return x.loc[[c]]
            except:
                pass

        a = [GENOME.joinCHROM(xs(a, c), xs(b, c), how=how) for c in CHROMS]
        a = [x for x in a if x is not None]
        a = pd.concat(a)
        a.index.names = ['CHROM', 'POS']
        return a

    @staticmethod
    def safeConcat(a, keys=None):
        return pd.concat([x for x in a if x is not None], keys=keys)

    @staticmethod
    def filterGapChr(a, chr, GAP):
        b = a.loc[chr]
        gap = GAP.loc[chr]
        gap['len'] = gap.end - gap.start
        return b.drop(pd.concat([maskChr(b, i) for _, i in gap.iterrows()]).index)

    @staticmethod
    def filterGapChr(a, CHROM, gap):
        b = a.loc[CHROM]
        return b.drop(pd.concat([maskChr(b, i) for _, i in gap.loc[CHROM].iterrows()]).index)

    @staticmethod
    def filterGap(a, assempbly=19, pad=200000):
        CHROMS = a.index.get_level_values('CHROM').unique()
        gap = loadGap(assempbly, pad)
        return pd.concat([GENOME.filterGapChr(a, chr, gap) for chr in CHROMS], keys=CHROMS)


class DBSNP():
    def __init__(self,hg):
        self.hg=hg
        self.idx=DBSNP.loadIDX(self.hg)
    @staticmethod
    def loadIDX(hg=37):
        f=home+'storage/Data/Human/dbSNP/151/GRCh{}/byBatch/byBatch.idx.gz'.format(hg)
        if hg==3738:f = home + 'storage/Data/Human/dbSNP/151/GRCh37/noINFO/hg19/1-22XYM/hg38/byBatch/byBatch.idx.gz'
        return pd.read_csv(f, sep='\t',header=None,names=['batch','start','end']).set_index('batch')

    @staticmethod
    def batch(hg,i):
        print( i,int(i))
        f = '/home/ubuntu/storage/Data/Human/dbSNP/151/GRCh{}/byBatch/{:02d}.gz'.format(hg, int(i))
        if hg==3738:
            f = '/home/ubuntu/storage/Data/Human/dbSNP/151/GRCh37/noINFO/hg19/1-22XYM/hg38/byBatch/{:02}.gz'.format(i)
        return pd.read_csv(f, sep='\t', header=None, index_col=0)


    def load(self,a):
        if isinstance(a,list):
            a=pd.Series(list(set(a)))
        if a.dtype!=int:
            a=a.apply(lambda x: int(x[2:]))
        def f( x):
            if x.size > 0:
                return x.rename(0).reset_index().set_index(0).sort_index().join(DBSNP.batch(self.hg,x.name),how='inner').reset_index()
        batches = pd.cut(a.values, [0] + self.idx['end'].tolist(), labels=self.idx.index)
        a.index = batches.tolist()
        b=a.groupby(level=0).apply(f).reset_index()
        if b.shape[0]:
            b=b[range(5)]
            b.columns=['ID', 'CHROM','POS','REF','ALT']
            # b= b..dropna();
            b.CHROM=b.CHROM.apply(INT);b.POS=b.POS.apply(int)
            b=b.drop_duplicates()
            b.ID=b.ID.apply(lambda x: 'rs'+str(x))
            return b.set_index('ID').sort_index()

    @staticmethod
    def loadCAD(risk,assembly):
        dataset=risk.dataset.iloc[0]
        f=home + 'CAD/raw/{}.dbSNP{}.df'.format(dataset,assembly)
        try:
            raise 0
            a=pd.read_pickle(f)
        except:
            ID=risk.ID
            ID=ID[ID.apply(lambda x: x.split(';')[0][:2]=='rs')].apply(lambda x: int(x.split(';')[0][2:]))
            a=DBSNP(assembly).load(ID)
            # a.to_pickle(f)
        return a



class Imputation:
    @staticmethod
    def mode(a):
        """Categorical Datafreme"""
        return  a.apply(lambda x: x.fillna(x.mode()[0]))

    @staticmethod
    def mean(a):
        """Real-valued Datafreme"""
        return a.apply(lambda x: x.fillna(x.mean()))

    @staticmethod
    def filter(a,maxMissing):
        return a.loc[:,a.isnull().mean()<maxMissing]

    @staticmethod
    def soft(a,max_iters=20,maxMissing=1):
        from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
        a=Imputation.filter(a,maxMissing)
        return pd.DataFrame(SoftImpute(max_iters=max_iters, verbose=False).complete(a.values),index=a.index,columns=a.columns)



class Learner:
    def __init__(self,X):
        self.names=X.columns
        self.model=[]
        self.train=[]
        self.test=[]
        self.deci=[]

    def addPerfoemance(self,test,train,model):
        from copy import deepcopy

        self.model+=[deepcopy(model)]
        self.test += [test]
        self.train += [train]
    def merge(self):
        self.train=pd.concat(self.train,1).T
        self.test = pd.concat(self.test, 1).T

        self.res=pd.concat([self.train, self.test], 1, keys=['Train', 'Test']).reorder_levels([1, 0], axis=1).sort_index(1)
        self.res.loc['mean']=self.res.mean()
        try:
            self.w=pd.DataFrame(map(lambda x: x.coef_[0], self.model),columns=self.names).T
        except:pass

    @staticmethod
    def AUC(y,yp,plot=False):
        from sklearn.metrics import roc_curve, auc
        roc=roc_curve(y,yp)
        roc = pd.DataFrame(list(roc[:2])).T
        x=auc(roc[0], roc[1])
        if plot:
            Learner.plotLabels(y,yp)
            import pylab as plt
            plt.suptitle('AUC={:.3f}'.format(x))
        return x
    @staticmethod
    def defaultModels(cl):
        from sklearn import svm
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.linear_model import LinearRegression
        if cl == 'LinearRegression':
            cl = LinearRegression()
        if cl == 'LR':
            cl=LogisticRegression()
        if cl is None or cl == 'LinearSVM':
            cl = svm.SVC(kernel='linear', C=1,random_state=0,cache_size=10000)
        if cl is None or cl == 'NonLinearSVM':
            cl = svm.SVC( C=10,cache_size=10000)
        if cl == 'RandomForest':
            cl = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=1000)
        if cl == 'LDA':
            cl = LinearDiscriminantAnalysis()
        if cl == 'QDA':
            cl = QuadraticDiscriminantAnalysis()
        return cl

    @staticmethod
    def pred(model,X):
        try:
            yp = model.decision_function(X)
        except:
            yp = model.predict(X)
        return yp

    @staticmethod
    def getCVAUC(X,y,cl=None,k=5,train_size=2000,test_size=None,random_state=np.random.randint(1000),verbosity=0):
        from sklearn.model_selection import StratifiedShuffleSplit
        a = Learner(X)
        cl=Learner.defaultModels(cl)
        splitter=StratifiedShuffleSplit(n_splits=k,test_size=test_size, train_size=train_size,random_state=random_state)
        for i, j in splitter.split(np.zeros(X.shape[0]),y.values):
            # print np.sort(i)[:10],np.sort(j)[:10]
            if verbosity:
                print( 'Train shape:',(X.iloc[i]).shape)
            model=cl.fit(X.iloc[i], y.iloc[i])
            if verbosity:
                print('Test  shape:',(X.iloc[j]).shape)
            yp=Learner.pred(model,X.iloc[j])
            yptr=Learner.pred(model, X.iloc[i])
            a.addPerfoemance(train=Learner.RankingMetrics(y.iloc[i].values,yptr),test=Learner.RankingMetrics(y.iloc[j].values,yp),model=model)
        a.merge()
        if verbosity:
            print(  a.res)
            print(a.w)
        return a
    @staticmethod
    def nullAUC(frac=0.01):
        n=int(10**np.ceil(-np.log10(frac)))
        y=pd.Series(1,index=range(n))
        y.iloc[:int(n*frac)]=0
        Learner.AUC(y,y*0+1)
    @staticmethod
    def MRR(y, yp):
        df = pd.DataFrame([y, yp]).T
        df['r'] = df[df.columns[1]].rank(ascending=False)
        return 1. / df[df.iloc[:, 0] == 1].r.min()

    @staticmethod
    def RankingMetrics(y,yp):
        from sklearn.metrics import average_precision_score
        a = pd.DataFrame([y, yp]).T
        a.columns=[0,1]
        a=a.sort_values([1],ascending=False)
        def apAt( q):
            i=int(a.shape[0]*q)+1
            ai=a.iloc[:i]
            k='AP{:d}p'.format(int(q*100))
            if q==1: k='AP'
            return pd.Series({k:average_precision_score(ai[0].values, ai[1].values)})
        Q=[1,0.5,0.25,0.1,0.05,0.01]
        return pd.concat(list(map(apAt,Q))+[pd.Series({'AUC':Learner.AUC(y,yp),'MRR':Learner.MRR(y,yp)})])

    @staticmethod
    def scale(X):return X / (X.max() - X.min())
    @staticmethod
    def Cox2(a,summary=True,scale=False,plot=True,strata=None,args={},ax=None):
        import lifelines as ll
        if scale:
            c=a.columns[list(map(lambda x: x not in ['time','event'],a.columns))]
            a[c]=Learner.scale(a[c])
        cph= ll.CoxPHFitter().fit(a,duration_col='time', event_col='event', strata=strata,**args)
        sd = cph.summary.iloc[:, -2:].applymap(np.exp)
        cph.HR = (cph.summary['exp(coef)'] / ((sd['upper 0.95'] - sd['lower 0.95']) )).rename('HR95CI')
        if plot: Learner.plotCox(cph.summary['exp(coef)'], 'Cindex={:.3f}'.format(cph.score_),ax=ax)
        if summary:
            cph=cph.summary

        return cph

    @staticmethod
    def Cox3(a,summary=True,scale=False,plot=True,strata=None,args={}):
        if scale:
            c=a.columns[list(map(lambda x: x not in ['time','event'],a.columns))]
            a[c]=Learner.scale(a[c])
        cph= CPH().fit(df=a,duration_col='time', event_col='event', strata=strata,**args)
        sd = cph.summary.iloc[:, -2:].applymap(np.exp)
        cph.HR = (cph.summary['exp(coef)'] / ((sd['upper 0.95'] - sd['lower 0.95']) )).rename('HR95CI')
        if plot: Learner.plotCox(cph.summary['exp(coef)'], 'Cindex={:.3f}'.format(cph.score_))
        if summary:
            cph=cph.summary

        return cph


    @staticmethod
    def KM(Y):
        """
        :param Y: dataframe with time and event
        :return: Survival
        """
        import  lifelines as ll
        km = ll.KaplanMeierFitter().fit(Y.time, Y.event)
        return km.survival_function_.iloc[:,0]

    @staticmethod
    def NA(Y):
        """
        Nelson-Alen
        :param Y:
        :return: Cummulitive Hazard
        """
        import lifelines as ll
        na = ll.NelsonAalenFitter().fit(Y.time, Y.event)
        return na.cumulative_hazard_.iloc[:, 0]

    @staticmethod
    def kYearRisk(cumulative_hazard, since, cph=None,k=10):
        if cph is  not None: cumulative_hazard=cph.baseline_cumulative_hazard_.iloc[:,0]
        surv = cumulative_hazard.apply(lambda x: np.exp(-x))
        return surv[surv.index < since].iloc[-1] - surv[surv.index < since + k].iloc[-1]

    @staticmethod
    def Incidence(A,strata=None,full=True):
        if strata is not None: return A.groupby(strata).apply(Learner.Incidence).unstack(0)
        a=A.copy(True)
        # a.time = a.time.round().astype(int)
        def atrisk(a):
            b = (a.shape[0] - CDFCounts(a.time))
            b.iloc[1:] = b.iloc[:-1];
            b.iloc[0] = a.shape[0]
            return b

        def casesat(a):
            return a.groupby('time').event.sum()

        c,r=casesat(a) , atrisk(a)
        i=(casesat(a) / atrisk(a))
        if full:
            c= pd.concat([c,r,i],1,keys=['CasesAt','AtRisk','Incidence'])
            c['Prevalence'] = c.CasesAt.cumsum()
            # c['Censored'] = pd.concat(
            #     [pd.Series(c.AtRisk.diff().abs().dropna().values, c.index[:-1]), c.iloc[[-1]].AtRisk]).astype(
            #     int) - c.CasesAt
            return c
        return i

    @staticmethod
    def Incidence2(A, strata=None, full=True):
        if strata is not None: return A.groupby(strata).apply(Learner.Incidence).unstack(0)
        a = A.copy(True)
        # a.time = a.time.round().astype(int)

        def atrisk(a):
            b = (a.kernel.sum() - a.groupby('time').kernel.sum().cumsum())
            b.iloc[1:] = b.iloc[:-1];
            b.iloc[0] = a.kernel.sum()
            return b

        def casesat(a):
            return a.groupby('time').soft.sum()


        c, r = casesat(a), atrisk(a)

        i = (casesat(a) / atrisk(a))
        if full:
            c = pd.concat([c, r, i], 1, keys=['CasesAt', 'AtRisk', 'Incidence'])
            c['Prevalence'] = c.CasesAt.cumsum()
            # c['Censored'] = pd.concat(
            #     [pd.Series(c.AtRisk.diff().abs().dropna().values, c.index[:-1]), c.iloc[[-1]].AtRisk]).astype(
            #     int) - c.CasesAt
            return c
        return i

    @staticmethod
    def CoxPred2(cph,a,strata=None,verbose=False,plot=True):
        import UTILS.cindex as ci
        df=cph.predict_partial_hazard(a).rename(columns={0: 'pred'})
        df.index=a.index
        df=df.join(a[['time', 'event']+ [[strata],[]][strata is None]])
        c=ci.concordance(df, strata, verbose=verbose)
        auc=Learner.AUC(df.event,df.pred,plot=plot)
        metrics=pd.Series({'AUC':auc,'Cindex':c})
        return metrics,df

    @staticmethod
    def Cox(X,Y=None,eventi=0,scale=False,plot=False):
        if Y is None:
            Y=X[['event','time']]

        if scale:
            X=Learner.scale(X)
        y=Y.copy(True)
        y.iloc[:, eventi]=y.iloc[:,eventi].astype(bool)
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        y = sarray(y.loc[X.index])
        cph = CoxPHSurvivalAnalysis(alpha=0.01).fit(X, y)
        cph.w = pd.Series(cph.coef_, index=X.columns)
        cph.cindex = cph.score(X, y)
        if plot:
            Learner.plotCox(cph.w,'Cindex={:.3f}'.format(cph.cindex));
        return cph

    @staticmethod
    def plotCox(w,title=None,ax=None):
        import pylab as plt
        if ax is None: ax=AX()
        w.plot.bar( ax=ax)
        if title is not None:   ax.set_title(title)
    @staticmethod
    def CoxPred(model,X,Y,eventi=0,lifelines=True,arya=False):
        y = Y.copy(True)
        y.iloc[:, eventi] = y.iloc[:, eventi].astype(bool)
        y = sarray(y.loc[X.index])
        if lifelines:
            try:
                pred= pd.Series(model.predict(X), index=X.index, name='pred')
            except:
                pred = model.predict_partial_hazard(X)[0].rename('pred')
            df = pd.concat([Y,pred],1)
            return Learner.concodtance(df)
        if arya:
            df = pd.concat([Y, pd.Series(model.predict(X), index=X.index, name='pred')], 1)
            return Learner.c_index(df)
        return model.score(X,y)

    @staticmethod
    def predLifelines(X,model):
        """
        :param X: Dataframe of input
        :param model: Lifelines model that is a series with first level index of  ['mean' , 'coef']
        :return:
        """
        return np.exp((X[model['mean'].index] - model['mean']).dot(model['coef'])).rename('pred')

    @staticmethod
    def LR(X,y,cl='LDA'):
        from sklearn.linear_model import LogisticRegression
        model = Learner.defaultModels(cl).fit(X, y)
        return model,Learner.AUC(y,model.decision_function(X))


    @staticmethod
    def parseFeatures(x):
        try:
            return x.astype(int)
        except:
            try:
                return x.astype(float)
            except:
                return x.astype('category').cat.codes
    @staticmethod
    def gridSVM(X,y,cv,rangeG =10. ** np.arange(-6, 6, 2),rangeC = 10. ** np.arange(-4, 5, 2),nproc=1,cl=None,verbose=False):
        from itertools import product
        print('N params: ',len(rangeC) *len( rangeG))
        args = product([X],[y],[cv], rangeC, rangeG,[cl],[verbose])
        import multiprocessing
        if nproc>1:
            res=multiprocessing.Pool(nproc).map(SVMtrainhelper, args)
        else:
            res=map(SVMtrainhelper, args)
        return pd.concat(res)

    @staticmethod
    def c_index(df):
        dfp = df[df['event'] == 1]
        concordant = 0
        discordant = 0
        tied_risk = 0
        for row in dfp.itertuples():
            if row.event:  # an event should have a higher score
                comparables = df.loc[(df.time > row.time), 'pred'].values
                con = (comparables < row.pred).sum()
            else:  # a non-event should have a lower score
                comparables = dfp.loc[(dfp.time < row.time), 'pred'].values
                con = (comparables > row.pred).sum()
            concordant += con
            tie = (comparables == row.pred).sum()
            tied_risk += tie
            discordant += comparables.size - con - tie
        cindex = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
        return cindex

    @staticmethod
    def plotLabels(y,yp):
        import pylab as plt
        import seaborn as sns
        Y = pd.concat([y.rename('Label'), yp.rename('Prediction')], 1)
        ax = plt.subplots(1, 3, figsize=(10, 6), dpi=120, sharey=True)[1]
        sns.stripplot(data=Y, y='Prediction', x='Label', ax=ax[0], alpha=0.2)
        sns.boxplot(data=Y, y='Prediction', x='Label', ax=ax[1])
        Y['x'] = 0
        sns.violinplot(data=Y, y='Prediction', x='x', hue='Label', ax=ax[2], split=True)
        ax[1].set_ylabel('')
        ax[2].set_ylabel('')
        ax[2].set_xlabel('')
        ax[0].set_xlabel('')
        plt.tight_layout(pad=0.1)


class LogisticReg:
    """
        https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
    Wrapper Class for Logistic Regression which has the usual sklearn instance
    in an attribute self.model, and pvalues, z scores and estimated
    errors for each coefficient in

    self.z_scores
    self.p_values
    self.sigma_estimates

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):

        from sklearn import linear_model
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        import scipy.stats as stat
        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        F_ij = np.dot((X / denom[:, None]).T, X)  ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.array(
            [np.sqrt(Cramer_Rao[i, i]) for i in range(Cramer_Rao.shape[0])])  # sigma for each coefficient
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values

        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij

class Sampler:
    @staticmethod
    def ageMatch(A,ncontrol=100000,ageBinSize=5,verbose=False):
        """
        :param A: a dataframe with 2 columns: [age,X]  where X is a binary column for case/control status
        :param ncontrol: num of controls, for ncontrol=0, it takes case size
        :return: agematched dataset
        """
        np.random.seed(0)

        a=A.copy(True)
        a.index.name='eid'
        y=a.columns[a.columns!='age'][0]
        a['ageq']= a.age.apply(lambda x: roundto(x,ageBinSize))
        norm = lambda x: x / x.sum()
        d1= norm(a.set_index(y).ageq.groupby(level=0).apply(lambda x: x.value_counts()).unstack(1).T)
        d1.index.name = 'age'
        case=a[a[y] == 1].ageq;control = a[a[y] == 0].ageq
        if ncontrol == 0:ncontrol=case.shape[0]
        cc = control.reset_index().set_index('ageq')
        n=(PMF(case)*ncontrol).round().astype(int);n.loc[n.idxmax()]+=n.sum()-ncontrol
        def ss2(x):
            try:
                return cc.loc[x.name].iloc[np.random.choice(cc.loc[x.name].shape[0], min(cc.loc[x.name].shape[0], x.iloc[0]),replace=False)]
            except:
                return pd.DataFrame(pd.Series(None,name='eid'))
        ss=lambda x: cc.loc[x.name].iloc[np.random.choice(cc.loc[x.name].shape[0], min(cc.loc[x.name].shape[0], x.iloc[0]),replace=False)]
        i=case.index.tolist()+n.groupby(level=0).apply(ss2).dropna().eid.dropna().astype(int).tolist()
        if verbose:
            d2= norm(a.loc[i].set_index(y).ageq.groupby(level=0).apply(lambda x: x.value_counts()).unstack(1).T)
            d2.index.name = d1.index.name
            dd=pd.concat([d1,d2],1,keys=['Before','After'])
            print(dd)
            print(dd.After[1]-dd.After[0])
        return A.loc[i]

    @staticmethod
    def genderMatch(a,ncontrol=100000,verbose=False):
        np.random.seed(0)
        a.index.name = 'eid'
        y = a.columns[a.columns != 'Gender'][0]
        case = a[a[y] == 1].Gender;
        norm = lambda x: x / x.sum()
        control = a[a[y] == 0].Gender
        d1 = norm(a.set_index(y).Gender.groupby(level=0).apply(lambda x: x.value_counts()).unstack(1).T)
        d1.index.name='Gender'
        n = (PMF(case) * ncontrol).round().astype(int);
        cc = control.reset_index().set_index('Gender')
        ss = lambda x: cc.loc[x.name].iloc[np.random.choice(cc.loc[x.name].shape[0], min(cc.loc[x.name].shape[0], x.iloc[0]), replace=False)]
        i = case.index.tolist() + n.groupby(level=0).apply(ss).eid.tolist()
        if verbose:
            d2= norm(a.loc[i].set_index(y).Gender.groupby(level=0).apply(lambda x: x.value_counts()).unstack(1).T)
            d2.index.name = d1.index.name
            print(pd.concat([d1,d2],1,keys=['Before','After']))
        return a.loc[i]

    @staticmethod
    def ageGenderMatch(A,ncontrol=100000,ageBinSize=5,verbose=True):
        """
        :param A: dataframe with the columns [X,Gender,age] ,where X is the case/control status
        :param ncontrol: number of controls after subsampling and age matching
        :param ageBinSize:
        :param verbose:
        :return:
        """
        np.random.seed(0)
        a = A.copy(True)
        a.index.name = 'eid'
        y = a.columns[(a.columns != 'age')& (a.columns != 'Gender')][0]
        a['ageq'] = a.age.apply(lambda x: roundto(x, ageBinSize))
        norm = lambda x: x / x.sum()


        c=['Gender','ageq',y]
        d = norm(a.set_index(c).groupby(level=range(3)).size().unstack([y])).sort_index(1)

        case = a[a[y] == 1][c];
        control = a[a[y] == 0][c]


        if ncontrol == 0: ncontrol = case.shape[0]
        cc = control.reset_index().set_index(['Gender','ageq']).sort_index()
        n = (ncontrol * d[1].dropna()).astype(int)

        ss = lambda x: cc.loc[x.name].iloc[np.random.choice(cc.loc[x.name].shape[0], min(cc.loc[x.name].shape[0], x.iloc[0]), replace=False)]
        i = case.index.tolist() + n.groupby(level=[0, 1]).apply(ss).sort_index().eid.tolist()
        if verbose:
            d2= norm(a.loc[i].set_index(c).groupby(level=range(3)).size().unstack([y])).sort_index(1)
            d=pd.concat([d, d2], 1, keys=['Before', 'After']).fillna(0)
            d.index.names=['Gender','Age']
        return A.loc[i]

    @staticmethod
    def ss(x, n,random_state=0):
        np.random.seed(random_state)
        return x.iloc[np.random.choice(x.shape[0], min(n, x.shape[0]), replace=False)]

    @staticmethod
    def bootrstrap(x,f,K=10,frac=0.8,n=None,random_state=0):
        if n is None:n=int(x.shape[0]*frac)
        return pd.Series(index=range(K)).groupby(level=0).apply(lambda random_state: f(Sampler.ss(x,n,random_state.name)) )

    @staticmethod
    def random(a,ncontrol=0,n=None,seed=None):
        """
        :param a: a binary series (case/control), where its index is the subject id
        :param ncontrol: subsamples controls, when ncontrol=0 balances the dataset
        :param n: random sample from dataset
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        if n is not None:
            return a.iloc[np.random.choice(a.shape[0],min(n,a.shape[0]),replace=False)]
        if ncontrol == 0: ncontrol = a[a==1].sum()
        def ss(x,n): return x.iloc[np.random.choice(x.shape[0],min(n,x.shape[0]),replace=False)]
        i=a[a==1].index.tolist()+ ss((a[a==0],ncontrol)).index.tolist()
        return a.loc[i]

    @staticmethod
    def randomStratified(a,n):
        """
        :param a: a binary series (case/control), where its index is the subject id
        :param n: takes a random sample, keeping case/control ratio
        :return:
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=2, train_size=n,random_state=np.random.randint(1000))
        for i, j in splitter.split(np.zeros(a.shape[0]), a.values):
            break
        return a.iloc[i]

    @staticmethod
    def balancedTrain(a, nTrain=3000, ratio=1./3, random_state=0):
        b = a.reset_index().set_index('event').eid
        import random
        case = b.loc[1].sample(frac=1, random_state=random_state)
        control = b.loc[0].sample(frac=1, random_state=random_state)
        i = int(nTrain * ratio)
        j = int(nTrain * (1 - ratio))
        train = pd.concat([case.iloc[:i], control.iloc[:j]])
        test = pd.concat([case.iloc[i:], control.iloc[j:]])
        return a.loc[train], a.loc[test]

def SVMtrainhelper(args):
    from sklearn import svm
    from sklearn.model_selection import  cross_validate
    X, y,cv, C, gamma,cl,verbose = args
    print(verbose,cl)
    if cl is None:
        cl = svm.SVC(C=C, gamma=gamma, class_weight='balanced', tol=1e-2, cache_size=2000);
    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy',#'PError':'balanced_accuracy',
               'AP':'average_precision','Percision':'precision','Recall':'recall'}
    df=pd.DataFrame(cross_validate(cl, X, y, cv=cv,scoring=scoring))
    df['C']=C
    df['gamma']=gamma
    df['k']=df.index
    if verbose:
        pd.options.display.max_columns = 20;
        pd.options.display.expand_frame_repr = False
        print(pd.DataFrame(df.mean()).T.set_index(['C', 'gamma']))
    return df

def sarray(x):
    dtype = x.dtypes.to_dict()
    try:
        dtype=dtype.iteritems()
    except:
        dtype = dtype.items()
    dtype=list(dtype)
    return np.array(list(map(tuple, x.values.tolist())), dtype=dtype)


class CAT:
    @staticmethod
    def onehot( x,c=None,dropLast=False):
        if len(x.shape)==1:
            z= renameColumns(pd.get_dummies(x), str(x.name) + '_')
        else:
            z= pd.concat([x.drop(c,1),CAT.onehot(x[c])],1)
        if dropLast :
            z=z.iloc[:,:-1]
        return z


    @staticmethod
    def CatNonNegative(x, Q=[0.25, 0.75], bins=None, zero=True):
        include_lowest = False
        if zero:
            xx = x[x > 0]
        else:
            include_lowest = True
            xx = x
        if bins is None:
            bins = pd.Series(index=Q).groupby(level=0).apply(lambda q: xx.quantile(q.name))
        else:
            bins = pd.Series(bins)
        bins = pd.Series([0] + bins.tolist() + [x.max()]).drop_duplicates().tolist()
        lab = range(1, len(bins))
        y = pd.cut(x, bins, include_lowest=include_lowest, labels=lab).astype(float)
        yc = pd.cut(x, bins, include_lowest=include_lowest)
        cats = pd.Series(yc.cat.categories, index=lab)
        if zero:
            y[x == 0] = 0
            cats.loc[0] = 0
        cats[-1] = 'NA'
        y[x.isnull()] = -1
        # print(cats)
        return y.astype(int), cats.sort_index()

    @staticmethod
    def OR(x, y, cl='LR', cov=None):
        z = pd.DataFrame(x).join(y.rename('y'), how='inner')
        if cov is not None: z = z.join(cov)
        xx = z.drop('y', 1)
        w, auc = Learner.LR(xx, z.y,cl)
        auc = pd.Series([auc], index=['performance'])
        ci=pd.concat([pd.Series(w.coef_[0], index=xx.columns).apply(np.exp), auc])
        if cov is not None:
            ci= pd.concat([ci,CAT.OR(cov,y)],keys=['All','Cov'])
        return ci

    @staticmethod
    def HR(x, y, cov=None):
        z = pd.DataFrame(x).join(y, how='inner')
        if cov is not None: z = z.join(cov)
        cph = Learner.Cox2(z,summary=False)
        cindex = pd.Series([cph.score_], index=['performance'])
        ci=pd.concat([cph.summary['exp(coef)'], cindex])
        if cov is not None:
            ci= pd.concat([ci,CAT.HR(cov,y)],keys=['All','Cov'])
        return ci

    @staticmethod
    def both(x, y, cov=None):
        return pd.concat([CAT.HR(x, y, cov), CAT.OR(x, y.event, 'LDA', cov),CAT.OR(x, y.event,'LR', cov)], 1, keys=['HR', 'OR (LDA)','OR (LR)'])

    @staticmethod
    def ORCat(x, y, cov=None, cats=None):
        xx = x.fillna(-1).astype(int)

        pmf = PMF(xx)
        z = pd.concat(
            [CAT.HR(pd.get_dummies(xx), y, cov), CAT.OR(pd.get_dummies(xx), y.iloc[:, 0], cov), pmf, xx.value_counts()], 1,
            keys=['HR', 'OR', 'Prop', 'N'])
        if cats is not None: z = pd.concat([cats.rename(x.name), z], 1)
        z.index.name = x.name
        return z


def combinations(seq,minSize=1):
    from itertools import combinations, chain
    a= map(list, list(chain(*map(lambda n: combinations(seq, n), range(len(seq)+1)))))
    return [x for x in a if len(x)>=minSize]


def GBD():
    f='/home/ubuntu/storage/Data/Human/GBD/CAD.df'
    try:
        aa=pd.read_pickle(f)
    except:
        aa=pd.read_csv('/home/ubuntu/storage/Data/Human/GBD/GBD_USA2016_CAD_Incidence_Prevalence_Rate.csv', index_col=[0,3])['sex val'.split()]
        aa.sex=aa.sex.replace({'Male':1,'Female':0})
        aa=aa.set_index('sex',append=True)['val'].unstack(level=0)
        aa.to_pickle(f)
    return aa

def ICD10(a=None,index=False):
    import pandas as pd
    if a is None:
        return pd.read_csv('/home/ubuntu/storage/Data/Human/ICD10/codes.txt', sep=';', header=None, index_col=7)[8]
    aa=pd.Series(a.index).rename('ICD10').reset_index()
    i=aa.columns[:-1].tolist()

    pd.options.display.max_colwidth = 1000
    aa=aa.set_index(aa.columns[-1]).join(ICD10().rename('Desc')).reset_index().set_index(i)

    if index:
        print('hi')
        aa=pd.DataFrame(a).join(aa.set_index('ICD10'))
    return aa
def ICD10Super():
    "https://www.icd10data.com/ICD10CM/Codes"
    return pd.DataFrame(pd.Series("""A00-B99  Certain infectious and parasitic diseases
    C00-D49  Neoplasms
    D50-D89  Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
    E00-E89  Endocrine, nutritional and metabolic diseases
    F01-F99  Mental, Behavioral and Neurodevelopmental disorders
    G00-G99  Diseases of the nervous system
    H00-H59  Diseases of the eye and adnexa
    H60-H95  Diseases of the ear and mastoid process
    I00-I99  Diseases of the circulatory system
    J00-J99  Diseases of the respiratory system
    K00-K95  Diseases of the digestive system
    L00-L99  Diseases of the skin and subcutaneous tissue
    M00-M99  Diseases of the musculoskeletal system and connective tissue
    N00-N99  Diseases of the genitourinary system
    O00-O9A  Pregnancy, childbirth and the puerperium
    P00-P96  Certain conditions originating in the perinatal period
    Q00-Q99  Congenital malformations, deformations and chromosomal abnormalities
    R00-R99  Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
    S00-T88  Injury, poisoning and certain other consequences of external causes
    V00-Y99  External causes of morbidity
    Z00-Z99  Factors influencing health status and contact with health services""".split('\n')).apply(lambda x: x.split('  ')).tolist()).set_index(0)[1]

def superDisease(heart=True):
    h= pd.DataFrame(pd.Series("""K00-K95  Digestive
        O00-O99  Pregnancy
        I00-I09  Rheumatic
        I10-I16  Hypertensive
        I20-I25  Ischemic
        I26-I28  Pulmonary
        I30-I52  OtherHeart
        I60-I69  Cerebrovascular
        I70-I79  Arteries
        I80-I89  Veins
        I95-I99  OtherCirc""".split('\n')).apply(lambda x: x.strip().split('  ')).tolist()).set_index(1)[0]
    o= pd.Series({'Diabetes':'E08-E13','Neoplasms':'C00-D49','Mental':'F01-F99','Respiratory':'J00-J99','Metabolic':'E70-E88','Thyroid':'E00-E07','Symptoms':'R00-R09'})
    return pd.concat([o,h]).drop('Ischemic')
def binaryHist(A):
    a=A.astype(int)
    z=a.iloc[:,0].astype(str)
    for c in a.columns[1:]:
        z=z+a[c].astype(str)
    return z.value_counts().rename('.'.join(map(str,a.columns)))

def AX(FIG=False,figsize=(8, 6)):
    import pylab as plt
    fig,ax=plt.subplots(1, 1,figsize= figsize, dpi=144)
    if FIG: return fig
    return ax

from lifelines import CoxPHFitter
class CPH(CoxPHFitter):
    def __init__(self):
        super(CPH, self).__init__()

    def kYearRisk(self,t0,k,r):
        """
        :param t0: time in which kYearRish is being calculated, (Age of client)
        :param k: number of years
        :param r: relative risk
        :return: risk
        """
        s=self.baseline_survival_.iloc[:,0]
        t1=t0+k
        return  s[s.index < t0].iloc[-1]**r -s[s.index < t1].iloc[-1]**r # risk(t) =1-surv(t)

    def fit(self,**kwargs):
        super(CPH, self).fit(**kwargs)
        return self

def plinkGT(CHROM,POS=None,verbose=False,cat=False,fold=False,ids=None):
    if POS is None:
        if isinstance(CHROM,list):
            return pd.concat(map(plinkGT,CHROM),1)
        CHROM,POS=CHROM
    path=UKBB_PATH
    import uuid

    f='/tmp/'+str(uuid.uuid4())

    cmd=home+'bin/plink2 --bpfile {3}chr{0} --from-bp {1} --to-bp {1}  --chr {0} --out {2} --recode A > /dev/null 2>&1'.format(CHROM,POS,f,path)
    if verbose:
        print(cmd)
    os.system( cmd);
    a=pd.read_csv(f+'.raw',sep='\t').set_index('IID').iloc[:,-1].round().astype(np.int8)
    os.system('rm -f ' + f + '.*')
    if cat:a=a.astype('category')
    # a.columns=map(lambda x: '_'.join(x.split('_')[:-1]),a.columns)
    a.name= '_'.join( a.name.split('_')[:-1])
    if fold and not cat:
        if a.mean()>1:
            a=2-a
    if ids is not None:
        a=a.loc[np.intersect1d(ids,a.index)]
    return a

def loadParquetHelper(x):
    y=pd.read_parquet(x)
    if 'CPRA' in y.columns:
        y=y.set_index('CPRA')
    return y

def loadParquet(f,nProc=8):
    import  glob,multiprocessing
    files = glob.glob(f+'/*.parquet')
    pool = multiprocessing.Pool(nProc)
    a= pd.concat(map(loadParquetHelper, files))
    pool.terminate()
    return a



def saveDF(df,fname,index=True):
    df.to_pickle(fname)
    df.to_csv(fname.replace('.df','.tsv.gz'),sep='\t',compression='gzip',index=index)



class LDPrune():
    def __init__(self,f='/home/ubuntu/aux3/GWAS/train/AF/UKBB_Males.gz'):
        print('Loading')
        a = pd.read_csv(f, sep='\t', header=None)
        a=a.join(MAF(a[5]).round(3).rename('xf'))
        a['i']=a.xf.astype('category').cat.codes
        a=a[[2,'i']].set_index('i')[2]
        self.a=a

    @staticmethod
    def prune(f='/home/ubuntu/aux3/GWAS/train/LDPrune/ld.500.snps.ld.gz',cutoff=0.9):
        k=pd.read_csv(f,sep='\t',header=None)
        k=pd.DataFrame(k[0].apply(lambda x: x.split()).tolist()).T.set_index(0).T['SNP_A SNP_B R'.split()].set_index('SNP_A SNP_B'.split()).R.astype(float).abs()
        # k.value_counts().sort_index()
        # i=k.sort_values().index[300]
        # c=i[0].split('_')[0]
        # z=pd.concat([utl.plinkGT(c,i[0].split('_')[1]),utl.plinkGT(c,i[1].split('_')[1])],1).T.reset_index(drop=True).T
        # utl.binaryHist(z),z.mean()

        k=k[k.abs()>cutoff]
        if k.min()<-cutoff:
            print('Warning neglinkage found in',f)
        kk=k.astype(bool)
        keep,drop=[],[]
        for i, row in kk.iloc[:].groupby(level=0):
            if i not in drop:
                keep+=[i]
                drop+=TI(kk.loc[i]).tolist()
            elif i not in keep:
                drop += TI(kk.loc[i]).tolist()
        keep,drop=set(keep),set(drop)
        drop=[x for x in drop if x not in np.intersect1d(keep,drop)]
        return drop

    @staticmethod
    def runBin(i,ids):
        fout = home + 'train/LDPrune/ld.{}.snps'.format(int(i))
        try:
            print(i,'loaded')
            return pd.read_csv(fout+'.drop')[0].tolist()
        except:
            pd.Series(ids).to_csv(fout, index=False)
            pl='/home/ubuntu/bin/plink'
            bed='/home/ubuntu/aux3/GWAS/train/all'
            execute( '{} -bfile {}  --r gz  --extract {}   -out {}'.format(pl,bed,fout,fout),returnDF=False)
            drop=LDPrune.prune(fout+'.ld.gz')
            pd.Series(drop).to_csv(fout+'.drop',index=False)
            return drop

    def run(self):
        args=[(x,y.tolist()) for x,y in self.a.groupby(level=[0])]
        from multiprocessing import Pool
        DROP=pd.concat(Pool(6).map(LDHelper,args))
        D=pd.DataFrame(DROP.tolist()).stack().drop_duplicates().dropna().reset_index(drop=True)
        saveDF(D,home+'train/LDPrune/DROP.df',index=False)


    @staticmethod
    def getCP(x):
        y = pd.DataFrame(x.apply(lambda x: x.split('_')).tolist(), columns='CHROM POS REF ALT'.split())
        y.POS = y.POS.astype(int)
        y.CHROM = y.CHROM.apply(INT)
        return pd.Series(x.values, y.set_index(['CHROM', 'POS']).index).sort_index()

    @staticmethod
    def filter():
        ld = pd.read_csv(home + 'train/LDPrune/DROP.tsv.gz', header=None)[0]
        ld = pd.DataFrame(LDPrune.getCP(ld).rename('ld'))

        all = pd.read_csv(home + 'train/all.bim', header=None, sep='\t')[1]
        all = pd.DataFrame(LDPrune.getCP(all).rename('all'))

        c = GENOME.merge([all, ld], None)

        c[c.ld.isnull()]['all'].reset_index(drop=True).to_csv(home + 'train/LDPrune/KEEP.tsv.gz', index=False,
                                                              compression='gzip')
        # utl.LDPrune().run();utl.LDPrune().filter()

def LDHelper(args):
    i, ids=args
    i=int(i)
    try:
        return pd.Series([LDPrune.runBin(i,ids)],index=[i])
    except:
        print('error in',i)
        return []

def reinstallAILibs():
    execute('/databricks/python/bin/pip install --upgrade git+https://airanmehr:d21eb460800eb76f46c664e7601689d69ead20dd@github.hli.io/airanmehr/AILibs.git', returnDF=False)
    print('AILibs sucessfully was installed. Now detach and reattach your Notebook.')


class PLINKGWAS():

    @staticmethod
    def load(X,CHROMS=range(1,23)+['X'],col='P',GWASID=1,cutoff=None,takelog=True,
             path= UKBB_PATH+'GWAS/train/gwas{}/'):
        """
        :param col: OR or P
        """
        CP = 'CHROM POS'.split()
        path =path.format(GWASID)
        fin=path+'{}.df'.format(X)
        try:
            a=pd.read_pickle()
        except:
            f=lambda x: np.abs(np.log10(x))
            if not takelog: f=lambda x: x
            one=lambda x: pd.read_csv('{}{}/chr{}.PHENO1.glm.logistic'.format(path,X,x),sep='\t').rename(columns={'#CHROM':'CHROM'}).dropna()
            two = lambda x: pd.read_csv('{}{}/chr{}.PHENO1.glm.linear'.format(path, X, x), sep='\t').rename(columns={'#CHROM': 'CHROM'}).dropna()
            try:
                a=pd.concat(map(one,CHROMS))
            except:
                a = pd.concat(map(two, CHROMS))
            a[CP]=a[CP].applymap(INT)
            a=a.set_index(CP)[[col, 'ID']]
            a[col] = a[col].apply(f)
            if cutoff is not None:
                a= a[a[col]>cutoff]
            a.to_pickle(fin)
        return a