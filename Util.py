'''
Copyleft Dec 17, 2015 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from __future__ import print_function

import numba
from UTILS import *
from UTILS.BED import *
from UTILS.VCF import *
from UTILS.EE import *
from UTILS.Genome import *


np.set_printoptions(linewidth=140, precision=5, suppress=True)
import os

try:
    import readline

except:
    pass

def googleDocURL(name,url):return '=HYPERLINK("{}","{}")'.format(url,name)


def FoldOn(y,foldOn):
    x = y.copy(True)
    if not (foldOn is None):
        fold = x[foldOn] < 0.5
        x.loc[fold, :] = 1 - x.loc[fold, :]
    return x


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




def burden():
    f= PATH.UKBB+ 'chr21.vcf.gz.ann.gz.LOF.df'
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


def makeCategory(df,field):
    try:
        df[field]=df[field].astype('category')
    except:
        pass
    return df






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
        a=pd.read_csv(PATH.Dmel+'gene_association.fb',sep='\t',comment='!',header=None).set_index(1)[4].rename('GO')
        a.index.name=None
        if relation:
            return a
        else:
            return a.groupby(level=0).apply(lambda x: set(x.tolist()))

    @staticmethod
    def geneCoordinates(assembly=5,allOrganisms=False):
        # ftp://ftp.flybase.net/releases/FB2014_03/precomputed_files/genes/gene_map_table_fb_2014_03.tsv.gz
        # ftp://ftp.flybase.net/releases/FB2016_04/precomputed_files/genes/gene_map_table_fb_2016_04.tsv.gz
        fname=PATH.Dmel+('gene_map_table_fb_2016_04.tsv','gene_map_table_fb_2014_03.tsv')[assembly==5]
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

    loadSNPID = lambda: pd.read_csv(home+'storage/Data/Dmelanogaster/dm5.vcf', sep='\t', usecols=range(5),
                                    header=None, comment='#', names=['CHROM', 'POS', 'ID', 'REF', 'ALT']).set_index(
        ['CHROM', 'POS'])










def loadGenes(Intervals=True):
    a=pd.read_csv(PATH.data+'Human/WNG_1000GP_Phase3/gene_info.csv')[['chrom','pop','gene','POS_hg19']].rename(columns={'chrom':'CHROM','POS_hg19':'POS'})
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
    gap = pd.read_csv(PATH.data + 'Human/gaps/hg{}.gap'.format(assempbly), sep='\t')[
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

def lite1d(a,q=0.9,cutoff=None):
    return a[a>a.quantile(q)]



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
            path=PATH.scan + 'SFS/{}.{}.df'
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
        a = pos(GENOME.merge([HS,HL,SL])).fillna(0)
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
    a=pd.read_pickle(PATH.data + 'Human/scan/WNG.df').set_index('pop').loc[[pop]]
    a.start-=padding;a.start[a.start<0]=0
    a.end += padding
    return a

def mergeResults(path='/home/arya/POP/HAT/CEU+CHB/CEU/',f='chr{}.xpehh.CEU.CHB.gz',out='xpehh.CEU.CHB',CHROMS=range(1,23),outpath=None):
    if outpath==None: outpath=path
    os.system('rm -f ' + path + out)
    for c in CHROMS:os.system('zcat {} >> {}'.format( path + f.format(c),outpath+out))
    os.system('bgzip -c {0} > {1} && tabix -p vcf {1} && rm -f {0}'.format(outpath+out,outpath+out+'.gz'))




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


def loadUKBB_ANN(CHROM=None,path=PATH.UKBB):
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
        """
        conda install -c conda-forge scikit-allel

        """
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
    """
    conda install -c bioconda mygene 
    """
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


def loadPlinkFreq(x,pop,AC=False,ID=False,name=False,level=False,CHROM=False,flat=False,splitID=False,path=PATH.data+'Human/1KG/hg38/bed/pgen/refFromFa/CPRA/AF/'):
    if pop=='UKBB':
        path=PATH.UKBB+'AF/'
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



def plinkGT(CHROM,POS=None,verbose=False,cat=False,fold=False,ids=None):
    if POS is None:
        if isinstance(CHROM,list):
            return pd.concat(map(plinkGT,CHROM),1)
        CHROM,POS=CHROM
    path=PATH.UKBB
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
             path= PATH.UKBB+'GWAS/train/gwas{}/'):
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