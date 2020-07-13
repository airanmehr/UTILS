from UTILS import *
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
            from UTILS.Util import Dmel
            chromLen=Dmel.getChromLen(ver)
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
    def xmap_bed(Interval=None,variants=None,hgFrom=19, hgTo=38,removeXPchromSNPs=True,keepOnlyPos=False,chainPath=home+'storage/Data/Human/CrossMap-0.2.5/chains',xmap='/home/arya/miniconda2/bin/CrossMap.py',verbose=False):
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
        cmd = "{} bed  {} {} {}".format(xmap,chainfile, in_file, out_file)
        # if verbose:
        #     print cmd
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