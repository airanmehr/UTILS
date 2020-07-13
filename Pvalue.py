from UTILS import *
from UTILS.Learner import getDensity


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
        odds = lambda x: x.iloc[1] / x.iloc[0]
        return odds(pval.crosstab(cc))

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
        return pd.crosstab(cc.iloc[:, 0], cc.iloc[:, 1])

    @staticmethod
    def qval(p, concat=False):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        qvalue = importr("qvalue")
        pvals = ro.FloatVector(p.values)
        rcode = 'qobj <-qvalue(p=%s,  lambda=seq(0.05, 0.95, 1))' % (pvals.r_repr())
        res = ro.r(rcode)
        q = pd.Series(list(ro.r('qobj$qvalue')), index=p.index)
        if concat: q = pd.concat([p, q], 1, keys=['pval', 'qval'])
        return q

    # c=utl.scan.Genome(a,f=lambda x: utl.chi2SampleMeanPval(x,1))
    @staticmethod
    def zscore(x):
        return ((x - x.mean()) / x.std()).astype(float)

    @staticmethod
    def zscoreChr(x):
        return x.groupby(level='CHROM').apply(pval.zscore).astype(float)

    @staticmethod
    def chi2(x, df=1):
        return - np.log10(np.exp(st.chi2.logsf(x, df)))

    @staticmethod
    def chi2SampleMean(x, df, dftot=None):
        if not x.size: return None
        df = df * x.size
        mu = x.mean()
        if dftot is not None: df = dftot
        return -np.log10(np.exp(st.gamma.logsf(mu, df / 2., scale=2. / x.size)))

    @staticmethod
    def zpval(z):
        return -pd.Series(1 - st.norm.cdf(pval.zscore(z).values) + 1e-16, index=z.index).apply(np.log)

    @staticmethod
    def zgenome(x):
        return -pd.Series(1 - st.norm.cdf(pval.zscoreChr(x).values) + 1e-16, index=x.index).apply(np.log)

    # @staticmethod
    # zgenome2tail= lambda x: -pd.Series(1-st.norm.cdf(pval.zscoreChr(x).abs().values)+ 1e-16,index=x.index).apply(np.log)

    @staticmethod
    def z2tail(x):
        return -pd.Series(1 - st.norm.cdf(pval.zscore(x).abs().values) + 1e-16, index=x.index).apply(np.log)

    @staticmethod
    def gammachi2Test(x, df):
        return -st.chi2.logsf(x, df), -st.gamma.logsf(x, df / 2., scale=2.), -st.gamma.logsf(x / df, df / 2.,
                                                                                             scale=2. / df)

    @staticmethod
    def fisher(A):
        import rpy2.robjects as robjects
        if isinstance(A, pd.DataFrame):
            a = A.values
        else:
            a = A
        if a.shape[0] == 2:
            r = 'fisher.test(rbind(c({},{}),c({},{})), alternative="less")$p.value'
            return robjects.r(r.format(a[0, 0], a[0, 1], a[1, 0], a[1, 1]))[0]
        elif a.shape[0] == 3:
            r = 'fisher.test(rbind(c({},{}),c({},{}),c({},{})), alternative="less")$p.value'
            return robjects.r(r.format(a[0, 0], a[0, 1], a[1, 0], a[1, 1], a[2, 0], a[2, 1]))[0]

    @staticmethod
    def chi2ContingencyDF(A):
        a = A.dropna()
        try:
            return pval.chi2Contingency(pval.crosstab(a), True)
        except:
            pass

    @staticmethod
    def chi2ContingencyDFApply(A, ycol):
        cols = A.drop(ycol, 1).columns
        return pd.Series(cols, index=cols).apply(lambda x: pval.chi2ContingencyDF(A[[ycol, x]]))

    @staticmethod
    def chi2Contingency(A, log=False):
        import scipy as sc
        if isinstance(A, pd.DataFrame):
            a = A.values
        else:
            a = A
        p = sc.stats.chi2_contingency(a, correction=False)[1]
        if log: p = np.round(abs(np.log10(p)), 2)
        return p

    @staticmethod
    def empirical(A, Z, positiveStatistic=True):  # Z is null scores
        if positiveStatistic:
            a = A[A > 0].sort_values()
            z = Z[Z > 0].sort_values().values
        else:
            a = A.sort_values()
            z = Z.sort_values().values
        p = np.zeros(a.size)
        j = 0
        N = z.size
        for i in range(a.size):
            while j < N:
                if a.iloc[i] <= z[j]:
                    p[i] = N - j + 1
                    break
                else:
                    j += 1
            if j == N: p[i] = 1
        return -pd.concat([pd.Series(p, index=a.index).sort_index() / (Z.size + 1), A[A == 0] + 1]).sort_index().apply(
            np.log10)

    @staticmethod
    def CMH(x, num_rep=3):
        import rpy2.robjects as robjects
        r = robjects.r
        response_robj = robjects.IntVector(x.reshape(-1))
        dim_robj = robjects.IntVector([2, 2, num_rep])
        response_rar = robjects.r['array'](response_robj, dim=dim_robj)
        testres = r['mantelhaen.test'](response_rar);
        pvalue = testres[2][0];
        return pvalue

    @staticmethod
    def CMHcd(cd, DisCoverage=True, eps=1e-20, negLog10=True, damp=1):
        name = 'CMH ' + '-'.join(cd.columns.get_level_values('GEN').unique().values.astype(str))
        a = cd + damp
        num_rep = cd.shape[1] / (2 * cd.columns.get_level_values('GEN').unique().size)
        if DisCoverage:
            a.loc[:, pd.IndexSlice[:, :, 'D']] = (a.xs('D', level=2, axis=1) - a.xs('C', level=2, axis=1)).values
        a = a.apply(lambda x: pval.CMH(x.values.reshape(num_rep, 2, 2)), axis=1).rename(name) + eps
        if negLog10: a = -a.apply(np.log10)
        return a

    @staticmethod
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

    @staticmethod
    def getPvalFisher(AllGenes, putativeList, myList):
        cont = pd.DataFrame(pval.getContingencyTable(AllGenes=AllGenes, putativeList=putativeList, myList=myList));
        pval = -np.log10(1 - pval.fisher(cont.values))
        return pval, cont


class Enrichment:
    @staticmethod
    def load_GO_fly():
        fin = PATH.data + "GO/GO.fly.df"
        try:
            return pd.read_pickle(fin)
        except:
            go = Enrichment.loadAssociations().dropna().groupby(level=0).apply(
                lambda x: pd.Series(list(x.iloc[0]))).reset_index().drop('level_1', 1)

            go.to_pickle(fin)

    @staticmethod
    def load_go_names():
        from goatools.obo_parser import GODag
        fin = PATH.data + "GO/GO.names.df"
        try:
            raise 0
            return pd.read_pickle(fin)
        except:
            obodag = GODag(PATH.data + "GO/go-basic.obo")
            ret = []
            for k in obodag.keys():
                v = obodag[k]
                ret += [[k, v.name, v.namespace]]
            pd.DataFrame(ret, columns=['go', 'name', 'namespace']).to_pickle(fin)
    @staticmethod
    def GOEA(bg, study, assoc=None, alpha=0.05, propagate=False):
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
        print('bg={} stydy={}'.format(len(bg), len(study)))
        from goatools.go_enrichment import GOEnrichmentStudy
        from goatools.obo_parser import GODag
        if assoc is None:
            assoc = Enrichment.loadAssociations()
        obodag = GODag(PATH.data + "GO/go-basic.obo")
        goea = GOEnrichmentStudy(bg, assoc.to_dict(), obodag, propagate_counts=propagate, alpha=alpha,
                                 methods=['fdr_bh'])
        goea_results_all = goea.run_study(study)
        goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < alpha]
        import tempfile
        # print goea_results_sig
        try:
            with tempfile.NamedTemporaryFile()as f:
                goea.wr_tsv(f.name, goea_results_sig)
                df = pd.read_csv(f.name, sep='\t')
            return df
        except:
            print('No Association found!')

    @staticmethod
    def loadAssociations(species='fly'):
        taxid = {'fly': 7227, 'human': 9606, 'mouse': 10090, 'rat': 10116}
        from goatools.associations import read_ncbi_gene2go
        aa = pd.Series(read_ncbi_gene2go(PATH.data + "GO/gene2go", taxids=[taxid[species]]))
        if species == 'fly':
            bb = pd.read_pickle(PATH.data + 'GO/fly.mygene.df')
            bb.index = map(int, bb.index)
            aa = bb.join(aa.rename('GO')).set_index("FLYBASE")['GO']
        return aa

    @staticmethod
    def getGeneName(geneIDs=None, species='human'):
        try:
            return pd.read_pickle(PATH.data + 'GO/{}.mygene.symbol.df'.format(species))
        except:
            import mygene
            names = mygene.MyGeneInfo().querymany(geneIDs, scopes="entrezgene,flybase", species=species,
                                                  as_dataframe=True, fields='all')
            names.to_pickle(PATH.data + 'GO/{}.mygene.df'.format(species))

            return names

    @staticmethod
    def GOtablPrint(a):
        return a.join(
            a.study_items.apply(lambda x: ', '.join(Enrichment.getGeneName().loc[x.split(', ')].tolist())).rename(
                'genes')).drop(
            ['enrichment', '# GO', 'ratio_in_study', 'p_uncorrected', 'ratio_in_pop', 'study_items'],
            axis=1).sort_values(
            ['NS', 'p_fdr_bh']).set_index('NS').rename(columns={'study_count': 'count'})

    @staticmethod
    def loadGowinda(path='/home/arya/out/real/gowinda/', fname='cands.final.out.snp.tsv'):
        gowinda = pd.read_csv(path + fname, sep='\t', header=None)[[0, 4, 5, 6, 7, 8, 9]]
        gowinda.columns = ['GO ID', '-log($p$-value)', 'Hits', 'Num of Genes', 'Total Genes', 'GO Term', 'Genes']
        return gowinda

    @staticmethod
    def saveGowinda(cands, all, path=PATH.out + 'real/gowinda/', fname='cands.final.txt'):
        cands.sort_index().reset_index().drop_duplicates().dropna().to_csv(path + fname, sep='\t', header=None,
                                                                           index=False)
        all.sort_index().reset_index().drop_duplicates().dropna().to_csv(path + 'allsnps.txt', sep='\t', header=None,
                                                                         index=False)
