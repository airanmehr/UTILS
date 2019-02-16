import numpy as np
import pandas as pd
from UTILS import *
from UTILS.Util import   AX,roundto
from UTILS.Cox import CPH
# def getDensity(X,width='silverman'):# not width can be string {scott or silverman} or positive real
def getDensity(X,width='scott'):# not width can be string {scott or silverman} or positive real
    from scipy.stats import gaussian_kde
    return gaussian_kde(X, bw_method=width)

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