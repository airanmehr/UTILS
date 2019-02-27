'''
Copyleft Oct 10, 2015 Arya Iranmehr, PhD Student, Bafna's Lab, UC San Diego,  Email: airanmehr@gmail.com
'''

from __future__ import division

import numpy as np;
import pandas as pd;
np.set_printoptions(linewidth=140, precision=5, suppress=True)
import subprocess, uuid, os,sys
import pylab as plt
import UTILS.Util as utl

stdout_old=sys.stdout;sys.stdout=open('/dev/null','w');import simuPOP as sim;sys.stdout=stdout_old # to avoid simuPop welcome message!


def sig(x): return 1./(1+np.exp(-x));
def logit(p): return (np.inf if p==1  else  np.log(p/(1.-p)))




a='';


def fff(msg):
    global a
    a += msg


class MSMS:
    @staticmethod
    def Simulate(n=200, mu=2*1e-9, L=50000, Ne=1e6,r=1e-9,verbose=False,seed=None,intPos=False):
        L=int(L)
        a= MSMS.Song(F=n, mu=mu, L=L, Ne=Ne, r=r,verbose=verbose,seed=seed)
        c=pd.Series(a.columns)
        if c.round().value_counts().max()==1:
            a.columns=c.round().astype(int)
        elif c.astype(int).value_counts().max()==1:
            a.columns = c.astype(int)
        if intPos:
            a.columns=map(int,np.sort(np.random.choice(L, a.shape[1], replace=False)))
        return a

    @staticmethod
    def Song(F=200, mu=2*1e-9, L=50000, Ne=1e6,r=4e-9, uid=None, theta=None, msmsFile=None, dir=None,verbose=False,seed=None):
        """
        Everything is exactly the sam
        """
#         print 'mu: {} r:{} NE:{} ,theta={} '.format(mu,r,Ne,4*Ne*mu*L), theta
        if msmsFile is not None:
            pop=MSMS.load(filename=msmsFile)[0]
        else:
            if theta:
                pop=MSMS.MSMS(n=F, numReps=1, theta=theta, rho=2*Ne*(L-1)*r, L=L, Ne=Ne, uid=uid, dir=dir,verbose=verbose,seed=seed)[0]
            else:
                pop=MSMS.MSMS(n=F, numReps=1, theta=2*Ne*mu*L, rho=2*Ne*(L-1)*r, L=L, Ne=Ne, uid=uid, dir=dir,verbose=verbose,seed=seed)[0]
        pop.r=r
        pop.Ne=Ne
        pop.L=L
        return pop
    
    @staticmethod
    def MSMS(n, numReps, theta, rho, L, Ne=None,uid=None,oneMutationEvery=None, dir=dir,verbose=False,seed=None):
        """
        Returns a list of dataframe for each replicate
        """
        if dir is None:
            dir= utl.PATH.simout;dir+= 'msms/';
        os.system('mkdir -p ' +dir)
        if oneMutationEvery is not None:
            nSS=L/oneMutationEvery
            theta=nSS/sum(1./np.arange(1,n))
        if uid is None:
            uid=str(uuid.uuid4())
        unique_filename = dir+uid+'.msms'
        if seed is None:
            seed=''
        else:
            seed=' -seed {} '.format(seed)
        cmd="java -jar -Xmx2g ~/bin/msms/lib/msms.jar -ms {} {} -t {:.0f} -r {:.0f} {:.0f} -oFP 0.000000000000E00 {} > {}".format(n, numReps, theta, rho, L, seed,unique_filename)
        if verbose:
            print cmd
        subprocess.call(cmd,shell=True)
        return MSMS.load(unique_filename)

    @staticmethod
    def getSeed(filename):
        file=open(filename);cmd=np.array(file.readline().strip().split(' '));seed=file.readline().strip()
        return seed

    @staticmethod
    def load(filename):
        n, R, L, posUnderSelection = MSMS.getParams(open(filename).readline())
        lines=np.array(map(str.strip,open(filename).readlines()) )
        posIdx= np.where(map(lambda x: x[:len('positions:')]=='positions:',lines))[0]
        try:
            theta = lines[np.where(map(lambda x: 'ThetaW Estimate Summaray:' in x, lines))[0][0]].split(':')[1].strip()
        except:
            theta = None
        POS=[map(lambda x: (float(x)*L), lines[ii].split()[1:]) for ii in posIdx]
        dfs=[pd.DataFrame(map(list ,lines[i +1 +range(n)]),columns=pos ) for i,pos in zip(posIdx,POS)]
        for df in dfs:
            df[df!='0']=1
            df[df=='0']=0
            df.L = L
            if posUnderSelection is not None:
                df.posUnderSelection = posUnderSelection * L
            if theta is not None:
                df.stat = pd.Series(theta.split(), index=['W', 'Pi', 'D']).astype(float)
        return  dfs

    @staticmethod
    def getParams(line):
        """
        Args:
            params: takes the first line of msmsm file

        Returns:
            n,R,L: number of individuals in the sample, the number of the replicates, genome length

        """
        params=np.array(line.strip().split(' '))
        offset=np.where(map(lambda x: 'ms'in x,  params))[0][0]
        if params[offset+1] == '-N':
            i=3
        else:
            i=1
        posUnderSelection = None
        if '-Sp' in params:  posUnderSelection = float(params[np.where(params == '-Sp')[0][0] + 1])
        return int(params[offset + i]), int(params[offset + i + 1]), int(
                params[np.where(params == '-r')[0][0] + 2]), posUnderSelection

    @staticmethod
    def fixDuplicatePositions(pos,L):
        pos=pd.Series(range(len(pos)),index=pos)
        posHits=pos.index.value_counts()
        invalidPOS=posHits[posHits>1]
        if not invalidPOS.shape[0]:
            return pos.index.values
        for invalidPos in invalidPOS.index:
            mini=pos.loc[invalidPos].min()
            maxi=pos.loc[invalidPos].max()
            lowerBound=pos[pos==mini-1].index.max()
            upperBound=pos[pos==maxi+1].index.min(); 
            if maxi==pos.shape[0]-1: upperBound=L
            if mini==0: lowerBound=0
            validRange=np.arange((upperBound-lowerBound)/2) # only second and third quartiles,
            offset=validRange+validRange.shape[0]/2 # first qunatulw
            newPos=pos.index.values;
            newPos[mini:maxi+1]=np.sort(np.random.choice(offset,pos.loc[invalidPos].shape[0],replace=False))+lowerBound
            pos.index=newPos
        assert pos.index.value_counts().max()==1
        return pos.index.values

    @staticmethod
    def Selection(msms, Ne, n, numReplicates, theta, rho, window_size, s, origin_count, posUnderSelection, gens, path):
        seed = ''
        for ii, gen in enumerate(gens):
            fname = path + '{}.msms'.format(int(gen))
            if (not ii) and s != 0:
                # while (nu0 < 0.95) or (nu0 > 0.99):
                cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SI {} 1 {} -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SForceKeep -SFC   -oTW  >{}".format(
                        msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, gen / (4. * Ne),
                                                                             origin_count / Ne,
                        posUnderSelection, ('-seed {}'.format(seed), '')[seed is ''], fname)
                os.system(cmd)
            else:
                cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SI {} 1 {} -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SFC -SForceKeep  -oTW  >{}".format(
                        msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, gen / (4. * Ne),
                                                                             origin_count / Ne,
                        posUnderSelection, ('-seed {}'.format(seed), '')[seed is ''], fname)
                os.system(cmd)
            if not ii: seed = MSMS.getSeed(fname)

    @staticmethod
    def SelectionFinale(msms, Ne, n, numReplicates, theta, rho, window_size, s, origin_count, posUnderSelection, gens,
                            path):
        seed = ''
        nu0 = 0
        for ii, gen in enumerate(gens):
            fname = path + '{}.msms'.format(int(gen))
            if (not ii) and s != 0:
                while (nu0 < 0.9):
                    cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SI {} 1 {} -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SForceKeep -SFC   -oTW  >{}".format(
                            msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, gen / (4. * Ne),
                                                                                 origin_count / Ne,
                            posUnderSelection, ('-seed {}'.format(seed), '')[seed is ''], fname)
                    os.system(cmd)

                    nu0 = MSMS.load(fname)[0].mean(0).loc[25000]
            else:
                cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SI {} 1 {} -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SFC -SForceKeep  -oTW  >{}".format(
                        msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, gen / (4. * Ne),
                                                                             origin_count / Ne,
                        posUnderSelection, ('-seed {}'.format(seed), '')[seed is ''], fname)
                os.system(cmd)
            if not ii: seed = MSMS.getSeed(fname)

    @staticmethod
    def SelectionNu(msms, Ne, n, numReplicates, theta, rho, window_size, s, posUnderSelection, nu, path=None):
        seed = ''
        if path is None: path = '~/tmp.msms'
        fname = path + '{}.msms'.format(nu)
        cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SF 0 {}  -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SFC   -oTW  >{}".format(
                msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, nu, posUnderSelection,
                ('-seed {}'.format(seed), '')[seed is ''], fname)
        print cmd
        os.system(cmd)
        return MSMS.load(fname)

    @staticmethod
    def SelectionNuForward(msms, Ne, n, numReplicates, theta, rho, window_size, s, origin_count, posUnderSelection,
                               gens, path):
        nu0 = 0
        for ii, gen in enumerate(gens):
            fname = path + '{}.msms'.format(gen)
            if (not ii) and s != 0:
                while (nu0 < 0.95) or (nu0 > 0.99):
                    cmd = "{} -N {} -ms {} {} -t {} -r {} {:.0f} -SAA {} -SaA {} -SI {} 1 {} -Sp {} -oOC -Smark -oFP 0.000000000000E00 {}  -SFC   -oTW  >{}".format(
                            msms, Ne, n, numReplicates, theta, rho, window_size, 2 * Ne * s, Ne * s, gen / (4. * Ne),
                                                                                 origin_count / Ne,
                            posUnderSelection, ('-seed {}'.format(seed), '')[seed is ''], fname)
                    os.system(cmd)
                    nu0 = MSMS.load(fname)[0].mean(0).loc[25000]
            print nu0, gen, cmd
            if not ii: seed = MSMS.getSeed(fname)


class Simulation:
    @staticmethod
    def setSeed(seed):
        if seed is None: return
        sim.setRNG('rand', seed + 1);
        np.random.seed(seed)



    @staticmethod
    def load(ExperimentName, s=0.1, L=50000, experimentID=0, nu0=0.005, isFolded=False, All=False, startGeneration=0,
             maxGeneration=50, numReplicates=3, numSamples=5, step=10, replicates=None, coverage=np.inf):
        path='{}{}/simpop/'.format(utl.PATH.simout, ExperimentName) + Simulation.getSimulationName(s=s, L=L, experimentID=experimentID, initialCarrierFreq=nu0, isFolded=isFolded) + '.pkl'
        sim= pd.read_pickle(path)
        sim.savedPath=path
        if replicates is not None:          sim.setReplicates(sorted(replicates))
        elif numReplicates is not None:     sim.setReplicates(range(numReplicates))

        if coverage != np.inf:
            sim.Xi = sim.X
            sim.X = sim.C.loc[coverage] / sim.D.loc[coverage].astype(float)
            sim.X = np.array(map(lambda x: utl.roundto(x, 5), sim.X.reshape(-1) * 1e4)).reshape(sim.X.shape) / 1e4
            sim.CD=sim.getCD(coverage)
            sim.CD.columns.names=['REP','GEN','READ']

        if not All: sim.setSamplingTimes(maxGeneration=min(maxGeneration,sim.getGenerationTimes()[-1]),numSamples=numSamples,step=step,startGeneration=startGeneration)
        return sim

    @staticmethod
    def getSimulationName(s,L,experimentID,initialCarrierFreq,isFolded,msms=False):
        if msms:
            return 'L{:.0f}K.{:04.0f}'.format(L/1000,experimentID)
        if s:
            return 'Nu{:E}.s{:E}.L{:.0f}K.{:04.0f}{}'.format(np.round(float(initialCarrierFreq), 3), s, L / 1000,
                                                             experimentID, ('', '.Folded')[isFolded])
        else:
            return 'Nu{:E}.s{:E}.L{:.0f}K.{:04.0f}{}'.format(0, s * 100, L / 1000, experimentID,
                                                             ('', '.Folded')[isFolded])

    def setReplicates(self,replicates):
        self.numReplicates=len(replicates)
        self.X=self.X[:,:,replicates]
        self.C = self.C.apply(lambda x: x[:, :, replicates])
        self.D = self.D.apply(lambda x: x[:, :, replicates])

    def __init__(self, outpath=utl.PATH.simout, N=1000, generationStep=10, maxGeneration=None,
                 s=0.05, r=4e-9, Ne=1e6, mu=2e-9, F=200, h=0.5, L=50000, startGeneration=0, numReplicates=3, H0=None,
                 foldInitialAFs=False, save=True, foutName=None,
                 doForwardSimulationNow=True, experimentID=-1,
                 msmsFile=None,initialCarrierFreq=0, ExperimentName=None, simulateNeutrallyFor=0,
                 initialNeutralGenerations=0, ignoreInitialNeutralGenerations=True,
                 makeSureSelectedSiteDontGetLost=True, onlyKeep=None, verbose=0, sampingTimes=None, minIncrease=0,
                 model=None,initDiploidPop=None,posUnderSelection=-1,haplotypes=False,seed=None,recombinator=None
                 ):
        """
        A General Simulation Class; with params
        H0: Dataframe F x m for F individuals and m segregation sites ;  Initial Haplotypes; dataframe with columns as positions 
        """
        self.recombinator=recombinator
        if seed is not None:
            Simulation.setSeed(seed)
        self.s = s;
        self.r = r;
        self.Ne = Ne;
        self.mu = mu;
        self.F = F;
        self.h = h;
        self.L = int(L);
        self.startGeneration = startGeneration;
        self.numReplicates = numReplicates;
        self.posUnderSelection = -1
        self.initDiploidPop = initDiploidPop
        self.initialCarrierFreq= initialCarrierFreq if initialCarrierFreq else 1./self.F
        if foutName is not None:
            self.uid=foutName
            self.uidMSMS=None
        elif experimentID>=0:
            self.uid=Simulation.getSimulationName(self.s, self.L, self.experimentID, initialCarrierFreq=self.initialCarrierFreq, isFolded=self.foldInitialAFs)
            self.uidMSMS=Simulation.getSimulationName(self.s, self.L, self.experimentID, initialCarrierFreq=self.initialCarrierFreq, isFolded=self.foldInitialAFs,msms=True)
        else:
            self.uid=str(uuid.uuid4())
            self.uidMSMS=self.uid


        if H0 is None:
            self.simulateH0()
            H0=self.H0
        else:
            self.setH0(H0);

        if posUnderSelection >= 0:
            if self.positions is None:
                self.positions=map(int, self.initDiploidPop.lociPos())
            self.set_posUnderSelection(posUnderSelection)
        assert ExperimentName != None
        self.save=save
        self.model=model
        self.minIncrease = minIncrease
        self.samplingTimes=sampingTimes
        self.initialNeutralGenerations=initialNeutralGenerations
        self.onlyKeep=onlyKeep
        self.makeSureSelectedSiteDontGetLost=makeSureSelectedSiteDontGetLost
        self.ignoreInitialNeutralGenerations=ignoreInitialNeutralGenerations
        self.msmsFile=msmsFile;self.outpath=outpath; self.outpath=outpath ; self.N=N; self.generationStep=generationStep; self.maxGeneration= maxGeneration;
        self.foldInitialAFs=foldInitialAFs;self.doForwardSimulationNow=doForwardSimulationNow;self.experimentID=experimentID
        self.simulateNeutrallyFor=simulateNeutrallyFor

        self.setH0(H0);
        if not os.path.exists(self.outpath) : os.makedirs(self.outpath)
        self.outpath+=ExperimentName
        if not os.path.exists(self.outpath) : os.makedirs(self.outpath)
        self.outpathmsms=self.outpath+'/msms/';self.outpath+='/simpop/'
        if not os.path.exists(self.outpath) : os.makedirs(self.outpath)
        if not os.path.exists(self.outpathmsms) : os.makedirs(self.outpathmsms)
        if self.maxGeneration is None: self.maxGeneration=Simulation.getFixationTime(self.s, Ne=self.F, roundto10=True)
        self.theta=2*self.Ne*self.mu*self.L
        self.pops=[]

        if self.model is None:
            import simuPOP.demography as dmg
            self.model=dmg.LinearGrowthModel(T=self.maxGeneration, N0=self.N, NT=self.N)

        if self.doForwardSimulationNow:
            self.forwardSimulation()

    @staticmethod
    def simulateSingleLoci(nu0=0.005, T=100, s=0.1, N=1000,verbose=True,h=0.5,seed=None):
        if verbose:
            print '.',
        step = 1
        Simulation.setSeed(seed)
        pop = sim.Population(size=N, ploidy=2, loci=[1],infoFields=['fitness']);sim.initGenotype(pop, prop=[1-nu0,nu0]);simulator = sim.Simulator(pop.clone(), rep=1);
        # sim.stat(pop, alleleFreq=[0]);        print pop.dvars().alleleFreq[0][1]
        global a;a = "0;;{}\n".format(nu0)
        simulator.evolve(initOps=[sim.InitSex()],
                         preOps=sim.MapSelector(loci=0, fitness={(0, 0): 1, (0, 1): 1 + s *h, (1, 1): 1 + s}),
                         matingScheme=sim.RandomMating(), postOps=[sim.Stat(alleleFreq=[0], step=step),
                                                                   sim.PyEval("'%d;;' % (gen+1)", reps=0, step=step,
                                                                              output=fff), sim.PyEval(
                    r"'{}\n'.format(map(lambda x: round(x[1],5),alleleFreq.values())[0])", step=step, output=fff)],
                         gen=T)
        return pd.DataFrame(zip(*map(lambda x: x.split(';;'), a.strip().split('\n')))).T.set_index(0)[1].astype(float)



    def createInitialDiploidPopulation(self):
        """
        initHaps : np 2D array which m x nSS where m i number of individual haps and nSS is number of SS
        return a homozygote diploid population which every haplotype is copied n times
        """
        if self.initDiploidPop is not None: return self.initDiploidPop
        assert int(2*self.N/self.F)==2*self.N/float(self.F)  # N should be  a multiplier of F
        nSS=self.H0.shape[1];n=int(self.N/self.F)
        try:
            pop = sim.Population(size=self.N, ploidy=2, loci=nSS,lociPos=list(self.positions), infoFields='fitness')
        except:
            import traceback
            print(traceback.format_exc())
            print list(self.positions), nSS,n,self.H0.shape[0]
            exit()
        assert (self.N % self.H0.shape[0]) ==0
        H= [[list(h.values),list(h.values)] for _ in range(n) for _,h in self.H0.iterrows()]
        for (i,h) in zip(pop.individuals(),H): # for each indv assing first and second chromosome
            i.setGenotype(h[0],0 );i.setGenotype(h[1],1 ) #homozygote population of diploid
        # sim.stat(pop, alleleFreq=range(nSS));print np.array([pop.dvars().alleleFreq[x][1] for x in range(nSS)])
        return pop

    @staticmethod
    def getGT(pop, i=None, pos=None):
        if i == None and pos == None:
            df = pd.concat([pd.DataFrame([list(i.genotype(0)) for i in pop.individuals()]),
                            pd.DataFrame([list(i.genotype(1)) for i in pop.individuals()])],
                           keys=[0, 1]).sort_index().reorder_levels([1, 0]).sort_index()
            df.columns = map(int, pop.lociPos())
            return df
        i = np.where(np.array(pop.lociPos()).astype(int) == pos)[0][0]
        a, b = [], []
        for ind in pop.individuals():
            a += [ind.genotype(0)[i]]
            b += [ind.genotype(1)[i]]
        return pd.concat([pd.Series(a), pd.Series(b)], keys=[0, 1]).reorder_levels([1, 0]).sort_index()
    @staticmethod
    def createDiploidPopulationFromDataFrame(df):
        """
        initHaps : np 2D array which m x nSS where m i number of individual haps and nSS is number of SS
        return a homozygote diploid population which every haplotype is copied n times
        """
        pop = sim.Population(size=df.shape[0]/2, ploidy=2, loci=df.shape[1], lociPos=list(df.columns), infoFields='fitness')
        for j,i in enumerate(pop.individuals()): # for each indv assing first and second chromosome
            i.setGenotype(df.loc[j].loc[0].tolist(),0 );i.setGenotype(df.loc[j].loc[1].tolist(),1 )
        return pop

    @staticmethod
    def _simualtePop(pop, s=0, h=0.5, r=2e-8, siteUnderSelection=0,gen=1,recombinator=None,seed=None):
        "Gets population and returns population"
        Simulation.setSeed(seed)
        simulator = sim.Simulator(pop.clone(), rep=1)
        if recombinator is None:recombinator=sim.Recombinator(intensity=r)
        simulator.evolve(
            initOps=[sim.InitSex()],
            preOps=sim.MapSelector(loci=siteUnderSelection, fitness={(0, 0): 1, (0, 1): 1 + s * h, (1, 1): 1 + s}),
            matingScheme=sim.RandomMating(ops=recombinator),
            gen=gen)
        return simulator.population(0).clone()

    @staticmethod
    def _simualte(pop,s,h,r,siteUnderSelection,positions,startGeneration,generationStep,maxGeneration,model=None,makeSureSelectedSiteDontGetLost=True):
        "Gets population and returns Dataframe, Static method"
        N = int(pop.popSize())
        if model is None:
            import simuPOP.demography as dmg
            model = dmg.LinearGrowthModel(T=maxGeneration, N0=N, NT=N)
        simulator = sim.Simulator(pop.clone(), rep=1)
        global a;a = ""
        pops=[]
        step=1# this is slow but safe, dont change it
        simulator.evolve(
            initOps=[sim.InitSex()],
            preOps=sim.MapSelector(loci=siteUnderSelection, fitness={(0, 0): 1, (0, 1): 1 + s * h, (1, 1): 1 + s}),
            matingScheme=sim.RandomMating(ops=sim.Recombinator(intensity=r),subPopSize=model),
            postOps=[sim.Stat(alleleFreq=range(int(pop.numLoci()[0])), step=step), sim.PyEval("'Gen %4d;;' % (gen+1)", reps=0,step= step, output=fff), sim.PyEval(r"'{},'.format(map(lambda x: round(x[1],5),alleleFreq.values()))", step=step, output=fff),sim.PyOutput('\n', reps=-1, step=step, output=fff)],
            gen = maxGeneration)
        # idx=np.arange(self.generationStep-1,self.maxGeneration,self.generationStep)+self.initialNeutralGenerations
        print a
        _,data=zip(*map(lambda x: x.split(';;'),a.strip().split('\n')))
        data=np.array(map(eval,data))[:,0,:]
        print data
        # if data[-1, self.siteUnderSelection] >= self.initialCarrierFreq + self.minIncrease or self.s == 0 or not self.makeSureSelectedSiteDontGetLost:
        if data[-1, siteUnderSelection] or s == 0 or not makeSureSelectedSiteDontGetLost:
            try:
                pops+=[simulator.extract(0) ]
            except:
                print 'Error'
            return data[int(startGeneration/generationStep):,:]
        else:
            return Simulation._simualte()


    def simualte(self):
        "Gets population and returns Dataframe, Class method"
        import simuPOP.demography as dmg
        # model=dmg.ExponentialGrowthModel(T=50, N0=1000, NT=200)
        simulator = sim.Simulator(self.initDiploidPop.clone(), rep=1)
        # sim.dump(self.initDiploidPop)
        global a;a = ""
        if self.recombinator is None:
            self.recombinator=sim.Recombinator(intensity=self.r)
        step=1# this is slow but safe, dont change it
        simulator.evolve(
            initOps=[sim.InitSex()],
            preOps=sim.MapSelector(loci=self.siteUnderSelection, fitness={(0,0):1, (0,1):1+self.s*self.h, (1,1):1+self.s}),
            matingScheme=sim.RandomMating(ops=self.recombinator,subPopSize=self.model),

            postOps=[sim.Stat(alleleFreq=range(len(self.positions)), step=step),
                     sim.PyEval("'Gen %4d;;' % (gen+1)", reps=0,step= step, output=fff), sim.PyEval(r"'{},'.format(map(lambda x: round(x[1],5),alleleFreq.values()))", step=step, output=fff),sim.PyOutput('\n', reps=-1, step=step, output=fff)],
            gen = self.maxGeneration)
        # idx=np.arange(self.generationStep-1,self.maxGeneration,self.generationStep)+self.initialNeutralGenerations
        _,data=zip(*map(lambda x: x.split(';;'),a.strip().split('\n')))
        data=np.array(map(eval,data))[:,0,:]
        # if data[-1, self.siteUnderSelection] >= self.initialCarrierFreq + self.minIncrease or self.s == 0 or not self.makeSureSelectedSiteDontGetLost:
        if data[-1, self.siteUnderSelection] or self.s == 0 or not self.makeSureSelectedSiteDontGetLost:
            try:
                self.pops+=[simulator.extract(0) ]
            except:
                print 'Error'
            return data[int(self.startGeneration/self.generationStep):,:]
        else:
            # print pd.Series(data[:, self.siteUnderSelection])
            return self.simualte()
    
    def simulateH0(self):
        self.H0=MSMS.Song(F=self.F, L=self.L, Ne=self.Ne, r=self.r, mu=self.mu,uid=self.uidMSMS)
        
        
    def set_siteUnderSelection(self,x):
        self.siteUnderSelection=x
        self.posUnderSelection=self.positions[self.siteUnderSelection]
    
    def set_posUnderSelection(self,x):
        self.posUnderSelection=x
        self.siteUnderSelection=np.where(self.positions==self.posUnderSelection)[0][0]
    
    def setH0(self,H0):
        self.H0=H0
        self.positions=self.H0.columns.values
        self.F=self.H0.shape[0]

    def set_BeneficialLoci(self,selectionOnRandomSite=False,siteUnderSelection=None,posUnderSelection =None):
        if selectionOnRandomSite:
            self.set_siteUnderSelection(np.random.randint(0,self.H0.shape[1]))
        elif siteUnderSelection is not None:
            self.set_siteUnderSelection(siteUnderSelection)
        elif posUnderSelection is not None:
            self.set_siteUnderSelection(posUnderSelection)
        else:
            if not self.s:
                self.set_siteUnderSelection(self.X0.argmax())
            else:
                sites=np.sort(np.where(self.X0== self.initialCarrierFreq)[0]);
                if not len(sites):
                    sites=np.sort(np.where(( self.X0 <= self.initialCarrierFreq +0.025) & ( self.X0 >= self.initialCarrierFreq -0.025) ) [0]);
                    if not len(sites):
                        print 'Try again. No site at freq ',self.initialCarrierFreq, self.uid; return
                self.set_siteUnderSelection(sites[np.random.randint(0,len(sites))])

    def createInitHaps(self):
        assignPositions=True
        if self.H0 is None:
            H0 = MSMS.Song(F=self.F, L=self.L, Ne=self.Ne, r=self.r, mu=self.mu, uid=self.uidMSMS,
                           msmsFile=self.msmsFile, dir=self.outpathmsms)
        else:
            H0 = self.H0
            assignPositions=False
        if self.foldInitialAFs:
            idx = H0.mean(0) > 0.5
            H0.iloc[:, idx.values] = 1 - H0.iloc[:, idx.values]
        self.setH0(H0)
        if assignPositions:
            self.positions_msms = self.H0.columns.values.copy(True)
            self.positions = sorted(np.random.choice(self.L, self.H0.shape[1], replace=False))
        self.H0 = pd.DataFrame(self.H0.values, columns=self.positions)
        self.X0 = self.H0.mean(0).values

    def forwardSimulation(self):
        """
        returns np 3D array T x nSS x R which T=|{t_1,t_2,..}| (nnumber of times), nSS is number of SS , and R is the number of replicates  
        """
        import numpy as np
        # df = pd.DataFrame([list(i.genotype(j)) for j in range(2) for i in self.initDiploidPop.individuals()])
        if self.posUnderSelection<0 and self.initDiploidPop is None:
            self.createInitHaps()
            self.set_BeneficialLoci()
            self.initDiploidPop=self.createInitialDiploidPopulation()
        elif self.initDiploidPop is None:
            self.createInitHaps()
            self.initDiploidPop = self.createInitialDiploidPopulation()
            # self.X0=self.H0.mean().values
        else:
            self.X0=Simulation.getGT(self.initDiploidPop).mean().values

        # df = pd.DataFrame([list(i.genotype(j)) for j in range(2) for i in self.initDiploidPop.individuals()])
        # print pd.concat([df.mean(),self.H0.mean().reset_index(drop=True)],1)
        self.X=np.array([self.simualte() for _ in range(self.numReplicates)]).swapaxes(0, 2).swapaxes(0, 1)
        self.X=np.append(np.tile(self.X0[:,None],(1,self.X.shape[2]))[None,:,:],self.X,axis=0)

        self.sampleDepths()
        if self.save:
            pd.to_pickle(self,self.outpath+self.uid+'.pkl')
        # self.createDF()


     
    def getGenerationTimes(self,step=None,includeZeroGeneration=True):
        if step is None: step=self.generationStep
        times= np.arange(0,self.maxGeneration-self.startGeneration+1,step)
        if includeZeroGeneration:
            return times
        else:
            return times[1:]
    
    def getTrueGenerationTimes(self,step=None,includeZeroGeneration=True):
        if step is None: step=self.generationStep
        times= np.arange(self.startGeneration,self.maxGeneration+1,step)
        if includeZeroGeneration:
            return times
        else:
            return times[1:]

    
    @staticmethod
    def getFixationTime(s,Ne=200,roundto10=True):
        if s==0: s=0.01
        t=-4*int(logit(1./Ne)/s)
        if roundto10:
            return (t//10 +1)*10
        else:
            return t
         
    @staticmethod
    def sampleInitSamplingTime(s,Ne=200,phase=0,samplingWindow=50,startOfEpoch=False):
        fix=Simulation.getFixationTime(s, Ne=Ne)
        if phase==0:    lower,upper=(0, fix-samplingWindow)
        if phase==1:    lower,upper=(0, fix/3-samplingWindow)
        if phase==2:    lower,upper=(fix/3, 2*fix/3-samplingWindow)
        if phase==3:    lower,upper=(2*fix/3, fix-samplingWindow)
        if startOfEpoch:
            rnd=lower
        else:
            rnd=np.random.randint(lower,max(lower,upper)+1)
        return int(rnd)//10 *10
    @staticmethod
    def sampleStartTimesforAlls(samplingWindow=50):
        S=[0.1, 0.05, 0.02, 0.01,0]
        for phase in [1,2,3]:
            pd.DataFrame([[Simulation.sampleInitSamplingTime(s, phase=phase, samplingWindow=samplingWindow, startOfEpoch=True) for _ in range(100)] for s in S], index=S).T.to_pickle('/home/arya/out/startSamplingTimes.phase{}.sampleWin{}.pkl'.format(phase, samplingWindow))
        
    def setSamplingTimes(self,maxGeneration=None,numSamples=5,step=None,startGeneration=None):
            GT=pd.Series(range(len(self.getTrueGenerationTimes(includeZeroGeneration=True))),index=self.getTrueGenerationTimes(includeZeroGeneration=True))
            if startGeneration is not None:   self.startGeneration=startGeneration
            if maxGeneration is not None:   self.maxGeneration = maxGeneration
            if step is not None:self.generationStep=step
            else:   self.generationStep=(self.maxGeneration-self.startGeneration)/numSamples
            i = GT.loc[self.getTrueGenerationTimes(includeZeroGeneration=True)[:self.X.shape[0]]].values
            self.X = self.X[i, :, :]
            self.C = self.C.apply(lambda x: x[i, :, :])
            self.D = self.D.apply(lambda x: x[i, :, :])
            self.X0=self.X[0,:,0]

    @staticmethod
    def getSamplingTimeBasedOnFreq(sim,phase,samplingWin=50):
        carrier_freq=[0.1,0.5,0.9][phase-1]
        a= np.where(sim.X[:,sim.siteUnderSelection,:].mean(1)>carrier_freq)[0]
        ft=sim.getTrueGenerationTimes().max()
        if len(a):
            t= sim.getTrueGenerationTimes()[np.where(sim.X[:,sim.siteUnderSelection,:].mean(1)>carrier_freq)[0].min()]
        else:
            t=sim.getTrueGenerationTimes().max()
        return min(t,ft-samplingWin)

    @staticmethod
    def Load(s=0.1, experimentID=0, nu0=0.005, numReplicates=3, step=10, ModelName='TimeSeries', samplingWindow=50,
             L=50000, depthRate=30):
        if not s: nu0=0.005
        sim = Simulation.load(s=s, experimentID=experimentID % 100, nu0=nu0, numReplicates=numReplicates, step=step,
                              ExperimentName=ModelName, All=True, L=L, replicates=range(numReplicates),
                              coverage=depthRate)
        sim.experimentID=experimentID
        startGen=0
        sim.setSamplingTimes(maxGeneration=min(startGen+samplingWindow,sim.getTrueGenerationTimes()[-1]),step=step,startGeneration=startGen)
        sim.createDF()
        return sim

    def getHardSweepMutations(self):
        MAF=1./self.H0.shape[0]
        dups=self.H0[self.H0.duplicated()]
        x0=pd.Series(self.X0, index=self.positions)
        hard=[]
        for _,dup in dups.iterrows():
            numDup=self.H0.apply(lambda x:(x==dup).all(),axis=1).sum()
            hard=np.append(hard, (dup*x0==numDup*MAF).replace({False:None}).dropna().index.values)
        hard=np.sort(np.append(hard,(x0==MAF).replace({False:None}).dropna().index.values).astype(int))
        return hard

    @property
    def df(self):
        reps=range(self.numReplicates)
        self.df=pd.concat([pd.DataFrame(self.X[:,:,r],columns=self.positions,index=pd.MultiIndex.from_product([[r],range(self.X.shape[0])],names=['REP','TIME'])).T for r in reps],axis=1)
        if self.numReplicates==1:
            self.df=self.df[0]
        return self.df
    def computeCDi(self, EE, depthRate):
        E = EE.loc[depthRate]
        index = pd.Series(range(E.shape[0]), E.index)
        C = pd.concat([pd.DataFrame(self.C.loc[depthRate][:, :, r], columns=self.H0.columns,
                                    index=pd.MultiIndex.from_product([[r], self.getTrueGenerationTimes()],
                                                                     names=['REP', 'GEN'])).T for r in
                       range(self.numReplicates)], axis=1)
        D = pd.concat([pd.DataFrame(self.D.loc[depthRate][:, :, r], columns=self.H0.columns,
                                    index=pd.MultiIndex.from_product([[r], self.getTrueGenerationTimes()],
                                                                     names=['REP', 'GEN'])).T for r in
                       range(self.numReplicates)], axis=1)
        self.cd = pd.concat([pd.Series(zip(C[i], D[i])) for i in C.columns], axis=1)
        self.cd.columns = C.columns;
        self.cd.index = C.index
        self.cdi = self.cd.applymap(lambda x: index.loc[x])

    def sampleDepths(self,depths = [30, 100, 300]):
        self.D = pd.Series(None, index=depths)
        self.C = pd.Series(None, index=depths)
        for depthRate in depths:
            self.D.loc[depthRate] = np.random.poisson(depthRate,
                                                      self.X.shape[0] * self.X.shape[1] * self.X.shape[2]).reshape(
                self.X.shape).astype(object)
            self.C.loc[depthRate] = np.array([np.random.binomial(d, x) for x, d in
                                              zip(self.X.reshape(-1), self.D.loc[depthRate].reshape(-1))]).reshape(
                    self.X.shape).astype(object)

    @staticmethod
    def sampleDepthX(X,cov):
        D= np.random.poisson(cov,X.size)
        C= np.array([np.random.binomial(d, x) for x, d in zip(X, D)])
        return C,D
    @staticmethod
    def sampleDepthXSeries(X,cov):
        C,D=Simulation.sampleDepthX(X.values,cov)
        a=pd.DataFrame([C,D],columns=X.index,index=['C','D']).T
        return a

    @staticmethod
    def computeCDdf(a, E):
        index = pd.Series(range(E.shape[0]), E.index)
        def f(x):
            try:
                return index.loc[x]
            except:
                return -1
        z=a.groupby(level=[0,1],axis=1).apply(lambda x: x.apply(lambda y:(y.iloc[0],y.iloc[1]),1)).applymap(f)
        return z[(z<0).sum(1)==0]
    def getCD(self,coverage):
        T=self.getTrueGenerationTimes()
        Ti=T
        if T[-1]!=self.C[coverage].shape[0]-1: Ti=range(self.C[coverage].shape[0])
        C=pd.concat([pd.DataFrame(self.C[coverage][Ti,:,i],columns=self.positions,index=T).T for i in range(self.numReplicates)],1,keys=range(self.C[coverage].shape[2]))
        D=pd.concat([pd.DataFrame(self.D[coverage][Ti,:,i],columns=self.positions,index=T).T for i in range(self.numReplicates)],1,keys=range(self.C[coverage].shape[2]))
        CD=pd.concat([C,D],1,keys=['C','D']).reorder_levels([1,2,0],1).sort_index(1)
        CD.columns.names=['REP','GEN','READ']
        return CD

    @staticmethod
    def Recombinator(rate, loci):
        """
        Recombination at loci, after variant index. Loci can take value in [0, NumSNPs-1]
        Args:
            rate: recombination rate
            loci: index of the loci in which rec is is being performed
        Returns: recombinator which is an argument of Simulation, _simulation2 and evolve. It can be list of loci
        """
        if not isinstance(loci, list):
            loci = [loci]
        return sim.Recombinator(intensity=rate, loci=loci)


class POP:
    @staticmethod
    def createISOGenicDiploidPopulation(df):
        """
        initHaps : np 2D array which m x nSS where m i number of individual haps and nSS is number of SS
        return a homozygote diploid population which every haplotype is copied n times
        """
        pop = sim.Population(size=df.shape[0], ploidy=2, loci=df.shape[1], lociPos=list(df.columns),
                             infoFields='fitness')
        for (i, (_, h)) in zip(pop.individuals(), df.iterrows()):
            i.setGenotype(h.tolist(), 0);
            i.setGenotype(h.tolist(), 1)
        return pop

    @staticmethod
    def toDF(pop):
        x = pd.concat(map(pd.DataFrame, [map(list, [i.genotype(0), i.genotype(1)]) for i in pop.allIndividuals()]),
                      keys=range(pop.popSize()))
        x.columns = list(pop.lociPos())
        return x

    @staticmethod
    def freq(pop):
        sim.stat(pop, alleleFreq=range(pop.numLoci()[0]), vars=['alleleFreq'])
        return pd.Series(pd.DataFrame(pop.vars()['alleleFreq']).loc[1].reindex().values,map(int,pop.lociPos())).fillna(0)

    @staticmethod
    def Haplotypes(pop,counts=False,unique=True):
        if isinstance(pop,sim.Population):
            a=POP.toDF(pop)
        else:
            a=pop

        H=a.reset_index(drop=True)
        H.columns=map(int,H.columns)
        b=H.loc[H.sum(1).sort_values().index].astype(str).apply(lambda x: ''.join(x), 1).reset_index(drop=True)
        if counts:
            return b.value_counts().sort_index()
        else:
            if unique:
                b=b.drop_duplicates()
            return b.loc[b.sort_values().index].reset_index(drop=True)


    @staticmethod
    def establish(H, ba, k=5):
        N = H.shape[0]
        car = H[H[ba] == 1]
        n = car.shape[0]
        return pd.concat([car.iloc[np.random.choice(n, k)], H.iloc[np.random.choice(N, N - k)]]).reset_index(drop=True)

class Drift:
    @staticmethod
    def nextGeneration(N,x):
        return (np.random.random(N)<=x).mean()

    @staticmethod
    def sampleReads(D,x):
        return [Drift.sampleReadsDerived(D,x),D]

    @staticmethod
    def sampleReadsDerived(D,x):
        return (np.random.random(D)<=x).sum()

    @staticmethod
    def simulateAF(N,x,T):
        Xt=[]
        for i in range(1, T[-1]+1):
            x=Drift.nextGeneration(N,x)
            if i in T:Xt.append(x)
        return Xt

    @staticmethod
    def simulatePoolCD(N,n,cd):
        x=cd[0].C/float(cd[0].D)
        D=cd.xs('D',level=1)
        Xt=[]
        for i in range(1, D.index[-1]+1):
            x=Drift.nextGeneration(N,x)
            if i in D.index:
                y=Drift.nextGeneration(n,x)
                Xt.append(Drift.sampleReads(D[i], y))
        return pd.DataFrame([[cd[0].C,cd[0].D]]+Xt,index=D.index,columns=['C','D'])

    @staticmethod
    def simulatePoolDerivd(N,n,cd):
        x=cd[0].C/float(cd[0].D)
        D=cd.xs('D',level=1)
        Xt=[]
        for i in range(1, D.index[-1]+1):
            x=Drift.nextGeneration(N,x)
            if i in D.index:
                Xt+=[Drift.sampleReadsDerived(D[i], Drift.nextGeneration(n,x))]
        return [cd[0].C]+Xt

    @staticmethod
    def simulatePools(N,cd,M):
        return pd.concat([Drift.simulatePool(N,cd) for _ in range(M)],keys=range(M))


    @staticmethod
    def simulateAFs(N,x,T,M):
        return pd.DataFrame([Drift.simulateAF(N,x,T) for _ in range(M)],columns=T)




