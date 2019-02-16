import numpy as np
import pandas as pd
# conda install -c conda-forge lifelines

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