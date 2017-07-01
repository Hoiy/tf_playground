from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

"""
Deprecated, seems has bug..
Credit http://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
"""

def parallel_apply(df, func):
    dfGrouped = df.groupby(df.index % multiprocessing.cpu_count())
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(dfApply)(group, func) for name, group in dfGrouped)
    return pd.concat(retLst)

def dfApply(df, func):
    return df.apply(func)


def testFunc(x):
    return x*2


def test():
    df = pd.DataFrame({ 'a': pd.Series(range(1000000)) } )
    df['b'] = parallel_apply(df, testFunc)
    print(df.sample(10))

if __name__=='__main__':
    test()
