import os,pickle,sys,warnings
os.environ["PYTHONWARNINGS"]="ignore"; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from scipy.stats import kurtosis,skew
from sklearn.metrics import adjusted_rand_score as ari,silhouette_score
from sklearn.mixture import GaussianMixture as G
pd.set_option("display.width",220,"display.max_columns",30)
b=pickle.load(open(sys.argv[1],"rb")); C=b["C_2030"]; Ck=b["C_2030_knn"]; names=np.array(b["names"]); idx=list(b["index"]); pop=b["features_2030"]["population"].values; cols=b["feature_cols"]; rank=b["rank_2030"]
def M(k,cov,reg,init,s): return G(k,covariance_type=cov,reg_covar=reg,init_params=init,n_init=5,max_iter=500,random_state=s)
def lab(c,X,s=42): return M(*c,s).fit(X).predict(X)
def ok(l,k):
    n=np.bincount(l,minlength=k); return n.min()>=8 and n.max()<=95
def stab(c,base):
    r=lab(c,C,base); g=np.random.default_rng(base); o=[]
    for t in range(15):
        i=g.choice(len(C),152,replace=False); o.append(ari(r[i],lab(c,C[i],base*100+t)))
    return np.mean(o)
def rep(c,par):
    st=par(delayed(lab)(c,C,s) for s in range(30))
    return {"cfg":str(c),"bic":round(M(*c,42).fit(C).bic(C)),"sil":round(np.mean([silhouette_score(C,x) for x in st]),3),"stab":round(np.mean(par(delayed(stab)(c,x) for x in [1,2,3])),3),"size_pass":f"{sum(ok(x,c[0]) for x in st)}/30","knn":round(ari(lab(c,C),lab(c,Ck)),3),"ari_chosen":round(ari(lab(c,C),lab(CH,C)),3),"sizes":sorted(np.bincount(lab(c,C)).tolist(),reverse=True)}
CH=(5,"spherical",1e-3,"random")
if __name__=="__main__":
    par=Parallel(n_jobs=-1)
    N=[CH,(5,"spherical",1e-6,"random"),(5,"spherical",1e-2,"random"),(5,"spherical",.1,"random"),(5,"spherical",1e-3,"kmeans"),(5,"spherical",1e-3,"k-means++"),(4,"spherical",1e-3,"random"),(6,"spherical",1e-3,"random"),(5,"diag",1e-3,"random")]
    print(pd.DataFrame([rep(c,par) for c in N]).to_string(index=False))
    m=M(*CH,42).fit(C); l=m.predict(C); p=m.predict_proba(C).max(1)
    prof=pd.DataFrame(C,columns=cols).groupby(l).mean().round(1); prof.insert(0,"size",np.bincount(l)); prof.insert(1,"variance",np.round(m.covariances_,2)); print(prof.T.to_string())
    s=pd.Series(l,index=idx)
    for t in sorted(set(l)):
        x=l==t; print(t,x.sum(),", ".join(names[x][np.argsort(-pop[x])][:14]))
    print("top10",[(r["name"],int(s[c])) for c,r in rank.head(10).iterrows()]); print("median rank",rank.join(s.rename("t")).groupby("t")["rank"].median().to_dict(),"ranked count",rank.join(s.rename("t")).groupby("t").size().to_dict())
    print("prob>=.9",round((p>=.9).mean(),3),"<.7",round((p<.7).mean(),3),"median",round(np.median(p),3))
    print("least certain",[(names[i],round(p[i],2)) for i in np.argsort(p)[:6]])
    print("params spherical/diag/full at k=5:",5*12+4,5*22+4,5*77+4)
    print("kurtosis",dict(zip(cols,np.round(kurtosis(C),1)))); print("within-type kurtosis median",round(np.median([kurtosis(C[l==t],axis=None) for t in set(l)]),2))
    b_=[(k,cov,round(min(M(k,cov,1e-3,"random",42).fit(C).bic(C) for _ in [0]))) for k in [4,5,6] for cov in ["spherical","diag","full"]]; print(b_)
