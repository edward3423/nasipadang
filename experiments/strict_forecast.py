import os, sys, warnings
os.environ["PYTHONWARNINGS"]="ignore"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor as ET, HistGradientBoostingRegressor as HGB, RandomForestRegressor as RF
from sklearn.impute import SimpleImputer as SI
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline as mp
from sklearn.preprocessing import StandardScaler as SS
from sklearn.svm import SVR
pd.set_option("display.width",230,"display.max_columns",30,"display.max_rows",80)
raw=pd.read_excel(sys.argv[1],sheet_name="Data"); meta=pd.read_excel(sys.argv[1],sheet_name="Country"); meta=meta[meta["Region"]!="Aggregates"].set_index("Country code")
raw=raw[raw["Country code"].isin(meta.index)]; YEARS=[c for c in raw.columns if str(c).isdigit()]
w=raw.set_index(["Country code","Series name"])[YEARS].replace("..",np.nan)
S={"population":"Population","urban":"Urban population","gni":"GNI per capita (Atlas $)","yield":"Cereal yield (kg per hectare)","co2":"CO2 emissions per capita (metric tons)","u5":"Under-five mortality rate (per 1,000)"}
L={k:np.log(w.xs(v,level="Series name").apply(pd.to_numeric,errors="coerce").reindex(meta.index).rename(columns=int).pipe(lambda d:d.where(d>0))) for k,v in S.items()}
inc=pd.Categorical(meta["Income group"]).codes; reg=pd.Categorical(meta["Region"]).codes
def F(t): return pd.DataFrame({"income":inc,"region":reg,**{k:v[t].values for k,v in L.items()}},index=meta.index)
M={"Ridge":lambda:mp(SI(strategy="median"),SS(),Ridge(5.)),"KNN":lambda:mp(SI(strategy="median"),SS(),KNN(10,weights="distance")),"SVR":lambda:mp(SI(strategy="median"),SS(),SVR(C=1.,epsilon=.05)),
"RandomForest":lambda:mp(SI(strategy="median"),RF(500,min_samples_leaf=3,random_state=42,n_jobs=-1)),"ExtraTrees":lambda:mp(SI(strategy="median"),ET(500,min_samples_leaf=3,random_state=42,n_jobs=-1)),
"GradBoost":lambda:HGB(max_depth=3,learning_rate=.05,max_iter=200,min_samples_leaf=10,random_state=42),"MLP":lambda:mp(SI(strategy="median"),SS(),MLPRegressor(hidden_layer_sizes=(32,),alpha=1.,max_iter=3000,random_state=42))}
ORI={"population":2000,"urban":2000,"gni":2000,"u5":2000,"yield":1999,"co2":1998}
rows=[];pred={};boot=[]
rng=np.random.default_rng(42)
for tg,o in ORI.items():
    v=L[tg]; end=o+10; tr0=1990
    if tg in("yield","co2"): pass
    Xtr=F(tr0); ytr=(v[o]-v[tr0])*10/(o-tr0); ok=ytr.notna(); Xtr,ytr=Xtr[ok],ytr[ok]
    Xte=F(o); yte=v[end]-v[o]; ok=yte.notna(); Xte,yte=Xte[ok],yte[ok]; b=v[o][ok]
    r={"target":tg,"test":f"{o}>{end}","train":f"{tr0}>{o}","n_tr":len(ytr),"n_te":len(yte)}
    P={"NoChange":np.zeros(len(yte)),"AvgChange":np.full(len(yte),ytr.mean()),"TrendLine":((v[o]-v[o-5])*2)[ok].fillna(ytr.mean()).values}
    for n,mk in M.items(): P[n]=mk().fit(Xtr,ytr).predict(Xte)
    for n,p in P.items(): r[n]=np.sqrt(np.mean((yte.values-p)**2))
    pred[tg]=(P,yte,b)
    r2={"target":tg}
    for n,p in P.items():
        r2[n]=spearmanr(yte.values,p)[0] if np.std(p)>0 else np.nan
    rows.append((r,r2))
    best=P["RandomForest"]; e=(yte.values-P["TrendLine"])**2-(yte.values-best)**2; idx=rng.integers(0,len(e),(2000,len(e))); boot.append((tg,e.mean(),*np.quantile(e[idx].mean(1),[.025,.975])))
print("== RMSE of 10-year log change (strict: train 1990s, test 2000s) ==")
print(pd.DataFrame([r for r,_ in rows]).round(3).to_string(index=False))
print("\n== Do they rank who changes most? Spearman, predicted vs actual change ==")
print(pd.DataFrame([x for _,x in rows]).round(2).to_string(index=False))
print("\nRF vs TrendLine, squared-error gain, 95% bootstrap:",[(a,round(b,3),round(c,3),round(d,3)) for a,b,c,d in boot])
print("\n== Market-pick backcast: composite of city customers, income, child survival, cereal yield. Forecast 2000 -> 2010 vs actual 2010 ==")
FEAT={"urban":1,"gni":1,"u5":-1,"yield":1}
act={k:(pred[k][2]+pred[k][1]) for k in FEAT}
idx=None
for k in FEAT:
    idx=act[k].index if idx is None else idx.intersection(act[k].index)
def comp(d): return sum((pd.Series(d[k]).loc[idx].rank(pct=True)*s) for k,s in FEAT.items())/len(FEAT)
actual=comp(act); res=[]
for n in ["NoChange","AvgChange","TrendLine","Ridge","KNN","SVR","RandomForest","ExtraTrees","GradBoost","MLP"]:
    d={k:pd.Series(pred[k][2].values+pred[k][0][n],index=pred[k][2].index) for k in FEAT}; c=comp(d)
    top=lambda s,m:set(s.nlargest(m).index)
    A=(c.rank()-comp({k:pred[k][2] for k in FEAT}).rank()); B=(actual.rank()-comp({k:pred[k][2] for k in FEAT}).rank())
    res.append({"model":n,"rank_corr":spearmanr(c,actual)[0],"top10":len(top(c,10)&top(actual,10)),"top20":len(top(c,20)&top(actual,20)),"top30":len(top(c,30)&top(actual,30)),"rank_mean_err":(c.rank()-actual.rank()).abs().mean(),"movers_corr":spearmanr(A,B)[0] if A.std()>0 else np.nan})
print("countries",len(idx)); print(pd.DataFrame(res).round(3).to_string(index=False))
