import os, sys, warnings, traceback, collections
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cs_lib as L

if __name__ == "__main__":
    C = L.B()["C_2030"]
    G = [g for g in L.grid() if g["fam"] == "hdbscan" and not g.get("assign_noise")]
    errs = collections.Counter()
    first = {}
    for g in G:
        try:
            Z, _ = L.embed(C, g["ver"])
            L.fit_method(g, Z, 1)
        except Exception as e:
            sig = (type(e).__name__, str(e)[:80], g["metric"], g["method"], g["cse"], g["ms"])
            errs[sig[:2] + (g["metric"], g["method"])] += 1
            if sig[:2] not in first:
                first[sig[:2]] = (g, traceback.format_exc())
    print(len(G), errs)
    for k, v in first.items():
        print(k, v[0])
        print(v[1][-1500:])
