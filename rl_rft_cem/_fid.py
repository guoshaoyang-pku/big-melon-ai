import numpy as np, common
import _model_orig as O
import model as N
from _bench_model import build_board
from _bench_dense import build_dense_board

def gaps(mod, st, seeds, K=16, trials=12):
    ds=[]; dn=[]
    for s in seeds:
        a,b = mod.measure_fidelity_gap(st, K=K, trials=trials, seed=s)
        ds.append(a); dn.append(b)
    return np.mean(ds), np.std(ds), np.mean(dn), np.std(dn)

seeds = list(range(6))
for name, st in [("small(17)", build_board(45,7,16)), ("dense(72)", build_dense_board())]:
    print("=== fidelity gap on %s (mean+-std over %d seeds) ==="%(name,len(seeds)))
    go = gaps(O, st, seeds); gn = gaps(N, st, seeds)
    print("  OLD: dscore %.2f+-%.2f  dcount %.2f+-%.2f"%go)
    print("  NEW: dscore %.2f+-%.2f  dcount %.2f+-%.2f"%gn)
