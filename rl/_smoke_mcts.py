import numpy as np, common
from net import build_net, pick_device
from mcts import MCTS
from model import SuikaModel, sample_fruit
from _bench_model import build_board

cfg = {"arch":"mlp","K":16,"max_fruits":80,"hidden":128,"n_layers":2}
dev = pick_device("cpu")          # keep off GPU for a tiny smoke
net = build_net(cfg).to(dev).eval()
K=16
st = build_board(n_drops=20, seed=2, K=K)
print("start fruits=%d score=%d"%(len(st["fruits"]), st["score"]))
mcts = MCTS(net, dev, K=K, num_simulations=16, fast_model=True, seed=0, eval_batch=8)
m = SuikaModel(K=K)
rng = np.random.default_rng(0)
total=0
for mv in range(5):
    if st.get("game_over"):
        print("game over at move",mv); break
    pi, root = mcts.run(st, add_noise=True)
    a = int(np.argmax(pi))
    nf = sample_fruit(rng)
    st, gain, over = m.step(st, a, nf, fast=False)
    total += gain
    print("move %d: action=%2d pi_max=%.3f gain=%6.1f fruits=%2d score=%d over=%s"
          %(mv, a, float(pi.max()), gain, len(st["fruits"]), st["score"], over))
print("SMOKE_OK total_gain=%.1f"%total)
