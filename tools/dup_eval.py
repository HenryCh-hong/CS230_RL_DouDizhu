# Duplicate-deal evaluation with per-deal outcomes (probe for the CS230 review).
# Usage: dup_eval.py <douzero_root> <A> <B> <n_deals> <seed> <workers>
# A/B are 'rlcard', 'random', or a checkpoint dir containing landlord*.ckpt.
import sys, os, copy, pickle, multiprocessing as mp
import numpy as np

def make_deals(n, seed):
    deck = [i for i in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
    rng = np.random.RandomState(seed); deals = []
    for _ in range(n):
        d = deck.copy(); rng.shuffle(d)
        deal = {'landlord': d[:20], 'landlord_up': d[20:37], 'landlord_down': d[37:54], 'three_landlord_cards': d[17:20]}
        for k in deal: deal[k].sort()
        deals.append(deal)
    return deals

def spec(agent, pos):
    return agent if agent in ('rlcard', 'random') else os.path.join(agent, pos + '.ckpt')

def worker(args):
    root, ll, peas, chunk = args
    sys.path.insert(0, root)
    import torch; torch.set_num_threads(1)
    from douzero.env.game import GameEnv
    from douzero.evaluation.simulation import load_card_play_models
    players = load_card_play_models({'landlord': spec(ll, 'landlord'), 'landlord_up': spec(peas, 'landlord_up'), 'landlord_down': spec(peas, 'landlord_down')})
    env = GameEnv(players); out = []
    for idx, deal in chunk:
        env.card_play_init(copy.deepcopy(deal))
        while not env.game_over: env.step()
        ll_pts = (2 if env.winner == 'landlord' else -2) * 2 ** env.bomb_num   # landlord's points; peasant team gets the negative
        out.append((idx, env.winner == 'landlord', ll_pts, env.bomb_num))
        env.reset()
    return out

def play(root, ll, peas, deals, workers):
    chunks = [[(i, deals[i]) for i in range(w, len(deals), workers)] for w in range(workers)]
    with mp.get_context('spawn').Pool(workers) as pool:
        res = [r for part in pool.map(worker, [(root, ll, peas, c) for c in chunks]) for r in part]
    res.sort(); return np.array([r[1] for r in res]), np.array([r[2] for r in res], float)

if __name__ == '__main__':
    root, A, B, n, seed, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    deals = make_deals(n, seed)
    llwin1, L1 = play(root, A, B, deals, workers)   # A = landlord, B = peasants
    llwin2, L2 = play(root, B, A, deals, workers)   # B = landlord, A = peasants
    wp_d = (llwin1.astype(float) + (~llwin2).astype(float)) / 2   # per-deal WP of A
    adp_d = (L1 - L2) / 2                                          # per-deal ADP of A
    rng = np.random.RandomState(0); B_ = 2000; idx = rng.randint(0, n, (B_, n))
    ci = lambda v: np.percentile(v[idx].mean(1), [2.5, 97.5])
    print(f'A={os.path.basename(A.rstrip("/"))} vs B={os.path.basename(B.rstrip("/"))} on {n} deals x 2 seatings (seed {seed})')
    print(f'  overall WP {wp_d.mean():.3f}  95% CI [{ci(wp_d)[0]:.3f}, {ci(wp_d)[1]:.3f}]')
    print(f'  overall ADP {adp_d.mean():.3f}  95% CI [{ci(adp_d)[0]:.3f}, {ci(adp_d)[1]:.3f}]   per-deal paired SD {adp_d.std(ddof=1):.3f}')
    print(f'  A as Landlord: WP {llwin1.mean():.3f} ADP {L1.mean():.3f} | A as Peasants: WP {(~llwin2).mean():.3f} ADP {(-L2).mean():.3f}')
    np.savez(os.path.join(os.getcwd(), f'dup_{os.path.basename(A.rstrip("/"))}_vs_{os.path.basename(B.rstrip("/"))}_{n}_{seed}.npz'), L1=L1, L2=L2, llwin1=llwin1, llwin2=llwin2)
