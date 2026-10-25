#!/usr/bin/env python3
"""
bdls.py -- weighted local search (NuMVC-style, with annealing-flavoured random walk) for small covers.
See LOCALSEARCH.md.

  python bdls.py N M [--time SEC] [--seed S] [--W] [--restarts R]
"""
import argparse
import random
import time

from .bdcover import balanced_vectors, build_constraints, bits, verify_rowsets


def neg(x):
    return tuple(-a for a in x)


def instance(n, m, W=False):
    vecs = balanced_vectors(n)
    Ks, nmin, _ = build_constraints(n, m, vecs, lambda s: None, allow_empty=W,
                                    minimal_limit=10 ** 9 if n <= 7 else 0)
    lines = [x for x in vecs if x > neg(x)]
    lid = {}
    for j, x in enumerate(lines):
        lid[x] = j
        lid[neg(x)] = j
    cons = [sorted({lid[vecs[i]] for i in bits(K)}) for K in Ks[:nmin]]
    return vecs, lines, cons


class LS:
    def __init__(self, cons, L, rng):
        self.cons, self.L, self.rng = cons, L, rng
        C = len(cons)
        self.inc = [[] for _ in range(L)]
        for ci, c in enumerate(cons):
            for v in c:
                self.inc[v].append(ci)
        self.w = [1] * C
        self.cnt = [0] * C
        self.inS = [False] * L
        self.score = [0] * L                        # gain if not in S, loss if in S
        self.unc = list(range(C))                   # uncovered constraints
        self.pos = list(range(C))
        for v in range(L):
            self.score[v] = len(self.inc[v])
        self.age = [0] * L
        self.size = 0

    def _unc_remove(self, c):
        i = self.pos[c]
        last = self.unc[-1]
        self.unc[i] = last
        self.pos[last] = i
        self.unc.pop()

    def _unc_add(self, c):
        self.pos[c] = len(self.unc)
        self.unc.append(c)

    def add(self, v):
        self.inS[v] = True
        self.size += 1
        loss = 0
        for c in self.inc[v]:
            self.cnt[c] += 1
            k = self.cnt[c]
            if k == 1:
                self._unc_remove(c)
                wc = self.w[c]
                loss += wc
                for u in self.cons[c]:
                    if u != v:
                        self.score[u] -= wc
            elif k == 2:
                for u in self.cons[c]:
                    if u != v and self.inS[u]:
                        self.score[u] -= self.w[c]
                        break
        self.score[v] = loss

    def remove(self, v):
        self.inS[v] = False
        self.size -= 1
        gain = 0
        for c in self.inc[v]:
            self.cnt[c] -= 1
            k = self.cnt[c]
            if k == 0:
                self._unc_add(c)
                wc = self.w[c]
                gain += wc
                for u in self.cons[c]:
                    if u != v:
                        self.score[u] += wc
            elif k == 1:
                for u in self.cons[c]:
                    if self.inS[u]:
                        self.score[u] += self.w[c]
                        break
        self.score[v] = gain

    def bump(self):
        for c in self.unc:
            self.w[c] += 1
            for u in self.cons[c]:
                self.score[u] += 1

    def S(self):
        return [v for v in range(self.L) if self.inS[v]]


def greedy(ls):
    while ls.unc:
        v = max(range(ls.L), key=lambda x: (not ls.inS[x]) * ls.score[x])
        ls.add(v)


def search(cons, L, time_limit, rng, log, init=None):
    ls = LS(cons, L, rng)
    for v in (init or []):
        ls.add(v)
    greedy(ls)
    best = ls.S()
    log(f"  initial cover (init + greedy completion): {len(best)}")
    t0, step, last_add, last_rm = time.time(), 0, -1, -1
    while time.time() - t0 < time_limit:
        if not ls.unc:
            if ls.size < len(best) or not best:
                best = ls.S()
                log(f"  {time.time() - t0:7.1f}s  step {step:8d}: cover of size {len(best)}")
            v = min((x for x in range(L) if ls.inS[x]), key=lambda x: (ls.score[x], ls.age[x]))
            ls.remove(v)
            continue
        step += 1
        cand = [x for x in range(L) if ls.inS[x] and x != last_add]
        v = min(cand, key=lambda x: (ls.score[x], ls.age[x]))
        ls.remove(v)
        ls.age[v] = step
        last_rm = v
        c = ls.unc[rng.randrange(len(ls.unc))]
        opts = [u for u in cons[c] if u != last_rm] or cons[c]
        if rng.random() < 0.02:                     # occasional random walk (annealing-style noise)
            u = rng.choice(opts)
        else:
            u = max(opts, key=lambda x: (ls.score[x], -ls.age[x]))
        ls.add(u)
        ls.age[u] = step
        last_add = u
        ls.bump()
    log(f"  {step} steps in {time.time() - t0:.0f}s")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--W", action="store_true")
    ap.add_argument("--init-embed", action="store_true",
                    help="start from all lines with last coordinate 0 (the embedded (n-1,m) cover)")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    vecs, lines, cons = instance(a.n, a.m, a.W)
    log(f"n={a.n} m={a.m}: {len(lines)} lines, {len(cons)} constraints")
    best = None
    for r in range(a.restarts):
        init = [j for j, x in enumerate(lines) if x[-1] == 0] if a.init_embed else None
        S = search(cons, len(lines), a.time / a.restarts, random.Random(a.seed + r), log, init)
        if best is None or len(S) < len(best):
            best = S
    idx = {x: i for i, x in enumerate(vecs)}
    Sv = [idx[lines[j]] for j in best]
    ok = verify_rowsets(Sv, vecs, a.n, a.m, W=a.W) is None
    log(f"best cover {len(best)}; verify_rowsets: {ok}")


if __name__ == "__main__":
    main()
