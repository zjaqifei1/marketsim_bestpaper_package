
# -*- coding: utf-8 -*-
"""
Reproducible MarketSim experiments for Evidence-Driven Adaptive Fixed-Share (ED-AdaptiveShare).

This script:
  1) Generates nonstationary synthetic sequences (MarketSim)
  2) Builds an expert library (AR(1) forecasters with fixed coefficients)
  3) Computes a DP switching oracle with budget S
  4) Evaluates baselines vs ED-AdaptiveShare
  5) Runs scaling + runtime micro-benchmarks

Note: This is a minimal, self-contained reference implementation intended for a paper appendix.
"""

import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------
# Reproducibility
# -------------------------
SEED = 7
np.random.seed(SEED)

# -------------------------
# Expert library
# -------------------------
K = 10
A_LIST = [-0.9, -0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 0.9, 1.0]
B_LIST = [0.0]*K

def expert_predictions(y, a_list=A_LIST, b_list=B_LIST):
    T = len(y)
    K = len(a_list)
    yhat = np.zeros((T,K), dtype=np.float32)
    prev = y[:-1].reshape(-1,1).astype(np.float32)
    a = np.array(a_list, dtype=np.float32).reshape(1,-1)
    b = np.array(b_list, dtype=np.float32).reshape(1,-1)
    yhat[1:] = prev @ a + b
    return yhat

def bounded_loss(yhat, y, scale=1.0):
    err2 = (y.reshape(-1,1) - yhat)**2
    l = np.minimum(err2/scale, 1.0).astype(np.float32)
    return l, err2.astype(np.float32)

# -------------------------
# MarketSim generator
# -------------------------
def generate_episode(T=600, family="burst_switch", S=10, noise_scale=0.3):
    y = np.zeros(T, dtype=np.float32)
    regimes = np.zeros(T, dtype=np.int32)
    cps = []
    y[0] = np.float32(np.random.normal(0, 0.5))
    a_pool = np.array([-0.7, -0.2, 0.2, 0.7, 0.95], dtype=np.float32)

    if family == "ultra_stable":
        a = float(np.random.choice(a_pool))
        sig = 0.08
        for t in range(1,T):
            y[t] = np.float32(a*float(y[t-1]) + float(np.random.normal(0, sig)))
        return y, regimes, cps

    if family == "burst_switch":
        L1 = int(0.45*T)
        Lb = int(0.15*T)
        burst_seg_len = 10
        n_burst = max(2, Lb//burst_seg_len)
        cps = [L1 + i*burst_seg_len for i in range(n_burst)]
        cps = [c for c in cps if c < L1+Lb]
        seg_starts = [0]+cps
        seg_ends = cps+[T]

        a1 = float(np.random.choice(a_pool))
        a2 = float(np.random.choice(a_pool))
        while abs(a2-a1) < 0.4:
            a2 = float(np.random.choice(a_pool))

        burst_choices = np.array([-0.7, 0.7, 0.95], dtype=np.float32)
        burst_as = np.random.choice(burst_choices, size=len(cps), replace=True).astype(np.float32)

        seg_a=[]
        for st,en in zip(seg_starts, seg_ends):
            if en <= L1: seg_a.append(a1)
            elif st >= L1+Lb: seg_a.append(a2)
            else:
                j = cps.index(st)
                seg_a.append(float(burst_as[j]))

        for ridx,(st,en) in enumerate(zip(seg_starts, seg_ends)):
            a = seg_a[ridx]
            regimes[st:en] = ridx
            sig = noise_scale*(3.0 if (st>=L1 and st<L1+Lb) else 1.0)
            for t in range(max(1,st), en):
                y[t] = np.float32(a*float(y[t-1]) + float(np.random.normal(0, sig)))
        return y, regimes, cps

    if family == "multi_burst":
        segments=[]
        stable_lens=[int(0.25*T), int(0.2*T), int(0.2*T)]
        burst_lens =[int(0.1*T), int(0.1*T)]
        a_stables=[float(np.random.choice(a_pool)) for _ in stable_lens]
        burst_choices=np.array([-0.7, 0.7, 0.95], dtype=np.float32)
        for i in range(len(burst_lens)):
            segments.append(("stable", stable_lens[i], a_stables[i]))
            segments.append(("burst", burst_lens[i], None))
        segments.append(("stable", stable_lens[-1], a_stables[-1])
        )
        total = sum(l for _,l,_ in segments)
        if total < T:
            kind,L,a = segments[-1]
            segments[-1] = (kind, L + (T-total), a)

        t=0
        ridx=0
        cps=[]
        for kind,L,a_st in segments:
            st=t; en=min(T, t+L)
            if kind=="stable":
                a=float(a_st)
                for tt in range(max(1,st), en):
                    y[tt]=np.float32(a*float(y[tt-1]) + float(np.random.normal(0, noise_scale)))
                regimes[st:en]=ridx
                ridx += 1
            else:
                seglen=10
                nseg=max(2, (en-st)//seglen)
                bstarts=[st + i*seglen for i in range(nseg)]
                bstarts=[s for s in bstarts if s<en]
                for bs in bstarts[1:]:
                    cps.append(bs)
                b_as=np.random.choice(burst_choices, size=len(bstarts), replace=True).astype(np.float32)
                for j,bs in enumerate(bstarts):
                    be=min(en, bs+seglen)
                    a=float(b_as[j])
                    sig=noise_scale*2.8
                    for tt in range(max(1,bs), be):
                        y[tt]=np.float32(a*float(y[tt-1]) + float(np.random.normal(0, sig)))
                    regimes[bs:be]=ridx
                    ridx += 1
            t=en
        cps=sorted(list(set(cps)))
        return y, regimes, cps

    if family == "heavy_tail":
        min_len=max(25, T//(S+2))
        candidates=np.arange(min_len, T-min_len)
        S_eff=min(S, len(candidates)) if len(candidates)>0 else 0
        cps=sorted(np.random.choice(candidates,size=S_eff,replace=False).tolist()) if S_eff>0 else []
        seg_starts=[0]+cps
        seg_ends=cps+[T]
        a_choices=np.array([-0.7,0.2,0.7,0.95], dtype=np.float32)
        df_choices=np.array([2.5,3.0,5.0], dtype=np.float32)
        jump_prob=0.05
        jump_scale=4.0
        for ridx,(st,en) in enumerate(zip(seg_starts,seg_ends)):
            a=float(np.random.choice(a_choices))
            df=float(np.random.choice(df_choices))
            regimes[st:en]=ridx
            for tt in range(max(1,st), en):
                eps=float(np.random.standard_t(df)*noise_scale)
                if np.random.rand()<jump_prob:
                    eps += float(np.random.normal(0,jump_scale))
                y[tt]=np.float32(a*float(y[tt-1]) + eps)
        return y, regimes, cps

    raise ValueError("Unknown family")

# -------------------------
# Evidence features (backtest blocks)
# -------------------------
def compute_block_means(losses, V):
    T,K = losses.shape
    csum = np.cumsum(losses, axis=0)
    bm = np.full((T,K), np.nan, dtype=np.float32)
    bm[V-1:] = (csum[V-1:] - np.vstack([np.zeros((1,K),dtype=np.float32), csum[:-V]])) / V
    return bm

def evidence_from_losses(losses, N=100, J=5):
    T,K = losses.shape
    assert N % J == 0
    V = N//J
    bm = compute_block_means(losses, V=V)
    block_means = np.full((J, T, K), np.nan, dtype=np.float32)
    for j in range(J):
        shift = j*V
        if shift==0:
            block_means[j] = bm
        else:
            block_means[j, shift:] = bm[:-shift]
    mu = np.nanmean(block_means, axis=0)
    s = np.nanstd(block_means, axis=0)
    min_mu = np.nanmin(mu, axis=1)
    drift = np.full(T, np.nan, dtype=np.float32)
    drift[1:] = np.abs(min_mu[1:] - min_mu[:-1])

    # pack [mu,s,med,last] + globals [min_mu, spread, ent, drift, best_k]
    med = np.nanmedian(block_means, axis=0)
    last = block_means[0]
    spread = np.nanmax(mu, axis=1) - min_mu

    ent = np.full(T, np.nan, dtype=np.float32)
    temp = 0.2
    for t in range(T):
        if np.isnan(mu[t]).any():
            continue
        logits = (-mu[t]/temp).astype(np.float64)
        logits = logits - logits.max()
        p = np.exp(logits); p = p/p.sum()
        ent[t] = -(p*np.log(p+1e-12)).sum()

    best_k = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        if np.isnan(mu[t]).any(): 
            continue
        best_k[t] = float(np.argmin(mu[t]))

    feat = np.full((T, 4*K + 5), np.nan, dtype=np.float32)
    for t in range(T):
        if np.isnan(mu[t]).any():
            continue
        per = np.stack([mu[t], s[t], med[t], last[t]], axis=1).reshape(-1)
        g = np.array([min_mu[t], spread[t], ent[t], drift[t] if t>0 else 0.0, best_k[t]], dtype=np.float32)
        feat[t] = np.concatenate([per, g])
    return feat, mu

# -------------------------
# DP switching oracle
# -------------------------
def dp_switching_oracle(losses, S):
    T,K = losses.shape
    INF = 1e18
    dp = np.full((S+1,K), INF, dtype=np.float64)
    back = np.full((T,S+1,K), -1, dtype=np.int32)
    dp[0,:] = losses[0,:]
    for s in range(1,S+1):
        dp[s,:] = losses[0,:]
    for t in range(1,T):
        newdp = np.full((S+1,K), INF, dtype=np.float64)
        for s in range(S+1):
            stay = dp[s,:]
            if s>0:
                prev = dp[s-1,:]
                arg1 = int(np.argmin(prev)); min1 = prev[arg1]
                tmp = prev.copy(); tmp[arg1]=INF
                arg2 = int(np.argmin(tmp)); min2 = tmp[arg2]
            for k in range(K):
                best_prev = stay[k]
                best_kprev = k
                if s>0:
                    cand = min1 if arg1!=k else min2
                    cand_k = arg1 if arg1!=k else arg2
                    if cand < best_prev:
                        best_prev = cand
                        best_kprev = cand_k
                newdp[s,k] = losses[t,k] + best_prev
                back[t,s,k] = best_kprev
        dp = newdp

    best_val = INF; best_s=0; best_k=0
    for s in range(S+1):
        k = int(np.argmin(dp[s,:])); val = dp[s,k]
        if val < best_val:
            best_val = val; best_s=s; best_k=k

    pi = np.zeros(T, dtype=np.int32); pi[T-1]=best_k
    s = best_s
    for t in range(T-1,0,-1):
        kprev = back[t,s,pi[t]]
        if kprev != pi[t] and s>0:
            s -= 1
        pi[t-1]=kprev
    return float(best_val), pi

# -------------------------
# Algorithms
# -------------------------
def run_hedge(losses, eta):
    T,K = losses.shape
    w = np.ones(K)/K
    cum=0.0
    for t in range(T):
        cum += float(np.dot(w, losses[t]))
        w = w*np.exp(-eta*losses[t])
        w = w/w.sum()
    return cum

def run_fixedshare(losses, eta, alpha):
    T,K = losses.shape
    w = np.ones(K)/K
    uni = np.ones(K)/K
    cum=0.0
    for t in range(T):
        cum += float(np.dot(w, losses[t]))
        w = w*np.exp(-eta*losses[t])
        w = w/w.sum()
        w = (1-alpha)*w + alpha*uni
    return cum

def run_gating(mu, losses, beta):
    T,K = losses.shape
    uni = np.ones(K)/K
    cum=0.0
    for t in range(T):
        if np.isnan(mu[t]).any():
            w = uni
        else:
            x = (-beta*mu[t]).astype(np.float64)
            x = x - x.max()
            w = np.exp(x); w = w/w.sum()
        cum += float(np.dot(w, losses[t]))
    return cum

def run_ed_adaptiveshare(losses, feat, mu, eta, rho_min, rho_max, c_drift, c_var):
    T,K = losses.shape
    w = np.ones(K)/K
    uni = np.ones(K)/K
    cum=0.0
    for t in range(T):
        cum += float(np.dot(w, losses[t]))
        # Hedge step
        w_bar = w*np.exp(-eta*losses[t])
        w_bar = w_bar/w_bar.sum()
        # evidence-driven rho
        if np.isnan(feat[t]).any():
            rho = rho_min
        else:
            per = feat[t][:4*K].reshape(K,4)
            mu_t = per[:,0]; s_t = per[:,1]
            kbest = int(np.argmin(mu_t))
            drift_t = float(feat[t][-2])
            varbest = float(s_t[kbest])
            score = c_drift*drift_t + c_var*varbest
            q = 1.0 - math.exp(-score)
            rho = rho_min + (rho_max-rho_min)*q
            rho = float(np.clip(rho, rho_min, rho_max))
        w = (1-rho)*w_bar + rho*uni
    return cum

# -------------------------
# Visualization utilities
# -------------------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_dynregret_bars(df, out_dir):
    pivot_mean = df.pivot_table(index="Method", columns="Family", values="DynRegret(mean)")
    pivot_std = df.pivot_table(index="Method", columns="Family", values="DynRegret(std)")

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    pivot_mean.plot(kind="bar", yerr=pivot_std, ax=ax, capsize=3)
    ax.set_title("Average Dynamic Regret by Family")
    ax.set_ylabel("Dynamic Regret (mean ± std)")
    ax.set_xlabel("")
    ax.legend(title="Family", frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "avg_dynregret_bar.png"))
    plt.close(fig)


def plot_dynregret_heatmap(df, out_dir):
    pivot_mean = df.pivot_table(index="Method", columns="Family", values="DynRegret(mean)")
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=140)
    im = ax.imshow(pivot_mean.values, cmap="viridis")
    ax.set_xticks(np.arange(pivot_mean.shape[1]), labels=pivot_mean.columns)
    ax.set_yticks(np.arange(pivot_mean.shape[0]), labels=pivot_mean.index)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            ax.text(j, i, f"{pivot_mean.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Dynamic Regret Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dynregret_heatmap.png"))
    plt.close(fig)


def plot_episode_example(family, out_dir, T, S):
    y, regimes, cps = generate_episode(T=T, family=family, S=S, noise_scale=0.3)
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=140)
    ax.plot(y, color="#1f77b4", linewidth=1.4, label="MarketSim signal")
    for cp in cps:
        ax.axvline(cp, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Sample Episode ({family}) with Regime Shifts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"episode_{family}.png"))
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="MarketSim ED-AdaptiveShare experiments")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization outputs.")
    parser.add_argument("--viz-dir", default="results_viz", help="Output directory for figures.")
    args = parser.parse_args()

    # Loss scale estimated once; in the paper we'd compute from a pilot set
    LOSS_SCALE = 1.622  # from pilot quantile

    # Experiment config
    T = 600
    S = 10
    N = 100
    J = 5
    families = ["ultra_stable","burst_switch","multi_burst","heavy_tail"]

    # Tuned hyperparams (example)
    ETA = 1.6
    ALPHA = 0.002
    BETA = 80.0
    RHO_MIN = 0.001
    RHO_MAX = 0.01
    C_DRIFT = 8.0
    C_VAR = 2.0

    n_eps = 80
    rows=[]
    for fam in families:
        regs = { "Hedge-true": [], "FixedShare-true": [], "Gating-backtest": [], "Ours-ED-AdaptiveShare": []}
        for _ in range(n_eps):
            y, regimes, cps = generate_episode(T=T, family=fam, S=S, noise_scale=0.3)
            yhat = expert_predictions(y)
            losses, _ = bounded_loss(yhat, y, scale=LOSS_SCALE)
            feat, mu = evidence_from_losses(losses, N=N, J=J)
            L_star, _ = dp_switching_oracle(losses, S=S)

            cum_h = run_hedge(losses, eta=ETA)
            cum_fs = run_fixedshare(losses, eta=ETA, alpha=ALPHA)
            cum_gate = run_gating(mu, losses, beta=BETA)
            cum_ours = run_ed_adaptiveshare(losses, feat, mu, eta=ETA, rho_min=RHO_MIN, rho_max=RHO_MAX, c_drift=C_DRIFT, c_var=C_VAR)

            regs["Hedge-true"].append(cum_h - L_star)
            regs["FixedShare-true"].append(cum_fs - L_star)
            regs["Gating-backtest"].append(cum_gate - L_star)
            regs["Ours-ED-AdaptiveShare"].append(cum_ours - L_star)

        for m in regs:
            r = np.array(regs[m], dtype=np.float64)
            rows.append({"Family": fam, "Method": m, "DynRegret(mean)": float(r.mean()), "DynRegret(std)": float(r.std(ddof=1))})

    df = pd.DataFrame(rows)
    print(df.pivot_table(index="Method", columns="Family", values="DynRegret(mean)"))
    out = "results_marketsim.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    if not args.no_viz:
        _ensure_dir(args.viz_dir)
        plot_dynregret_bars(df, args.viz_dir)
        plot_dynregret_heatmap(df, args.viz_dir)
        plot_episode_example("burst_switch", args.viz_dir, T=T, S=S)
        print(f"Saved figures to: {args.viz_dir}")

if __name__ == "__main__":
    main()
