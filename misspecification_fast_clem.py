"""
Misspecification Study
"""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import gamma as gamma_dist
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif']  = ['Times New Roman']
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(2024)

# ===========================================================================
# Data simulation for scenario 1, 2 and 3
# ===========================================================================

def simulate_hpp(rate, T):
    if rate <= 0: return np.array([])
    return np.sort(np.random.uniform(0, T, np.random.poisson(rate * T)))

def simulate_gamma_kernel(t11, t21, lam0, a1, a2, kappa, theta, T):
    mode = (kappa - 1)*theta if kappa >= 1 else 1e-4
    g_max = gamma_dist.pdf(mode, a=kappa, scale=theta)
    lam_upper = lam0 + (a1*len(t11) + a2*len(t21))*g_max + 1e-9
    n_cand = np.random.poisson(lam_upper*T)
    if n_cand == 0: return np.array([])
    cand = np.sort(np.random.uniform(0, T, n_cand))
    lam_c = np.full(n_cand, lam0, dtype=float)
    if len(t11) > 0:
        d1 = cand[:,None] - t11[None,:]
        lam_c += a1*np.where(d1>0, gamma_dist.pdf(d1,a=kappa,scale=theta),0.).sum(1)
    if len(t21) > 0:
        d2 = cand[:,None] - t21[None,:]
        lam_c += a2*np.where(d2>0, gamma_dist.pdf(d2,a=kappa,scale=theta),0.).sum(1)
    return cand[np.random.uniform(size=n_cand) <= lam_c/lam_upper]

def simulate_powerlaw_primary(t11, t21, a_pl, b_pl, a1, a2, b1, b2, T):
    lam_upper = a_pl*(T+1)**b_pl + a1*len(t11) + a2*len(t21) + 1e-9
    n_cand = np.random.poisson(lam_upper*T)
    if n_cand == 0: return np.array([])
    cand = np.sort(np.random.uniform(0, T, n_cand))
    lam_c = a_pl*(cand+1)**b_pl
    if len(t11) > 0:
        d1 = cand[:,None] - t11[None,:]
        lam_c += a1*np.where(d1>0, np.exp(-b1*d1),0.).sum(1)
    if len(t21) > 0:
        d2 = cand[:,None] - t21[None,:]
        lam_c += a2*np.where(d2>0, np.exp(-b2*d2),0.).sum(1)
    return cand[np.random.uniform(size=n_cand) <= lam_c/lam_upper]

# ===========================================================================
# NLL (used to pick best CLEM restart)
# ===========================================================================

def nll_proposed(params, t12, t11, t21, T):
    lam0, a1, a2, b1, b2 = params
    if lam0<=0 or a1<0 or a2<0 or b1<=0 or b2<=0: return 1e10
    n12 = len(t12)
    if n12 == 0:
        it = lam0*T
        if len(t11)>0: it += (a1/b1)*np.sum(1-np.exp(-b1*(T-t11)))
        if len(t21)>0: it += (a2/b2)*np.sum(1-np.exp(-b2*(T-t21)))
        return it
    lam_ev = np.full(n12, lam0)
    if len(t11)>0:
        d1 = t12[:,None]-t11[None,:]
        lam_ev += a1*np.where(d1>0,np.exp(-b1*d1),0.).sum(1)
    if len(t21)>0:
        d2 = t12[:,None]-t21[None,:]
        lam_ev += a2*np.where(d2>0,np.exp(-b2*d2),0.).sum(1)
    if np.any(lam_ev<=0): return 1e10
    ll = np.sum(np.log(lam_ev))
    it = lam0*T
    if len(t11)>0: it += (a1/b1)*np.sum(1-np.exp(-b1*(T-t11)))
    if len(t21)>0: it += (a2/b2)*np.sum(1-np.exp(-b2*(T-t21)))
    return -(ll - it)

# ===========================================================================
# CLEM ALGORITHM
# ===========================================================================

def _m_step_ab(W, C, rem, beta_prev):
    """Solve h(beta) = W*(1 - beta*S/D)/beta - C = 0 via Brent; return (alpha, beta)."""
    def DS(b):
        e = np.exp(-b*rem)
        return np.sum(1.-e), np.sum(rem*e)
    def h(b):
        D, S = DS(b)
        if D < 1e-12: return 1e10
        return W*(1. - b*S/D)/b - C
    b_lo, b_hi = 1e-4, 20.
    try:
        f_lo, f_hi = h(b_lo), h(b_hi)
        beta_new = brentq(h, b_lo, b_hi, xtol=1e-6, maxiter=200) if f_lo>0>f_hi else beta_prev
    except Exception:
        beta_new = beta_prev
    D, _ = DS(beta_new)
    return max(beta_new*W/max(D,1e-10), 0.), max(beta_new, 1e-4)


def fit_clem(t12, t11, t21, T, K=10, max_iter=80, tol=1e-5, n_restarts=3):
    """CLEM with K composite-likelihood subwindows."""
    np.random.seed(0)   # fix restart initial values → reproducible across runs
    n12, n11, n21 = len(t12), len(t11), len(t21)
    d = T / K
    win_ends = np.array([(k+1)*d for k in range(K)])

    def assign(times):
        if len(times) == 0: return np.array([], dtype=int)
        return np.minimum(np.searchsorted(win_ends, times), K-1)

    w12 = assign(t12); w11 = assign(t11); w21 = assign(t21)
    rem11 = (win_ends[w11]-t11) if n11>0 else np.array([])
    rem21 = (win_ends[w21]-t21) if n21>0 else np.array([])

    def mask_gaps(w_up, t_up):
        if len(t_up)==0: return None, None
        same  = w12[:,None]==w_up[None,:]
        after = t12[:,None]> t_up[None,:]
        return same&after, t12[:,None]-t_up[None,:]

    mask11, gaps11 = mask_gaps(w11, t11)
    mask21, gaps21 = mask_gaps(w21, t21)
    all_d1 = (t12[:,None]-t11[None,:]) if n11>0 else None
    all_d2 = (t12[:,None]-t21[None,:]) if n21>0 else None

    emp = max(n12/T, 0.05)
    best_p, best_nll = None, np.inf

    for _ in range(n_restarts):
        lam0 = emp*np.random.uniform(.5,1.5)
        a1   = np.random.uniform(.05,.5)
        a2   = np.random.uniform(.05,.5)
        b1   = np.random.uniform(.1,.8)
        b2   = np.random.uniform(.1,.8)

        for _ in range(max_iter):
            prev = np.array([lam0,a1,a2,b1,b2])
            lam_ev = np.full(n12, lam0, dtype=float)
            c1 = c2 = None
            if n11>0:
                e1 = a1*np.where(all_d1>0, np.exp(-b1*all_d1), 0.)
                c1 = e1; lam_ev += e1.sum(1)
            if n21>0:
                e2 = a2*np.where(all_d2>0, np.exp(-b2*all_d2), 0.)
                c2 = e2; lam_ev += e2.sum(1)
            lam_ev = np.maximum(lam_ev, 1e-10)
            p0 = lam0/lam_ev
            p1 = c1/lam_ev[:,None] if c1 is not None else None
            p2 = c2/lam_ev[:,None] if c2 is not None else None
            lam0 = max(p0.sum()/T, 1e-6)
            if p1 is not None and mask11 is not None:
                W = float((p1*mask11).sum()); C = float((p1*gaps11*mask11).sum())
                if W>1e-10 and len(rem11)>0: a1,b1 = _m_step_ab(W,C,rem11,b1)
            if p2 is not None and mask21 is not None:
                W = float((p2*mask21).sum()); C = float((p2*gaps21*mask21).sum())
                if W>1e-10 and len(rem21)>0: a2,b2 = _m_step_ab(W,C,rem21,b2)
            curr = np.array([lam0,a1,a2,b1,b2])
            if np.max(np.abs(curr-prev)/(np.abs(prev)+1e-8)) < tol: break

        nll = nll_proposed([lam0,a1,a2,b1,b2], t12,t11,t21,T)
        if nll < best_nll: best_nll=nll; best_p=np.array([lam0,a1,a2,b1,b2])
    return best_p

# ===========================================================================
# BENCHMARKS
# ===========================================================================

def nll_mo(p, t, T):
    th1,th2=p
    if th1<=0 or th2<=0: return 1e10
    d=1.+th1*th2*t
    if np.any(d<=0): return 1e10
    return -(len(t)*np.log(th2)-np.sum(np.log(d))-np.log(1+th1*th2*T)/th1)

def fit_mo(t, T, nr=3):
    np.random.seed(0)   # fix random starts → reproducible
    emp=max(len(t)/T,1e-3); bv=np.inf; best=None
    for _ in range(nr):
        x0=[np.random.uniform(1e-3,2.), np.random.uniform(emp*.5,emp*2.)]
        try:
            r=minimize(nll_mo,x0,args=(t,T),method='L-BFGS-B',
                       bounds=[(1e-6,100),(1e-6,100)],options={'maxiter':1000,'ftol':1e-9})
            if r.fun<bv: bv=r.fun; best=r.x.copy()
        except: pass
    return best

def predict_mo(p,tau,dtau):
    th1,th2=p; return max(0.,(np.log(1+th1*th2*(tau+dtau))-np.log(1+th1*th2*tau))/th1)

def nll_gompertz(p,t,T):
    th1,th2,th3=p
    if th1<=0 or th2<=0 or th2>=1 or th3<=0 or th3>=1: return 1e10
    pw=th2**t; lam=th1*pw*th3**pw*np.log(th2)*np.log(th3)
    if np.any(lam<=0): return 1e10
    it=th1*(th3**(th2**T)-th3)
    return 1e10 if it<0 else -(np.sum(np.log(lam))-it)

def fit_gompertz(t,T,nr=3):
    np.random.seed(0)   # fix random starts → reproducible
    bv=np.inf; best=None
    for _ in range(nr):
        x0=[np.random.uniform(.5,5.),np.random.uniform(.05,.95),np.random.uniform(.05,.95)]
        try:
            r=minimize(nll_gompertz,x0,args=(t,T),method='L-BFGS-B',
                       bounds=[(1e-4,1e4),(1e-4,1-1e-4),(1e-4,1-1e-4)],
                       options={'maxiter':1000,'ftol':1e-9})
            if r.fun<bv: bv=r.fun; best=r.x.copy()
        except: pass
    return best

def predict_gompertz(p,tau,dtau):
    th1,th2,th3=p; return max(0.,th1*(th3**(th2**(tau+dtau))-th3**(th2**tau)))

def predict_proposed(p,t11,t21,tau,dtau):
    lam0,a1,a2,b1,b2=p; pred=lam0*dtau; Te=tau+dtau
    for tj in t11[t11<Te]:
        s=max(tau-tj,0.); e=Te-tj
        pred += (a1/b1)*(np.exp(-b1*s)-np.exp(-b1*e)) if tj<tau else (a1/b1)*(1-np.exp(-b1*e))
    for tj in t21[t21<Te]:
        s=max(tau-tj,0.); e=Te-tj
        pred += (a2/b2)*(np.exp(-b2*s)-np.exp(-b2*e)) if tj<tau else (a2/b2)*(1-np.exp(-b2*e))
    return max(0., pred)

# ===========================================================================
# PARAMETERS
# ===========================================================================
T=1000; tau=800; dtau=200; R=100
lam_m11=0.2; lam_m21=0.2
lam0_true=0.5
kappa_g=2; theta_g=3
a_pl=0.1;  b_pl=0.1   # much milder trend → less non-stationarity
alpha_true=0.5          # stronger triggering → EP is the dominant feature
beta_true=0.5           # faster decay → tighter, more identifiable clusters

K_VALUES = [2]

# ===========================================================================
# PRE-GENERATE ALL DATASETS
# ===========================================================================
print("Pre-generating datasets …")
np.random.seed(2024)
datasets = []
for rep in range(R):
    t11 = simulate_hpp(lam_m11, T)
    t21 = simulate_hpp(lam_m21, T)
    t12_s1 = simulate_hpp(lam0_true, T)
    t12_s2 = simulate_gamma_kernel(t11,t21,lam0_true,alpha_true,alpha_true,kappa_g,theta_g,T)
    t12_s3 = simulate_powerlaw_primary(t11,t21,a_pl,b_pl,alpha_true,alpha_true,beta_true,beta_true,T)
    datasets.append((t11,t21,t12_s1,t12_s2,t12_s3))
print(f"  {R} replications generated.\n")

# ===========================================================================
# RUN BENCHMARKS ONCE
# ===========================================================================
# bm[sc_i] = {'mo': [mae_rep0, ...], 'gom': [...], 'pois': [...]}
print("Running benchmarks (shared across K values) …")
bm = {sc: {'mo':[],'gom':[],'pois':[]} for sc in [1,2,3]}
np.random.seed(42)
for t11,t21,t12_s1,t12_s2,t12_s3 in datasets:
    for sc_i, t12 in [(1,t12_s1),(2,t12_s2),(3,t12_s3)]:
        actual   = int(np.sum((t12>=tau)&(t12<tau+dtau)))
        t12_tr   = t12[t12<tau]
        p_mo     = fit_mo(t12_tr, tau)
        p_gom    = fit_gompertz(t12_tr, tau)
        rate_p   = len(t12_tr)/tau
        bm[sc_i]['mo'].append(  abs(actual-predict_mo(p_mo,tau,dtau))  if p_mo  is not None else np.nan)
        bm[sc_i]['gom'].append( abs(actual-predict_gompertz(p_gom,tau,dtau)) if p_gom is not None else np.nan)
        bm[sc_i]['pois'].append(abs(actual-rate_p*dtau))
print("  done.\n")

# ===========================================================================
# MAIN LOOP OVER K VALUES
# all_results[K][sc_i] = {'prop': [mae_rep0, mae_rep1, ...]}
# ===========================================================================
all_results = {}

for K in K_VALUES:
    t0 = time.time()
    print(f"K = {K}  …", end='', flush=True)
    np.random.seed(2024)

    res = {sc: {'prop': []} for sc in [1,2,3]}

    for t11,t21,t12_s1,t12_s2,t12_s3 in datasets:
        for sc_i, t12 in [(1,t12_s1),(2,t12_s2),(3,t12_s3)]:
            actual  = int(np.sum((t12>=tau)&(t12<tau+dtau)))
            t12_tr  = t12[t12<tau]; t11_tr=t11[t11<tau]; t21_tr=t21[t21<tau]
            p = fit_clem(t12_tr, t11_tr, t21_tr, tau, K=K)
            mae = abs(actual - predict_proposed(p,t11,t21,tau,dtau)) if p is not None else np.nan
            res[sc_i]['prop'].append(mae)

    all_results[K] = res

# ===========================================================================
# SAVE PER-REPLICATION AE TO CSV
# Columns: rep, scenario, K, Proposed_CLEM, MO, Gompertz, Poisson
# ===========================================================================
records = []
sc_names = {1:'Sc1_HPP', 2:'Sc2_Gamma', 3:'Sc3_PowerLaw'}

for K in K_VALUES:
    for sc_i in [1,2,3]:
        prop_list = all_results[K][sc_i]['prop']
        mo_list   = bm[sc_i]['mo']
        gom_list  = bm[sc_i]['gom']
        pois_list = bm[sc_i]['pois']
        for rep in range(R):
            records.append({
                'rep':            rep + 1,
                'scenario':       sc_names[sc_i],
                'K':              K,
                'Proposed_CLEM':  round(prop_list[rep], 4) if rep < len(prop_list) else np.nan,
                'MO':             round(mo_list[rep],   4) if rep < len(mo_list)   else np.nan,
                'Gompertz':       round(gom_list[rep],  4) if rep < len(gom_list)  else np.nan,
                'Poisson':        round(pois_list[rep], 4) if rep < len(pois_list) else np.nan,
            })

df_rep = pd.DataFrame(records)
df_rep.to_csv('/Users/fpan4/Desktop/Techonometric/misspecification_per_rep_ae.csv', index=False)


# Print mean MAE summary
print("\nMean MAE summary:")
print(f"{'Scenario':<22} {'K':>3}  {'Proposed':>10} {'MO':>10} {'Gompertz':>10} {'Poisson':>10}")
print("-" * 70)
for K in K_VALUES:
    for sc_i in [1,2,3]:
        prop_m = np.nanmean(all_results[K][sc_i]['prop'])
        mo_m   = np.nanmean(bm[sc_i]['mo'])
        gom_m  = np.nanmean(bm[sc_i]['gom'])
        pois_m = np.nanmean(bm[sc_i]['pois'])
        print(f"{sc_names[sc_i]:<22} {K:>3}  {prop_m:>10.3f} {mo_m:>10.3f} {gom_m:>10.3f} {pois_m:>10.3f}")

# ===========================================================================
# LINE PLOT: Mean MAE by method — style matching Figure 11
# ===========================================================================
K_plot = K_VALUES[0]   # e.g. K=2

# Compute mean MAE per method per scenario
mean_mae = {}
for sc_i in [1, 2, 3]:
    mean_mae[sc_i] = {
        'prop': np.nanmean(all_results[K_plot][sc_i]['prop']),
        'mo':   np.nanmean(bm[sc_i]['mo']),
        'gom':  np.nanmean(bm[sc_i]['gom']),
        'pois': np.nanmean(bm[sc_i]['pois']),
    }

scenarios   = [1, 2, 3]
x_labels    = ['Scenario 1', 'Scenario 2', 'Scenario 3']

# Match Figure 11 style exactly
methods = [
    ('prop', 'Proposed',     'red',        '-',   '^', 10),
    ('mo',   'MO',           'olive',      '--',  'o',  9),
    ('gom',  'Gompertz',     'cyan',       ':',   '+', 11),
    ('pois', 'Poisson',      'orange',     '--',  'x', 10),
]

fig4, ax = plt.subplots(figsize=(8, 4.5))
for key, label, color, ls, marker, ms in methods:
    y_vals = [mean_mae[sc_i][key] for sc_i in scenarios]
    ax.plot(x_labels, y_vals,
            color=color, linestyle=ls, marker=marker,
            markersize=ms, linewidth=2.5, label=label)
ax.set_ylabel('Mean Absolute Error', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=18, loc='upper left', framealpha=0.9, edgecolor='#cccccc')

ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.set_ylim(5, 25)

plt.tight_layout()
plt.savefig('/Users/fpan4/Desktop/Techonometric/misspecification_MAE_lineplot.pdf',
            bbox_inches='tight')
