#!/usr/bin/env python3 -I
"""Generate LaTeX table bodies for the manuscript from results/*.csv.
Writes results/tables.tex (snippets separated by %% ==== markers).
Only formatting happens here; no number is recomputed beyond ratios/medians of CSV columns."""
import csv, os, statistics as st
R = os.path.join(os.path.dirname(__file__), "..", "results")
f = float
def rd(name): return list(csv.DictReader(open(os.path.join(R, name))))
def sci(x, d=1):
    if x == 0: return "0"
    s = f"{x:.{d}e}"; m, e = s.split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def dec(x, d=3): return f"{x:.{d}f}"
def rat(x): return f"{x:.2f}" if x < 1 else f"{x:.1f}"  # speedup ratios: 2 decimals below 1
out = []
def emit(tag, lines): out.append(f"%% ==== {tag}\n" + "\n".join(lines) + "\n")

# ---------------- runtime (6.2/6.3) ----------------
if os.path.exists(os.path.join(R, "runtime.csv")):
    rt = rd("runtime.csv")
    if "time_mstep_s" in rt[0]:
        for d in ("2", "3"):
            lines = []
            for kname in ("indep", "gauss0.5", "clayton2"):
                for K in ("100", "1000"):
                    for size in sorted({r["size"] for r in rt if r["d"] == d}, key=lambda s: int(s.split("x")[0])):
                        rows = {r["mstep"]: r for r in rt if r["d"] == d and r["data"] == kname and r["K"] == K and r["size"] == size}
                        if len(rows) < 3: continue
                        sp, do, ne = rows["sinkhorn"], rows["dou"], rows["newton"]
                        lab = {"indep": "Indep.", "gauss0.5": "Gauss.\\ $0.5$", "clayton2": "Clayton $2$"}[kname]
                        lines.append(f"{lab} & {K} & ${size.replace('x', '\\times ')}$ & "
                                     f"{dec(f(sp['time_mstep_s']),4)} & {rat(f(do['time_mstep_s'])/f(sp['time_mstep_s']))} & {rat(f(ne['time_mstep_s'])/f(sp['time_mstep_s']))} & "
                                     f"{dec(f(do['time_s'])/f(sp['time_s']),2)} & {dec(f(ne['time_s'])/f(sp['time_s']),2)} & "
                                     f"{sp['iter']}/{ne['iter']} & {dec(1e3*f(sp['time_s'])/int(sp['iter']),2)}/{dec(1e3*f(ne['time_s'])/int(ne['iter']),2)} & {int(sp['inner'])/int(sp['iter']):.1f}/{int(do['inner'])/int(do['iter']):.1f}/{int(ne['inner'])/int(ne['iter']):.1f} \\\\")
            emit(f"runtime-d{d}", lines)
        # accuracy table: SP-EM vs reference, Newton-EM vs reference
        lines = []
        for d in ("2", "3"):
            for kname in ("indep", "gauss0.5", "clayton2"):
                for K in ("100", "1000"):
                    for size in sorted({r["size"] for r in rt if r["d"] == d}, key=lambda s: int(s.split("x")[0])):
                        rows = {r["mstep"]: r for r in rt if r["d"] == d and r["data"] == kname and r["K"] == K and r["size"] == size}
                        if len(rows) < 3: continue
                        sp, ne = rows["sinkhorn"], rows["newton"]
                        lab = {"indep": "Indep.", "gauss0.5": "Gauss.\\ $0.5$", "clayton2": "Clayton $2$"}[kname]
                        lines.append(f"{lab} & {K} & ${size.replace('x', '\\times ')}$ & {sci(f(sp['err_w']))} & {sci(f(ne['err_w']))} & "
                                     f"{sci(f(sp['gapJ_norm']))} & {sci(f(ne['gapJ_norm']))} & {sci(f(sp['dens_sup']))} & {sci(f(ne['dens_sup']))} & {sci(abs(f(sp['llf'])-f(ne['llf']))/f(K))} \\\\")
        emit("accuracy", lines)

# ---------------- bound at fixed points (6.4) ----------------
# curv.csv (exp_curv.jl): curvature-free certified bound on the loss, local-curvature estimate of the distance
b = rd("curv.csv")
lines = []
for r in b:
    if r["data"] != "gauss0.5": continue
    lines.append(f"${r['size'].replace('x', '\\times ')}$ & {r['K']} & {dec(f(r['eps']))} & {dec(f(r['eta']))} & {sci(f(r['rK']))} & {sci(f(r['taylor_bound_rK']))} & "
                 f"{sci(f(r['gap_L']))} & {sci(f(r['cert_L']))} & {sci(f(r['err_w']))} & {sci(f(r['est_w']))} & {sci(f(r['mu_sp']))} \\\\")
emit("bound-gauss", lines)
# summary ranges over all data kinds
def rng(key, rows, fmt=sci):
    v = [f(r[key]) for r in rows]; return f"{fmt(min(v))} to {fmt(max(v))}"
def rrng(v): return f"{sci(min(v))} to {sci(max(v))}"
lines = []
for size in ("2x2", "5x5", "10x10", "5x5x5"):
    rows = [r for r in b if r["size"] == size]
    cert = [f(r["cert_L"]) / f(r["gap_L"]) for r in rows]
    est = [f(r["est_w"]) / f(r["err_w"]) for r in rows]
    taylor = [f(r["taylor_bound_rK"]) / f(r["rK"]) for r in rows]
    lines.append(f"${size.replace('x', '\\times ')}$ & {rng('eps', rows, dec)} & {dec(min(taylor),1)} to {dec(max(taylor),1)} & {rng('gap_L', rows)} & {rng('cert_L', rows)} & "
                 f"{rrng(cert)} & {rng('err_w', rows)} & {rat(min(est))} to {rat(max(est))} \\\\")
emit("bound-summary", lines)

# ---------------- stopped iterates ----------------
s = rd("curv_stopped.csv")
lines = []
for r in s:
    if r["data"] != "gauss0.5": continue
    lines.append(f"${r['size'].replace('x', '\\times ')}$ & {sci(f(r['abstol']),0)} & {r['iter']} & {sci(f(r['tau']))} & {dec(f(r['eps']))} & {sci(f(r['RK']))} & {sci(f(r['rK']))} & "
                 f"{sci(f(r['gap_L']))} & {sci(f(r['cert_L_RK']))} & {sci(f(r['cert_L_rK']))} & {sci(f(r['err_w']))} & {sci(f(r['est_w']))} \\\\")
emit("stopped-gauss", lines)

# ---------------- initial values (6.5) ----------------
ini = rd("init_summary.csv")
lines = []
lab = {"indep": "Indep.", "gauss0.5": "Gauss.\\ $0.5$", "gauss0.9": "Gauss.\\ $0.9$", "clayton2": "Clayton $2$", "clayton8": "Clayton $8$"}
for m in ("3", "5"):
    for k in lab:
        sp = next(r for r in ini if r["m"] == m and r["data"] == k and r["mstep"] == "sinkhorn")
        ne = next(r for r in ini if r["m"] == m and r["data"] == k and r["mstep"] == "newton")
        lines.append(f"{m} & {lab[k]} & {sci(f(sp['max_pair_dist']))} & {sci(f(sp['J_range']))} & {sp['iter_min']}--{sp['iter_max']} & "
                     f"{sci(f(ne['max_pair_dist']))} & {sci(f(ne['J_range']))} & {ne['iter_min']}--{ne['iter_max']} \\\\")
emit("init", lines)

# ---------------- uranium (6.6) ----------------
u = rd("uranium_pairs.csv")
pairs_order = sorted({r["pair"] for r in u}, key=lambda p: -f(next(r for r in u if r["pair"] == p)["tau"]))
lines = []
for m in ("3", "5", "7"):
    sp = {r["pair"]: r for r in u if r["m"] == m and r["mstep"] == "sinkhorn"}
    do = {r["pair"]: r for r in u if r["m"] == m and r["mstep"] == "dou"}
    ne = {r["pair"]: r for r in u if r["m"] == m and r["mstep"] == "newton"}
    P = list(sp)
    msd = [f(do[p]["time_mstep_s"]) / f(sp[p]["time_mstep_s"]) for p in P]
    msn = [f(ne[p]["time_mstep_s"]) / f(sp[p]["time_mstep_s"]) for p in P]
    td = [f(do[p]["time_s"]) / f(sp[p]["time_s"]) for p in P]
    tn = [f(ne[p]["time_s"]) / f(sp[p]["time_s"]) for p in P]
    share = [f(sp[p]["time_mstep_s"]) / f(sp[p]["time_s"]) for p in P]
    sharen = [f(ne[p]["time_mstep_s"]) / f(ne[p]["time_s"]) for p in P]
    dl = [f(sp[p]["dllf_per_K"]) for p in P]
    conv = (sum(sp[p]["converged"] == "true" for p in P), sum(do[p]["converged"] == "true" for p in P), sum(ne[p]["converged"] == "true" for p in P))
    lines.append(f"{m} & {dec(st.median(msd),1)} ({dec(min(msd),1)}--{dec(max(msd),1)}) & {dec(st.median(msn),1)} ({dec(min(msn),1)}--{dec(max(msn),1)}) & "
                 f"{dec(st.median(td),2)} ({dec(min(td),2)}--{dec(max(td),2)}) & {dec(st.median(tn),2)} ({dec(min(tn),2)}--{dec(max(tn),2)}) & "
                 f"{dec(st.median(share),2)}/{dec(st.median(sharen),2)} & {sci(st.median(dl))} / {sci(max(dl))} & {conv[0]}/{conv[1]}/{conv[2]} \\\\")
emit("uranium-summary", lines)
# full table (Appendix B): m = 3 and 5
lines = []
for p in pairs_order:
    cells = [p.replace("-", "--"), f"{f(next(r for r in u if r['pair']==p)['tau']):+.3f}"]
    for m in ("3", "5"):
        sp = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "sinkhorn")
        do = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "dou")
        ne = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "newton")
        dag = lambda r: "" if r["converged"] == "true" else "$^\\dagger$"
        cells += [dec(f(sp["time_mstep_s"])*1e3, 1), rat(f(do["time_mstep_s"])/f(sp["time_mstep_s"])), rat(f(ne["time_mstep_s"])/f(sp["time_mstep_s"])),
                  f"{sp['iter']}{dag(sp)}/{ne['iter']}{dag(ne)}", sci(f(sp["dllf_per_K"]))]
    lines.append(" & ".join(cells) + " \\\\")
emit("uranium-full", lines)
# order dependence: representative pairs Co-Sc, U-Li, K-Sc at m = 3,5,7
lines = []
for p in ("Co-Sc", "U-Li", "K-Sc"):
    for i, m in enumerate(("3", "5", "7")):
        sp = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "sinkhorn")
        do = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "dou")
        ne = next(r for r in u if r["pair"] == p and r["m"] == m and r["mstep"] == "newton")
        dag = lambda r: "" if r["converged"] == "true" else "$^\\dagger$"
        lines.append(f"{p.replace('-', '--') if i == 0 else ''} & {m} & {dec(f(sp['time_mstep_s'])*1e3,1)} & {dec(f(do['time_mstep_s'])*1e3,1)} & {dec(f(ne['time_mstep_s'])*1e3,1)} & "
                     f"{rat(f(do['time_mstep_s'])/f(sp['time_mstep_s']))} & {rat(f(ne['time_mstep_s'])/f(sp['time_mstep_s']))} & {dec(f(sp['time_s']),3)} & {dec(f(ne['time_s']),3)} & {sp['iter']}{dag(sp)}/{do['iter']}{dag(do)}/{ne['iter']}{dag(ne)} & {dec(f(sp['llf']),2)} \\\\")
emit("uranium-orders", lines)
# monotonicity
mo = rd("uranium_monotone.csv")
lines = []
for m in ("3", "5"):
    for ms, lab2 in (("sinkhorn", "SP-EM"), ("newton", "Newton-EM")):
        rr = [r for r in mo if r["m"] == m and r["mstep"] == ms]
        nd = [int(r["ndec"]) for r in rr]
        lines.append(f"{m} & {lab2} & {sum(n > 0 for n in nd)} & {sum(nd)} & {max(nd)} & {sci(max(f(r['maxdec']) for r in rr))} \\\\")
emit("uranium-monotone", lines)
# bound on uranium (fixed points), representative rows (uranium_curv.csv from exp_curv.jl)
ub = rd("uranium_curv.csv")
lines = []
for p in ("Co-Sc", "Sc-Ti", "K-Cs", "U-Li", "Li-Co"):
    for i, m in enumerate(("3", "5", "7")):
        r = next(r for r in ub if r["pair"] == p and r["m"] == m)
        lines.append(f"{p.replace('-', '--') if i == 0 else ''} & {m} & {dec(f(r['eps']))} & {dec(f(r['eta']))} & {sci(f(r['rK']))} & {sci(f(r['gap_L']))} & {sci(f(r['cert_L']))} & "
                     f"{sci(f(r['err_w']))} & {sci(f(r['est_w']))} & {sci(f(r['mu_sp']))} \\\\")
emit("uranium-bound", lines)
# d = 3
u3 = rd("uranium_d3.csv")
lines = []
for m in ("3", "5"):
    rows = {r["mstep"]: r for r in u3 if r["m"] == m}
    sp, do, ne = rows["sinkhorn"], rows["dou"], rows["newton"]
    lines.append(f"{m} & {dec(f(sp['time_mstep_s'])*1e3,1)} & {dec(f(do['time_mstep_s'])*1e3,1)} & {dec(f(ne['time_mstep_s'])*1e3,1)} & {dec(f(sp['time_s']),3)} & {dec(f(do['time_s']),3)} & {dec(f(ne['time_s']),3)} & "
                 f"{sp['iter']}/{do['iter']}/{ne['iter']} & {sci(f(sp['err_w']))} & {sci(f(ne['err_w']))} & {sci(f(sp['gapJ_norm']))} & {sci(f(sp['dllf_per_K']))} \\\\")
emit("uranium-d3", lines)

if os.path.exists(os.path.join(R, "large.csv")):
    lg = rd("large.csv"); lines = []
    for d in ("2", "3", "4"):
        sizes = sorted({r["size"] for r in lg if r["d"] == d}, key=lambda s: int(s.split("x")[0]))
        for kname in ("indep", "gauss0.5", "clayton2"):
            for size in sizes:
                rows = {r["mstep"]: r for r in lg if r["d"] == d and r["data"] == kname and r["size"] == size}
                if len(rows) < 3: continue
                sp, do, ne = rows["sinkhorn"], rows["dou"], rows["newton"]
                lab = {"indep": "Indep.", "gauss0.5": "Gauss.\\ $0.5$", "clayton2": "Clayton $2$"}[kname]
                dag = lambda r: "" if r["converged"] == "true" else "$^\\dagger$"
                lines.append(f"{lab} & ${size.replace('x', '\\times ')}$ & {sp['p']} & {dec(f(sp['time_s']),2)} & {rat(f(do['time_s'])/f(sp['time_s']))} & {rat(f(ne['time_s'])/f(sp['time_s']))} & "
                             f"{dec(100*f(sp['time_mstep_s'])/f(sp['time_s']),0)}/{dec(100*f(ne['time_mstep_s'])/f(ne['time_s']),0)} & {rat(f(do['time_mstep_s'])/f(sp['time_mstep_s']))} & {rat(f(ne['time_mstep_s'])/f(sp['time_mstep_s']))} & "
                             f"{sp['iter']}{dag(sp)}/{ne['iter']}{dag(ne)} & {dec(1e3*f(sp['time_s'])/int(sp['iter']),1)}/{dec(1e3*f(ne['time_s'])/int(ne['iter']),1)} & {sci(f(sp['dllf_per_K']))} \\\\")
    emit("large", lines)

# ---------------- direct Newton solver vs EM-type iterations (6.2) ----------------
dr = rd("direct.csv")
lines = []
for size in ("5x5", "10x10", "20x20", "30x30", "5x5x5", "10x10x10", "5x5x5x5"):
    rows = [r for r in dr if r["size"] == size]
    its = [int(r["it_direct"]) for r in rows]
    a = [f(r["t_sp"]) / f(r["t_direct"]) for r in rows]; b = [f(r["t_newtonem"]) / f(r["t_direct"]) for r in rows]
    lines.append(f"${size.replace('x', '\\times ')}$ & {rows[0]['p']} & {min(its)} to {max(its)} & {rat(min(a))} to {rat(max(a))} & {rat(min(b))} to {rat(max(b))} & "
                 f"{rng('gap_sp', rows)} & {rng('gap_newtonem', rows)} \\\\")
emit("direct", lines)

open(os.path.join(R, "tables.tex"), "w").write("\n".join(out))
print("\n".join(out))
