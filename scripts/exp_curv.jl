# Sections 6.4 and 6.6 (bound tables): curvature-free bound on the objective loss
# (Theorem 1 (ii), Corollary 1) and the local-curvature estimate of the distance.
# Recomputes the fixed points of exp_bound.jl / exp_stopped.jl / exp_uranium.jl with the
# same seeds and checks them against the existing CSVs; no timing is involved.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
readrows(name) = begin
    L = readlines(joinpath(V2RESULTS, name)); h = split(L[1], ",")
    [Dict(zip(h, split(l, ","))) for l in L[2:end]]
end
close_to(a, s; rtol = 1e-6, atol = 1e-13) = abs(a - parse(Float64, s)) <= rtol * abs(a) + atol
mu_at(Bf, ns, Bt, w) = curvature_constants(Bf, ns, Bt, ALPHA, w).mu_actual
cfac(ns) = sqrt(2 / maximum(ns))                    # ||what - w||_2 <= sqrt(2 / max_k n_k)

kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
configs = [(2, 2), (2, 5), (2, 10), (3, 5)]

# ---- fixed points (Theorem 1) ----
old = readrows("bound.csv"); rows = Any[]; k = 0
for (d, m) in configs, (kind, par, kname) in kinds, K in (200, 800, 3200)
    global k += 1; o = old[k]
    ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 2000 + K)
    Bs = ntuple(i -> basis_matrix(U[i], ns[i]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + prod(ns) * ALPHA
    fp = spem_fixedpoint(Bf, ns, Bt; alpha = ALPHA)
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Lhat = objJ(mp.w, Bf, ALPHA) / SK
    rs = residuals(Bf, ns, Bt, ALPHA, fp.w)
    err = norm(fp.w - mp.w); gap = Lhat - objJ(fp.w, Bf, ALPHA) / SK
    @assert o["data"] == kname && o["size"] == join(ns, "x") && o["K"] == string(K)
    @assert close_to(rs.eps, o["eps"]) && close_to(rs.rK, o["rK"]) && close_to(err, o["err_w"]) && close_to(gap, o["gap_L"]) "mismatch with bound.csv row $k"
    mu = mu_at(Bf, ns, Bt, fp.w); taylor = exp(rs.eta) * rs.eps^2 / 2; cert = cfac(ns) * rs.rK
    @assert rs.rK <= taylor * (1 + 1e-9) && gap <= cert "bound violated"
    push!(rows, (d, kname, K, join(ns, "x"), rs.eps, rs.eta, rs.rK, taylor, err, gap, mu, rs.rK / mu, cert, cfac(ns) * taylor))
    @printf("d=%d %-9s K=%-4d %-8s eps=%.3e eta=%.3e rK=%.2e | gap=%.2e cert=%.2e | err=%.2e est=%.2e mu(wSP)=%.1e\n",
            d, kname, K, join(ns, "x"), rs.eps, rs.eta, rs.rK, gap, cert, err, rs.rK / mu, mu)
end
write_csv(joinpath(V2RESULTS, "curv.csv"),
          ["d","data","K","size","eps","eta","rK","taylor_bound_rK","err_w","gap_L","mu_sp","est_w","cert_L","cert_L_taylor"], rows)

# ---- stopped iterates (Corollary 1) ----
old = readrows("bound_stopped.csv"); rows = Any[]; k = 0
for (d, m) in configs, (kind, par, kname) in kinds
    K = 800; ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 2000 + K)
    Bs = ntuple(i -> basis_matrix(U[i], ns[i]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + prod(ns) * ALPHA
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Lhat = objJ(mp.w, Bf, ALPHA) / SK
    for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
        global k += 1; o = old[k]
        r = emfit(Bf, ns, :sinkhorn; alpha = ALPHA, abstol = atol, reltol = Inf, maxiter = 50_000, tol_inner = 1e-10)
        rr = residuals(Bf, ns, Bt, ALPHA, r.w)
        e = norm(r.w - mp.w); g = Lhat - r.J / SK
        @assert o["data"] == kname && o["size"] == join(ns, "x") && o["iter"] == string(r.iter)
        @assert close_to(rr.RK, o["RK"]) && close_to(rr.rK, o["rK_direct"]) && close_to(e, o["err_w"]) && close_to(g, o["gap_L"]; atol = 1e-12) "mismatch with bound_stopped.csv row $k"
        mu = mu_at(Bf, ns, Bt, r.w)
        @assert rr.rK <= rr.RK * (1 + 1e-9) && g <= cfac(ns) * rr.rK + 1e-12 "bound violated"
        push!(rows, (d, kname, K, join(ns, "x"), atol, r.iter, r.conv, rr.tau, rr.eps, rr.RK, rr.rK, e, g, mu, rr.rK / mu, cfac(ns) * rr.RK, cfac(ns) * rr.rK))
        @printf("d=%d %-9s %-7s atol=%.0e iter=%6d tau=%.1e RK=%.1e rK=%.1e | gap=%.1e cert(RK)=%.1e cert(rK)=%.1e | err=%.1e est=%.1e\n",
                d, kname, join(ns, "x"), atol, r.iter, rr.tau, rr.RK, rr.rK, g, cfac(ns) * rr.RK, cfac(ns) * rr.rK, e, rr.rK / mu)
    end
end
write_csv(joinpath(V2RESULTS, "curv_stopped.csv"),
          ["d","data","K","size","abstol","iter","converged","tau","eps","RK","rK","err_w","gap_L","mu_w","est_w","cert_L_RK","cert_L_rK"], rows)

# ---- uranium pairs (fixed points) ----
X, names = load_uranium(); K = size(X, 1); nv = length(names)
U = [pseudo_obs(X[:, i]) for i in 1:nv]
old = readrows("uranium_bound.csv"); rows = Any[]; k = 0
for i in 1:nv, j in i+1:nv, m in (3, 5, 7)
    global k += 1; o = old[k]
    ns = (m, m); rng = MersenneTwister(20250301 + 100i + j + m)
    Bs = (basis_matrix(U[i], m), basis_matrix(U[j], m)); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + m^2 * ALPHA
    w0 = random_start(rng, ns)
    r = emfit(Bf, ns, :sinkhorn; alpha = ALPHA, w0 = w0)
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Jhat = objJ(mp.w, Bf, ALPHA)
    fp = spem_fixedpoint(Bf, ns, Bt; alpha = ALPHA, w0 = r.w)
    rf = residuals(Bf, ns, Bt, ALPHA, fp.w)
    err = norm(fp.w - mp.w); gap = (Jhat - objJ(fp.w, Bf, ALPHA)) / SK
    pair = names[i] * "-" * names[j]
    @assert o["pair"] == pair && o["m"] == string(m)
    @assert close_to(rf.eps, o["eps_fp"]) && close_to(rf.rK, o["rK_fp"]) && close_to(err, o["err_fp"]) && close_to(gap, o["gap_fp"]) "mismatch with uranium_bound.csv row $k"
    mu = mu_at(Bf, ns, Bt, fp.w); taylor = exp(rf.eta) * rf.eps^2 / 2; cert = cfac(ns) * rf.rK
    @assert rf.rK <= taylor * (1 + 1e-9) && gap <= cert "bound violated"
    push!(rows, (pair, m, o["tau"], rf.eps, rf.eta, rf.rK, taylor, err, gap, mu, rf.rK / mu, cert))
    @printf("%-6s m=%d eps=%.3f rK=%.2e | gap=%.2e cert=%.2e | err=%.2e est=%.2e mu(wSP)=%.1e\n", pair, m, rf.eps, rf.rK, gap, cert, err, rf.rK / mu, mu)
end
write_csv(joinpath(V2RESULTS, "uranium_curv.csv"),
          ["pair","m","tau","eps","eta","rK","taylor_bound_rK","err_w","gap_L","mu_sp","est_w","cert_L"], rows)
