# Section 6.4: verification of Theorem 1 and Corollary 1.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
configs = [(2, 2), (2, 5), (2, 10), (3, 5)]          # (d, cells per axis)
rows = Any[]; rows_stop = Any[]
for (d, m) in configs, (kind, par, kname) in kinds, K in (200, 800, 3200)
    ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 2000 + K)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + prod(ns) * ALPHA
    fp = spem_fixedpoint(Bf, ns, Bt; alpha = ALPHA)
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Lhat = objJ(mp.w, Bf, ALPHA) / SK
    check_reference(mp.w, [fp.w], Bf, ALPHA)
    cc = curvature_constants(Bf, ns, Bt, ALPHA, mp.w)
    rs = residuals(Bf, ns, Bt, ALPHA, fp.w)
    err = norm(fp.w - mp.w); gap = Lhat - objJ(fp.w, Bf, ALPHA) / SK
    bw(mu) = exp(rs.eta) * rs.eps^2 / (2mu); bL(mu) = exp(2rs.eta) * rs.eps^4 / (8mu)
    @assert err <= bw(cc.mu_data) * (1 + 1e-6) && gap <= bL(cc.mu_data) * (1 + 1e-6) + 1e-14 "bound violated"
    push!(rows, (d, kname, K, join(ns, "x"), fp.iter, fp.tau, rs.eps, rs.eta, rs.rK, exp(rs.eta) * rs.eps^2 / 2,
                 cc.mu_cons, cc.mu_data, cc.mu_actual, cc.lam_GK, err, bw(cc.mu_cons), bw(cc.mu_data), bw(cc.mu_actual),
                 gap, bL(cc.mu_cons), bL(cc.mu_data), bL(cc.mu_actual), rs.rK / cc.mu_actual))
    @printf("d=%d %-9s K=%-4d %-8s eps=%.3e eta=%.3e rK=%.2e | err=%.2e bnd(data)=%.2e bnd(act)=%.2e | gap=%.2e bndL(data)=%.2e | mu cons=%.1e data=%.1e act=%.1e\n",
            d, kname, K, join(ns, "x"), rs.eps, rs.eta, rs.rK, err, bw(cc.mu_data), bw(cc.mu_actual), gap, bL(cc.mu_data),
            cc.mu_cons, cc.mu_data, cc.mu_actual)
    # stopped iterates (Corollary 1): outer tolerances on the change of J_K
    for atol in (1e-2, 1e-4, 1e-6, 1e-8)
        r = emfit(Bf, ns, :sinkhorn; alpha = ALPHA, abstol = atol, reltol = 0.0, maxiter = 50_000)
        rr = residuals(Bf, ns, Bt, ALPHA, r.w)
        e = norm(r.w - mp.w); g = Lhat - r.J / SK
        @assert e <= rr.RK / cc.mu_data * (1 + 1e-6) && g <= rr.RK^2 / (2cc.mu_data) * (1 + 1e-6) + 1e-14
        push!(rows_stop, (d, kname, K, join(ns, "x"), atol, r.iter, rr.tau, rr.eps, rr.RK, rr.rK,
                          e, rr.RK / cc.mu_data, rr.rK / cc.mu_data, g, rr.RK^2 / (2cc.mu_data)))
    end
end
write_csv(joinpath(V2RESULTS, "bound.csv"),
          ["d","data","K","size","fp_iter","tau","eps","eta","rK","taylor_bound_rK",
           "mu_cons","mu_data","mu_actual","lam_GK","err_w","bound_w_cons","bound_w_data","bound_w_actual",
           "gap_L","bound_L_cons","bound_L_data","bound_L_actual","rK_over_mu_actual"], rows)
write_csv(joinpath(V2RESULTS, "bound_stopped.csv"),
          ["d","data","K","size","abstol","iter","tau","eps","RK","rK_direct","err_w","bound_w_RK","bound_w_rK","gap_L","bound_L_RK"], rows_stop)
