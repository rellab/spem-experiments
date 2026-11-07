# Section 6.4 (second part): Corollary 1 at stopped iterates, outer tolerance varied.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
configs = [(2, 2), (2, 5), (2, 10), (3, 5)]
rows = Any[]
for (d, m) in configs, (kind, par, kname) in kinds
    K = 800; ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 2000 + K)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + prod(ns) * ALPHA
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Lhat = objJ(mp.w, Bf, ALPHA) / SK
    cc = curvature_constants(Bf, ns, Bt, ALPHA, mp.w)
    for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
        r = emfit(Bf, ns, :sinkhorn; alpha = ALPHA, abstol = atol, reltol = Inf, maxiter = 50_000, tol_inner = 1e-10)
        rr = residuals(Bf, ns, Bt, ALPHA, r.w)
        e = norm(r.w - mp.w); g = Lhat - r.J / SK
        @assert e <= rr.RK / cc.mu_data * (1 + 1e-6) && g <= rr.RK^2 / (2cc.mu_data) * (1 + 1e-6) + 1e-14
        push!(rows, (d, kname, K, join(ns, "x"), atol, r.iter, r.conv, rr.tau, rr.eps, rr.RK, rr.rK,
                     e, rr.RK / cc.mu_data, rr.rK / cc.mu_data, rr.rK / cc.mu_actual, g, rr.RK^2 / (2cc.mu_data), cc.mu_data, cc.mu_actual))
        @printf("d=%d %-9s %-7s atol=%.0e iter=%6d tau=%.1e eps=%.3f RK=%.1e rK=%.1e | err=%.1e bnd=%.1e | gap=%.1e\n",
                d, kname, join(ns, "x"), atol, r.iter, rr.tau, rr.eps, rr.RK, rr.rK, e, rr.RK / cc.mu_data, g)
    end
end
write_csv(joinpath(V2RESULTS, "bound_stopped.csv"),
          ["d","data","K","size","abstol","iter","converged","tau","eps","RK","rK_direct","err_w","bound_w_RK","bound_w_rK","rK_over_mu_actual","gap_L","bound_L_RK","mu_data","mu_actual"], rows)
