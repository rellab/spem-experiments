# Section 6.2 (remark on direct solution): the MAP problem is concave, so it can also be solved
# without EM by a damped Newton method on the tangent space.  This script times that solver
# (cold start from the uniform weights, stop at ||P grad L_K|| <= 1e-6) against SP-EM and
# Newton-EM under the common stopping rule, on the data of exp_large.jl (K = 500).
# One warm-up call per method, then the best of three runs (one run when p >= 512).
# All methods run with one Julia thread and one BLAS thread (set in lib.jl).
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
configs = [(2, 5), (2, 10), (2, 20), (2, 30), (3, 5), (3, 10), (4, 5)]
K = 500; rows = Any[]
function best_of(f, reps)
    best = Inf; r = nothing
    for _ in 1:reps
        t = @elapsed rr = f()
        if t < best; best = t; r = rr; end
    end
    best, r
end
for (d, m) in configs, (kind, par, kname) in kinds
    ns = ntuple(_ -> m, d); p = prod(ns); reps = p >= 512 ? 1 : 3
    U = sample_copula(kind, K, d; param = par, seed = 5000 + d)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + p * ALPHA
    w0 = fill(1 / p, p)
    emfit(Bf, ns, :sinkhorn; alpha = ALPHA, maxiter = 20); emfit(Bf, ns, :newton; alpha = ALPHA, maxiter = 20)
    map_newton(Bf, ns, Bt; alpha = ALPHA, w0 = w0, tol = 1e-6, maxit = 2)            # warm-up (JIT)
    tsp, rsp = best_of(() -> emfit(Bf, ns, :sinkhorn; alpha = ALPHA), reps)
    tne, rne = best_of(() -> emfit(Bf, ns, :newton; alpha = ALPHA), reps)
    tdn, dn = best_of(() -> map_newton(Bf, ns, Bt; alpha = ALPHA, w0 = w0, tol = 1e-6), reps)
    Jdn = objJ(dn.w, Bf, ALPHA); Jbest = max(Jdn, rsp.J, rne.J)
    gap(J) = (Jbest - J) / SK
    push!(rows, (d, kname, K, join(ns, "x"), p, size(Bt, 2), tsp, rsp.iter, rsp.conv, gap(rsp.J), tne, rne.iter, rne.conv, gap(rne.J),
                 tdn, dn.iter, dn.gradnorm, gap(Jdn)))
    @printf("d=%d %-9s %-9s p=%4d | SP %.3fs it=%4d gap=%.1e | NewtonEM %.3fs it=%4d gap=%.1e | direct %.3fs it=%3d g=%.1e gap=%.1e | SP/direct=%.2f\n",
            d, kname, join(ns, "x"), p, tsp, rsp.iter, gap(rsp.J), tne, rne.iter, gap(rne.J), tdn, dn.iter, dn.gradnorm, gap(Jdn), tsp / tdn)
    flush(stdout)
end
write_csv(joinpath(V2RESULTS, "direct.csv"),
          ["d","data","K","size","p","dimT","t_sp","it_sp","conv_sp","gap_sp","t_newtonem","it_newtonem","conv_newtonem","gap_newtonem",
           "t_direct","it_direct","grad_direct","gap_direct"], rows)
