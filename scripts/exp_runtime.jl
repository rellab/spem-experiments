# Section 6.2/6.3: runtime, iteration counts and accuracy of the three M-steps
# on synthetic data (d = 2, 3), alpha = 1e-3, common protocol.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
rows = Any[]
function timed(Bf, ns, ms)
    big = prod(ns) >= 512
    emfit(Bf, ns, ms; alpha = ALPHA)                        # warm-up (JIT)
    best = Inf; r = nothing
    for _ in 1:(big ? 1 : 3)
        t = @elapsed rr = emfit(Bf, ns, ms; alpha = ALPHA)
        if t < best; best = t; r = rr; end
    end
    best, r
end
for d in (2, 3), (kind, par, kname) in kinds, K in (100, 1000), m in (2, 5, 10)
    ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 1000 + K)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns)
    fits = Dict(ms => timed(Bf, ns, ms) for ms in (:sinkhorn, :dou, :newton))
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA)
    worst = check_reference(mp.w, [fits[ms][2].w for ms in keys(fits)], Bf, ALPHA)
    Jhat = objJ(mp.w, Bf, ALPHA); SK = K + prod(ns) * ALPHA
    for ms in (:sinkhorn, :dou, :newton)
        t, r = fits[ms]
        dd = density_diff(r.w, mp.w, ns; G = d == 2 ? 50 : 16)
        push!(rows, (d, kname, K, join(ns, "x"), String(ms), t, r.tmstep, r.iter, r.inner, r.conv,
                     r.llf, r.J, r.ndec, r.maxdec, norm(r.w - mp.w), (Jhat - r.J) / SK, dd.sup, dd.l2, mp.gradnorm, worst))
        @printf("d=%d %-9s K=%-4d %-9s %-8s t=%.4f tM=%.4f iter=%4d inner=%6d conv=%s  |w-what|=%.2e dJ/S=%.2e\n",
                d, kname, K, join(ns, "x"), ms, t, r.tmstep, r.iter, r.inner, r.conv, norm(r.w - mp.w), (Jhat - r.J) / SK)
    end
end
write_csv(joinpath(V2RESULTS, "runtime.csv"),
          ["d","data","K","size","mstep","time_s","time_mstep_s","iter","inner","converged","llf","J","ndec","maxdec",
           "err_w","gapJ_norm","dens_sup","dens_l2","map_gradnorm","ref_worst"], rows)
