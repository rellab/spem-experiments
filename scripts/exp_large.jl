# Section 6.2 (addition): larger orders / dimensions where the M-step dominates.
# Same protocol as exp_runtime.jl (alpha = 1e-3, common stopping rule), K = 500,
# one warm-up run then a single timed run per method.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
configs = [(2, 10), (2, 20), (2, 30), (3, 5), (3, 10), (4, 5)]
K = 500; rows = Any[]
for (d, m) in configs, (kind, par, kname) in kinds
    ns = ntuple(_ -> m, d)
    U = sample_copula(kind, K, d; param = par, seed = 5000 + d)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + prod(ns) * ALPHA
    fits = Dict{Symbol,Any}()
    for ms in (:sinkhorn, :dou, :newton)
        emfit(Bf, ns, ms; alpha = ALPHA, maxiter = 20)                 # warm-up (JIT)
        t = @elapsed r = emfit(Bf, ns, ms; alpha = ALPHA)
        fits[ms] = (t, r)
    end
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA)
    check_reference(mp.w, [fits[ms][2].w for ms in keys(fits)], Bf, ALPHA)
    Jhat = objJ(mp.w, Bf, ALPHA)
    for ms in (:sinkhorn, :dou, :newton)
        t, r = fits[ms]
        push!(rows, (d, kname, K, join(ns, "x"), prod(ns), String(ms), t, r.tmstep, r.iter, r.inner, r.conv,
                     r.llf, r.J, norm(r.w - mp.w), (Jhat - r.J) / SK, abs(r.llf - fits[:newton][2].llf) / K))
        @printf("d=%d %-9s %-12s %-8s t=%.3f tM=%.3f (%.0f%%) iter=%4d inner/it=%.1f  |w-what|=%.1e dJ/S=%.1e\n",
                d, kname, join(ns, "x"), ms, t, r.tmstep, 100r.tmstep / t, r.iter, r.inner / r.iter, norm(r.w - mp.w), (Jhat - r.J) / SK)
    end
end
write_csv(joinpath(V2RESULTS, "large.csv"),
          ["d","data","K","size","p","mstep","time_s","time_mstep_s","iter","inner","converged","llf","J","err_w","gapJ_norm","dllf_per_K"], rows)
