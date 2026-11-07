# Section 6.5: sensitivity to the initial value (d = 2).
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:gauss, 0.9, "gauss0.9"),
         (:clayton, 2.0, "clayton2"), (:clayton, 8.0, "clayton8")]
K = 500; rows = Any[]; summ = Any[]
for m in (3, 5), (kind, par, kname) in kinds
    ns = (m, m)
    U = sample_copula(kind, K, 2; param = par, seed = 3000)
    Bs = ntuple(k -> basis_matrix(U[k], ns[k]), 2); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns)
    rng = MersenneTwister(77)
    starts = [fill(1.0 / prod(ns), prod(ns)); [random_start(rng, ns) for _ in 1:8]]
    starts = vcat([fill(1.0 / prod(ns), prod(ns))], [random_start(rng, ns) for _ in 1:8])
    for ms in (:sinkhorn, :newton)
        ws = Vector{Vector{Float64}}(); Js = Float64[]; its = Int[]
        for (s, w0) in enumerate(starts)
            r = emfit(Bf, ns, ms; alpha = ALPHA, w0 = w0, abstol = 1e-10, reltol = 1e-13, maxiter = 50_000, tol_inner = 1e-10)
            push!(ws, r.w); push!(Js, r.J); push!(its, r.iter)
            push!(rows, (m, kname, String(ms), s, r.iter, r.conv, r.J))
        end
        maxd = maximum(norm(ws[a] - ws[b]) for a in 1:length(ws) for b in 1:length(ws))
        push!(summ, (m, kname, String(ms), maxd, maximum(Js) - minimum(Js), minimum(its), maximum(its), median(its)))
        @printf("m=%d %-9s %-8s max|dw|=%.2e  range J=%.2e  iters %d..%d\n", m, kname, ms, maxd, maximum(Js) - minimum(Js), minimum(its), maximum(its))
    end
end
write_csv(joinpath(V2RESULTS, "init_runs.csv"), ["m","data","mstep","start","iter","converged","J"], rows)
write_csv(joinpath(V2RESULTS, "init_summary.csv"), ["m","data","mstep","max_pair_dist","J_range","iter_min","iter_max","iter_median"], summ)
