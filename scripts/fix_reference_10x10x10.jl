# One-off correction (2026-11-20).  Before this date constraint_matrices returned a tangent basis
# with 971 instead of 972 columns for the 10x10x10 order (rank misjudged by nullspace), so the
# reference MAP estimate of those configurations was computed on a slightly too small subspace.
# The EM fits and their timings do not use the tangent basis.  This script recomputes, with the
# corrected basis, the columns of runtime.csv and large.csv that depend on the reference
# (err_w, gapJ_norm, dens_sup, dens_l2, map_gradnorm, ref_worst) for the 10x10x10 rows, checks
# that the refitted EM solutions reproduce the stored iter / llf / J, and leaves every other
# entry, including all timings, unchanged.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
kinds = [(:indep, 0.0, "indep"), (:gauss, 0.5, "gauss0.5"), (:clayton, 2.0, "clayton2")]
ns = (10, 10, 10); d = 3; p = prod(ns)
_, _, Bt = constraint_matrices(ns)
function patch!(file, Ks, seedf, cols, withdens)
    path = joinpath(V2RESULTS, file); L = readlines(path); h = split(L[1], ","); ix(c) = findfirst(==(c), h)
    for (kind, par, kname) in kinds, K in Ks
        U = sample_copula(kind, K, d; param = par, seed = seedf(K))
        Bs = ntuple(k -> basis_matrix(U[k], ns[k]), d); Bf = basis_tensor(Bs, ns); SK = K + p * ALPHA
        fits = Dict(ms => emfit(Bf, ns, ms; alpha = ALPHA) for ms in (:sinkhorn, :dou, :newton))
        mp = map_newton(Bf, ns, Bt; alpha = ALPHA)
        worst = check_reference(mp.w, [fits[ms].w for ms in keys(fits)], Bf, ALPHA); Jhat = objJ(mp.w, Bf, ALPHA)
        for ms in (:sinkhorn, :dou, :newton)
            r = fits[ms]
            li = findfirst(l -> (f = split(l, ","); f[ix("data")] == kname && f[ix("K")] == string(K) && f[ix("size")] == "10x10x10" && f[ix("mstep")] == String(ms)), L)
            f = split(L[li], ",")
            @assert f[ix("iter")] == string(r.iter) && isapprox(parse(Float64, f[ix("J")]), r.J; rtol = 1e-10) && isapprox(parse(Float64, f[ix("llf")]), r.llf; rtol = 1e-10) "refit differs from stored fit: $file $kname K=$K $ms"
            new = Dict("err_w" => norm(r.w - mp.w), "gapJ_norm" => (Jhat - r.J) / SK, "map_gradnorm" => mp.gradnorm, "ref_worst" => worst)
            if withdens; dd = density_diff(r.w, mp.w, ns; G = 16); new["dens_sup"] = dd.sup; new["dens_l2"] = dd.l2; end
            @printf("%-12s %-9s K=%-4d %-8s err_w %s -> %.6e   gap %s -> %.6e\n", file, kname, K, ms, f[ix("err_w")], new["err_w"], f[ix("gapJ_norm")], new["gapJ_norm"])
            for c in cols; f[ix(c)] = string(new[c]); end
            L[li] = join(f, ",")
        end
    end
    open(path, "w") do io; for l in L; println(io, l); end; end
    println("# patched ", file)
end
patch!("runtime.csv", (100, 1000), K -> 1000 + K, ["err_w", "gapJ_norm", "dens_sup", "dens_l2", "map_gradnorm", "ref_worst"], true)
patch!("large.csv", (500,), K -> 5000 + d, ["err_w", "gapJ_norm"], false)
