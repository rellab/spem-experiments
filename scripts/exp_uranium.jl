# Section 6.6: Cook-Johnson uranium exploration data (7 elements, K = 655).
# Protocol follows Nakayama (2026): mid-rank pseudo-observations, shared random
# start per (pair, m), alpha = 1e-3, common stopping rule, repeats with AB/BA order.
include(joinpath(@__DIR__, "..", "src", "lib.jl"))
const ALPHA = 1e-3
X, names = load_uranium(); K = size(X, 1); nv = length(names)
U = [pseudo_obs(X[:, k]) for k in 1:nv]
@printf("uranium: K=%d  variables=%s\n", K, join(names, ","))
kendall(u, v) = begin  # O(K^2) is fine for K = 655
    s = 0; n = length(u)
    for i in 1:n-1, j in i+1:n; s += sign((u[i]-u[j])*(v[i]-v[j])); end
    2s / (n * (n - 1))
end
REPEATS = 3
rows = Any[]; brows = Any[]
function fitpair(Bf, ns, w0, ms)
    t = @elapsed r = emfit(Bf, ns, ms; alpha = ALPHA, w0 = w0)
    t, r
end
pairs = [(i, j) for i in 1:nv for j in i+1:nv]
for (i, j) in pairs, m in (3, 5, 7)
    ns = (m, m); rng = MersenneTwister(20250301 + 100i + j + m)
    Bs = (basis_matrix(U[i], m), basis_matrix(U[j], m)); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + m^2 * ALPHA
    w0 = random_start(rng, ns)
    tau_k = kendall(U[i], U[j])
    for ms in (:sinkhorn, :dou, :newton); fitpair(Bf, ns, w0, ms); end      # warm-up
    res = Dict{Symbol,Any}(); times = Dict(ms => Float64[] for ms in (:sinkhorn, :dou, :newton))
    for rep in 1:REPEATS
        order = isodd(rep) ? (:dou, :newton, :sinkhorn) : (:sinkhorn, :newton, :dou)
        for ms in order
            t, r = fitpair(Bf, ns, w0, ms); push!(times[ms], t); res[ms] = r
        end
    end
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Jhat = objJ(mp.w, Bf, ALPHA)
    check_reference(mp.w, [res[ms].w for ms in keys(res)], Bf, ALPHA)
    pair = names[i] * "-" * names[j]
    for ms in (:sinkhorn, :dou, :newton)
        r = res[ms]
        push!(rows, (pair, m, tau_k, String(ms), median(times[ms]), r.tmstep, r.iter, r.inner, r.conv, r.llf, r.J, r.ndec, r.maxdec,
                     norm(r.w - mp.w), (Jhat - r.J) / SK, abs(r.llf - res[:newton].llf) / K))
    end
    # bound quantities at the SP-EM stopped iterate and at the exact fixed point
    rr = residuals(Bf, ns, Bt, ALPHA, res[:sinkhorn].w); cc = curvature_constants(Bf, ns, Bt, ALPHA, mp.w)
    fp = spem_fixedpoint(Bf, ns, Bt; alpha = ALPHA, w0 = res[:sinkhorn].w)
    rf = residuals(Bf, ns, Bt, ALPHA, fp.w)
    push!(brows, (pair, m, tau_k, rr.eps, rr.eta, rr.tau, rr.RK, norm(res[:sinkhorn].w - mp.w), rr.RK / cc.mu_data,
                  rf.eps, rf.eta, rf.rK, norm(fp.w - mp.w), exp(rf.eta) * rf.eps^2 / (2cc.mu_data),
                  (Jhat - objJ(fp.w, Bf, ALPHA)) / SK, exp(2rf.eta) * rf.eps^4 / (8cc.mu_data), cc.mu_cons, cc.mu_data, cc.mu_actual))
    @printf("%-6s m=%d tau=%+.3f  t: SP=%.3f Dou=%.3f Newton=%.3f  iter %d/%d/%d  |dl|/K=%.1e  eps=%.3f err=%.2e bound=%.2e\n",
            pair, m, tau_k, median(times[:sinkhorn]), median(times[:dou]), median(times[:newton]),
            res[:sinkhorn].iter, res[:dou].iter, res[:newton].iter, abs(res[:sinkhorn].llf - res[:newton].llf) / K,
            rf.eps, norm(fp.w - mp.w), exp(rf.eta) * rf.eps^2 / (2cc.mu_data))
end
write_csv(joinpath(V2RESULTS, "uranium_pairs.csv"),
          ["pair","m","tau","mstep","time_s","time_mstep_s","iter","inner","converged","llf","J","ndec","maxdec","err_w","gapJ_norm","dllf_per_K"], rows)
write_csv(joinpath(V2RESULTS, "uranium_bound.csv"),
          ["pair","m","tau","eps_stop","eta_stop","tau_stop","RK_stop","err_stop","bound_stop",
           "eps_fp","eta_fp","rK_fp","err_fp","bound_w_fp","gap_fp","bound_L_fp","mu_cons","mu_data","mu_actual"], brows)
# trivariate sub-basket {Co, Sc, Ti}
sub = [findfirst(==(s), names) for s in ("Co", "Sc", "Ti")]
rows3 = Any[]
for m in (3, 5)
    ns = (m, m, m); rng = MersenneTwister(4242 + m)
    Bs = ntuple(k -> basis_matrix(U[sub[k]], m), 3); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns); SK = K + m^3 * ALPHA
    w0 = random_start(rng, ns)
    for ms in (:sinkhorn, :dou, :newton); fitpair(Bf, ns, w0, ms); end
    res = Dict{Symbol,Any}(); times = Dict(ms => Float64[] for ms in (:sinkhorn, :dou, :newton))
    for rep in 1:REPEATS, ms in (isodd(rep) ? (:dou, :newton, :sinkhorn) : (:sinkhorn, :newton, :dou))
        t, r = fitpair(Bf, ns, w0, ms); push!(times[ms], t); res[ms] = r
    end
    mp = map_newton(Bf, ns, Bt; alpha = ALPHA); Jhat = objJ(mp.w, Bf, ALPHA)
    check_reference(mp.w, [res[ms].w for ms in keys(res)], Bf, ALPHA)
    for ms in (:sinkhorn, :dou, :newton)
        r = res[ms]
        push!(rows3, ("Co-Sc-Ti", m, String(ms), median(times[ms]), r.tmstep, r.iter, r.inner, r.conv, r.llf, r.J, r.ndec, r.maxdec,
                      norm(r.w - mp.w), (Jhat - r.J) / SK, abs(r.llf - res[:newton].llf) / K))
        @printf("Co-Sc-Ti m=%d %-8s t=%.3f iter=%d inner=%d |w-what|=%.2e dJ/S=%.2e\n", m, ms, median(times[ms]), r.iter, r.inner, norm(r.w - mp.w), (Jhat - r.J) / SK)
    end
end
write_csv(joinpath(V2RESULTS, "uranium_d3.csv"),
          ["set","m","mstep","time_s","time_mstep_s","iter","inner","converged","llf","J","ndec","maxdec","err_w","gapJ_norm","dllf_per_K"], rows3)

# monotonicity of J_K along the iteration with an accurately solved M-step (inner tol 1e-10)
mrows = Any[]
for (i, j) in pairs, m in (3, 5)
    ns = (m, m); rng = MersenneTwister(20250301 + 100i + j + m)
    Bs = (basis_matrix(U[i], m), basis_matrix(U[j], m)); Bf = basis_tensor(Bs, ns)
    w0 = random_start(rng, ns)
    for ms in (:sinkhorn, :newton)
        r = emfit(Bf, ns, ms; alpha = ALPHA, w0 = w0, tol_inner = 1e-10)
        push!(mrows, (names[i] * "-" * names[j], m, String(ms), r.iter, r.conv, r.ndec, r.maxdec))
    end
end
write_csv(joinpath(V2RESULTS, "uranium_monotone.csv"), ["pair","m","mstep","iter","converged","ndec","maxdec"], mrows)
