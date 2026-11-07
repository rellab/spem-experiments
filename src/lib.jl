# Shared library for the experiments in
#   H. Okamura, K. Nakayama, T. Dohi, "Sinkhorn Projection EM for Bernstein Copulas".
#
# Self-contained general-d EM harness for the Bernstein copula with the MAP
# objective J_K(w) = l_K(w) + alpha * sum log w_j, alpha > 0.
#
#   M-steps (all project the SAME q = (N + alpha)/S_K onto the constraint set):
#     :sinkhorn   forward KL projection  (SP-EM)                 -- Algorithm 1
#     :dou        reverse KL projection, alternating 1-D bisection (Dou et al.)
#     :newton     reverse KL projection, Newton on the concave dual
#
#   Reference MAP estimate  w_hat : damped Newton on J_K restricted to the
#   tangent space (high precision), used for the accuracy and bound studies.
#
#   Bound quantities of Theorem 1 / Corollary 1: z, eps, eta, tau, r_K,
#   conservative and data-dependent curvature constants, actual curvature.
#
# Notation follows the manuscript: K observations, orders ns = (n_1,...,n_d),
# p = prod(ns) weights, basis b_j(z) = prod_k beta_{i_k, n_k}(u_k).

using LinearAlgebra, Random, Printf, Statistics, Distributions, DelimitedFiles

const V2RESULTS = normpath(joinpath(@__DIR__, "..", "results"))
const V2DATA    = normpath(joinpath(@__DIR__, "..", "data"))

# ---------------------------------------------------------------------------
# basis and E-step
# ---------------------------------------------------------------------------
"""Order-statistic Beta densities beta_{i,n}(u) = n C(n-1,i-1) u^{i-1}(1-u)^{n-i}, i=1..n."""
function basis_matrix(u::AbstractVector, n::Int)
    K = length(u); B = Matrix{Float64}(undef, K, n)
    @inbounds for i in 1:n
        c = n * binomial(n - 1, i - 1)
        for t in 1:K
            ut = clamp(u[t], 1e-12, 1 - 1e-12)
            B[t, i] = c * ut^(i - 1) * (1 - ut)^(n - i)
        end
    end
    B
end

"""Full K x p matrix of basis products b_j(z_t) (column-major over CartesianIndices(ns))."""
function basis_tensor(Bs, ns)
    d = length(ns); K = size(Bs[1], 1); idxs = CartesianIndices(ns)
    Bfull = Matrix{Float64}(undef, K, length(idxs))
    @inbounds for (q, I) in enumerate(idxs), t in 1:K
        v = 1.0
        for k in 1:d; v *= Bs[k][t, I[k]]; end
        Bfull[t, q] = v
    end
    Bfull
end

"""E-step: expected counts N (vector, length p) and log-likelihood l_K(w)."""
function estep(w::AbstractVector, Bfull::AbstractMatrix)
    K, p = size(Bfull)
    N = zeros(p); ll = 0.0
    @inbounds for t in 1:K
        s = 0.0
        for q in 1:p; s += w[q] * Bfull[t, q]; end
        ll += log(s); invs = 1.0 / s
        for q in 1:p; N[q] += w[q] * Bfull[t, q] * invs; end
    end
    N, ll
end

loglik(w, Bfull) = sum(log, Bfull * w)
objJ(w, Bfull, alpha) = loglik(w, Bfull) + alpha * sum(log, w)

# ---------------------------------------------------------------------------
# constraint set helpers
# ---------------------------------------------------------------------------
otherdims(d, k) = Tuple(i for i in 1:d if i != k)
axis_marg(W, k) = vec(sum(W; dims = otherdims(ndims(W), k)))
bshape(d, k, nk) = ntuple(i -> i == k ? nk : 1, d)

"""Full (redundant) constraint matrix A0 (sum_k n_k rows) and rhs; and an
orthonormal basis Bt of the tangent space ker A0."""
function constraint_matrices(ns)
    d = length(ns); idxs = CartesianIndices(ns); p = length(idxs)
    A0 = zeros(sum(ns), p); b0 = zeros(sum(ns)); off = 0
    for k in 1:d
        for (q, I) in enumerate(idxs); A0[off + I[k], q] = 1.0; end
        b0[off+1:off+ns[k]] .= 1.0 / ns[k]; off += ns[k]
    end
    Bt = nullspace(A0)                         # orthonormal columns
    A0, b0, Bt
end

marg_err(W) = maximum(maximum(abs.(axis_marg(W, k) .- 1.0 / size(W, k))) for k in 1:ndims(W))

# ---------------------------------------------------------------------------
# M-steps. Each returns (W, state, inner_iterations).
# ---------------------------------------------------------------------------
function proj_sinkhorn(Q::AbstractArray; u0 = nothing, maxiter = 50_000, tol = 1e-6)
    d = ndims(Q); ns = size(Q)
    u = u0 === nothing ? [ones(ns[k]) for k in 1:d] : deepcopy(u0)
    W = copy(float(Q))
    for k in 1:d; W .*= reshape(u[k], bshape(d, k, ns[k])); end
    it = 0
    for _ in 1:maxiter
        it += 1
        for k in 1:d
            a = (1.0 / ns[k]) ./ axis_marg(W, k)
            W .*= reshape(a, bshape(d, k, ns[k])); u[k] .*= a
        end
        marg_err(W) < tol && break
    end
    W, u, it
end

function _bisect(f, lo, hi; tol = 1e-12, maxit = 200)
    flo = f(lo); fhi = f(hi); ex = 0
    while flo * fhi > 0 && ex < 80
        hi *= 2.0; fhi = f(hi); ex += 1
    end
    flo * fhi > 0 && return NaN
    for _ in 1:maxit
        c = 0.5 * (lo + hi); fc = f(c)
        (abs(fc) < tol || (hi - lo) < tol) && return c
        flo * fc < 0 ? (hi = c; fhi = fc) : (lo = c; flo = fc)
    end
    0.5 * (lo + hi)
end

"""Dou et al. style: cycle over axes, solve each multiplier by 1-D bisection."""
function proj_dou(Q::AbstractArray; lam0 = nothing, maxiter = 5000, tol = 1e-6)
    d = ndims(Q); ns = size(Q)
    lam = lam0 === nothing ? [fill(1.0, ns[k]) for k in 1:d] : deepcopy(lam0)
    Sbuf = similar(float(Q))
    rebuild_S!() = (fill!(Sbuf, 0.0);
        for k in 1:d; Sbuf .+= reshape(lam[k], bshape(d, k, ns[k])); end)
    it = 0
    for _ in 1:maxiter
        it += 1
        for k in 1:d
            rebuild_S!()
            Rrest = Sbuf .- reshape(lam[k], bshape(d, k, ns[k]))
            for j in 1:ns[k]
                idx = ntuple(i -> i == k ? j : Colon(), d)
                Nsl = vec(Q[idx...]); Rsl = vec(Rrest[idx...])
                lo = -minimum(Rsl) + 1e-9
                f = l -> sum(Nsl ./ (l .+ Rsl)) - 1.0 / ns[k]
                lam[k][j] = _bisect(f, lo, max(1e6, 2 * abs(lam[k][j]) + 1.0))
            end
        end
        for k in 2:d
            mk = mean(lam[k]); lam[k] .-= mk; lam[1] .+= mk
        end
        rebuild_S!()
        marg_err(float(Q) ./ Sbuf) < tol && break
    end
    rebuild_S!()
    float(Q) ./ Sbuf, lam, it
end

"""Same reverse-KL projection, solved by Newton on the concave dual."""
function proj_newton(Q::AbstractArray; lam0 = nothing, tol = 1e-6, maxit = 200)
    d = ndims(Q); ns = size(Q)
    off = cumsum([0; collect(ns[1:end-1])]); D = sum(ns)
    vidx(k, j) = off[k] + j
    idxs = collect(CartesianIndices(ns)); Qv = float(Q)
    lam = lam0 === nothing ? ones(D) : copy(lam0)
    Svec(l) = [sum(l[vidx(k, I[k])] for k in 1:d) for I in idxs]
    function gval(l)
        S = Svec(l); any(S .<= 0) && return Inf
        s = 0.0
        for k in 1:d, j in 1:ns[k]; s += l[vidx(k, j)] / ns[k]; end
        s - sum(Qv[q] * log(S[q]) for q in 1:length(idxs))
    end
    it = 0
    for _ in 1:maxit
        it += 1
        S = Svec(lam); g = zeros(D); H = zeros(D, D)
        @inbounds for k in 1:d, j in 1:ns[k]; g[vidx(k, j)] = 1.0 / ns[k]; end
        @inbounds for (q, I) in enumerate(idxs)
            n_s = Qv[q] / S[q]; n_s2 = Qv[q] / S[q]^2
            for k in 1:d
                vk = vidx(k, I[k]); g[vk] -= n_s
                for l in 1:d; H[vk, vidx(l, I[l])] += n_s2; end
            end
        end
        # stop on the SAME criterion as the other solvers: marginal violation
        Wc = similar(Qv); @inbounds for (q, I) in enumerate(idxs); Wc[I] = Qv[q] / S[q]; end
        marg_err(Wc) < tol && break
        Δ = -((H + 1e-10 * I(D)) \ g)
        t = 1.0; g0 = gval(lam); slope = dot(g, Δ)
        while (gnew = gval(lam .+ t .* Δ)) > g0 + 1e-4 * t * slope
            t *= 0.5; t < 1e-14 && break
        end
        lam .+= t .* Δ
    end
    S = Svec(lam); W = similar(Qv)
    @inbounds for (q, I) in enumerate(idxs); W[I] = Qv[q] / S[q]; end
    W, lam, it
end

function mstep(msym::Symbol, Q, state; tol = 1e-6)
    if msym === :sinkhorn
        W, st, it = proj_sinkhorn(Q; u0 = state, tol = tol)
    elseif msym === :dou
        W, st, it = proj_dou(Q; lam0 = state, tol = tol)
    elseif msym === :newton
        W, st, it = proj_newton(Q; lam0 = state, tol = tol)
    else
        error("unknown mstep $msym")
    end
    W ./= sum(W)
    W, st, it
end

# ---------------------------------------------------------------------------
# EM fit (common harness for SP-EM, Dou-EM, Newton-EM)
# ---------------------------------------------------------------------------
"""
EM with the chosen M-step. Stopping (same as the uranium protocol): every
`steps` iterations compare J_K with its value `steps` iterations earlier and
stop when |dJ| <= abstol and |dJ| <= reltol*|J|. Returns a NamedTuple with the
final weights (vector), timings are measured by the caller.
"""
function emfit(Bfull, ns, msym::Symbol; alpha = 1e-3, w0 = nothing,
               maxiter = 2500, steps = 10, abstol = 1e-6, reltol = 1e-8,
               tol_inner = 1e-6, track = false)
    p = prod(ns); K = size(Bfull, 1); SK = K + p * alpha
    w = w0 === nothing ? fill(1.0 / p, p) : copy(vec(w0))
    state = nothing; inner = 0; iter = 0; conv = false; tm = 0.0
    Jprev = -Inf; Jcheck = -Inf; ndec = 0; maxdec = 0.0; traj = Float64[]
    J = NaN
    for it in 1:maxiter
        iter = it
        N, ll = estep(w, Bfull)
        J = ll + alpha * sum(log, w)
        track && push!(traj, J)
        if J < Jprev - 1e-9
            ndec += 1; maxdec = max(maxdec, Jprev - J)
        end
        Jprev = J
        if it % steps == 1 || steps == 1
            if isfinite(Jcheck) && abs(J - Jcheck) <= abstol && abs(J - Jcheck) <= reltol * abs(J)
                conv = true; break
            end
            Jcheck = J
        end
        Q = reshape((N .+ alpha) ./ SK, ns)
        tm += @elapsed (Wn, state, nin) = mstep(msym, Q, state; tol = tol_inner)
        inner += nin
        w = vec(Wn)
    end
    ll = loglik(w, Bfull)
    (w = w, iter = iter, inner = inner, conv = conv, llf = ll, tmstep = tm,
     J = ll + alpha * sum(log, w), ndec = ndec, maxdec = maxdec, traj = traj)
end

"""SP-EM iterated until tau(w) = ||P z(w)|| <= tol_fp (a near-exact fixed point)."""
function spem_fixedpoint(Bfull, ns, Bt; alpha = 1e-3, w0 = nothing, maxiter = 50_000,
                         tol_fp = 1e-11, tol_inner = 1e-10)
    p = prod(ns); K = size(Bfull, 1); SK = K + p * alpha
    w = w0 === nothing ? fill(1.0 / p, p) : copy(vec(w0))
    state = nothing; iter = 0; tau = NaN
    for it in 1:maxiter
        iter = it
        N, _ = estep(w, Bfull)
        q = (N .+ alpha) ./ SK
        z = log.(q ./ w); tau = norm(Bt' * z)
        tau <= tol_fp && break
        Wn, state, _ = mstep(:sinkhorn, reshape(q, ns), state; tol = tol_inner)
        w = vec(Wn)
    end
    (w = w, iter = iter, tau = tau)
end

# ---------------------------------------------------------------------------
# high-precision MAP estimate: damped Newton on J_K over the tangent space
# ---------------------------------------------------------------------------
function map_newton(Bfull, ns, Bt; alpha = 1e-3, w0 = nothing, tol = 1e-12, maxit = 2000)
    p = prod(ns); K = size(Bfull, 1); SK = K + p * alpha
    if w0 === nothing      # warm start: exact EM (Newton M-step) with tight tolerances
        r0 = emfit(Bfull, ns, :newton; alpha = alpha, maxiter = 3000, abstol = 1e-12, reltol = 1e-14, tol_inner = 1e-10)
        w0 = r0.w
    end
    w = copy(vec(w0))
    Jf(x) = any(x .<= 0) ? -Inf : (sum(log, Bfull * x) + alpha * sum(log, x)) / SK
    it = 0; gn = NaN
    for _ in 1:maxit
        it += 1
        c = Bfull * w
        g = (Bfull' * (1.0 ./ c) .+ alpha ./ w) ./ SK            # grad L_K = q/w
        gt = Bt' * g; gn = norm(gt)
        gn <= tol && break
        H = -(Bfull' * (Bfull ./ (c .^ 2)) .+ Diagonal(alpha ./ w .^ 2)) ./ SK
        Ht = Symmetric(Bt' * H * Bt)
        dy = -(Ht \ gt)
        dot(gt, dy) <= 0 && (dy = gt)                              # fallback: ascent direction
        dx = Bt * dy
        t = 1.0; J0 = Jf(w); slope = dot(gt, dy)
        while Jf(w .+ t .* dx) < J0 + 1e-4 * t * slope
            t *= 0.5; t < 1e-20 && break
        end
        t < 1e-20 && break
        w .+= t .* dx
    end
    (w = w, iter = it, gradnorm = gn)
end

"""Check that w_hat is not beaten by any candidate (sanity for the reference solution)."""
function check_reference(what, cands, Bfull, alpha)
    Jh = objJ(what, Bfull, alpha)
    worst = maximum(objJ(c, Bfull, alpha) - Jh for c in cands; init = -Inf)
    worst > 1e-8 && @warn "reference MAP beaten by a candidate" worst
    worst
end

# ---------------------------------------------------------------------------
# quantities of Theorem 1 / Corollary 1
# ---------------------------------------------------------------------------
"""Curvature constants: conservative, data-dependent (Prop. 2), and the actual
lambda_min of the tangent Hessian at w (for diagnostics)."""
function curvature_constants(Bfull, ns, Bt, alpha, w)
    K, p = size(Bfull); SK = K + p * alpha; nmax = maximum(ns)
    mu_cons = alpha * nmax^2 / SK
    # M_t = max_{w in W} w'b(z_t) <= (1/nmax) * sum of the nmax largest b_j(z_t)
    M = [sum(partialsort(view(Bfull, t, :), 1:nmax; rev = true)) / nmax for t in 1:K]
    Aall = (Bt' * (Bfull ./ M)')                        # d x K  (a_t columns)
    GK = (Aall * Aall') ./ K
    lam_GK = eigmin(Symmetric(GK))
    mu_data = (K / SK) * lam_GK + mu_cons
    c = Bfull * w
    H = (Bfull' * (Bfull ./ (c .^ 2)) .+ Diagonal(alpha ./ w .^ 2)) ./ SK
    mu_actual = eigmin(Symmetric(Bt' * H * Bt))
    (mu_cons = mu_cons, mu_data = mu_data, mu_actual = mu_actual, lam_GK = lam_GK)
end

"""Residual quantities at w: z, eps, eta, tau, r_K = ||P grad L_K||, R_K."""
function residuals(Bfull, ns, Bt, alpha, w)
    K, p = size(Bfull); SK = K + p * alpha
    N, _ = estep(w, Bfull)
    q = (N .+ alpha) ./ SK
    z = log.(q ./ w)
    eps = norm(z); eta = maximum(abs, z); tau = norm(Bt' * z)
    g = q ./ w
    rK = norm(Bt' * g)
    RK = tau + exp(eta) / 2 * eps^2
    (eps = eps, eta = eta, tau = tau, rK = rK, RK = RK)
end

"""Normalized objective L_K and density discrepancies on a grid (d = 2 or 3)."""
function density_diff(w1, w2, ns; G = 50)
    d = length(ns)
    gx = ((1:G) .- 0.5) ./ G
    Bg = ntuple(k -> basis_matrix(collect(gx), ns[k]), d)
    idxs = CartesianIndices(ns)
    pts = Iterators.product(ntuple(_ -> 1:G, d)...)
    supd = 0.0; l2 = 0.0; cnt = 0
    W1 = reshape(w1, ns); W2 = reshape(w2, ns)
    for P in pts
        s1 = 0.0; s2 = 0.0
        for I in idxs
            v = 1.0
            for k in 1:d; v *= Bg[k][P[k], I[k]]; end
            s1 += W1[I] * v; s2 += W2[I] * v
        end
        dd = abs(s1 - s2); supd = max(supd, dd); l2 += dd^2; cnt += 1
    end
    (sup = supd, l2 = sqrt(l2 / cnt))
end

# ---------------------------------------------------------------------------
# synthetic copula samples (exactly on the copula scale)
# ---------------------------------------------------------------------------
function sample_copula(kind::Symbol, K::Int, d::Int; param = 0.0, seed = 42)
    rng = MersenneTwister(seed)
    if kind === :indep
        return [rand(rng, K) for _ in 1:d]
    elseif kind === :gauss
        rho = param
        Σ = fill(rho, d, d); for i in 1:d; Σ[i, i] = 1.0; end
        L = cholesky(Σ).L
        Z = (L * randn(rng, d, K))'
        return [cdf.(Normal(), Z[:, k]) for k in 1:d]
    elseif kind === :clayton
        θ = param
        V = rand(rng, Gamma(1 / θ, 1.0), K)
        return [(1 .+ rand(rng, Exponential(), K) ./ V) .^ (-1 / θ) for _ in 1:d]
    else
        error("unknown copula $kind")
    end
end

"""Random feasible start: uniform random matrix projected by Sinkhorn (shared by all methods)."""
function random_start(rng, ns)
    W = rand(rng, ns...) .+ 1e-12
    W, _, _ = proj_sinkhorn(W; tol = 1e-12, maxiter = 100_000)
    vec(W ./ sum(W))
end

# ---------------------------------------------------------------------------
# uranium data
# ---------------------------------------------------------------------------
function load_uranium()
    raw, header = readdlm(joinpath(V2DATA, "uranium.csv"), ',', Float64, '\n'; header = true)
    names = String.(strip.(replace.(vec(header), '"' => "")))
    raw, names
end

"""Mid-rank pseudo-observations u_t = R_t/(K+1), ties get the average rank."""
function pseudo_obs(x::AbstractVector)
    K = length(x); r = zeros(K)
    ord = sortperm(x); i = 1
    while i <= K
        j = i
        while j < K && x[ord[j+1]] == x[ord[i]]; j += 1; end
        r[ord[i:j]] .= (i + j) / 2
        i = j + 1
    end
    r ./ (K + 1)
end

# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
function write_csv(path, header::Vector{String}, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(string.(r), ","))
        end
    end
    println("# wrote ", path)
end

# ---------------------------------------------------------------------------
# self-check: the three M-step solvers and the reference MAP solver agree
# ---------------------------------------------------------------------------
function self_check()
    Random.seed!(42); ed = 0.0
    for _ in 1:5
        N = rand(4, 3) .+ 0.1
        Wd = first(proj_dou(N; tol = 1e-13, maxiter = 50_000))
        ed = max(ed, norm(first(proj_newton(N ./ sum(N); tol = 1e-12, maxit = 300)) .- Wd))
    end
    @printf("[self-check d=2]  Dou vs Newton reverse-KL projection: %.1e\n", ed)
    @assert ed < 1e-6
    # the exact projections agree with each other and with the MAP Newton on a tiny problem
    U = sample_copula(:gauss, 200, 2; param = 0.5)
    ns = (3, 3); Bs = ntuple(k -> basis_matrix(U[k], ns[k]), 2); Bf = basis_tensor(Bs, ns)
    _, _, Bt = constraint_matrices(ns)
    r1 = emfit(Bf, ns, :newton; maxiter = 20000, abstol = 1e-13, reltol = 1e-15, tol_inner = 1e-12)
    r2 = map_newton(Bf, ns, Bt)
    @printf("  EM(Newton M-step) vs direct MAP Newton: |dw|=%.1e  |dJ|=%.1e\n",
            norm(r1.w - r2.w), abs(r1.J - objJ(r2.w, Bf, 1e-3)))
    @assert norm(r1.w - r2.w) < 1e-6
    println("  => harness consistent.")
end
