using Random
using LinearAlgebra
using Plots

const ⊗ = kron

H = (1 / √2) * [1 1; 1 -1]

I = [1 0; 0 1]

s0 = [1, 0]
s1 = [0, 1]

function prob(x, i)
    return x[i] * x[i]'
end

function measure(x, n)
    p = rand() # ∈ [0, 1]
    P = cumsum([prob(x, i) for i = 1:2^n])

    i = searchsortedfirst(P, p)

    return i
    # return digits(i - 1; base = 2, pad = n)
end

function measure(x, n, N)
    return [measure(x, n) for _ = 1:N]
end

function show_results(X, n)
    bins = ["|$(join(reverse!(digits(i - 1; base = 2, pad = n))))⟩" for i = 1:2^n]

    histogram(X, label="Samples", bins = 2^n, xlims=(1,2^n+1), xticks=(1:2^n, bins))
end