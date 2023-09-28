using Random
using LinearAlgebra
using Plots

const ⊗ = kron

H = (1 / √2) * [1 1; 1 -1]
X = [0 1; 1 0]
I = [1 0; 0 1]

C(U) =  [[I; zeros(2, 2)];;[zeros(2, 2); U]]

CX = C(X)
CH = C(H)

s0 = [1, 0]
s1 = [0, 1]

function prob(x, i::Integer)
    return x[i] * x[i]'
end

function measure(x)
    N = length(x)
    p = rand() # ∈ [0, 1]
    P = cumsum([prob(x, i) for i = 1:N])

    i = searchsortedfirst(P, p)

    return i
end

function measure(x, N)
    return [measure(x) for _ = 1:N]
end

function show_results(X, n::Integer)
    bins = ["|$(join(reverse!(digits(i - 1; base = 2, pad = n))))⟩" for i = 1:2^n]

    histogram(X, label="Samples", bins = 2^n, xlims=(1,2^n+1), xticks=(1:2^n, bins))
end