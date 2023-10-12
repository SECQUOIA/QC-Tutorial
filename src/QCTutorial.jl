module QCTutorial

using LinearAlgebra

export ⊗
export State, Gate, H, X, I

const ⊗ = kron
const ℂ = ComplexF64

struct State
    vector::Vector{ℂ}
end

function State(v::Vector{T}) where {T<:Number}
    @assert ispow2(length(v))

    return State(convert(Vector{ℂ}, v))
end

function State(k::Integer, n::Integer)
    @assert 0 ≤ k < 2^n

    v = zeros(2^n)
    v[k + 1] = 1

    return State(v)
end

function Base.show(io::IO, ψ::State)
    n = round(Int, log2(length(ψ.vector)))

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]

        if α != 0
            if isreal(α)
                c = real(α)
            end

            if isimag(α)
                c = "$(imag(α))im"
            end

            push!(basis, "$(c) |$(join(reverse!(digits(i - 1; base = 2, pad = n))))⟩")
        end
    end

    return join(io, basis, " + ")
end

function Base.show(io::IO, ::MIME"text/latex", ψ::State)
    n = round(Int, log2(length(ψ.vector)))

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]

        if α != 0
            if isreal(α)
                c = real(α)
            end

            if isimag(α)
                c = "$(imag(α)) i"
            end

            push!(basis, "$(c) \\vert{}$(join(reverse!(digits(i - 1; base = 2, pad = n))))\\rangle{}")
        end
    end

    return print(io, "\$", join(basis, " + "), "\$")
end

struct Gate
    matrix::Matrix{ℂ}
end

function Gate(M::Matrix{T}) where {T<:Number}
    @assert ispow2(size(M, 1))
    @assert ispow2(size(M, 2))

    return Gate(convert(Matrix{ℂ}, M))
end

const _H = Gate((1 / √2) * [1 1; 1 -1])
const _X = Gate([0 1; 1 0])
const _I = Gate([1 0; 0 1])

function H(n::Integer)
    if n < 0
        return inv(H(-n))
    elseif n == 0
        return _I
    elseif n == 1
        return _H
    else
        return _H ⊗ H(n - 1)
    end
end

function Base.kron(ψ::State, ϕ::State)
    return State(kron(ψ.vector, ϕ.vector))
end

function Base.kron(U::Gate, V::Gate)
    return Gate(kron(U.matrix, V.matrix))
end

function Base.:(*)(U::Gate, c::Number)
    return Gate(U.matrix * c)
end

function Base.:(*)(c::Number, U::Gate)
    return Gate(U.matrix * c)
end

function Base.:(*)(ψ::State, c::Number)
    return State(ψ.vector * c)
end

function Base.:(*)(c::Number, ψ::State)
    return State(ψ.vector * c)
end

function Base.:(*)(U::Gate, ψ::State)
    return State(U.matrix * ψ.vector)
end

function Base.:(*)(U::Gate, V::Gate)
    return Gate(U.matrix * V.matrix)
end

function Base.:(+)(ψ::State, ϕ::State)
    return State(ψ.vector + ϕ.vector)
end

function Base.:(+)(U::Gate, V::Gate)
    return Gate(U.matrix + V.matrix)
end

function Base.:(-)(ψ::State, ϕ::State)
    return State(ψ.vector - ϕ.vector)
end

function Base.:(-)(U::Gate, V::Gate)
    return Gate(U.matrix - V.matrix)
end

end # module QCTuorial
