module QCTutorial

using LinearAlgebra

export ⊗, ∧
export shift
export Ket, @ket_str
export Bra, @bra_str
export Gate, H, X, I

const ⊗ = kron
const ℂ = ComplexF64

const DISPLAY_PRECISION = 3

abstract type State end

struct Ket <: State
    vector::Vector{ℂ}
end

function Ket(v::AbstractVector{T}) where {T<:Number}
    @assert ispow2(length(v))

    return Ket(convert(Vector{ℂ}, v))
end

struct Bra <: State
    vector::Adjoint{ℂ,Vector{ℂ}}
end

function Bra(v::Adjoint{T,V}) where {T<:Number,V<:AbstractVector{T}}
    @assert ispow2(length(v))

    return Bra(adjoint(convert(Vector{ℂ}, v)))
end

vector(ψ::Ket) = ψ.vector
vector(ψ::Bra) = ψ.vector

adjoint(ψ::Ket) = Bra(vector(ψ)')
adjoint(ψ::Bra) = Ket(vector(ψ)')

function Base.length(ψ::State)
    return length(vector(ψ))
end

function nbits(ψ::State)
    return round(Int, log2(length(ψ)))
end

function Ket(k::Integer, n::Integer = 1)
    @assert 0 ≤ k < 2^n

    v = zeros(2^n)

    v[k + 1] = 1

    return Ket(v)
end

function Bra(k::Integer, n::Integer = 1)
    @assert 0 ≤ k < 2^n

    v = zeros(2^n)

    v[k + 1] = 1

    return Bra(v)
end

function Ket(s::AbstractString)
    n = length(s)
    i = parse(Int, s; base = 2)

    return Ket(i, n)
end

function Bra(s::AbstractString)
    n = length(s)
    i = parse(Int, s; base = 2)

    return Bra(i, n)
end

macro ket_str(bits::AbstractString)
    return quote
        Ket($(esc(bits)))
    end
end

macro bra_str(bits::AbstractString)
    return quote
        Bra($(esc(bits)))
    end
end

function Base.show(io::IO, ψ::Ket)
    n = nbits(ψ)

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]
        c = round(α; digits = DISPLAY_PRECISION)

        if !iszero(c)
            s = if isreal(c)
                if isone(real(c))
                    ""
                else
                    "$(real(c))"
                end
            elseif isimag(c)
                "$(imag(c))im"
            end

            push!(basis, "$(s) |$(join(reverse!(digits(i - 1; base = 2, pad = n))))⟩")
        end
    end

    return join(io, basis, " + ")
end

function Base.show(io::IO, ψ::Bra)
    n = nbits(ψ)

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]
        c = round(α; digits = DISPLAY_PRECISION)

        if !iszero(c)
            s = if isreal(c)
                if isone(real(c))
                    ""
                else
                    "$(real(c))"
                end
            elseif isimag(c)
                "$(imag(c))im"
            end

            push!(basis, "$(s) ⟨$(join(reverse!(digits(i - 1; base = 2, pad = n))))|")
        end
    end

    return join(io, basis, " + ")
end

function Base.show(io::IO, ::MIME"text/latex", ψ::Ket)
    n = round(Int, log2(length(ψ.vector)))

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]
        c = round(α; digits = DISPLAY_PRECISION)

        if !iszero(c)
            s = if isreal(c)
                if isone(real(c))
                    ""
                else
                    "$(real(c))"
                end
            elseif isimag(c)
                "$(imag(c))im"
            end

            push!(basis, "$(s) \\vert{}$(join(reverse!(digits(i - 1; base = 2, pad = n))))\\rangle{}")
        end
    end

    return print(io, "\$", join(basis, " + "), "\$")
end

function Base.show(io::IO, ::MIME"text/latex", ψ::Bra)
    n = round(Int, log2(length(ψ.vector)))

    basis = []

    isimag(x) = !isreal(x) && iszero(real(x))

    for i = eachindex(ψ.vector)
        α = ψ.vector[i]
        c = round(α; digits = DISPLAY_PRECISION)

        if !iszero(c)
            s = if isreal(c)
                if isone(real(c))
                    ""
                else
                    "$(real(c))"
                end
            elseif isimag(c)
                "$(imag(c))im"
            end

            push!(basis, "$(s) \\langle{}$(join(reverse!(digits(i - 1; base = 2, pad = n))))\\vert{}")
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

const _I = Gate([1 0; 0 1])
const _H = Gate((1 / √2) * [1 1; 1 -1])
const _X = Gate([0 1; 1 0])

function gate(k::Integer, U::Gate)
    if k < 0
        return inv(H(-k))
    elseif k == 0
        return _I
    elseif k == 1
        return U
    elseif k % 2 == 0
        V = gate(k ÷ 2, U)

        return V ⊗ V
    else # k % 2 == 1
        V = gate(k ÷ 2, U)

        return V ⊗ V ⊗ U
    end
end

I(n::Integer) = gate(n, _I)
H(n::Integer) = gate(n, _H)
X(n::Integer) = gate(n, _X)

function Base.kron(ψ::S, ϕ::S) where {S<:State}
    return S(kron(vector(ψ), vector(ϕ)))
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

function Base.:(*)(ψ::S, c::Number) where {S<:State}
    return S(ψ.vector * c)
end

function Base.:(*)(c::Number, ψ::S) where {S<:State}
    return S(ψ.vector * c)
end

function Base.:(*)(ϕ::Bra, ψ::Ket)
    return vector(ϕ) * vector(ψ)
end

function Base.:(*)(U::Gate, ψ::Ket)
    return Ket(U.matrix * vector(ψ))
end

function Base.:(*)(ψ::Bra, U::Gate)
    return Bra(vector(ψ) * U.matrix)
end

function Base.:(*)(U::Gate, V::Gate)
    return Gate(U.matrix * V.matrix)
end

function Base.:(+)(ψ::S, ϕ::S) where {S<:State}
    return S(vector(ψ) + vector(ϕ))
end

function Base.:(+)(U::Gate, V::Gate)
    return Gate(U.matrix + V.matrix)
end

function Base.:(-)(ψ::S, ϕ::S) where {S<:State}
    return S(ψ.vector - ϕ.vector)
end

function Base.:(-)(U::Gate, V::Gate)
    return Gate(U.matrix - V.matrix)
end

function outer(ψ::Ket, ϕ::Bra)
    @assert length(ψ) == length(ϕ) "Dimension mismatch"

    return Gate(vector(ψ) * vector(ϕ))
end

function outer(ψ::S, c::Number) where {S<:State}
    return S(vector(ψ) * c)
end

function outer(c::Number, ψ::S) where {S<:State}
    return S(vector(ψ) * c)
end

const ∧ = outer

# Shift Operator
function shift(n::Integer)
    σ₋ = ket"0" ∧ bra"0"
    σ₊ = ket"1" ∧ bra"1"
    
    m = ceil(Int, log2(n))
    
    Δ₋ = zeros(ComplexF64, 2^m, 2^m)
    Δ₊ = zeros(ComplexF64, 2^m, 2^m)


    for i = 0:(n-1)
        δᵢ = Bra(i, m) # ⟨i|
        
        if i > 0
            δ₋ = Ket(i-1, m) # |i-1⟩
            Δ₋ .+= (δ₋ ∧ δᵢ).matrix
        end
        
        if i < n - 1
            δ₊ = Ket(i+1, m) # |i+1⟩
            Δ₊ .+= (δ₊ ∧ δᵢ).matrix
        end
    end
        
    return σ₋ ⊗ Gate(Δ₋) + σ₊ ⊗ Gate(Δ₊)
end

end # module QCTuorial
