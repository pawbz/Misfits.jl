using Misfits
using Test
using ForwardDiff
using BenchmarkTools
using LinearAlgebra
using Calculus
using LinearMaps

# =================================================
# generalized least-squares
# =================================================
x=abs2.(randn(10,10))
Q=x*x'
pa=Misfits.P_gls(LinearMap(Q))
x=randn(10);
y=randn(10);
dfdx1=similar(x);
@btime Misfits.func_grad!(dfdx1,x,y, pa)
@btime Misfits.func_grad!(nothing,x,y, pa)
f(x)=Misfits.func_grad!(nothing,x,y, pa)
dfdx2=Calculus.gradient(f,x)

@test dfdx1 ≈ dfdx2

# =================================================
# squared_euclidean!
# =================================================
x=randn(1000,10);
y=randn(1000,10);
w=randn(1000,10);
dfdx1=similar(x);
dfdx2=similar(x);

@btime Misfits.error_squared_euclidean!(dfdx1,x,y,w)
@btime Misfits.error_squared_euclidean!(nothing,x,y,w)
f(x)=Misfits.error_squared_euclidean!(nothing,x,y,w)
ForwardDiff.gradient!(dfdx2,f, x);

@test dfdx1 ≈ dfdx2


# =================================================
# error_after_scaling
# =================================================
# real 1D
x = randn(10,1); α = randn(1); y = α[1] .* x;
J, α1 = Misfits.error_after_scaling(x,y)
@test isapprox(α1.*x, y)
#@test_approx_eq J 0.0

# real 2D
x = randn(10,10); α = randn(1); y = α[1] .* x;
J, α1 = Misfits.error_after_scaling(x,y)
@test isapprox(α1.*x, y)

# complex 2D
x = complex.(randn(10,10), randn(10,10));
α = complex.(randn(1), randn(1)); y = α[1] .* x;
J, α1 = Misfits.error_after_scaling(x,y)
@test isapprox(α1.*x, y)


# =================================================
# weighted norm
# =================================================
x=randn(100,10);
w=randn(100,10);
dfdx1=similar(x);
@btime Misfits.error_weighted_norm!(dfdx1,x,w)
xvec=vec(x)
dfdx2=similar(xvec);
#ForwardDiff.gradient!(dfdx2,x -> Misfits.error_weighted_norm!(nothing, reshape(x,100,10), w), xvec);
dfdx2=Calculus.gradient(x -> Misfits.error_weighted_norm!(nothing, reshape(x,100,10), w), xvec)

@test vec(dfdx1) ≈ vec(dfdx2)


# =================================================
# test derivative_vector_magnitude
# =================================================

# some func
function f(x, z)
    y=x./norm(x)
    J=sum((y-z).^2)
    return J
end

# test derivative_vector_magnitude
function g!(g, x, z)
    xn=norm(x)
    rmul!(x, inv(xn))
    g1=similar(g)
    for i in eachindex(g1)
        g1[i]=2. * (x[i]-z[i])
    end
         rmul!(x, xn)
         nx=length(x)
         X=zeros(nx,nx)
         @time Misfits.derivative_vector_magnitude!(g,g1,x,X)
    return g
end

x=randn(10)
z=randn(10)
g1=zero(x)
f1(x)=f(x,z)
ForwardDiff.gradient!(g1,f1, x)
g2=zero(x)
@time g!(g2,x,z)
@test g1 ≈ g2


# error invariant of translation or global phase
#x=randn(100); y=randn() .* circshift(x,20)
#J, α = Misfits.error_after_autocorr_scaling(x,y)
#@test J < 1e-15

#
#x=randn(100,10);
#dfdx1=similar(x);
#@time Misfits.error_pairwise_corr_dist(dfdx1,x)
#xvec=vec(x)
#dfdx2=similar(xvec);
#Inversion.finite_difference!(x -> Misfits.error_pairwise_corr_dist(nothing, reshape(x,100,10)), xvec, dfdx2, :central)
#
#@test dfdx1 ≈ reshape(dfdx2,100,10)
#
#
#x=randn(10,10);
#dfdx1=similar(x);
#Misfits.error_autocorr_pairwise_corr_dist(dfdx1,x)
#xvec=vec(x)
#dfdx2=similar(xvec);
#Inversion.finite_difference!(x -> Misfits.error_autocorr_pairwise_corr_dist(nothing, reshape(x,10,10)), xvec, dfdx2, :central)
#
#@test dfdx1 ≈ reshape(dfdx2,10,10)


