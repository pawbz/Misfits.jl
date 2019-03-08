module Misfits

using ForwardDiff
using Distances
using StatsBase
using LinearAlgebra
using LinearMaps


"""
Distance between x[:,i] and x[:,j]
cosine_dist(x, y)=1 - dot(x, y) / (norm(x) * norm(y))	
corr_dist(x, y)=cosine_dist(x - mean(x), y - mean(y))	
"""
function error_pairwise_corr_dist(dfdx,x)
	R=pairwise(CorrDist(), x)
	J=sum(R)

	if(!(dfdx===nothing))
		f(x)=error_pairwise_corr_dist(nothing,x)
		ForwardDiff.gradient!(dfdx, f, x);
	end
	return J
end

function error_corr_dist!(dfdx, x, y)
	xn=norm(x)
	yn=norm(y)

	dotxy=0.0
	for i in eachindex(x)
		dotxy+=x[i]*y[i]
	end

	J = 1. - (dotxy/xn/yn)
	
	if(!(dfdx===nothing))
		for i in eachindex(x)
			dfdx[i]=(dotxy*yn*x[i]/xn - y[i]*yn*xn)/(xn*yn)/(xn*yn)
		end
	end
	return J
end


function error_autocorr_pairwise_corr_dist(dfdx,x)
	X=fft(x, [1]);
	# remove phase
	#XC=real(X .* conj(X))
	J=error_pairwise_corr_dist(nothing, X)
	J=norm(J)

	if(!(dfdx===nothing))
		GXC=similar(X)
		error_pairwise_corr_dist(GXC,X)
		#GX = GXC .* conj(X)
		dfdx[:]=real.(ifft(GXC, [1]))
	end

	return J
end


function error_weighted_norm(dfdx,x,w)
	J=0.0
	for i in eachindex(x)
		J += (x[i]*w[i])*(x[i]*w[i])
	end
	J=sqrt(J)
	if(!(dfdx===nothing))
		f(x)=error_weighted_norm(nothing,x,w)
		ForwardDiff.gradient!(dfdx, f, x);
	end
	return J
end



"""
Compute the error with a normalized vector ̂̂x 
̂x=x/|x|

function 
* `ghat` : gradient with respect to ̂x
* `g`	 : output gradient with respect to x
* `X`	 : preallocated matrix of dimension (nx,nx) if necessary
"""
function derivative_vector_magnitude!(g,ghat,x,X=nothing) 
	xn=norm(x)
	nx=length(x)
	rmul!(x, inv(xn))  

	# compute the outer product of 
	if(!(X===nothing))      
		mul!(X,x,transpose(x))        
	else    
		X=x*transpose(x)
	end         
	for i in 1:nx
		X[i,i]=X[i,i]-1. 
	end
	rmul!(X,-inv(xn))
	mul!(g,X,ghat)    
	rmul!(x, xn) 
end

"""
Normalized least-squares error between two arrays after 
estimating a scalar that best fits on to another.
Return misfit and α such that αx-y is minimum.
Normalization is done with respect to the 
norm of y.
"""
function error_after_scaling(
			     x::AbstractArray{T},
			     y::AbstractArray{T}
			     ) where {T}
	any(size(x) ≠ size(y)) && error("x and y different sizes") 
	sxx=T(0.0)
	sxy=T(0.0)
	for i in eachindex(x)
		sxx += x[i]*x[i]
		sxy += x[i]*y[i]
	end
	α = sxy * inv(sxx)
	
	if(!(iszero(α)))
		rmul!(x, α)
		J = error_squared_euclidean!(nothing,  x,   y,   nothing, norm_flag=true)
		rmul!(x, inv(α))
	else
		J = zero(T)
	end

	return J, α
end


"""
Measure the least-squares distance between auto correlations of x and y, 
"""
function error_after_normalized_autocor(x::AbstractArray{Float64}, y::AbstractArray{Float64})
	nt=size(x,1)

	ax=autocor(x, 0:nt-1, demean=true) # normalizd autocorr from StatsBase
	ay=autocor(y, 0:nt-1, demean=true)

	return error_squared_euclidean!(nothing,  ax,   ay, nothing, norm_flag=true)
end



"""
Compute the L2 distance between two arrays: ``\\sqrt{\\sum_{i=1}^n |a_i - b|^2}``.
* `norm_flag` : optional; normalizes the distance with the norm of `y` (don't use in the inner loops)
"""
function error_squared_euclidean!(dfdx,  x,   y,   w; norm_flag=false)
	J=zero(eltype(x))
	if(w===nothing)
		for i in eachindex(x)
			@inbounds J += abs2(x[i]-y[i])
		end
	else
		for i in eachindex(x)
			@inbounds J += w[i] * abs2(x[i]-y[i])
		end
	end
	if(!(dfdx === nothing))
		if(w===nothing)
			for i in eachindex(x)
				@inbounds dfdx[i] = 2.0 * (x[i]-y[i])
			end
		else
			for i in eachindex(x)
				@inbounds dfdx[i] = 2.0 * w[i] * (x[i]-y[i])
			end
		end
	end
	if(norm_flag) # second norm of y
		ynorm=zero(Float64)
		for i in eachindex(y)
			@inbounds ynorm += abs2(y[i])
		end
		# divide the functional with ynorm
		J /= ynorm
		if(!(dfdx === nothing)) 
			rmul!(dfdx, inv(ynorm)) # divide dfdx with ynorm too
		end
	end
	return J
end

"""
Computed the weighted norm of 
"""
function error_weighted_norm!(dfdx,  x,   w)
	J=zero(Float64)
	if(w===nothing)
		for i in eachindex(x)
			J += (x[i]) * (x[i])
		end
	else
		for i in eachindex(x)
			J += w[i] * (x[i]) * (x[i])
		end
	end
	if(!(dfdx === nothing))
		if(w===nothing)
			for i in eachindex(x)
				dfdx[i] = 2.0 * (x[i])
			end
		else
			for i in eachindex(x)
				dfdx[i] = 2.0 * w[i] * (x[i])
			end
		end
	end
	return J
end

"""
Calculate the front load of dfdx
"""
function front_load!(dfdx,  x::AbstractMatrix)
	nt=size(x,1)
	nr=size(x,2)
	J=zero(Float64)
	for ir in 1:nr
		for it in 1:nt
			J += abs2(x[it,ir] * inv(nt-1) * (it-1))
		end
	end
	if(!(dfdx === nothing))
		for ir in 1:nr
			for it in 1:nt
				dfdx[it,ir] = 2.0 * (x[it,ir]) * 
					abs2(inv(nt-1) * (it-1))
			end
		end
	end
	return J
end

"""
Compute error b/w g1 and g2 after ingnoring time translation
allocates a lot of memory, dont use for big data
"""
function error_after_translation(g1,g2)
	err=[]
	for is in 1:size(g1,1)
		g11=circshift(g1,(is,0))
		push!(err,Misfits.error_after_scaling(g11,g2)[1])
	end
	return minimum(err)
end


"""
type for computing the generalized least-squares error
* `Q` : must be symmetric, inverse covariance matrix
"""
mutable struct P_gls{T}
	Q::LinearMaps.LinearMap{T} # inverse covariance matrix, must be symmetric
	r::Vector{T} # store residual temporarily
	Qr::Vector{T} # store Q*r temporarily
end

"""
# Q is a diagonal matrix with α
* `n` : size
"""
function P_gls(n,α)
	Q=LinearMap(
	     (y,x)->LinearAlgebra.mul!(y,x,α), 
	     (y,x)->LinearAlgebra.mul!(y,x,α), 
		      n, n, ismutating=true, issymmetric=true)
	r=zeros(n)
	Qr=zeros(n)
	return P_gls(Q,r,Qr)
end

function P_gls(Q)
	n=size(Q,1)
	r=zeros(n)
	Qr=zeros(n)
	return P_gls(Q,r,Qr)
end



function func_grad!(dfdx,  x,   y, pa::P_gls)
	# compute the difference
	for i in eachindex(x)
		pa.r[i]=x[i]-y[i]
	end

	mul!(pa.Qr, pa.Q, pa.r)

	J=pa.r' * pa.Qr

	if(!(dfdx === nothing))
		mul!(dfdx,pa.Q,pa.r)
		rmul!(dfdx,2.0)
	end
	return J
end


end # module
