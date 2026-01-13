
unsqueeze(x, dim) = reshape(x, (size(x)[1:dim-1]..., 1, size(x)[dim:end]...))

get_dims(x::AbstractArray) = size(x)
get_dims(x::AbstractVector{<:AbstractArray}) = [get_dims(i) for i in x]
get_dims(x::Tuple) = [get_dims(i) for (_, i) in pairs(x)]