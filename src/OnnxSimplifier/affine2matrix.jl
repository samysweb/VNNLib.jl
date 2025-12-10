

"""
Convert a theoretically affine operation op into representation op(x) = Ax + b (for vectorized x).

Only operations that allow for batched inputs are supported!

args:
    op - a theoretically affine operation that supports batch input
    x - an example input for op in the right shape

kwargs:
    batched - true iff the op supports batched arguments

returns:
    A, b s.t. vec(op(x)) == A*vec(x) + b
"""
function affop2mat(op, x; batched=true, verbosity=0)
    s_in = size(x)

    # want x with batch dim, but s_in without batch dim
    if s_in[end] != 1
        # append batch dimension to e.g. feed it through convolution
        x = reshape(x, s_in..., 1)
    else
        # don't want trailing 1, when we construct new shapes,
        # but directly want product-batch-dim
        s_in = s_in[1:end-1]
    end
    
    # get bias
    b = op(zero(x))

    s_out = size(b)

    if batched
        # identity matrix, but in shape of x with batch dim for every input
        eye = reshape(I(prod(s_in)), s_in..., prod(s_in))
        A = op(eye)
        A = reshape(A, prod(s_out), prod(s_in))
    else
        n_in = prod(s_in)
        A = zeros(prod(s_out), n_in)
        for i in 1:n_in
            eᵢ = (1:n_in) .== i  # i-th unit basis vector
            
            # build the matrix column by column
            A[:,i] .= op(reshape(eᵢ, size(x)...))
        end
    end

    b = reshape(b, prod(s_out))
    # for every entry, executing the layer also added the bias term to the result, but we only want the
    # linear contribution in the matrix
    A = A .- b  
    
    return A, b
end


"""
Convert a theoretically affine operation op(x₁, x₂, ..., xₙ) with multiple arguments into matrix representation A*[x₁; x₂; ... ; xₙ] + b.

args:
    op - theoretically affine operation supporting batched inputs
    args - example inputs for each input argument of op in the expected shapes

kwargs:
    batched - true iff op supports batched input

returns:
    A, b s.t. vec(op(x₁, x₂, ..., xₙ)) == A * vcat(vec(x₁), vec(x₂), ..., vec(xₙ)) + b
"""
function affop2mat(op, args...; batched=true, verbosity=0)
    As = []
    bs = []
    for (i, arg) in enumerate(args)
        
        verbosity > 0 && println("[affop2mat] iteration $i - size(arg) = ", size(arg))

        A, b = affop2mat(x -> op((j != i ? zero(args[j]) : x for j in 1:length(args))...), arg, batched=batched)
        push!(As, A)
        push!(bs, b)
    end

    A = hcat(As...)
    b = bs[end]
    return A, b    
end


"""
Converts broadcast for bivariate non-linear operations to matrices.

After conversion to dense, all inputs to all nodes are flattened vectors.
Before conversion, an element-wise operation between tensors of compatible shapes can be broadcasted.

    op.(x, y)

works for e.g. tensors of shapes (1, m, n) and (k, m, n).

Note that shapes such as (1, m, 1, n) and (i, 1, j, 1) are also compatible!

After flattening however, we have tensors of shapes (m*n,) and (k*m*n), which cannot be broadcast in the same way.
To this end, we expand the tensors by a matrix product s.t.

    vec(op.(x, y)) = op.(A*vec(x), B*vec(y))

args:
    x - first input tensor 
    y - second input tensor 

returns:
    A - expansion matrix for x
    B - expansion matrix for y
"""
function broadcast_to_mat(x, y)
    x_passive = trues(size(x)...)
    y_passive = trues(size(y)...)

    # just some operation that uses broadcasting
    z = x .* y
    out_size = prod(size(z))

    x_arg = reshape(1:prod(size(x)), size(x)...)
    y_arg = reshape(1:prod(size(y)), size(y)...)

    zx = vec(y_passive .* x_arg)
    zy = vec(x_passive .* y_arg)

    A = falses(out_size, prod(size(x)))
    for i in 1:size(A, 1)
        A[i, zx[i]] = 1
    end

    B = falses(out_size, prod(size(y)))
    for i in 1:size(B, 1)
        B[i, zy[i]] = 1
    end

    return A, B
end