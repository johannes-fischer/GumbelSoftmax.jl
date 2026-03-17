export sample_gumbel_softmax, sample_softmax

Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

function sample_gumbel(probs::CuArray, size; epsilon=1e-10)
    ret = CUDA.zeros(size...)
    rand!(ret)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

function sample_gumbel(probs, size; epsilon=1e-10)
    ret = rand(Float32, size...)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

"""
    sample_gumbel_softmax(; probs = nothing, logits = nothing, tau = 0.1, hard = true, epsilon = 1e-10)

Sample from the Gumbel-Softmax distribution. The Gumbel-Softmax distribution is a continuous relaxation of the
categorical distribution. It is defined as follows:

    Gumbel-Softmax(logits) = softmax((logits + Gumbel(0, 1)) / tau)

where tau is a temperature parameter that controls the smoothness of the distribution. The expected inputs are
either `probs` or `logits`. If `logits` is not provided, it is computed as `log(probs + epsilon)` where `epsilon`
is a small value to avoid numerical instability. The expected shape of `logits` is `(categorical_dimension, batch_dimensions...)`.
For example

```julia
logits = randn(10, 30, 64) # 10 classes, 30 distributions, batch of 64
z = sample_gumbel_softmax(logits=logits, tau=0.5)
sizeof(z) # (10, 30, 64)
```
The result one be one-hot encoded if `hard` is set to `true`. If `hard` is set to `false`, the result will be the soft output of the Softmax.
"""
function sample_gumbel_softmax(; probs = nothing, logits = nothing, tau = 0.1, hard = true, epsilon = 1e-10)
    tau = Float32(tau)
    epsilon = Float32(epsilon)
    if logits === nothing
        logits = log.(probs .+ epsilon)
    end
    y = logits + sample_gumbel(logits, size(logits), epsilon = epsilon)
    y_soft = softmax(y / tau, dims=1)
    if hard
        y_hard = (y_soft .== maximum(y_soft, dims = 1))
        ret = y_hard - stop_gradient(y_soft) + y_soft
    else
        ret = y_soft
    end
    return ret
end

"""
    sample_softmax(; probs = nothing, logits = nothing, tau = 0.1, hard = true, epsilon = 1e-10)

Baseline softmax sampling without Gumbel noise. Used for comparison with Gumbel-Softmax to illustrate the benefits
of the Gumbel-Softmax trick for differentiable discrete sampling.
"""
function sample_softmax(; probs = nothing, logits = nothing, tau = 0.1, hard = true, epsilon = 1e-10)
    tau = Float32(tau)
    epsilon = Float32(epsilon)
    if logits === nothing
        logits = log.(probs .+ epsilon)
    end
    y = logits  # No Gumbel noise added - baseline sampling
    y_soft = softmax(y / tau, dims=1)
    if hard
        y_hard = (y_soft .== maximum(y_soft, dims = 1))
        ret = y_hard - stop_gradient(y_soft) + y_soft
    else
        ret = y_soft
    end
    return ret
end