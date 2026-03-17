# Based on: https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py

export sample_rao_gumbel_softmax

function sample_conditional_gumbel(logits, D, k = 1)
    # logits, D: (n_classes, batch_dims...)
    # E: (n_classes, batch_dims..., k)
    E = rand(Exponential(1), size(logits)..., k)
    D = reshape(D, size(D)..., 1)
    Ei = sum(D .* E, dims = 1)
    Z = sum(exp.(logits), dims = 1)
    logits = reshape(logits, size(logits)..., 1)
    Z = reshape(Z, size(Z)..., 1)
    adjusted = D .* (-log.(Ei) .+ log.(Z)) .+ (1 .- D) .* (-log.(E ./ exp.(logits) .+ Ei ./ Z))
    return (logits .- stop_gradient(logits)) .+ stop_gradient(adjusted)
end

"""
    sample_rao_gumbel_softmax(; probs=nothing, logits=nothing, k=1, tau=1.0, I=nothing, epsilon=1e-10)

Sample from the Rao-Blackwellized Gumbel-Softmax distribution. See https://arxiv.org/abs/2010.04838 for more details.
The expected inputs are either `probs` or `logits`. If `logits` is not provided, it is computed as `log(probs + epsilon)` where `epsilon`
is a small value to avoid numerical instability. The expected shape of `logits` is `(categorical_dimension, batch_dimensions...)`.
`tau` is a temperature parameter that controls the smoothness of the distribution. `k` is the number of Monte Carlo samples.
"""
function sample_rao_gumbel_softmax(logits; k = 1, tau = 1.0, I = nothing)
    probs = softmax(logits)
    num_classes = size(logits, 1)
    if I === nothing
        # ForwardDiff.value strips Dual numbers so Categorical gets plain floats
        probs_val = ForwardDiff.value.(probs)
        I = [rand(Categorical(probs_val[:, idx])) for idx in CartesianIndices(size(probs_val)[2:end])]
    end
    D = onehotbatch(I, 1:num_classes)
    adjusted = sample_conditional_gumbel(logits, D, k)
    # k is always the last dim
    kdim = ndims(adjusted)
    surrogate = dropdims(mean(softmax(adjusted ./ tau), dims = kdim), dims = kdim)
    return (surrogate - stop_gradient(surrogate)) + stop_gradient(D)
end
function sample_rao_gumbel_softmax(;probs = nothing, logits = nothing, k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
    epsilon = Float32(epsilon)
    tau = Float32(tau)
    if logits === nothing
        logits = log.(probs .+ epsilon)
    end
    return sample_rao_gumbel_softmax(logits, k = k, tau = tau, I = I)
end