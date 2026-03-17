# This is a direct port from this code:
# https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py

export sample_rao_gumbel_softmax

function sample_conditional_gumbel(logits, D, k = 1)
    # E should be size (n_classes, n_samples, k)
    E = rand(Exponential(1), size(logits)..., k)
    # Ei should be size (1, n_samples, k)
    D = reshape(D, size(D)..., 1)
    Ei = sum(D .* E, dims = 1)
    # Z should be size (1, n_samples)
    Z = sum(exp.(logits), dims = 1)
    logits = reshape(logits, size(logits)..., 1)
    Z = reshape(Z, size(Z)..., 1)
    adjusted = D .* (-log.(Ei) .+ log.(Z)) .+ (1 .- D) .* (-log.(E ./ exp.(logits) .+ Ei ./ Z))
    return (logits .- stop_gradient(logits)) .+ stop_gradient(adjusted)
end

function sample_rao_gumbel_softmax_no_batch(logits; k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
    probs = softmax(logits)
    num_classes = size(logits, 1)
    if I === nothing
        # note: calling value here beacuse it can interfere with StochasticAD.jl derivative of the Categorical.
        I = [rand(Categorical(ForwardDiff.value.(probs[:, i]))) for i in axes(probs, 2)]
    end
    D = hcat(onehot.(I, Ref(1:num_classes))...)
    adjusted = sample_conditional_gumbel(logits, D, k)
    surrogate = mean(softmax(adjusted ./ tau), dims = 3)
    return ((surrogate-stop_gradient(surrogate))+stop_gradient(reshape(D, size(D)..., 1)))[:, :, 1]
end

"""
    sample_rao_gumbel_softmax(; probs=nothing, logits=nothing, k=1, tau=1.0, I=nothing, epsilon=1e-10)

Sample from the Rao-Blackwellized Gumbel-Softmax distribution. See https://arxiv.org/abs/2010.04838 for more details.
The expected inputs are either `probs` or `logits`. If `logits` is not provided, it is computed as `log(probs + epsilon)` where `epsilon`
is a small value to avoid numerical instability. The expected shape of `logits` is `(categorical_dimension, batch_dimensions...)`.
`tau` is a temperature parameter that controls the smoothness of the distribution. `k` is the number of Monte Carlo samples.
"""
function sample_rao_gumbel_softmax(logits; k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
    return slicemap(x -> sample_rao_gumbel_softmax_no_batch(x, k = k, tau = tau, I = I, epsilon = epsilon), logits, dims = (1, 2))
end
function sample_rao_gumbel_softmax(logits::Array{<:ForwardDiff.Dual}; k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
    return mapslices(x -> sample_rao_gumbel_softmax_no_batch(x, k = k, tau = tau, I = I, epsilon = epsilon), logits, dims = (1, 2))
end
function sample_rao_gumbel_softmax(;probs = nothing, logits = nothing, k = 1, tau = 1.0, I = nothing, epsilon = 1e-10)
    epsilon = Float32(epsilon)
    tau = Float32(tau)
    if logits === nothing
        logits = log.(probs .+ epsilon)
    end
    return sample_rao_gumbel_softmax(logits, k = k, tau = tau, I = I, epsilon = epsilon)
end