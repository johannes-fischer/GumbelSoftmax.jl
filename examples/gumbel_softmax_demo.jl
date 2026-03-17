"""
Demo script: Gumbel-Softmax with different temperatures

This script generates random logits and visualizes how Gumbel-Softmax
behaves at different temperature values compared to standard softmax.
"""

using GumbelSoftmax, Statistics, Plots, NNlib, Zygote

# Generate random logits
# Shape: (categorical_dim=5, batch_size=10)
logits = randn(5, 10)

# Different temperature values to explore
temperatures = [0.001, 0.01, 0.1, 1.0, 10.0, 100.]

# Compute standard softmax (baseline)
softmax_result = softmax(logits)
softmax_mean = mean(softmax_result, dims=2)[:, 1]

println("Gumbel-Softmax Temperature Demo")
println("=" ^ 50)
println("\nLogits shape: $(size(logits))")
println("Temperature values: $temperatures")
println("\nStandard softmax probabilities: $softmax_mean\n")

# Create visualization with Plots.jl
plt = plot(layout=(2, 3), size=(1200, 800))

for (idx, tau) in enumerate(temperatures)
    hard=false
    # Compute standard softmax (baseline)
    softmax_result = sample_softmax(logits=logits, tau=tau, hard=hard)
    softmax_mean = mean(softmax_result, dims=2)[:, 1]

    # Sample from Gumbel-Softmax with hard=false to get soft probabilities
    gumbel_samples = sample_gumbel_softmax(logits=logits, tau=tau, hard=hard)
    gumbel_mean = mean(gumbel_samples, dims=2)[:, 1]

    x = collect(1:5)
    w = 0.35

    bar!(plt, x .- w/2, softmax_mean[:]; bar_width=w, label="Softmax",
        title="τ = $tau", xlabel="Category", ylabel="Probability",
        ylim=(0, 1), grid=true, subplot=idx)
    bar!(plt, x .+ w/2, gumbel_mean[:]; bar_width=w, label="Gumbel-Softmax",
        subplot=idx)

    # Print statistics for this temperature
    distance = sum(abs.(softmax_mean[:] .- gumbel_mean[:]))
    println("Temperature τ = $tau")
    println("  Gumbel-Softmax: $gumbel_mean")
    println("  L1 distance from Softmax: $(round(distance; digits=4))\n")
end
savefig(plt, "examples/img/gumbel_softmax_demo.png")
display(plt)
println("Saved figure to examples/img/gumbel_softmax_demo.png")

# Additional demo: Hard vs Soft mode
println("\n" * "=" ^ 50)
println("Hard vs Soft Mode Comparison")
println("=" ^ 50)

using Zygote

tau = 0.1
soft_samples = sample_gumbel_softmax(logits=logits, tau=tau, hard=false)
hard_samples = sample_gumbel_softmax(logits=logits, tau=tau, hard=true)

soft_mean = mean(soft_samples, dims=2)[:, 1]
hard_mean = mean(hard_samples, dims=2)[:, 1]

# Compute gradients of category 1 probability w.r.t. logits
# (sum over all samples would be constant = batch_size, since softmax sums to 1)
using Random
seed = 300
soft_grad = let
    Random.seed!(seed)
    Zygote.gradient(l -> sum((sample_gumbel_softmax(logits=l, tau=tau, hard=false))[1,:]), logits)[1]
end
hard_grad = let
    Random.seed!(seed)
    Zygote.gradient(l -> sum((sample_gumbel_softmax(logits=l, tau=tau, hard=true))[1,:]), logits)[1]
end
soft_grad == hard_grad

println("\nAt τ = $tau:")
println("Soft mode (continuous): $soft_mean")
println("Hard mode (one-hot): $hard_mean")
println("\nSoft gradient (∂P(cat=1)/∂logits) for batch 1:\n  $(round.(soft_grad[:, 1], digits=4))")
println("Hard gradient (straight-through) for batch 1:\n  $(round.(hard_grad[:, 1], digits=4))")
println("\nSame seed → soft_grad == hard_grad: $(soft_grad == hard_grad)")
println("The straight-through estimator uses the soft (softmax) gradient in the backward pass,")
println("so hard and soft modes produce identical gradients given the same Gumbel noise.")
