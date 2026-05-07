"""
Demo script: Gumbel-Softmax with different temperatures

This script generates random logits and visualizes how Gumbel-Softmax
behaves at different temperature values compared to standard softmax.
"""

using GumbelSoftmax, Statistics, Plots, NNlib, Zygote

# Generate random logits
# Shape: (categorical_dim=5, batch_size=10)
N = 1000
logits = repeat(randn(5,1), 1, N)

# Different temperature values to explore
temperatures = [0.1, 0.4, 0.7, 1.0, 3.0, 10.0]
# temperatures = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Compute standard softmax (baseline, no temperature — Gumbel-Softmax converges to this in expectation)
plain_softmax_mean = mean(softmax(logits), dims=2)[:, 1]

println("Gumbel-Softmax Temperature Demo")
println("=" ^ 50)
println("\nLogits shape: $(size(logits))")
println("Temperature values: $temperatures")
println("\nStandard softmax probabilities: $plain_softmax_mean\n")

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
    w = 0.27

    bar!(plt, x .- w, plain_softmax_mean[:]; bar_width=w, label="Softmax",
        title="τ = $tau", xlabel="Category", ylabel="Probability",
        ylim=(0, 1), grid=true, subplot=idx)
    bar!(plt, x, softmax_mean[:]; bar_width=w, label="Softmax(τ)",
        subplot=idx)
    bar!(plt, x .+ w, gumbel_mean[:]; bar_width=w, label="Gumbel-Softmax(τ)",
        subplot=idx)

    # Print statistics for this temperature
    distance = sum(abs.(softmax_mean[:] .- gumbel_mean[:]))
    distance_plain = sum(abs.(plain_softmax_mean[:] .- gumbel_mean[:]))
    println("Temperature τ = $tau")
    println("  Gumbel-Softmax: $gumbel_mean")
    println("  L1 distance from Softmax(τ): $(round(distance; digits=4))")
    println("  L1 distance from Softmax:    $(round(distance_plain; digits=4))\n")
    plt
end
display(plt)
savefig(plt, "examples/img/gumbel_softmax_demo.png")
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
