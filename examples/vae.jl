##
using Pkg
Pkg.activate("examples")
using Distributions, GumbelSoftmax, CUDA, cuDNN, Flux, MLDatasets, Statistics, ProgressMeter, Plots

##
# parameters

device = gpu
latent_dim = 30
categorical_dim = 10

function get_mnist()
    xtrain, ytrain = MNIST(:train)[:]
    xtest, ytest = MNIST(:test)[:]
    return xtrain, ytrain, xtest, ytest
end
xtrain, _, xtest, _ = get_mnist()
input_dim = size(xtrain, 1) * size(xtrain, 2)
xtrain_flat = reshape(xtrain, input_dim, size(xtrain, 3)) |> device

encoder = Chain(
    Dense(28^2, 512, relu),
    Dense(512, 256, relu),
    Dense(256, latent_dim * categorical_dim, relu),
) |> device
decoder = Chain(
    Dense(latent_dim * categorical_dim, 256, relu),
    Dense(256, 512, relu),
    Dense(512, input_dim, sigmoid),
) |> device

##
function run_model(encoder, decoder, x, latent_dim, categorical_dim; sampler=sample_gumbel_softmax)
    q = encoder(x)
    q_unflatten = reshape(q, categorical_dim, latent_dim, :)
    z = sampler(logits=q_unflatten, tau=0.5)
    z = reshape(z, latent_dim * categorical_dim, :)
    return decoder(z), reshape(softmax(q_unflatten), categorical_dim * latent_dim, :)
end

function compute_loss(x, x_reconstructed, latent_z)
    bce = Flux.Losses.binarycrossentropy(x_reconstructed, x, agg=sum) ./ size(x, 2)
    log_ratio = log.(latent_z .* categorical_dim .+ 1e-10)
    kld = mean(sum(latent_z .* log_ratio, dims=1))
    return bce + kld
end

function train(encoder, decoder, xtrain, nepochs; sampler=sample_gumbel_softmax)
    loader = Flux.DataLoader((xtrain), batchsize=64, shuffle=true)
    opt = Flux.Adam(1e-3)
    enc_state = Flux.setup(opt, encoder)
    dec_state = Flux.setup(opt, decoder)
    losses = []
    @showprogress for epoch in 1:nepochs
        for x in loader
            grads = Flux.gradient(encoder, decoder) do enc, dec
                z_decoded, z_soft = run_model(enc, dec, x, latent_dim, categorical_dim; sampler=sampler)
                compute_loss(x, z_decoded, z_soft)
            end
            Flux.update!(enc_state, encoder, grads[1])
            Flux.update!(dec_state, decoder, grads[2])
            loss = let
                z_decoded, z_soft = run_model(encoder, decoder, x, latent_dim, categorical_dim; sampler=sampler)
                compute_loss(x, z_decoded, z_soft)
            end
            push!(losses, loss)
        end
    end
    return losses
end
##
# Train two models for comparison
println("Training with Gumbel-Softmax...")
encoder_gs = Chain(
    Dense(28^2, 512, relu),
    Dense(512, 256, relu),
    Dense(256, latent_dim * categorical_dim, relu),
) |> device
decoder_gs = Chain(
    Dense(latent_dim * categorical_dim, 256, relu),
    Dense(256, 512, relu),
    Dense(512, input_dim, sigmoid),
) |> device

losses_gs = train(encoder_gs, decoder_gs, xtrain_flat, 10; sampler=sample_gumbel_softmax)

println("\nTraining with Softmax baseline (no Gumbel noise)...")
encoder_s = Chain(
    Dense(28^2, 512, relu),
    Dense(512, 256, relu),
    Dense(256, latent_dim * categorical_dim, relu),
) |> device
decoder_s = Chain(
    Dense(latent_dim * categorical_dim, 256, relu),
    Dense(256, 512, relu),
    Dense(512, input_dim, sigmoid),
) |> device

losses_s = train(encoder_s, decoder_s, xtrain_flat, 10; sampler=sample_softmax)

##
# plot loss comparison
p_loss = plot(losses_gs, label="Gumbel-Softmax", xlabel="Iteration", ylabel="Loss", title="VAE Loss Comparison", lw=2)
plot!(p_loss, losses_s, label="Softmax (baseline)", lw=2)
savefig(p_loss, "examples/img/losses.png")
p_loss
##

# plot reconstruction examples (Gumbel-Softmax)
n_examples = 10
plots_list_gs = []
for i in 1:n_examples
    xt = xtest[:, :, i]
    p_orig = heatmap(transpose(xt), color=:grays, axis=false, title=(i == 1 ? "Original" : ""))
    push!(plots_list_gs, p_orig)

    xt_flat = reshape(xt, input_dim, 1) |> device
    x_reconsructed = run_model(encoder_gs, decoder_gs, xt_flat, latent_dim, categorical_dim; sampler=sample_gumbel_softmax)[1]
    x_reconsructed = reshape(x_reconsructed |> cpu, 28, 28)
    p_recon = heatmap(transpose(x_reconsructed), color=:grays, axis=false, title=(i == 1 ? "Gumbel Recon" : ""))
    push!(plots_list_gs, p_recon)
end
fig_recon_gs = plot(plots_list_gs..., layout=(n_examples, 2), size=(200, 500))
savefig(fig_recon_gs, "examples/img/reconstructed_gumbel.png")
fig_recon_gs
##

# plot reconstruction examples (Softmax baseline)
plots_list_s = []
for i in 1:n_examples
    xt = xtest[:, :, i]
    p_orig = heatmap(transpose(xt), color=:grays, axis=false, title=(i == 1 ? "Original" : ""))
    push!(plots_list_s, p_orig)

    xt_flat = reshape(xt, input_dim, 1) |> device
    x_reconsructed = run_model(encoder_s, decoder_s, xt_flat, latent_dim, categorical_dim; sampler=sample_softmax)[1]
    x_reconsructed = reshape(x_reconsructed |> cpu, 28, 28)
    p_recon = heatmap(transpose(x_reconsructed), color=:grays, axis=false, title=(i == 1 ? "Softmax Recon" : ""))
    push!(plots_list_s, p_recon)
end
fig_recon_s = plot(plots_list_s..., layout=(n_examples, 2), size=(200, 500))
savefig(fig_recon_s, "examples/img/reconstructed_softmax.png")
fig_recon_s
##

##
# plot sampled examples (Gumbel-Softmax)
n_samples = 64
M = n_samples * latent_dim
samples = rand(Categorical(0.1 ./ ones(categorical_dim)), M)
samples_oh = Float32.(reduce(hcat, Flux.onehot.(samples, Ref(1:categorical_dim))))
samples_oh = reshape(samples_oh, latent_dim * categorical_dim, n_samples) |> device
samples_decoded_gs = decoder_gs(samples_oh)
samples_decoded_gs = reshape(samples_decoded_gs |> cpu, 28, 28, n_samples)

plots_gen_gs = []
for index in 1:n_samples
    p_gen = heatmap(transpose(samples_decoded_gs[:, :, index]), color=:grays, axis=false)
    push!(plots_gen_gs, p_gen)
end
fig_gen_gs = plot(plots_gen_gs..., layout=(8, 8), size=(800, 800))
savefig(fig_gen_gs, "examples/img/generated_gumbel.png")
fig_gen_gs

##
# plot sampled examples (Softmax baseline)
samples_decoded_s = decoder_s(samples_oh)
samples_decoded_s = reshape(samples_decoded_s |> cpu, 28, 28, n_samples)

plots_gen_s = []
for index in 1:n_samples
    p_gen = heatmap(transpose(samples_decoded_s[:, :, index]), color=:grays, axis=false)
    push!(plots_gen_s, p_gen)
end
fig_gen_s = plot(plots_gen_s..., layout=(8, 8), size=(800, 800))
savefig(fig_gen_s, "examples/img/generated_softmax.png")
fig_gen_s