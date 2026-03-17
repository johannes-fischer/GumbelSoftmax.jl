##
# Lane Change MWE — Gumbel-Softmax with simulation-based loss
#
# Validates end-to-end learning of discrete lane choice (classification) and
# PD controller gains (regression) from trajectory regression loss only — no
# classification labels. Gradient flows through:
#   loss → simulated trajectory → PD controller → y_target selection → GumbelSoftmax → logits

using Pkg
Pkg.activate("examples")
using GumbelSoftmax, Flux, NNlib, Statistics, Plots, Random, Printf

## ─── Part 1: Constants & Environment Setup ───────────────────────────────────

const LANE_POSITIONS = Float32[-4.0, 0.0, 4.0]   # y offsets of 3 lanes (m)
const LANE_POS_COL = reshape(LANE_POSITIONS, 3, 1) # (3,1) for broadcasting
const V_X = 20.0f0       # constant longitudinal velocity (m/s)
const DT = 0.1f0         # simulation timestep (s)
const T_LC = 3.0f0       # cosine lane-change duration (s)
const M_HORIZON = 30     # prediction horizon (steps) = 3 seconds

## ─── Part 2: Coin Sequence Generation ────────────────────────────────────────

function generate_coin_sequence(total_distance; min_stretch=80.0, max_stretch=200.0)
    segments = NamedTuple{(:lane, :start_x, :end_x), Tuple{Int,Float64,Float64}}[]
    x = 0.0
    current_lane = rand(1:3)
    while x < total_distance
        len = min_stretch + rand() * (max_stretch - min_stretch)
        push!(segments, (lane=current_lane, start_x=x, end_x=x + len))
        x += len
        current_lane = rand(setdiff(1:3, current_lane))
    end
    return segments
end

## ─── Part 3: Feature Computation ─────────────────────────────────────────────

const NO_COIN_DIST = 1000.0f0  # large constant when no upcoming stretch

function compute_features(y, vy, x_pos, segments)
    # For each lane: (distance_to_start_or_0, distance_to_end_or_0)
    coin_feats = zeros(Float32, 6)
    for lane_id in 1:3
        # Find the segment for this lane that is active or next upcoming
        feat_start = NO_COIN_DIST
        feat_end = 0.0f0
        for seg in segments
            seg.lane != lane_id && continue
            if seg.start_x <= x_pos < seg.end_x
                # Currently active on this lane
                feat_start = 0.0f0
                feat_end = Float32(seg.end_x - x_pos)
                break
            elseif seg.start_x > x_pos
                # Next upcoming stretch on this lane
                feat_start = Float32(seg.start_x - x_pos)
                feat_end = 0.0f0
                break
            end
        end
        idx = (lane_id - 1) * 2
        coin_feats[idx + 1] = feat_start
        coin_feats[idx + 2] = feat_end
    end
    return Float32[y, vy, coin_feats...]
end

## ─── Part 4: Cosine Lane-Change Model (Data Generation Policy) ──────────────

# Returns the active lane id (the lane that currently has coins) at position x_pos
function active_coin_lane(x_pos, segments)
    for seg in segments
        if seg.start_x <= x_pos < seg.end_x
            return seg.lane
        end
    end
    # Past all segments — return last segment's lane
    return segments[end].lane
end

function simulate_expert(segments)
    # Total distance from coin sequence
    total_x = segments[end].end_x
    total_steps = Int(floor(total_x / (V_X * DT)))

    # State variables
    y = Float64(LANE_POSITIONS[active_coin_lane(0.0, segments)])
    vy = 0.0
    x_pos = 0.0

    # Lane-change state machine
    in_lane_change = false
    y_start = 0.0
    y_target = 0.0
    t_start = 0.0

    # Storage
    ys = Float64[]
    vys = Float64[]
    ays = Float64[]
    xs = Float64[]
    gt_lanes = Int[]  # ground truth active coin lane

    for step in 1:total_steps
        t = (step - 1) * Float64(DT)
        coin_lane = active_coin_lane(x_pos, segments)
        push!(gt_lanes, coin_lane)

        if !in_lane_change
            # Check if we need to change lanes
            current_lane_idx = argmin(abs.(LANE_POSITIONS .- Float32(y)))
            if current_lane_idx != coin_lane
                # Initiate lane change
                in_lane_change = true
                y_start = y
                y_target = Float64(LANE_POSITIONS[coin_lane])
                t_start = t
            end
        end

        if in_lane_change
            t_elapsed = t - t_start
            if t_elapsed >= Float64(T_LC)
                # Complete lane change
                y = y_target
                vy = 0.0
                in_lane_change = false
                ay = 0.0
            else
                # Analytical cosine profile (no numerical integration error)
                Δy = y_target - y_start
                phase = π * t_elapsed / Float64(T_LC)
                y = y_start + Δy / 2.0 * (1.0 - cos(phase))
                vy = Δy * π / (2.0 * Float64(T_LC)) * sin(phase)
                ay = Δy * π^2 / (2.0 * Float64(T_LC)^2) * cos(phase)
            end
        else
            ay = 0.0
        end

        push!(ys, y)
        push!(vys, vy)
        push!(ays, ay)
        push!(xs, x_pos)

        # Advance position (only x_pos needs integration; y/vy set analytically above)
        if !in_lane_change
            y = y + vy * Float64(DT) + 0.5 * ay * Float64(DT)^2
            vy = vy + ay * Float64(DT)
        end
        x_pos += Float64(V_X) * Float64(DT)
    end

    return ys, vys, ays, xs, gt_lanes
end

## ─── Part 5: Data Collection via Simulation ──────────────────────────────────

function collect_dataset(; total_distance=10_000.0, seed=42)
    Random.seed!(seed)
    segments = generate_coin_sequence(total_distance)
    ys, vys, ays, xs, gt_lanes = simulate_expert(segments)

    N = length(ys) - M_HORIZON
    X = zeros(Float32, 8, N)
    Y_pos = zeros(Float32, M_HORIZON, N)
    Y_vel = zeros(Float32, M_HORIZON, N)
    Y_acc = zeros(Float32, M_HORIZON, N)
    lane_labels = zeros(Int, N)

    for i in 1:N
        X[:, i] = compute_features(Float32(ys[i]), Float32(vys[i]), xs[i], segments)
        for j in 1:M_HORIZON
            Y_pos[j, i] = Float32(ys[i + j - 1])
            Y_vel[j, i] = Float32(vys[i + j - 1])
            Y_acc[j, i] = Float32(ays[i + j - 1])
        end
        lane_labels[i] = gt_lanes[i]
    end

    # Feature normalization (z-score) — coin distances can be O(1000), which kills ReLU neurons
    feat_mean = mean(X, dims=2)
    feat_std = std(X, dims=2)
    feat_std[feat_std .== 0] .= 1.0f0
    X_norm = (X .- feat_mean) ./ feat_std

    return X, X_norm, feat_mean, feat_std, Y_pos, Y_vel, Y_acc, lane_labels, segments
end

## ─── Part 6: Model Architecture ──────────────────────────────────────────────

# BoundedSigmoid: maps output to [lb, ub] via sigmoid
struct BoundedSigmoid{T}
    lb::T
    ub::T
end
Flux.@layer BoundedSigmoid trainable=()
(b::BoundedSigmoid)(x) = b.lb .+ (b.ub .- b.lb) .* sigmoid.(x)

function build_model()
    backbone = Chain(
        Dense(8, 64, relu),
        Dense(64, 64, relu),
    )
    class_head = Dense(64, 3)
    # 3 outputs: kp, kd, T_delay with per-output bounds
    pd_head = Chain(
        Dense(64, 3),
        BoundedSigmoid(Float32[.1, .1, 0.0], Float32[10.0, 10.0, 2.5]),
    )
    return (backbone=backbone, class_head=class_head, pd_head=pd_head)
end

function forward(model, x_norm, tau; sampler=sample_gumbel_softmax)
    h = model.backbone(x_norm)                     # (64, batch)
    logits = model.class_head(h)                    # (3, batch)
    lane_choice = sampler(logits=logits, tau=tau, hard=true)  # (3, batch)
    y_target = sum(lane_choice .* LANE_POS_COL, dims=1)  # (1, batch)
    pd_params = model.pd_head(h)                    # (3, batch) — [kp, kd, T_delay]
    return y_target, pd_params, logits
end

# Deterministic forward for evaluation — no Gumbel noise, just argmax of logits
function forward_eval(model, x_norm)
    h = model.backbone(x_norm)
    logits = model.class_head(h)                    # (3, batch)
    # One-hot from argmax (no noise)
    lane_choice = Float32.(logits .== maximum(logits, dims=1))
    y_target = sum(lane_choice .* LANE_POS_COL, dims=1)
    pd_params = model.pd_head(h)
    return y_target, pd_params, logits
end

## ─── Part 6b: Gather-based forward (no straight-through estimator) ──────────

# Matches the batched_gather pattern from DroneDatasetAnalysis:
# argmax selects discrete index, NNlib.gather picks the value from source.
# gather IS differentiable through the source values, but since LANE_POSITIONS
# is constant, the classifier naturally gets no gradient — same as in
# DroneDatasetAnalysis where gradients flow through the gathered vehicle tokens
# (learned) but not through the discrete indices (argmax).
function forward_gather(model, x_norm, tau; sampler=nothing)
    h = model.backbone(x_norm)                     # (64, batch)
    logits = model.class_head(h)                    # (3, batch)
    pd_params = model.pd_head(h)                    # (3, batch)
    # Argmax + gather: discrete selection via NNlib.gather
    lane_idx = [argmax(logits[:, j]) for j in axes(logits, 2)]  # Vector{Int}
    y_target = reshape(NNlib.gather(LANE_POSITIONS, lane_idx), 1, :)  # (1, batch)
    return y_target, pd_params, logits
end

## ─── Part 7: PD Simulation in Loss (Differentiable) ─────────────────────────

function simulate_pd(y0, vy0, y_target, kp, kd, T_delay, dt, steps; sharpness=50.0f0)
    y = y0
    vy = vy0
    # First step (initialize accumulators — no mutation, Zygote-compatible)
    # Differentiable time gate: sigmoid((t - T_delay) * sharpness)
    # Sharpness is annealed during training: soft → sharp for gradient flow
    t = 0.0f0
    gate = sigmoid.((t .- T_delay) .* sharpness)
    ay = gate .* (-kp .* (y .- y_target) .- kd .* vy)
    positions = y
    velocities = vy
    accelerations = ay
    y = y .+ vy .* dt .+ 0.5f0 .* ay .* dt^2
    vy = vy .+ ay .* dt
    # Remaining steps: accumulate via vcat (creates new arrays, no mutation)
    for step in 2:steps
        t = Float32(step - 1) * dt
        gate = sigmoid.((t .- T_delay) .* sharpness)
        ay = gate .* (-kp .* (y .- y_target) .- kd .* vy)
        positions = vcat(positions, y)
        velocities = vcat(velocities, vy)
        accelerations = vcat(accelerations, ay)
        y = y .+ vy .* dt .+ 0.5f0 .* ay .* dt^2
        vy = vy .+ ay .* dt
    end
    return positions, velocities, accelerations
end

## ─── Part 8: Loss Function ──────────────────────────────────────────────────

function compute_trajectory_loss(model, x_norm, x_raw, y_pos, y_vel, y_acc, tau;
                                 sharpness=50.0f0, sampler=sample_gumbel_softmax, forward_fn=forward)
    # Raw y0, vy0 for PD simulation (physical units)
    y0 = x_raw[1:1, :]
    vy0 = x_raw[2:2, :]
    # Normalized features for model input
    y_target, pd_params, logits = forward_fn(model, x_norm, tau; sampler=sampler)
    kp = pd_params[1:1, :]
    kd = pd_params[2:2, :]
    T_delay = pd_params[3:3, :]

    pred_pos, pred_vel, pred_acc = simulate_pd(y0, vy0, y_target, kp, kd, T_delay, DT, M_HORIZON; sharpness=sharpness)

    mse_pos = mean((pred_pos .- y_pos) .^ 2)
    mse_vel = mean((pred_vel .- y_vel) .^ 2)
    mse_acc = mean((pred_acc .- y_acc) .^ 2)

    return mse_pos + mse_vel + mse_acc
end

## ─── Part 9: Training Loop ──────────────────────────────────────────────────

function train_model(; n_epochs=100, batch_size=128, lr=1e-3, tau_start=1.0f0, tau_end=0.1f0,
                       sharpness_start=0.1f0, sharpness_end=50.0f0, seed=42,
                       sampler=sample_gumbel_softmax, forward_fn=forward, name="Gumbel-Softmax")
    println("Training with $name...")
    println("Collecting dataset...")
    X_raw, X_norm, feat_mean, feat_std, Y_pos, Y_vel, Y_acc, lane_labels, segments = collect_dataset(seed=seed)
    println("Dataset: $(size(X_raw, 2)) samples, $(length(segments)) coin segments")

    model = build_model()
    opt_state = Flux.setup(Adam(lr), model)

    # DataLoader provides: (X_raw, X_norm, Y_pos, Y_vel, Y_acc)
    loader = Flux.DataLoader((X_raw, X_norm, Y_pos, Y_vel, Y_acc), batchsize=batch_size, shuffle=true)

    losses = Float32[]
    accuracies = Float32[]
    taus = Float32[]

    for epoch in 1:n_epochs
        # Temperature annealing: linear decay from tau_start to tau_end
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(n_epochs - 1, 1)
        # Sharpness annealing: soft → sharp for T_delay gradient flow
        sharpness = sharpness_start + (sharpness_end - sharpness_start) * (epoch - 1) / max(n_epochs - 1, 1)

        epoch_loss = 0.0f0
        epoch_correct = 0
        epoch_total = 0

        for (xb_raw, xb_norm, yp, yv, ya) in loader
            grads = Flux.gradient(model) do m
                compute_trajectory_loss(m, xb_norm, xb_raw, yp, yv, ya, tau; sharpness=sharpness, sampler=sampler, forward_fn=forward_fn)
            end
            Flux.update!(opt_state, model, grads[1])

            # Logging (no gradient)
            l = compute_trajectory_loss(model, xb_norm, xb_raw, yp, yv, ya, tau; sharpness=sharpness, sampler=sampler, forward_fn=forward_fn)
            epoch_loss += l * size(xb_raw, 2)

            # Classification accuracy from raw features (coin_start == 0 means active)
            h = model.backbone(xb_norm)
            logits = model.class_head(h)
            preds = [argmax(logits[:, i]) for i in axes(logits, 2)]

            for j in axes(xb_raw, 2)
                # Features: [y, vy, coin_start_1, coin_end_1, coin_start_2, coin_end_2, coin_start_3, coin_end_3]
                # Active lane has coin_start == 0 (use raw features for this check)
                gt_lane = 0
                for lane_id in 1:3
                    idx = 2 + (lane_id - 1) * 2 + 1  # coin_start index
                    if xb_raw[idx, j] ≈ 0.0f0
                        gt_lane = lane_id
                        break
                    end
                end
                if gt_lane > 0
                    epoch_correct += (preds[j] == gt_lane) ? 1 : 0
                    epoch_total += 1
                end
            end
        end

        avg_loss = epoch_loss / size(X_raw, 2)
        acc = epoch_total > 0 ? epoch_correct / epoch_total : 0.0f0
        push!(losses, avg_loss)
        push!(accuracies, acc)
        push!(taus, tau)

        if epoch % 10 == 1 || epoch == n_epochs
            println("Epoch $epoch/$n_epochs | τ=$(round(tau, digits=3)) | sharpness=$(round(sharpness, digits=1)) | loss=$(round(avg_loss, digits=4)) | acc=$(round(acc*100, digits=1))%")
        end
    end

    return model, losses, accuracies, taus, X_raw, X_norm, feat_mean, feat_std, Y_pos, Y_vel, Y_acc, lane_labels
end

## ─── Run Training: Gumbel-Softmax ───────────────────────────────────────────

model_gs, losses_gs, acc_gs, taus_gs, X_raw, X_norm, feat_mean, feat_std, Y_pos, Y_vel, Y_acc, lane_labels = train_model(
    n_epochs=100, sampler=sample_gumbel_softmax, name="Gumbel-Softmax")

## ─── Run Training: Softmax baseline ─────────────────────────────────────────

model_s, losses_s, acc_s, taus_s, _, _, _, _, _, _, _, _ = train_model(
    n_epochs=100, sampler=sample_softmax, name="Softmax (no Gumbel noise)")

## ─── Run Training: Gather baseline (no straight-through) ────────────────────

model_g, losses_g, acc_g, taus_g, _, _, _, _, _, _, _, _ = train_model(
    n_epochs=100, forward_fn=forward_gather, name="Gather (argmax, no STE)")

## ─── Part 10: Evaluation — Comparison ────────────────────────────────────────

# Plot training curves: all three methods
p1 = plot(losses_gs, xlabel="Epoch", ylabel="Loss", title="Training Loss", lw=2, label="Gumbel-Softmax")
plot!(p1, losses_s, lw=2, label="Softmax (STE)", ls=:dash)
plot!(p1, losses_g, lw=2, label="Gather (no STE)", ls=:dot)
p2 = plot(acc_gs .* 100, xlabel="Epoch", ylabel="Accuracy (%)", title="Lane Classification Accuracy", lw=2, label="Gumbel-Softmax")
plot!(p2, acc_s .* 100, lw=2, label="Softmax (STE)", ls=:dash)
plot!(p2, acc_g .* 100, lw=2, label="Gather (no STE)", ls=:dot)
p3 = plot(taus_gs, xlabel="Epoch", ylabel="τ", title="Temperature Schedule", lw=2, label="tau", color=:orange)
p_train = plot(p1, p2, p3, layout=(1, 3), size=(1200, 350))
savefig(p_train, "examples/img/lane_change_training.png")
display(p_train)

## ─── Trajectory Comparison ──────────────────────────────────────────────────

# Compare both models on the same samples
function plot_trajectory_comparison(model_gs, model_s, model_g, X_raw, X_norm, Y_pos; n_samples=12)
    Random.seed!(99)
    idxs = sort(rand(1:size(X_raw, 2), n_samples))
    plots_list = []

    for i in idxs
        xb_norm = X_norm[:, i:i]
        xb_raw = X_raw[:, i:i]
        t_axis = (0:M_HORIZON-1) .* DT
        gt_pos = Y_pos[:, i]

        # Gumbel-Softmax
        yt_gs, pd_gs, log_gs = forward_eval(model_gs, xb_norm)
        pos_gs, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_gs,
            pd_gs[1:1,:], pd_gs[2:2,:], pd_gs[3:3,:], DT, M_HORIZON)
        lane_gs = argmax(log_gs[:, 1])

        # Softmax (STE)
        yt_s, pd_s, log_s = forward_eval(model_s, xb_norm)
        pos_s, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_s,
            pd_s[1:1,:], pd_s[2:2,:], pd_s[3:3,:], DT, M_HORIZON)
        lane_s = argmax(log_s[:, 1])

        # Gather (no STE)
        yt_g, pd_g, log_g = forward_eval(model_g, xb_norm)
        pos_g, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_g,
            pd_g[1:1,:], pd_g[2:2,:], pd_g[3:3,:], DT, M_HORIZON)
        lane_g = argmax(log_g[:, 1])

        p = plot(t_axis, gt_pos, label="GT", lw=2.5, xlabel="t (s)", ylabel="y (m)",
                 title="GS=$lane_gs S=$lane_s G=$lane_g", titlefontsize=8)
        plot!(p, t_axis, vec(pos_gs), label="Gumbel", lw=2, ls=:dash, color=:red)
        plot!(p, t_axis, vec(pos_s), label="Softmax", lw=2, ls=:dot, color=:purple)
        plot!(p, t_axis, vec(pos_g), label="Gather", lw=2, ls=:dashdot, color=:orange)
        hline!(p, LANE_POSITIONS, label="", color=:gray, ls=:dot, alpha=0.5)
        push!(plots_list, p)
    end

    n_rows = 3
    n_cols = ceil(Int, n_samples / n_rows)
    p_traj = plot(plots_list..., layout=(n_rows, n_cols), size=(400 * n_cols, 300 * n_rows))
    savefig(p_traj, "examples/img/lane_change_trajectories.png")
    display(p_traj)
    return p_traj
end

plot_trajectory_comparison(model_gs, model_s, model_g, X_raw, X_norm, Y_pos)

## ─── Confusion Matrix ───────────────────────────────────────────────────────

function print_confusion_matrix(model, X_norm, lane_labels)
    h = model.backbone(X_norm)
    logits = model.class_head(h)
    preds = [argmax(logits[:, i]) for i in axes(logits, 2)]

    # Build confusion matrix
    cm = zeros(Int, 3, 3)
    valid = 0
    for i in eachindex(preds)
        gt = lane_labels[i]
        if gt > 0
            cm[gt, preds[i]] += 1
            valid += 1
        end
    end

    println("\nConfusion Matrix (rows=GT, cols=Predicted):")
    println("         Pred 1  Pred 2  Pred 3")
    for i in 1:3
        row = join([@sprintf("%7d", cm[i, j]) for j in 1:3])
        println("  GT $i: $row")
    end
    total_correct = sum(cm[i, i] for i in 1:3)
    println("Overall accuracy: $(round(100 * total_correct / valid, digits=1))% ($total_correct / $valid)")
    return cm
end

println("\n── Gumbel-Softmax ──")
cm_gs = print_confusion_matrix(model_gs, X_norm, lane_labels)
println("\n── Softmax (STE) ──")
cm_s = print_confusion_matrix(model_s, X_norm, lane_labels)
println("\n── Gather (no STE) ──")
cm_g = print_confusion_matrix(model_g, X_norm, lane_labels)

## ─── Learned PD Gains ───────────────────────────────────────────────────────

function analyze_pd_gains(model, X_norm)
    h = model.backbone(X_norm)
    pd_params = model.pd_head(h)
    kp_vals = vec(pd_params[1, :])
    kd_vals = vec(pd_params[2, :])
    T_vals = vec(pd_params[3, :])

    # Analytical optimal kp for cosine profile: π²/(2·T_lc²)
    kp_analytical = Float32(π^2 / (2 * T_LC^2))

    println("\nLearned PD gains:")
    println("  kp: mean=$(round(mean(kp_vals), digits=3)), std=$(round(std(kp_vals), digits=3))")
    println("  kd: mean=$(round(mean(kd_vals), digits=3)), std=$(round(std(kd_vals), digits=3))")
    println("  T_delay: mean=$(round(mean(T_vals), digits=3)), std=$(round(std(T_vals), digits=3))")
    println("  Analytical kp ≈ $(round(kp_analytical, digits=3)) (π²/(2·T_lc²))")
end

println("\n── Gumbel-Softmax ──")
analyze_pd_gains(model_gs, X_norm)
println("\n── Softmax (STE) ──")
analyze_pd_gains(model_s, X_norm)
println("\n── Gather (no STE) ──")
analyze_pd_gains(model_g, X_norm)

## ─── Data Sanity Check: Plot Expert Trajectories ────────────────────────────

function plot_expert_trajectory(; total_distance=2000.0, seed=42)
    Random.seed!(seed)
    segments = generate_coin_sequence(total_distance)
    ys, _, _, xs, _ = simulate_expert(segments)

    p = plot(xs, ys, xlabel="x (m)", ylabel="y (m)", title="Expert Trajectory (cosine lane changes)",
             lw=1.5, label="trajectory", size=(1000, 300))
    hline!(p, LANE_POSITIONS, label="", color=:gray, ls=:dot, alpha=0.5)

    # Mark coin segments
    for seg in segments
        if seg.start_x < total_distance && seg.end_x > 0
            lane_y = LANE_POSITIONS[seg.lane]
            plot!(p, [seg.start_x, seg.end_x], [lane_y, lane_y], lw=4, alpha=0.3,
                  color=[:red, :blue, :green][seg.lane], label="")
        end
    end
    savefig(p, "examples/img/lane_change_expert.png")
    display(p)
    return p
end

plot_expert_trajectory()

## ─── Diagnostic: Accuracy by sample type ─────────────────────────────────────

function diagnostic_analysis(model_gs, model_s, model_g, X_raw, X_norm, Y_pos, lane_labels)
    h_all = model_gs.backbone(X_norm)
    logits_all = model_gs.class_head(h_all)
    pred_lanes = [argmax(logits_all[:, i]) for i in axes(logits_all, 2)]
    kp_all = vec(model_gs.pd_head(h_all)[1, :])

    # Classify samples: "lane-keeping" (vy≈0, vehicle near a lane center) vs "lane-changing"
    is_lane_keeping = Bool[]
    for i in axes(X_raw, 2)
        y_i = X_raw[1, i]
        vy_i = X_raw[2, i]
        near_lane = minimum(abs.(LANE_POSITIONS .- y_i)) < 0.5f0
        push!(is_lane_keeping, near_lane && abs(vy_i) < 0.3f0)
    end

    keeping_idx = findall(is_lane_keeping)
    changing_idx = findall(.!is_lane_keeping)

    # Accuracy by type
    function acc_for(idxs)
        correct = count(i -> pred_lanes[i] == lane_labels[i], idxs)
        return correct, length(idxs)
    end

    ck, nk = acc_for(keeping_idx)
    cc, nc = acc_for(changing_idx)
    println("\n─── Accuracy by sample type (Gumbel-Softmax) ───")
    println("  Lane-keeping samples: $(round(100*ck/nk, digits=1))% ($ck / $nk)")
    println("  Lane-changing samples: $(round(100*cc/nc, digits=1))% ($cc / $nc)")
    println("  kp (keeping): mean=$(round(mean(kp_all[keeping_idx]), digits=3))")
    println("  kp (changing): mean=$(round(mean(kp_all[changing_idx]), digits=3))")

    # Plot: pick wrong predictions and correct predictions, show GT vs predicted (all models)
    correct_idx = findall(i -> pred_lanes[i] == lane_labels[i] && lane_labels[i] > 0, eachindex(pred_lanes))
    wrong_idx = findall(i -> pred_lanes[i] != lane_labels[i] && lane_labels[i] > 0, eachindex(pred_lanes))
    n_wrong = min(4, length(wrong_idx))
    n_correct = min(5, length(correct_idx))
    rng = Random.MersenneTwister(123)
    sample_wrong = wrong_idx[sort(randperm(rng, length(wrong_idx))[1:n_wrong])]
    sample_correct = correct_idx[sort(randperm(rng, length(correct_idx))[1:n_correct])]
    sample_idxs = vcat(sample_wrong, sample_correct)

    plots_list = []
    for i in sample_idxs
        xb_norm = X_norm[:, i:i]
        xb_raw = X_raw[:, i:i]

        # Gumbel-Softmax model
        yt_gs, pd_gs, log_gs = forward_eval(model_gs, xb_norm)
        pos_gs, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_gs,
            pd_gs[1:1,:], pd_gs[2:2,:], pd_gs[3:3,:], DT, M_HORIZON)
        lane_gs = argmax(log_gs[:, 1])

        # Softmax (STE) model
        yt_s, pd_s, log_s = forward_eval(model_s, xb_norm)
        pos_s, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_s,
            pd_s[1:1,:], pd_s[2:2,:], pd_s[3:3,:], DT, M_HORIZON)
        lane_s = argmax(log_s[:, 1])

        # Gather (no STE) model
        yt_g, pd_g, log_g = forward_eval(model_g, xb_norm)
        pos_g, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_g,
            pd_g[1:1,:], pd_g[2:2,:], pd_g[3:3,:], DT, M_HORIZON)
        lane_g = argmax(log_g[:, 1])

        gt_lane = lane_labels[i]
        t_axis = (0:M_HORIZON-1) .* DT
        gt_pos = Y_pos[:, i]
        gs_ok = lane_gs == gt_lane ? "OK" : "W"

        kp = pd_gs[1, 1]; T_d = pd_gs[3, 1]
        p = plot(t_axis, gt_pos, label="GT", lw=2.5, xlabel="t (s)", ylabel="y (m)",
                 title="[$gs_ok] GT=$gt_lane GS=$lane_gs S=$lane_s G=$lane_g\nkp=$(round(kp,digits=2)) T=$(round(T_d,digits=2))",
                 titlefontsize=8)
        plot!(p, t_axis, vec(pos_gs), label="Gumbel", lw=2, ls=:dash, color=:red)
        plot!(p, t_axis, vec(pos_s), label="Softmax", lw=2, ls=:dot, color=:purple)
        plot!(p, t_axis, vec(pos_g), label="Gather", lw=2, ls=:dashdot, color=:orange)
        hline!(p, LANE_POSITIONS, label="", color=:gray, ls=:dot, alpha=0.4)
        hline!(p, [LANE_POSITIONS[gt_lane]], label="GT lane", color=:blue, alpha=0.3, lw=3)
        push!(plots_list, p)
    end

    n_total = length(sample_idxs)
    n_cols = min(3, n_total)
    n_rows = ceil(Int, n_total / n_cols)
    p_diag = plot(plots_list..., layout=(n_rows, n_cols), size=(450 * n_cols, 300 * n_rows))
    savefig(p_diag, "examples/img/lane_change_diagnostic.png")
    display(p_diag)
end

diagnostic_analysis(model_gs, model_s, model_g, X_raw, X_norm, Y_pos, lane_labels)

## ─── Diagnostic: Sequence around a lane switch ──────────────────────────────

# Find the first from→to lane switch and show frames at offsets around it
function plot_switch_sequence(model_gs, model_s, model_g, X_raw, X_norm, Y_pos, lane_labels;
                              from_lane=1, to_lane=3, offsets=-8:2:8)
    switch_idx = nothing
    for i in 2:length(lane_labels)
        if lane_labels[i-1] == from_lane && lane_labels[i] == to_lane
            switch_idx = i
            break
        end
    end
    switch_idx === nothing && error("No $from_lane→$to_lane switch found")
    println("Lane $from_lane→$to_lane switch at index $switch_idx")

    plots_list = []
    for off in offsets
        i = switch_idx + off
        (i < 1 || i > size(X_raw, 2)) && continue

        xb_norm = X_norm[:, i:i]
        xb_raw = X_raw[:, i:i]

        # Gumbel-Softmax model
        yt_gs, pd_gs, log_gs = forward_eval(model_gs, xb_norm)
        pos_gs, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_gs,
            pd_gs[1:1,:], pd_gs[2:2,:], pd_gs[3:3,:], DT, M_HORIZON)
        lane_gs = argmax(log_gs[:, 1])

        # Softmax (STE) model
        yt_s, pd_s, log_s = forward_eval(model_s, xb_norm)
        pos_s, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_s,
            pd_s[1:1,:], pd_s[2:2,:], pd_s[3:3,:], DT, M_HORIZON)
        lane_s = argmax(log_s[:, 1])

        # Gather (no STE) model
        yt_g, pd_g, log_g = forward_eval(model_g, xb_norm)
        pos_g, _, _ = simulate_pd(xb_raw[1:1,:], xb_raw[2:2,:], yt_g,
            pd_g[1:1,:], pd_g[2:2,:], pd_g[3:3,:], DT, M_HORIZON)
        lane_g = argmax(log_g[:, 1])

        gt_lane = lane_labels[i]

        # GT lanes active during the 30-step horizon
        horizon_end = min(i + M_HORIZON - 1, length(lane_labels))
        horizon_lanes = lane_labels[i:horizon_end]
        lane_str = join(unique(horizon_lanes), "→")

        t_axis = (0:M_HORIZON-1) .* DT
        gt_pos = Y_pos[:, i]
        gs_ok = lane_gs == gt_lane ? "OK" : "W"

        kp = pd_gs[1, 1]; T_d = pd_gs[3, 1]
        p = plot(t_axis, gt_pos, label="GT", lw=2.5, xlabel="t (s)", ylabel="y (m)",
                 title="[$gs_ok] off=$off GT=$gt_lane GS=$lane_gs S=$lane_s G=$lane_g\ncoins:$lane_str kp=$(round(kp,digits=2)) T=$(round(T_d,digits=2))",
                 titlefontsize=8, ylim=(minimum(LANE_POSITIONS)-1.5, maximum(LANE_POSITIONS)+1.5))
        plot!(p, t_axis, vec(pos_gs), label="Gumbel", lw=2, ls=:dash, color=:red)
        plot!(p, t_axis, vec(pos_s), label="Softmax", lw=2, ls=:dot, color=:purple)
        plot!(p, t_axis, vec(pos_g), label="Gather", lw=2, ls=:dashdot, color=:orange)
        hline!(p, LANE_POSITIONS, label="", color=:gray, ls=:dot, alpha=0.4)
        hline!(p, [LANE_POSITIONS[gt_lane]], label="GT lane", color=:blue, alpha=0.3, lw=3)
        push!(plots_list, p)

        println("  off=$(lpad(off,3)): GT=$gt_lane GS=$lane_gs S=$lane_s G=$lane_g | coins=$lane_str | y=$(round(X_raw[1,i],digits=2)) vy=$(round(X_raw[2,i],digits=2))")
    end

    n_cols = min(3, length(plots_list))
    n_rows = ceil(Int, length(plots_list) / n_cols)
    p_seq = plot(plots_list..., layout=(n_rows, n_cols), size=(450 * n_cols, 300 * n_rows))
    savefig(p_seq, "examples/img/lane_change_switch_sequence.png")
    display(p_seq)
    return p_seq
end

plot_switch_sequence(model_gs, model_s, model_g, X_raw, X_norm, Y_pos, lane_labels; from_lane=1, to_lane=3)
