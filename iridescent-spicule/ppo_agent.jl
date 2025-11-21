using Flux
using Zygote
using Statistics
using Random

"""
Proximal Policy Optimization (PPO) Agent for continuous control.

Learns optimal turn rate policies for fixed-wing agents.
"""

# Actor network: outputs mean and log_std for Gaussian policy
struct ActorNetwork
    layers::Chain
    log_std::Vector{Float32}
end

function ActorNetwork(state_dim::Int, action_dim::Int, hidden_dims::Vector{Int}=[64, 64])
    layers = Chain(
        Dense(state_dim, hidden_dims[1], relu),
        Dense(hidden_dims[1], hidden_dims[2], relu),
        Dense(hidden_dims[2], action_dim, tanh)  # Output in [-1, 1]
    )
    
    # Learnable log standard deviation
    log_std = zeros(Float32, action_dim)
    
    return ActorNetwork(layers, log_std)
end

function (actor::ActorNetwork)(state::Vector{Float32})
    mean = actor.layers(state)
    return mean, actor.log_std
end

# Critic network: outputs state value
function CriticNetwork(state_dim::Int, hidden_dims::Vector{Int}=[64, 64])
    return Chain(
        Dense(state_dim, hidden_dims[1], relu),
        Dense(hidden_dims[1], hidden_dims[2], relu),
        Dense(hidden_dims[2], 1)
    )
end

"""
PPO Agent with actor-critic architecture.
"""
mutable struct PPOAgent
    actor::ActorNetwork
    critic::Chain
    actor_opt::Any  # Use Any instead of Abstract type
    critic_opt::Any
    
    # Hyperparameters
    γ::Float64          # Discount factor
    λ::Float64          # GAE parameter
    clip_ratio::Float64 # PPO clip
    n_epochs::Int       # Update epochs
    batch_size::Int
    ω_max::Float64     # Action scaling
    
    # Experience buffer
    states::Vector{Vector{Float32}}
    actions::Vector{Float32}
    rewards::Vector{Float32}
    values::Vector{Float32}
    log_probs::Vector{Float32}
    dones::Vector{Bool}
end

"""
Create a new PPO agent.
"""
function PPOAgent(state_dim::Int, action_dim::Int;
                 γ::Float64=0.99,
                 λ::Float64=0.95,
                 clip_ratio::Float64=0.2,
                 learning_rate::Float64=3e-4,
                 n_epochs::Int=10,
                 batch_size::Int=64,
                 ω_max::Float64=0.5)
    
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)
    
    # Simple optimizer setup - just store learning rate for manual updates
    actor_opt = learning_rate
    critic_opt = Flux.setup(Adam(learning_rate), critic)
    
    return PPOAgent(
        actor, critic, actor_opt, critic_opt,
        γ, λ, clip_ratio, n_epochs, batch_size, ω_max,
        [], [], [], [], [], []
    )
end

"""
Sample action from policy (with exploration).
"""
function get_action(agent::PPOAgent, state::Vector{Float64})
    state_f32 = Float32.(state)
    
    # Get policy distribution
    mean, log_std =  agent.actor(state_f32)
    std = exp.(log_std)
    
    # Sample action from Gaussian
    action_normalized = mean[1] + std[1] * randn(Float32)
    action_normalized = clamp(action_normalized, -1.0f0, 1.0f0)
    
    # Scale to actual action range
    action = action_normalized * agent.ω_max
    
    # Compute log probability
    log_prob = -0.5f0 * ((action_normalized - mean[1]) / std[1])^2 - log_std[1] - 0.5f0 * log(2.0f0 * π)
    
    # Get value estimate
    value = agent.critic(state_f32)[1]
    
    return Float64(action), Float64(log_prob), Float64(value)
end

"""
Get deterministic action (for evaluation).
"""
function get_action_deterministic(agent::PPOAgent, state::Vector{Float64})
    state_f32 = Float32.(state)
    mean, _ = agent.actor(state_f32)
    action = Float64(mean[1] * agent.ω_max)
    return action
end

"""
Store experience in buffer.
"""
function store_transition!(agent::PPOAgent, state, action, reward, value, log_prob, done)
    push!(agent.states, Float32.(state))
    push!(agent.actions, Float32(action))
    push!(agent.rewards, Float32(reward))
    push!(agent.values, Float32(value))
    push!(agent.log_probs, Float32(log_prob))
    push!(agent.dones, done)
end

"""
Compute Generalized Advantage Estimation (GAE).
"""
function compute_gae(agent::PPOAgent)
    n = length(agent.rewards)
    advantages = zeros(Float32, n)
    returns = zeros(Float32, n)
    
    gae = 0.0f0
    for t in reverse(1:n)
        if t == n
            next_value = 0.0f0
        else
            next_value = agent.dones[t] ? 0.0f0 : agent.values[t+1]
        end
        
        delta = agent.rewards[t] + agent.γ * next_value - agent.values[t]
        gae = delta + agent.γ * agent.λ * (agent.dones[t] ? 0.0f0 : gae)
        
        advantages[t] = gae
        returns[t] = gae + agent.values[t]
    end
    
    # Normalize advantages
    advantages = (advantages .- mean(advantages)) ./ (std(advantages) + 1e-8)
    
    return advantages, returns
end

"""
Update policy using PPO with simplified manual updates.
"""
function update!(agent::PPOAgent)
    if length(agent.states) < agent.batch_size
        return  # Need enough samples
    end
    
    # Compute advantages
    advantages, returns = compute_gae(agent)
    
    # Convert to arrays
    states = hcat(agent.states...)'
    actions = agent.actions
    old_log_probs = agent.log_probs
    
    n_samples = length(agent.rewards)
    lr = agent.actor_opt  # Learning rate
    
    # Multiple epochs of updates
    for epoch in 1:agent.n_epochs
        indices = shuffle(1:n_samples)
        
        for batch_start in 1:agent.batch_size:n_samples
            batch_end = min(batch_start + agent.batch_size - 1, n_samples)
            batch_idx = indices[batch_start:batch_end]
            
            batch_states = states[batch_idx, :]'
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_actions = actions[batch_idx]
            
            # Update critic
            critic_loss, critic_grads = Flux.withgradient(agent.critic) do m
                values = vcat([m(batch_states[:, i])[1] for i in 1:size(batch_states, 2)]...)
                return Flux.mse(values, batch_returns)
            end
            Flux.update!(agent.critic_opt, agent.critic, critic_grads[1])
            
            # Update actor - avoid mutation inside gradient
            actor_loss, actor_grads = Flux.withgradient(agent.actor.layers) do layers
                # Compute log probs functionally (no mutation)
                log_probs = map(batch_idx) do idx
                    state = agent.states[idx]
                    action_idx = findfirst(==(idx), batch_idx)
                    action_norm = batch_actions[action_idx] / agent.ω_max
                    
                    mean = layers(state)
                    std = exp(agent.actor.log_std[1])
                    
                    log_prob = -0.5f0 * ((action_norm - mean[1]) / std)^2 - 
                               agent.actor.log_std[1] - 0.5f0 * log(2.0f0 * π)
                    return log_prob
                end
                
                ratios = exp.(log_probs .- batch_old_log_probs)
                surr1 = ratios .* batch_advantages
                surr2 = clamp.(ratios, 1.0f0 - agent.clip_ratio, 1.0f0 + agent.clip_ratio) .* batch_advantages
                
                return -mean(min.(surr1, surr2))
            end
            
            # Manual SGD update for actor layers (outside gradient computation)
            if actor_grads[1] !== nothing
                # New Flux returns named tuple grad structure
                grad_struct = actor_grads[1]
                
                # Manually update each parameter
                for (i, layer) in enumerate(agent.actor.layers)
                    if hasfield(typeof(grad_struct), :layers) &&  length(grad_struct.layers) >= i
                        layer_grad = grad_struct.layers[i]
                        if hasfield(typeof(layer_grad), :weight) && layer_grad.weight !== nothing
                            layer.weight .-= lr .* layer_grad.weight
                        end
                        if hasfield(typeof(layer_grad), :bias) && layer_grad.bias !== nothing
                            layer.bias .-= lr .* layer_grad.bias
                        end
                    end
                end
            end
        end
    end
    
    # Clear buffer
    empty!(agent.states)
    empty!(agent.actions)
    empty!(agent.rewards)
    empty!(agent.values)
    empty!(agent.log_probs)
    empty!(agent.dones)
end

# Test PPO agent
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== PPO Agent Test ===\n")
    
    state_dim = 12
    action_dim = 1
    
    agent = PPOAgent(state_dim, action_dim)
    
    println("Actor network: $(agent.actor.layers)")
    println("Critic network: $(agent.critic)")
    
    # Test action sampling
    test_state = randn(state_dim)
    action, log_prob, value = get_action(agent, test_state)
    
    println("\nTest state: $(test_state[1:3])...")
    println("Sampled action: $action")
    println("Log prob: $log_prob")
    println("Value estimate: $value")
    
    # Test deterministic action
    det_action = get_action_deterministic(agent, test_state)
    println("Deterministic action: $det_action")
end
