#!/usr/bin/env julia

"""
Training script for RL-based turn control.

Trains PPO agents to minimize localization error and maintain connectivity.
"""

using Printf
using Statistics
using Plots

include("rl_environment.jl")
include("ppo_agent.jl")

"""
Run training loop.
"""
function train(;
    n_episodes::Int=100,
    n_agents::Int=3,
    max_steps::Int=100,
    eval_freq::Int=10,
    seed::Int=42
)
    Random.seed!(seed)
    
    # Create environment and agent
    env = LocControlEnv(n_agents=n_agents, max_steps=max_steps, seed=seed)
    state_dim, action_dim = get_dims(env)
    agent = PPOAgent(state_dim, action_dim)
    
    println("\n" * "=" ^ 70)
    println("RL TRAINING: Turn Control for Cooperative Localization")
    println("=" ^ 70)
    println("Agents: $n_agents")
    println("State dim: $state_dim")
    println("Action dim: $action_dim")
    println("Episodes: $n_episodes")
    println("Max steps per episode: $max_steps")
    println("=" ^ 70 * "\n")
    
    # Training metrics
    episode_rewards = []
    episode_pos_errors = []
    episode_neighbors = []
    
    for episode in 1:n_episodes
        # Reset environment
        observations = reset!(env)
        
        episode_reward = 0.0
        episode_error = 0.0
        episode_neighbor_count = 0.0
        
        for step in 1:max_steps
            # Get actions for all agents
            actions = Float64[]
            for obs in observations
                action, log_prob, value = get_action(agent, obs)
                push!(actions, action)
                
                # Store transition for first agent only (simplification)
                if length(actions) == 1
                    # We'll store transitions for agent 1 only
                    # Multi-agent RL with shared policy
                end
            end
            
            # Step environment
            next_observations, rewards, dones, info = step!(env, actions)
            
            # Store transition for agent 1
            action_1, log_prob_1, value_1 = get_action(agent, observations[1])
            store_transition!(agent, observations[1], action_1, rewards[1], value_1, log_prob_1, dones[1])
            
            # Update metrics
            episode_reward += mean(rewards)
            episode_error += mean(info["position_errors"])
            episode_neighbor_count += mean(info["n_neighbors"])
            
            observations = next_observations
            
            if any(dones)
                break
            end
        end
        
        # Update policy
        update!(agent)
        
        # Record metrics
        avg_reward = episode_reward / max_steps
        avg_error = episode_error / max_steps
        avg_neighbors = episode_neighbor_count / max_steps
        
        push!(episode_rewards, avg_reward)
        push!(episode_pos_errors, avg_error)
        push!(episode_neighbors, avg_neighbors)
        
        # Print progress
        if episode % eval_freq == 0
            recent_rewards = mean(episode_rewards[max(1, episode-eval_freq+1):episode])
            recent_errors = mean(episode_pos_errors[max(1, episode-eval_freq+1):episode])
            recent_neighbors = mean(episode_neighbors[max(1, episode-eval_freq+1):episode])
            
            @printf("Episode %3d: Reward=%.2f, PosError=%.3f, Neighbors=%.1f\\n",
                    episode, recent_rewards, recent_errors, recent_neighbors)
        end
    end
    
    println("\n" * "=" ^ 70)
    println("TRAINING COMPLETE")
    println("=" ^ 70)
    println("Final avg reward: $(mean(episode_rewards[end-9:end]))")
    println("Final avg pos error: $(mean(episode_pos_errors[end-9:end]))")
    println("Final avg neighbors: $(mean(episode_neighbors[end-9:end]))")
    println("=" ^ 70 * "\n")
    
    return agent, episode_rewards, episode_pos_errors, episode_neighbors
end

"""
Evaluate trained policy.
"""
function evaluate(agent::PPOAgent; n_episodes::Int=5, n_agents::Int=3, max_steps::Int=100)
    env = LocControlEnv(n_agents=n_agents, max_steps=max_steps)
    
    total_reward = 0.0
    total_error = 0.0
    total_neighbors = 0.0
    
    for episode in 1:n_episodes
        observations = reset!(env)
        
        for step in 1:max_steps
            # Get deterministic actions
            actions = [get_action_deterministic(agent, obs) for obs in observations]
            
            # Step
            observations, rewards, dones, info = step!(env, actions)
            
            total_reward += mean(rewards)
            total_error += mean(info["position_errors"])
            total_neighbors += mean(info["n_neighbors"])
            
            if any(dones)
                break
            end
        end
    end
    
    n_total_steps = n_episodes * max_steps
    println("\n" * "=" ^ 70)
    println("EVALUATION RESULTS ($(n_episodes) episodes)")
    println("=" ^ 70)
    @printf("Avg Reward: %.3f\\n", total_reward / n_total_steps)
    @printf("Avg Position Error: %.3f\\n", total_error / n_total_steps)
    @printf("Avg Neighbors: %.1f\\n", total_neighbors / n_total_steps)
    println("=" ^ 70 * "\n")
end

"""
Plot training curves.
"""
function plot_training_curves(rewards, errors, neighbors; save_path="training_curves.png")
    p = plot(layout=(3, 1), size=(800, 800))
    
    plot!(p[1], rewards, xlabel="Episode", ylabel="Avg Reward",
          title="Training Reward", linewidth=2, legend=false)
    
    plot!(p[2], errors, xlabel="Episode", ylabel="Position Error (m)",
          title="Position Error", linewidth=2, legend=false)
    
    plot!(p[3], neighbors, xlabel="Episode", ylabel="Avg Neighbors",
          title="Communication Connectivity", linewidth=2, legend=false)
    
    savefig(p, save_path)
    println("✓ Training curves saved to $save_path")
end

# Run training
if abspath(PROGRAM_FILE) == @__FILE__
    agent, rewards, errors, neighbors = train(
        n_episodes=50,  # Start with 50 episodes for testing
        n_agents=3,
        max_steps=50,   # Shorter episodes for faster training
        eval_freq=5
    )
    
    # Evaluate
    evaluate(agent, n_episodes=5)
    
    # Plot
    plot_training_curves(rewards, errors, neighbors)
end
