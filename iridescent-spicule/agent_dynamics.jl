using LinearAlgebra
using Random

"""
Agent dynamics model for cooperative localization.

Fixed-wing aircraft (Dubins vehicle) model with:
    - Constant velocity
    - Turn rate (angular velocity) control

State vector (5D):
    x = [px, py, θ, v, ω]ᵀ
    
Where:
    - px, py: Position (2D)
    - θ: Heading angle
    - v: Speed (constant)
    - ω: Turn rate (angular velocity)
"""

mutable struct AgentState
    position::Vector{Float64}      # [px, py]
    heading::Float64               # θ
    speed::Float64                 # v
    turn_rate::Float64             # ω
    
    # Full state vector
    x::Vector{Float64}             # 5D state
    
    # EKF covariance
    P::Matrix{Float64}             # 5x5 covariance
    
    # Agent ID
    id::Int
end

"""
Create a fixed-wing agent with initial state.
"""
function create_agent(id::Int, 
                     pos::Vector{Float64},
                     heading::Float64=0.0,
                     speed::Float64=2.0)
    
    x = zeros(5)
    x[1:2] = pos
    x[3] = heading
    x[4] = speed  # constant velocity
    x[5] = 0.0    # initial turn rate
    
    # Initial covariance
    P = diagm([1.0, 1.0, 0.1, 0.01, 0.01])
    
    return AgentState(pos, heading, speed, 0.0, x, P, id)
end

"""
Get state transition matrix for Dubins vehicle (fixed-wing) model.

Nonlinear dynamics:
    ṗx = v * cos(θ)
    ṗy = v * sin(θ)
    θ̇ = ω
    v̇ = 0  (constant velocity)
    ω̇ = 0  (turn rate changes slowly, modeled as random walk)

Linearized discrete-time: x_{k+1} ≈ F * x_k + process_noise
"""
function get_state_transition_matrix(dt::Float64, θ::Float64, v::Float64)
    F = Matrix{Float64}(I, 5, 5)
    
    # Position updates (linearized around current heading)
    F[1, 3] = -v * sin(θ) * dt  # ∂px/∂θ
    F[1, 4] = cos(θ) * dt        # ∂px/∂v
    
    F[2, 3] = v * cos(θ) * dt    # ∂py/∂θ
    F[2, 4] = sin(θ) * dt        # ∂py/∂v
    
    # Heading update from turn rate
    F[3, 5] = dt                 # θ += ω * dt
    
    # Velocity and turn rate are constant (no dynamics)
    # F[4,4] = 1, F[5,5] = 1 (identity)
    
    return F
end

"""
Nonlinear state propagation for Dubins vehicle.
"""
function propagate_state_nonlinear(x::Vector{Float64}, dt::Float64)
    px, py, θ, v, ω = x
    
    # Exact integration for Dubins vehicle
    if abs(ω) < 1e-6
        # Straight line motion
        px_new = px + v * cos(θ) * dt
        py_new = py + v * sin(θ) * dt
        θ_new = θ
    else
        # Circular arc
        radius = v / ω
        dθ = ω * dt
        
        px_new = px + radius * (sin(θ + dθ) - sin(θ))
        py_new = py - radius * (cos(θ + dθ) - cos(θ))
        θ_new = θ + dθ
    end
    
    return [px_new, py_new, θ_new, v, ω]
end

"""
Get process noise covariance matrix for fixed-wing model.
"""
function get_process_noise(dt::Float64, σ_process::Float64=0.1)
    Q = diagm([
        σ_process^2 * dt^2,  # Position x uncertainty
        σ_process^2 * dt^2,  # Position y uncertainty
        σ_process^2 * dt^2,  # Heading uncertainty
        0.001,               # Speed uncertainty (nearly constant)
        σ_process^2          # Turn rate uncertainty
    ])
    
    return Q
end

"""
Propagate agent state forward in time using Dubins vehicle model.
"""
function propagate_state(agent::AgentState, dt::Float64, 
                        σ_process::Float64=0.1; add_noise::Bool=true)
    
    # Nonlinear propagation
    x_new = propagate_state_nonlinear(agent.x, dt)
    
    # Add process noise if requested
    if add_noise
        Q = get_process_noise(dt, σ_process)
        noise = sqrt.(diag(Q)) .* randn(5)
        x_new += noise
    end
    
    # Update agent state
    agent.x = x_new
    agent.position = x_new[1:2]
    agent.heading = x_new[3]
    agent.speed = x_new[4]
    agent.turn_rate = x_new[5]
    
    return agent
end

"""
Generate a trajectory for an agent.

Trajectory types:
    - :linear: Straight line with constant velocity
    - :circular: Circular path
    - :random_walk: Random walk
"""
function generate_trajectory(agent::AgentState, 
                            duration::Float64, 
                            dt::Float64;
                            trajectory_type::Symbol=:linear,
                            params::Dict=Dict())
    
    n_steps = round(Int, duration / dt)
    trajectory = Vector{Vector{Float64}}(undef, n_steps)
    
    for i in 1:n_steps
        if trajectory_type == :linear
            # Constant velocity motion
            agent.acceleration = zeros(2)
            
        elseif trajectory_type == :circular
            # Circular motion
            radius = get(params, :radius, 5.0)
            ω_circle = get(params, :angular_speed, 0.5)
            
            center = get(params, :center, [0.0, 0.0])
            t = i * dt
            
            # Position on circle
            agent.position = center + radius * [cos(ω_circle * t), sin(ω_circle * t)]
            agent.velocity = radius * ω_circle * [-sin(ω_circle * t), cos(ω_circle * t)]
            agent.acceleration = -radius * ω_circle^2 * [cos(ω_circle * t), sin(ω_circle * t)]
            agent.heading = ω_circle * t + π/2
            agent.angular_vel = ω_circle
            
        elseif trajectory_type == :random_walk
            # Random acceleration
            max_acc = get(params, :max_acceleration, 1.0)
            agent.acceleration = max_acc * randn(2)
        end
        
        propagate_state(agent, dt, 0.1, add_noise=false)
        trajectory[i] = copy(agent.position)
    end
    
    return trajectory
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Agent Dynamics Test ===\n")
    
    # Create agent
    agent = create_agent(1, [0.0, 0.0], [1.0, 0.5])
    println("Initial state:")
    println("  Position: $(agent.position)")
    println("  Velocity: $(agent.velocity)")
    println("  Heading: $(agent.heading)")
    
    # Propagate forward
    dt = 0.1
    for i in 1:10
        propagate_state(agent, dt, 0.1, add_noise=true)
    end
    
    println("\nAfter 1 second:")
    println("  Position: $(agent.position)")
    println("  Velocity: $(agent.velocity)")
    
    # Generate circular trajectory
    agent2 = create_agent(2, [5.0, 0.0], [0.0, 2.5])
    traj = generate_trajectory(agent2, 10.0, 0.1, 
                              trajectory_type=:circular,
                              params=Dict(:radius => 5.0, :angular_speed => 0.5))
    
    println("\nGenerated circular trajectory with $(length(traj)) points")
end
