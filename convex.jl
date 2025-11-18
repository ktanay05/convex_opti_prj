using Convex
using SCS
using LinearAlgebra
using Random
using Plots
import MathOptInterface

# ----- DATA -----
L = 30
Random.seed!(43)
B = [rand(3) for i in 1:L]
# Example: random radii between 0.4 and 0.9 for each beacon
    # Set a pseudo-true position for r
    r_true = [0.5, 0.5, 0.5]
    # Calculate true ranges (Euclidean distance) from each beacon to r_true
    rho_true = [norm(B[i] - r_true) for i in 1:L]
    # Add Gaussian noise to simulate measurement error
    noise_std = 0.5
    rho = rho_true .+ noise_std .* abs.(randn(L))

# ----- SDP MODEL -----
"""
Rewrite using Convex.jl and SCS
"""
r = Variable(3)
P = Semidefinite(3)
lambda = Variable(L)

constraints = []
push!(constraints, lambda >= 0)
for i in 1:L
    a = rho[i] - lambda[i]
    b = r - B[i]
    C = rho[i] * I(3)
    D = P
    E = lambda[i] * I(3)
    row1 = hcat(a, b', zeros(1,3))
    row2 = hcat(b, C, D)
    row3 = hcat(zeros(3,1), D, E)
    M = vcat(row1, row2, row3)
        push!(constraints, M ⪰ 0)
end

problem = maximize(logdet(P), constraints)
solve!(problem, SCS.Optimizer)


# Always plot beacons, estimated position, spheres, and ellipsoid if solution is available
if problem.status == MathOptInterface.OPTIMAL || problem.status == MathOptInterface.FEASIBLE_POINT
    r_sol = evaluate(r)
    P_sol = evaluate(P)
    println("Best location estimate (r):")
    println(r_sol)
    println("\nEllipsoid shape matrix (P):")
    println(P_sol)

    # ----- PLOTTING -----
    plt = plot(title="Beacons and Maximum Volume Ellipsoid",
               legend=false, size=(600,600), aspect_ratio=:equal)

    # Plot beacons as spheres
    θ = range(0, 2π, length=80)
    ϕ = range(0, π, length=40)
    #for i in 1:L
    #    local xs = Float64[]
    #    local ys = Float64[]
        # Plot each beacon as a semi-transparent sphere surface
    #    for i in 1:L
    #        X = [B[i][1] + rho[i] * sin(p)*cos(t) for t in θ, p in ϕ]
    #        Y = [B[i][2] + rho[i] * sin(p)*sin(t) for t in θ, p in ϕ]
    #        Z = [B[i][3] + rho[i] * cos(p) for t in θ, p in ϕ]
    #        surface!(plt, X, Y, Z, color=:purple, alpha=0.0001, legend=false)
    #    end

    # Plot ellipsoid surface points
    xs = Float64[]; ys = Float64[]; zs = Float64[]
    for t in θ, p in ϕ
        u = [sin(p)*cos(t), sin(p)*sin(t), cos(p)]    # unit sphere point
        pt = r_sol + P_sol * u                       # map via ellipsoid
        push!(xs, pt[1]); push!(ys, pt[2]); push!(zs, pt[3])
    end
    scatter!(plt, xs, ys, zs, markersize=2, markercolor=:red, alpha=0.6, label="Maximum Volume Ellipsoid")

    # Show estimated center as a large red dot
    scatter!(plt, [r_sol[1]], [r_sol[2]], [r_sol[3]], markersize=2, color=:green, markerstrokewidth=0, label="Estimate")

    # Show beacon centers as black dots
    for i in 1:L
        scatter!(plt, [B[i][1]], [B[i][2]], [B[i][3]], markerstrokewidth=0, markersize=5, color=:black, label=nothing)
    end

    # Save three different views
    savefig(plt, "ellipsoid_plot_default.png")
    println("Plot saved as ellipsoid_plot_default.png")

    # Top view (z-axis up)
    plot!(plt, camera=(0, 90))
    savefig(plt, "ellipsoid_plot_top.png")
    println("Plot saved as ellipsoid_plot_top.png")

    # Side view (y-axis up)
    plot!(plt, camera=(90, 0))
    savefig(plt, "ellipsoid_plot_side.png")
    println("Plot saved as ellipsoid_plot_side.png")
else
    println("Problem not solved to optimality. Status: ", problem.status)
end



# Optional: print beacon info
println("\nBeacon positions:")
for i in 1:L
    println("Beacon $i: ", B[i], ", rho = ", rho[i])
end