using JuMP
using SCS
using Polyhedra
using CDDLib
using Graphs
using Plots
using LinearAlgebra
using Random
using Statistics
using VoronoiDelaunay

# Mapping functions
domain_min = 0.0
domain_max = 4.0
vd_min = 1.1
vd_max = 1.9

function to_vd(val)
    return vd_min + (val - domain_min) / (domain_max - domain_min) * (vd_max - vd_min)
end

function from_vd(val)
    return domain_min + (val - vd_min) / (vd_max - vd_min) * (domain_max - domain_min)
end

# ==============================================================================
# 1. Environment Setup & Decomposition
# ==============================================================================

include("convex_decomposition.jl")
using .ConvexDecomposition

# Helper to convert between types
function to_point(v::Vector{Float64})
    return ConvexDecomposition.Point(v[1], v[2])
end

function from_point(p::ConvexDecomposition.Point)
    return [p.x, p.y]
end

function generate_environment()
    # 1. Generate Convex Obstacles
    obstacles = []
    holes_points = Vector{Vector{ConvexDecomposition.Point}}()
    
    max_obstacles = 8
    tries = 0
    while length(obstacles) < max_obstacles && tries < 1000
        c = [0.5 + 3.0*rand(), 0.5 + 3.0*rand()]
        # Generate random points
        pts = [c .+ 0.6 .* (rand(2) .- 0.5) for _ in 1:10]
        
        # Compute Convex Hull using Polyhedra/CDDLib
        v = vrep(pts)
        p = polyhedron(v, CDDLib.Library())
        removevredundancy!(p)
        
        # Get ordered vertices for the hole
        # Polyhedra doesn't guarantee order, so we sort angularly (CCW)
        verts = collect(points(p))
        cen = mean(verts)
        sort!(verts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        
        # Check if valid and non-overlapping
        poly_obj = polyhedron(vrep(verts), CDDLib.Library())
        if volume(poly_obj) > 0.05 && all(isempty(intersect(poly_obj, o)) for o in obstacles)
            push!(obstacles, poly_obj)
            
            # Create inflated hole for decomposition (Safety Margin)
            # Scale vertices relative to center
            margin_scale = 1.2 # 20% inflation
            inflated_verts = [cen .+ (v .- cen) .* margin_scale for v in verts]
            
            # IMPORTANT: Holes must be Clockwise (CW) for the decomposition logic
            # (which assumes "Left" is inside the polygon).
            # sort! gave us CCW, so we reverse it.
            reverse!(inflated_verts)
            
            # Convert to Points for decomposition
            push!(holes_points, [to_point(v) for v in inflated_verts])
        end
        tries += 1
    end

    # 2. Define Domain (Free Space Boundary)
    # CCW order: (0,0) -> (4,0) -> (4,4) -> (0,4)
    domain_points = [
        ConvexDecomposition.Point(0.0, 0.0),
        ConvexDecomposition.Point(4.0, 0.0),
        ConvexDecomposition.Point(4.0, 4.0),
        ConvexDecomposition.Point(0.0, 4.0)
    ]

    # 3. Decompose
    # The convex_decompose function handles merging holes into the domain and then partitioning.
    polys_points = ConvexDecomposition.convex_decompose(domain_points, holes_points)
    
    # 4. Convert back to Polyhedra regions
    regions = []
    for pp in polys_points
        if length(pp) < 3; continue; end
        # Convert to Vector{Vector{Float64}}
        verts = [from_point(p) for p in pp]
        push!(regions, polyhedron(vrep(verts), CDDLib.Library()))
    end
    
    return obstacles, regions
end

# Retry loop
max_retries = 20
obstacles = []
regions = []
adj_matrix = Matrix{Bool}(undef, 0, 0)
start_region_idx = 0
goal_region_idx = 0
start_pos = [0.0, 0.0]
goal_pos = [0.0, 0.0]
connected = false

for attempt in 1:max_retries
    println("Attempt $attempt...")
    global obstacles, regions = generate_environment()
    
    if isempty(regions)
        continue
    end

    global n_regions = length(regions)
    global adj_matrix = zeros(Bool, n_regions, n_regions)

    # Build graph by checking shared boundaries
    # Two polygons are adjacent if they share an edge (segment)
    # We can check if they share at least 2 vertices.
    
    # Helper to get vertices of a region
    function get_verts(reg)
        vs = collect(points(vrep(reg)))
        # Snap to grid to avoid float issues? 
        # Or just use a tolerance equality
        return vs
    end
    
    region_verts = [get_verts(r) for r in regions]
    
    for i in 1:n_regions
        for j in i+1:n_regions
            # Check for shared edge
            # Count shared vertices
            shared = 0
            for v1 in region_verts[i]
                for v2 in region_verts[j]
                    if norm(v1 - v2) < 1e-4
                        shared += 1
                    end
                end
            end
            
            if shared >= 2
                adj_matrix[i, j] = true
                adj_matrix[j, i] = true
            end
        end
    end
    
    # Pick Start/Goal
    # Pick random regions? Or first and last?
    # Let's pick regions far apart.
    centroids = [mean(get_verts(r)) for r in regions]
    
    # Find pair with max distance
    max_dist = -1.0
    best_pair = (1, 1)
    for i in 1:n_regions, j in 1:n_regions
        d = norm(centroids[i] - centroids[j])
        if d > max_dist
            max_dist = d
            best_pair = (i, j)
        end
    end
    
    global start_region_idx = best_pair[1]
    global goal_region_idx = best_pair[2]
    
    # BFS Check
    q = [start_region_idx]
    visited = Set([start_region_idx])
    found = false
    while !isempty(q)
        u = popfirst!(q)
        if u == goal_region_idx
            found = true
            break
        end
        for v in 1:n_regions
            if adj_matrix[u, v] && !(v in visited)
                push!(visited, v)
                push!(q, v)
            end
        end
    end
    
    if found
        println("Connected environment found on attempt $attempt.")
        global connected = true
        global start_pos = centroids[start_region_idx]
        global goal_pos = centroids[goal_region_idx]
        break
    end
end

if !connected
    error("Could not generate a connected environment after $max_retries attempts.")
end

println("Generated $(length(obstacles)) obstacles and $(length(regions)) free regions.")
println("Start Region: $start_region_idx (Pos: $start_pos)")
println("Goal Region: $goal_region_idx (Pos: $goal_pos)")

# ==============================================================================
# 3. GCS Trajectory Optimization (MICP)
# ==============================================================================

# Parameters
bezier_degree = 2 # Quadratic Bezier
# 4. Define Optimization Model (SDP Relaxation)
# We use SCS for SDP. Note: SCS does not support integer variables, so we relax them.
model = Model(SCS.Optimizer)
set_silent(model)

# Variables
# Relaxed binary variables (0 <= y <= 1)
@variable(model, 0 <= y[1:n_regions] <= 1)
@variable(model, 0 <= z[1:n_regions, 1:n_regions] <= 1)

# Control points
@variable(model, x[1:n_regions, 0:bezier_degree, 1:2])

# Slack variables for L2 norm (t >= ||v||_2)
@variable(model, t[1:n_regions, 1:bezier_degree] >= 0)

# Big-M constant
M = 10.0

# Constraints

# 1. Flow Conservation
# Start
@constraint(model, sum(z[start_region_idx, j] for j in 1:n_regions if adj_matrix[start_region_idx, j]) - 
                   sum(z[j, start_region_idx] for j in 1:n_regions if adj_matrix[j, start_region_idx]) == 1)
# Goal
@constraint(model, sum(z[goal_region_idx, j] for j in 1:n_regions if adj_matrix[goal_region_idx, j]) - 
                   sum(z[j, goal_region_idx] for j in 1:n_regions if adj_matrix[j, goal_region_idx]) == -1)
# Intermediate
for i in 1:n_regions
    if i != start_region_idx && i != goal_region_idx
        @constraint(model, sum(z[i, j] for j in 1:n_regions if adj_matrix[i, j]) - 
                           sum(z[j, i] for j in 1:n_regions if adj_matrix[j, i]) == 0)
    end
end

# Connectivity & Capacity
for i in 1:n_regions
    # Outgoing
    @constraint(model, y[i] >= sum(z[i, j] for j in 1:n_regions if adj_matrix[i, j]))
    # Incoming
    @constraint(model, y[i] >= sum(z[j, i] for j in 1:n_regions if adj_matrix[j, i]))
end

# 2. Containment (H-rep)
for i in 1:n_regions
    poly = regions[i]
    # Get H-rep: Ax <= b
    h = hrep(poly)
    # We need to iterate over halfspaces
    for hp in halfspaces(h)
        n = hp.a
        b = hp.β
        for k in 0:bezier_degree
            # n' * x <= b + M(1-y)
            @constraint(model, dot(n, x[i, k, :]) <= b + M * (1 - y[i]))
        end
    end
    
    # Force x=0 if y=0
    for k in 0:bezier_degree
        for d in 1:2
            @constraint(model, x[i, k, d] <= M * y[i])
            @constraint(model, x[i, k, d] >= -M * y[i])
        end
    end
end

# 3. Continuity (C0)
for i in 1:n_regions
    for j in 1:n_regions
        if adj_matrix[i, j]
            # x[i,K] == x[j,0] if z[i,j]=1
            for d in 1:2
                @constraint(model, x[i, bezier_degree, d] - x[j, 0, d] <= M * (1 - z[i, j]))
                @constraint(model, x[i, bezier_degree, d] - x[j, 0, d] >= -M * (1 - z[i, j]))
            end
        else
            # Fix z=0 if not adjacent (already handled by bounds/loops but good for safety)
            fix(z[i, j], 0.0; force=true)
        end
    end
end

# 4. Heading Consistency (C1)
# v_out = x[i,K] - x[i,K-1]
# v_in = x[j,1] - x[j,0]
for i in 1:n_regions
    for j in 1:n_regions
        if adj_matrix[i, j]
            for d in 1:2
                diff = (x[i, bezier_degree, d] - x[i, bezier_degree-1, d]) - (x[j, 1, d] - x[j, 0, d])
                @constraint(model, diff <= M * (1 - z[i, j]))
                @constraint(model, diff >= -M * (1 - z[i, j]))
            end
        end
    end
end

# 5. Boundary Conditions
@constraint(model, x[start_region_idx, 0, 1] == start_pos[1])
@constraint(model, x[start_region_idx, 0, 2] == start_pos[2])
@constraint(model, x[goal_region_idx, bezier_degree, 1] == goal_pos[1])
@constraint(model, x[goal_region_idx, bezier_degree, 2] == goal_pos[2])

# 6. Objective: Minimize Path Length (L2 Norm) via SDP
# We want to minimize sum(t[i,k]) where t[i,k] >= ||x[i,k] - x[i,k-1]||_2
# Schur Complement:
# [ t*I      v ]
# [ v'       t ]  >= 0
# where v = x[i,k] - x[i,k-1]

for i in 1:n_regions
    for k in 1:bezier_degree
        # Define velocity vector v
        v = [x[i, k, 1] - x[i, k-1, 1], x[i, k, 2] - x[i, k-1, 2]]
        
        # Construct the matrix for PSD constraint
        # [ t  0  v1 ]
        # [ 0  t  v2 ]
        # [ v1 v2 t  ]
        # This is equivalent to t*I_2 - v*v'/t >= 0 => t^2 >= ||v||^2
        
        # JuMP requires the matrix to be symmetric
        # We build the upper triangle
        # M_sdp = [
        #   t[i,k]   0        v[1]
        #   0        t[i,k]   v[2]
        #   v[1]     v[2]     t[i,k]
        # ]
        
        # Using LinearAlgebra.Symmetric or just passing the matrix to PSDCone
        # Note: JuMP's PSDCone expects a vectorized form or a matrix.
        # Let's use the matrix form.
        
        # We need to be careful with variable references in the matrix.
        # [t 0 v1; 0 t v2; v1 v2 t]
        
        # Row 1
        m11 = t[i, k]
        m12 = 0.0
        m13 = v[1]
        
        # Row 2
        m21 = 0.0
        m22 = t[i, k]
        m23 = v[2]
        
        # Row 3
        m31 = v[1]
        m32 = v[2]
        m33 = t[i, k]
        
        # Construct matrix
        M_sdp = [m11 m12 m13; m21 m22 m23; m31 m32 m33]
        
        @constraint(model, M_sdp in PSDCone())
    end
end

@objective(model, Min, sum(t))

println("Solving GCS optimization with SCS (SDP Relaxation)...")
optimize!(model)
println("Status: ", termination_status(model))

# ==============================================================================
# 4. Visualization
# ==============================================================================

plt = plot(title="GCS Path Planning", size=(700, 700), legend=false, aspect_ratio=:equal)

# Plot Obstacles
for obs in obstacles
    pts = collect(points(vrep(obs)))
    if !isempty(pts)
        cen = mean(pts)
        sort!(pts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        push!(pts, pts[1]) 
        plot!(plt, [v[1] for v in pts], [v[2] for v in pts], seriestype=:shape, color=:red, fillalpha=0.5)
    end
end

# Plot Regions
for (idx, r) in enumerate(regions)
    pts = collect(points(vrep(r)))
    if !isempty(pts)
        cen = mean(pts)
        sort!(pts, by = v -> atan(v[2]-cen[2], v[1]-cen[1]))
        push!(pts, pts[1])
        plot!(plt, [v[1] for v in pts], [v[2] for v in pts], seriestype=:shape, color=:blue, fillalpha=0.1, linealpha=0.2)
        annotate!(plt, cen[1], cen[2], text("$idx", 8, :gray))
    end
end

# Plot Edges (Graph)
for i in 1:n_regions, j in 1:n_regions
    if adj_matrix[i, j]
        c1 = mean(collect(points(vrep(regions[i]))))
        c2 = mean(collect(points(vrep(regions[j]))))
        plot!(plt, [c1[1], c2[1]], [c1[2], c2[2]], color=:yellow, alpha=0.5, linewidth=1.5)
    end
end

# Plot Solution (Flow)
if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.ALMOST_OPTIMAL
    println("Solution found!")
    
    # Since we have a relaxed solution (flow), we plot all active regions/trajectories
    # weighted by their activation value y[i].
    
    for i in 1:n_regions
        val_y = value(y[i])
        if val_y > 0.01 # Threshold for visibility
            cps = [value.(x[i, k, :]) for k in 0:bezier_degree]
            
            # Plot Control Points
            # scatter!(plt, [p[1] for p in cps], [p[2] for p in cps], 
            #          color=:magenta, markersize=3, marker=:square, alpha=val_y, label="")
            
            # Plot Curve
            ts = range(0, 1, length=50)
            curve_pts = []
            for t in ts
                pt = (1-t)^2 .* cps[1] .+ 2*(1-t)*t .* cps[2] .+ t^2 .* cps[3]
                push!(curve_pts, pt)
            end
            
            plot!(plt, [p[1] for p in curve_pts], [p[2] for p in curve_pts], 
                  color=:green, linewidth=3*val_y, alpha=val_y, label="")
        end
    end
    
    scatter!(plt, [start_pos[1]], [start_pos[2]], color=:green, markersize=8, label="Start")
    scatter!(plt, [goal_pos[1]], [goal_pos[2]], color=:orange, markersize=8, label="Goal")
    
else
    println("No solution found. Status: ", termination_status(model))
end

display(plt)
savefig(plt, "gcs_solution.png")
println("Plot saved to gcs_solution.png")
