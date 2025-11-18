using Statistics
using Polyhedra
using CDDLib     # Polyhedra backend
using Graphs     # For graph construction
using Plots      # For plotting

# ---- 1. Obstacles and Free Regions ----

# Define obstacles (as convex polygons)
function make_obstacle(vertices)
    v = vrep(vertices)
    return polyhedron(v, CDDLib.Library())
end

obstacles = [
    make_obstacle([[1.0, 1.0], [1.5, 1.0], [1.5, 1.5], [1.0, 1.5]]),
    make_obstacle([[3.0, 3.0], [3.5, 3.0], [3.5, 3.5], [3.0, 3.5]])
]


# Generate a lot more convex regions (random rectangles)
using Random
Random.seed!(23)
num_regions = 25
region_size = 0.7
region_centers = [[rand()*4, rand()*4] for _ in 1:num_regions]
function make_rectangle(center, size)
    x, y = center
    s = size/2
    verts = [[x-s, y-s], [x+s, y-s], [x+s, y+s], [x-s, y+s]]
    v = vrep(verts)
    return polyhedron(v, CDDLib.Library())
end
regions = [make_rectangle(c, region_size) for c in region_centers]

# ---- 2. Graph Construction ----

n = length(regions)
g = Graphs.SimpleDiGraph(n)

# Connect adjacent regions (overlapping or touching)
function are_adjacent(r1, r2)
    !isempty(intersect(r1, r2))
end

for i = 1:n, j = 1:n
    if i != j && are_adjacent(regions[i], regions[j])
        Graphs.add_edge!(g, i, j)
    end
end

# Add 'start' and 'goal' positions as additional vertices
start_pos = [0.2, 0.2]
goal_pos = [3.8, 3.8]
Graphs.add_vertex!(g)   # n+1: start
Graphs.add_vertex!(g)   # n+2: goal

function in_region(pt, region)
    Polyhedra.in(region, pt)
end

for i in 1:n
    if in_region(start_pos, regions[i])
        Graphs.add_edge!(g, n+1, i)
    end
    if in_region(goal_pos, regions[i])
        Graphs.add_edge!(g, i, n+2)
    end
end

# ---- 3. Plotting ----

plt = plot(title="GCS Motion Planning Environment", size=(600,600), legend=false)


# Plot obstacles
for obs in obstacles
    verts = [point for point in Polyhedra.collect(Polyhedra.points(Polyhedra.vrep(obs)))]
    local xs = [v[1] for v in verts]
    local ys = [v[2] for v in verts]
    xs = push!(xs, xs[1]); ys = push!(ys, ys[1])
    plot!(xs, ys, color=:red, st=:shape, fillalpha=0.4)
end


# Plot free regions
for r in regions
    verts = [point for point in Polyhedra.collect(Polyhedra.points(Polyhedra.vrep(r)))]
    local xs = [v[1] for v in verts]
    local ys = [v[2] for v in verts]
    xs = push!(xs, xs[1]); ys = push!(ys, ys[1])
    plot!(xs, ys, color=:lightblue, st=:shape, fillalpha=0.3, lw=2)
end

# Plot region centers for clarity
centers = [vec(mean(reduce(hcat, [v for v in Polyhedra.collect(Polyhedra.points(Polyhedra.vrep(r)))]), dims=2)) for r in regions]
for c in centers
    scatter!([c[1]], [c[2]], marker=:circle, color=:blue)
end

# Plot edges between regions (as lines between centers)
for e in Graphs.edges(g)
    i, j = Graphs.src(e), Graphs.dst(e)
    if i <= n && j <= n
        x_coords = [centers[i][1], centers[j][1]]
        y_coords = [centers[i][2], centers[j][2]]
        plot!(x_coords, y_coords, lw=1, lc=:gray)
    end
end

# Plot start and goal positions
scatter!([start_pos[1]], [start_pos[2]], color=:green, marker=:star5, ms=12)
scatter!([goal_pos[1]], [goal_pos[2]], color=:orange, marker=:star5, ms=12)
# Plot path if it exists and is valid
if isdefined(Main, :region_path) && length(region_path) >= 2
    println("Region path (indices): ", region_path)
    valid_indices = [i for i in region_path[2:end-1] if 1 <= i <= length(centers)]
    intermediate_points = [centers[i] for i in valid_indices]
    all_points = [start_pos; intermediate_points; goal_pos]
    # Ensure all points are vectors of length 2
    path_points = [p isa AbstractVector && length(p) == 2 ? p : collect(p)[1:2] for p in all_points if length(p) >= 2]
    println("Path solution (coordinates):")
    for (i, pt) in enumerate(path_points)
        println("  Step $i: ", pt)
    end
    valid_indices = [i for i in region_path[2:end-1] if 1 <= i <= length(centers)]
    intermediate_points = [centers[i] for i in valid_indices]
    all_points = [start_pos; intermediate_points; goal_pos]
    # Ensure all points are vectors of length 2
    path_points = [p isa AbstractVector && length(p) == 2 ? p : collect(p)[1:2] for p in all_points if length(p) >= 2]
    xs = [p[1] for p in path_points]
    ys = [p[2] for p in path_points]
    plot!(xs, ys, lw=3, color=:black, marker=:circle, label="Path")
end

display(plt)
savefig(plt, "gcs_plot.png")
# ---- 4. Next Steps: Optimization ----

# To complete the GCS motion planner, you would:
# - Assign Bézier curve variables per region
# - Build convex constraints as per the GCS paper
# - Solve for path and trajectory using JuMP and a suitable solver
# - Perform rounding as per Section 4.2 in the source
# Please refer to optimization example provided earlier and the GitHub repository for a full implementation.
