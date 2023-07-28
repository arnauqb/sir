module SIR
export run, SAD, GS

using GraphNeuralNetworks, Distributions, GumbelSoftmax, Graphs

export SAD, GS

abstract type DiffMode end
struct SAD <: DiffMode end
struct GS
	tau::Float64
end
GS() = GS(0.1)

function sample_bernoulli(diff_mode::SAD, probs)
	probs = max.(min.(probs, 1.0), 0.0)
	return [rand(Bernoulli(p)) for p in probs]
end

function sample_bernoulli(diff_mode::GS, probs)
	probs_cat = max.(min.([probs 1.0 .- probs], 1.0), 0.0)
	return sample_gumbel_softmax(probs_cat, diff_mode.tau)[:, 1]
end

function initialize(graph, initial_fraction_infected, diff_mode)
	n_agents = nv(graph)
	gnn = GNNGraph(graph, ndata = (; S = ones(n_agents), I = zeros(n_agents), R = zeros(n_agents), aux = ones(n_agents)))
	probs = initial_fraction_infected * ones(n_agents)
	is_infected = sample_bernoulli(diff_mode, probs)
	gnn.ndata[:I] .= is_infected
	gnn.ndata[:S] .= gnn.ndata[:S] - is_infected
	return gnn
end

function step(gnn, beta, gamma, delta_t, diff_mode)
	# aggregate transmission * susceptibility
	trans = propagate((xi, xj, e) -> xi .* xj, gnn, +, xi = gnn.ndata[:S], xj = gnn.ndata[:I])
	n_neighbours = propagate((xi, xj, e) -> xi .+ xj, gnn, +, xi = gnn.ndata[:aux], xj = gnn.ndata[:aux]) ./ 2
	prob_infection = @. 1.0 - exp(-trans / n_neighbours * beta * delta_t)
	new_infected = sample_bernoulli(diff_mode, prob_infection)
	prob_recovery = gamma * gnn.ndata[:I]
	new_recovered = sample_bernoulli(diff_mode, prob_recovery)
	gnn.ndata[:S] .= gnn.ndata[:S] - new_infected
	gnn.ndata[:I] .= gnn.ndata[:I] + new_infected - new_recovered
	gnn.ndata[:R] .= gnn.ndata[:R] + new_recovered
	return gnn
end

function observe(gnn)
	return sum(gnn.ndata[:S]), sum(gnn.ndata[:I]), sum(gnn.ndata[:R])
end

function Base.run(g, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, diff_mode = SAD())
	S_t = zeros(n_timesteps)
	I_t = zeros(n_timesteps)
	R_t = zeros(n_timesteps)
	gnn = initialize(g, initial_fraction_infected, diff_mode)
	S_t[1], I_t[1], R_t[1] = observe(gnn)
	for t in 2:n_timesteps
		gnn = step(gnn, beta, gamma, delta_t, diff_mode)
		S_t[t], I_t[t], R_t[t] = observe(gnn)
	end
	return S_t, I_t, R_t
end
end # module

using .SIR, Graphs, Plots, Random

Random.seed!(1)

n_timesteps = 100
n_agents = 1000
initial_fraction_infected = 0.1
beta = 0.1
gamma = 0.05
diff_mode = Main.SIR.SAD()
delta_t = 1.0
graph = complete_graph(n_agents)

S_t, I_t, R_t = SIR.run(graph, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, diff_mode)

# plot the results normalized with labels, title and save to file.
plot(1:n_timesteps, S_t ./ n_agents, label = "S", title = "SIR Julia", xlabel = "t", ylabel = "fraction", legend = :bottomright, dpi = 150)
plot!(1:n_timesteps, I_t ./ n_agents, label = "I")
plot!(1:n_timesteps, R_t ./ n_agents, label = "R")
savefig("sir_julia.png")
