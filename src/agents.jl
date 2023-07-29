export SAD, GS, run

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
    gnn = ignore() do
	    GNNGraph(graph, ndata = (; aux = ones(n_agents)))
    end
	probs = initial_fraction_infected * ones(n_agents)
	is_infected = sample_bernoulli(diff_mode, probs)
    S = 1.0 .- is_infected
    I = is_infected
    R = zeros(n_agents)
	return gnn, S, I, R
end

function step(gnn, S, I, R, beta, gamma, delta_t, diff_mode)
	# aggregate transmission * susceptibility
	trans = propagate((xi, xj, e) -> xi .* xj, gnn, +, xi = S, xj = I)
	n_neighbours = propagate((xi, xj, e) -> xi .+ xj, gnn, +, xi = gnn.ndata[:aux], xj = gnn.ndata[:aux]) ./ 2
	prob_infection = @. 1.0 - exp(-trans / n_neighbours * beta * delta_t)
	new_infected = sample_bernoulli(diff_mode, prob_infection)
	prob_recovery = gamma * I
	new_recovered = sample_bernoulli(diff_mode, prob_recovery)
    S = S - new_infected
    I = I + new_infected - new_recovered
    R = R + new_recovered
	return S, I, R
end

function observe(S, I, R)
	return sum(S), sum(I), sum(R)
end

function Base.run(g, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, diff_mode = SAD())
	gnn, S, I, R = initialize(g, initial_fraction_infected, diff_mode)
	S_t, I_t, R_t = observe(S, I, R)
	for t in 2:n_timesteps
		S, I, R = step(gnn, S, I, R, beta, gamma, delta_t, diff_mode)
		S_t_i, I_t_i, R_t_i = observe(S, I, R)
        # concatenate
        S_t = [S_t; S_t_i]
        I_t = [I_t; I_t_i]
        R_t = [R_t; R_t_i]
	end
	return S_t, I_t, R_t
end