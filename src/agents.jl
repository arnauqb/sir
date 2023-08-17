export SAD, GS, run, AgentParameters

abstract type BernoulliGrad end
struct SAD <: BernoulliGrad end
struct GS <: BernoulliGrad
	tau::Float64
end
GS() = GS(0.1)

struct AgentParameters
    graph::Graphs.Graph
    beta::Float64
    gamma::Float64
    initial_fraction_infected::Float64
    bernoulli_grad::BernoulliGrad
    n_timesteps::Int64
    delta_t::Float64
    n_agents::Int64
end

using ChainRulesCore, ForwardDiff
stop_gradient(x) = ChainRulesCore.ignore_derivatives(x)
function stop_gradient(n::ForwardDiff.Dual{T}) where {T}
    ForwardDiff.Dual{T}(ForwardDiff.value(n), ForwardDiff.partials(1))
end
stop_gradient(n::Array{<:ForwardDiff.Dual}) = stop_gradient.(n)

function sample_bernoulli(bernoulli_grad::SAD, probs)
	return [rand(Bernoulli(p)) for p in probs]
end

function sample_bernoulli(bernoulli_grad::GS, probs)
    probs_cat = [probs 1.0 .- probs]
	return sample_gumbel_softmax(probs_cat, bernoulli_grad.tau)[:, 1]
end

function initialize(graph, initial_fraction_infected, bernoulli_grad)
	n_agents = nv(graph)
    gnn = ignore() do
	    GNNGraph(graph, ndata = (; aux = ones(n_agents)))
    end
	#probs = initial_fraction_infected * ones(n_agents)
	#is_infected = sample_bernoulli(bernoulli_grad, probs)
    n_infected = round(Int(n_agents * initial_fraction_infected))
    is_infected = vcat(ones(n_infected), zeros(n_agents - n_infected))
    S = 1.0 .- is_infected
    I = is_infected
    R = zeros(n_agents)
	return gnn, S, I, R
end

function step(gnn, S, I, R, beta, gamma, delta_t, bernoulli_grad)
	trans = propagate((xi, xj, e) -> xi .* xj, gnn, +, xi = S, xj = I)
	n_neighbours = propagate((xi, xj, e) -> xi .+ xj, gnn, +, xi = gnn.ndata[:aux], xj = gnn.ndata[:aux]) ./ 2
	prob_infection = @. 1.0 - exp(-trans / n_neighbours * beta * delta_t)
	new_infected = sample_bernoulli(bernoulli_grad, prob_infection)
	prob_recovery = @. 1.0 - exp(-gamma * delta_t * I)
	new_recovered = sample_bernoulli(bernoulli_grad, prob_recovery)
    S = S - new_infected
    I = I + new_infected - new_recovered
    R = R + new_recovered
	return S, I, R
end

function observe(S, I, R, N)
	return sum(S) / N, sum(I) / N, sum(R) / N
end

function Base.run(g, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, n_agents, bernoulli_grad = SAD())
	gnn, S, I, R = initialize(g, initial_fraction_infected, bernoulli_grad)
	S_t, I_t, R_t = observe(S, I, R, n_agents)
	for t in 2:n_timesteps
		S, I, R = step(gnn, S, I, R, beta, gamma, delta_t, bernoulli_grad)
		S_t_i, I_t_i, R_t_i = observe(S, I, R, n_agents)
        S_t = [S_t; S_t_i]
        I_t = [I_t; I_t_i]
        R_t = [R_t; R_t_i]
	end
	return S_t, I_t, R_t
end

function Base.run(p::AgentParameters)
    return run(p.graph, p.beta, p.gamma, p.initial_fraction_infected, p.n_timesteps, p.delta_t, p.n_agents, p.bernoulli_grad)
end