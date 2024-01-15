using SIR, Random, Graphs, StochasticAD, Zygote, PyPlot, ForwardDiff, GraphNeuralNetworks, JLD2, Statistics

include("./plotting.jl")

#Random.seed!(3)
## 

abstract type DiffMode end
struct FW <: DiffMode end
struct RV <: DiffMode end
struct DS <: DiffMode end

get_jacobian(alg::FW, f, x) = ForwardDiff.jacobian(f, x)
get_jacobian(alg::RV, f, x) = Zygote.jacobian(f, x)[1]
get_derivative(alg::FW, f, x) = ForwardDiff.derivative(f, x)
get_derivative(alg::RV, f, x) = Zygote.jacobian(f, x)[1]
get_derivative(alg::DS, f, x) = derivative_estimate(f, x)
get_jacobian(alg::DS, f, x) = derivative_estimate(f, x)

function get_agent_jacobian(diff_mode, bernoulli_grad, beta, agent_parameters)
    function f(x)
        sol = stack(SIR.run(agent_parameters.graph, x[1], agent_parameters.gamma, agent_parameters.initial_fraction_infected,
            agent_parameters.n_timesteps, agent_parameters.delta_t,
            agent_parameters.n_agents, bernoulli_grad))
        return sol
    end
    deriv = get_derivative(diff_mode, f, beta)
    ret = Matrix{Any}(undef, 2, 1)
    ret[1,1] = deriv[:,1]
    ret[2,1] = deriv[:,2]
    return ret
end

function get_parameters()
    n_timesteps = 50
    n_agents = 1000
    initial_fraction_infected = 0.1
    bernoulli_grad = SAD()
    beta = 0.1
    gamma = 0.0
    delta_t = 1.0
    graph = complete_graph(n_agents)
    agent_parameters = AgentParameters(graph, beta, gamma, initial_fraction_infected, bernoulli_grad, n_timesteps, delta_t, n_agents);
    ode_parameters = ODEParameters(beta, gamma, initial_fraction_infected, n_timesteps, delta_t);
    return agent_parameters, ode_parameters
end

function get_analytical_and_agent_curves()
    agent_parameters, _ = get_parameters();
    an_sol = run_si(agent_parameters.beta, agent_parameters.initial_fraction_infected,
        agent_parameters.n_agents, agent_parameters.n_timesteps, agent_parameters.delta_t)
    S_t, I_t = an_sol[:, 1], an_sol[:, 2];
    agents_sol = stack(SIR.run(agent_parameters))
    return [S_t, I_t], [agents_sol[:,1], agents_sol[:,2]]
end

curves = get_analytical_and_agent_curves();
f = plot_curves(curves, ["ode", "agents"]);
display(f)
##

function get_agent_jacobians()
    agent_parameters, _ = get_parameters();
    gs = mean([get_agent_jacobian(FW(), GS(), agent_parameters.beta, agent_parameters) for i in 1:1])
    sad = mean([get_agent_jacobian(FW(), SAD(), agent_parameters.beta, agent_parameters) for i in 1:1]);
    ds = mean([get_agent_jacobian(DS(), SAD(), agent_parameters.beta, agent_parameters) for i in 1:1]);
    return gs, sad, ds
end
##
agent_p, _ = get_parameters();
println("Computing gradients...")

analytical_jacobian = analytical_si_gradients(agent_p.beta, agent_p.initial_fraction_infected,
    agent_p.n_agents, agent_p.n_timesteps, agent_p.delta_t);

@time agent_jacobian_gs, agent_jacobian_sad, agent_jacobian_ds = get_agent_jacobians();
f = plot_jacobians([agent_jacobian_gs, agent_jacobian_sad, agent_jacobian_ds, analytical_jacobian],
	["agent GS", "agent SAD", "agent DS", "analytical"],
	figsize = (15, 7),
    qs=["S", "I"],
    xs=["β"])
display(f)
