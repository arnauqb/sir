using SIR, Random, Graphs, StochasticAD, Zygote, PyPlot, ForwardDiff
using Debugger

include("./analytical_sir_gradients.jl")
include("./plotting.jl")

#Random.seed!(3)
## 

abstract type DiffMode end
struct FW <: DiffMode end
struct RV <: DiffMode end

get_jacobian(alg::FW, f, x) = ForwardDiff.jacobian(f, x)
get_jacobian(alg::RV, f, x) = Zygote.jacobian(f, x)[1]

function get_agent_jacobian(diff_mode, bernoulli_grad, beta, gamma, agent_parameters)
	params = [beta, gamma]
	f = x -> SIR.run(agent_parameters.graph, x[1], x[2], agent_parameters.initial_fraction_infected,
		agent_parameters.n_timesteps, agent_parameters.delta_t,
		agent_parameters.n_agents, bernoulli_grad)
	dS_dBeta, dS_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[1], params))
	dI_dBeta, dI_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[2], params))
	dR_dBeta, dR_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[3], params))
	return [[dS_dBeta[:], dI_dBeta[:], dR_dBeta[:]] [dS_dGamma[:], dI_dGamma[:], dR_dGamma[:]]]
end

function get_ode_jacobian(diff_mode, beta, gamma, ode_parameters)
	params = [beta, gamma]
    f = x -> run_ode(x[1], x[2], ode_parameters.initial_fraction_infected, ode_parameters.n_timesteps, ode_parameters.delta_t)
	dS_dBeta, dS_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[1], params))
	dI_dBeta, dI_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[2], params))
	dR_dBeta, dR_dGamma = eachcol(get_jacobian(diff_mode, x -> f(x)[3], params))
	return [[dS_dBeta[:], dI_dBeta[:], dR_dBeta[:]] [dS_dGamma[:], dI_dGamma[:], dR_dGamma[:]]]
end

function get_parameters()
    n_timesteps = 25
    n_agents = 100
    initial_fraction_infected = 0.1
    bernoulli_grad = SAD()
    beta = 0.5
    gamma = 0.05
    delta_t = 1.0
    graph = complete_graph(n_agents)
    agent_parameters = AgentParameters(graph, beta, gamma, initial_fraction_infected, bernoulli_grad, n_timesteps, delta_t, n_agents);
    ode_parameters = ODEParameters(beta, gamma, initial_fraction_infected, n_timesteps, delta_t);
    return agent_parameters, ode_parameters
end

function get_ode_and_agent_curves()
    agent_parameters, ode_parameters = get_parameters();
    ode_sol = SIR.run_ode(ode_parameters)
    S_t, I_t, R_t = ode_sol[1, :], ode_sol[2, :], ode_sol[3, :];
    agents_sol = SIR.run(agent_parameters);
    return [S_t, I_t, R_t], agents_sol
end

curves = get_ode_and_agent_curves()
f = plot_curves(curves, ["ode", "agents"]);
display(f)
##

function get_agent_jacobians()
    agent_parameters, _ = get_parameters();
    gs = get_agent_jacobian(FW(), GS(), agent_parameters.beta, agent_parameters.gamma, agent_parameters);
    sad = get_agent_jacobian(FW(), SAD(), agent_parameters.beta, agent_parameters.gamma, agent_parameters);
    return gs, sad
end
##
println("Computing gradients...")
@time agent_jacobian_gs, agent_jacobian_sad = get_agent_jacobians();
    
#@Debugger.run gs = get_agent_jacobian(FW(), GS(), agent_parameters.beta, agent_parameters.gamma, agent_parameters);

I_t, S_t = curves[2][2], curves[2][1];
agent_parameters, _ = get_parameters();
analytical_jacobian = compute_analytical_gradient(agent_parameters.beta, agent_parameters.gamma, I_t, S_t, agent_parameters.delta_t, agent_parameters.n_timesteps);
#
f = plot_jacobians([agent_jacobian_gs, agent_jacobian_sad, analytical_jacobian],
	["agent GS", "agent SAD", "analytical"],
	figsize = (15, 7))
display(f)

###

#f = x -> SIR.run(agent_parameters.graph, x, agent_parameters.gamma, agent_parameters.initial_fraction_infected,
#	5, agent_parameters.delta_t,
#	agent_parameters.n_agents, GS())[2]
#f(10)

#println("="^50)
#deriv = ForwardDiff.derivative(f, 10)
#
#f, ax = subplots()
#ax.plot(deriv)
#display(f)

# si

sol_si = run_si(0.5, 0.1,  100, 50, 1)
S_t, I_t = sol_si[:, 1], sol_si[:, 2]

f, ax = subplots()
ax.plot(S_t, label = "S")
ax.plot(I_t, label = "I")
ax.legend()
display(f)
