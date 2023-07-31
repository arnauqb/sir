using SIR, Random, Graphs, StochasticAD, Zygote, PyPlot, ForwardDiff

function _get_dS_dBeta(
	beta, dS_dBeta_previous, dI_dBeta_previous, I_previous, S_previous, delta_t,
)
	return (
		dS_dBeta_previous
		-
		I_previous * S_previous * delta_t
		-
		beta * dI_dBeta_previous * S_previous * delta_t
		-
		beta * I_previous * dS_dBeta_previous * delta_t
	)
end


function _get_dI_dBeta(
	beta, gamma, dS_dBeta_previous, dI_dBeta_previous, I_previous, S_previous, delta_t,
)
	return (
		dI_dBeta_previous
		+ I_previous * S_previous * delta_t
		+ beta * dI_dBeta_previous * S_previous * delta_t
		+ beta * I_previous * dS_dBeta_previous * delta_t
		-
		gamma * dI_dBeta_previous * delta_t
	)
end


function _get_dR_dBeta(gamma, dR_dBeta_previous, dI_dBeta_previous, delta_t)
	return dR_dBeta_previous + gamma * dI_dBeta_previous * delta_t
end


function _get_dS_dGamma(
	beta, dS_dGamma_previous, dI_dGamma_previous, I_previous, S_previous, delta_t,
)
	return (
		dS_dGamma_previous
		-
		beta * dI_dGamma_previous * S_previous * delta_t
		-
		beta * I_previous * dS_dGamma_previous * delta_t
	)
end


function _get_dI_dGamma(
	beta, gamma, dS_dGamma_previous, dI_dGamma_previous, I_previous, S_previous, delta_t,
)
	return (
		dI_dGamma_previous
		+ beta * dI_dGamma_previous * S_previous * delta_t
		+ beta * I_previous * dS_dGamma_previous * delta_t
		-
		I_previous * delta_t
		-
		gamma * dI_dGamma_previous * delta_t
	)
end


function _get_dR_dGamma(gamma, dR_dGamma_previous, dI_dGamma_previous, I_previous, delta_t)
	return (
		dR_dGamma_previous + I_previous * delta_t + gamma * dI_dGamma_previous * delta_t
	)
end

function compute_analytical_gradient(beta, gamma, I_t, S_t, delta_t, n_timesteps)
	dS_dBeta_previous = 0.0
	dI_dBeta_previous = 0.0
	dR_dBeta_previous = 0.0
	dS_dGamma_previous = 0.0
	dI_dGamma_previous = 0.0
	dR_dGamma_previous = 0.0
	gradients = []

	for i ∈ 1:n_timesteps
		dS_dBeta = _get_dS_dBeta(
			beta,
			dS_dBeta_previous,
			dI_dBeta_previous,
			I_t[i],
			S_t[i],
			delta_t,
		)
		dI_dBeta = _get_dI_dBeta(
			beta,
			gamma,
			dS_dBeta_previous,
			dI_dBeta_previous,
			I_t[i],
			S_t[i],
			delta_t,
		)
		dR_dBeta = _get_dR_dBeta(
			gamma,
			dR_dBeta_previous,
			dI_dBeta_previous,
			delta_t,
		)
		dS_dGamma = _get_dS_dGamma(
			beta,
			dS_dGamma_previous,
			dI_dGamma_previous,
			I_t[i],
			S_t[i],
			delta_t,
		)
		dI_dGamma = _get_dI_dGamma(
			beta,
			gamma,
			dS_dGamma_previous,
			dI_dGamma_previous,
			I_t[i],
			S_t[i],
			delta_t,
		)
		dR_dGamma = _get_dR_dGamma(
			gamma,
			dR_dGamma_previous,
			dI_dGamma_previous,
			I_t[i],
			delta_t,
		)
		dS_dBeta_previous = dS_dBeta
		dI_dBeta_previous = dI_dBeta
		dR_dBeta_previous = dR_dBeta
		dS_dGamma_previous = dS_dGamma
		dI_dGamma_previous = dI_dGamma
		dR_dGamma_previous = dR_dGamma
		push!(gradients, [dS_dBeta, dI_dBeta, dR_dBeta, dS_dGamma, dI_dGamma, dR_dGamma])
	end
	return stack(gradients)
end

Random.seed!(3)

n_timesteps = 50
n_agents = 100
initial_fraction_infected = 0.1
diff_mode = GS()
beta = 0.1
gamma = 0.02
delta_t = 0.1
graph = complete_graph(n_agents)

function run_agents_sir(; beta=beta, graph = graph, gamma = gamma, initial_fraction_infected = initial_fraction_infected, n_timesteps = n_timesteps, delta_t = delta_t, diff_mode = diff_mode)
	S_t, I_t, R_t = SIR.run(graph, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, n_agents, diff_mode)
	return S_t, I_t, R_t
end

function run_ode_sir(p)
	sol = SIR.run_ode(p[1], p[2], initial_fraction_infected, n_agents, n_timesteps, delta_t)
	return stack(sol.u)[:,2:end]
end

function get_agent_jacobian(diff_mode)
	dS_dBeta = Zygote.jacobian(x -> run_agents_sir(beta = x, diff_mode = diff_mode)[1], beta)[1]
	dI_dBeta = Zygote.jacobian(x -> run_agents_sir(beta = x, diff_mode = diff_mode)[2], beta)[1]
	dR_dGamma = Zygote.jacobian(x -> run_agents_sir(gamma = x, diff_mode = diff_mode)[3], gamma)[1]
	return stack([dS_dBeta, dI_dBeta, dR_dGamma])
end

function get_agent_jacobian_mean_and_std(diff_mode, n_samples)
    jacobians = [get_agent_jacobian(diff_mode) for i = 1:n_samples]
    return mean(jacobians), std(jacobians) / (n_samples-1)
end

function get_agent_scatters(diff_mode, n_samples)
    jacobians = [get_agent_jacobian(diff_mode) for i = 1:n_samples]
    return jacobians
end

S_t_agents, I_t_agents, R_t_agents = run_agents_sir()
ode_sol = run_ode_sir([beta, gamma]);
S_t, I_t, R_t = ode_sol[1, :], ode_sol[2, :], ode_sol[3, :]

# plot S, I, R in same axis

## 

f, ax = subplots(1, 1, figsize = (5, 5))
t_range = 1:n_timesteps
ax.plot(t_range, S_t, label = "S")
ax.plot(t_range, I_t, label = "I")
ax.plot(t_range, R_t, label = "R")
ax.set_xlabel("Time-step")
ax.set_ylabel("Fraction of agents")
ax.legend()
#f

##

analytical_gradient = compute_analytical_gradient(beta, gamma, I_t, S_t, delta_t, n_timesteps)

n_samples = 5
println("Gumbel softmax...")
@time gs_jacobians = get_agent_scatters(GS(0.1), n_samples)
println("SAD...")
@time sad_jacobians = get_agent_scatters(SAD(), n_samples)
println("ODE gradients...")
@time ode_gradient = Zygote.jacobian(run_ode_sir, [beta, gamma])[1]
ode_gradient = reshape(ode_gradient, (3, n_timesteps, 2))



## plot results

f, ax = subplots(1, 3, figsize = (15, 5))
t_range = 1:n_timesteps
scatter_alpha = 0.4
ax[1].set_title(L"\frac{\partial S}{\partial \beta}")
ax[2].set_title(L"\frac{\partial I}{\partial \beta}")
ax[3].set_title(L"\frac{\partial R}{\partial \gamma}")

ax[1].plot(t_range, analytical_gradient[1, :], label = "Analytical", color = "black")
ax[1].scatter([], [], color = "C0", alpha = scatter_alpha, label="GS")
ax[1].scatter([], [], color = "C1", alpha = scatter_alpha, label="SAD")
for jacob in gs_jacobians
    ax[1].scatter(t_range, jacob[:, 1], color = "C0", alpha = scatter_alpha)
end
for jacob in sad_jacobians
    ax[1].scatter(t_range, jacob[:, 1], color = "C1", alpha = scatter_alpha)
end
ax[1].plot(t_range, ode_gradient[1, :, 1], label = "ODE", color = "C2")

ax[2].plot(t_range, analytical_gradient[2, :], label = "Analytical", color = "black")
ax[2].plot(t_range, ode_gradient[2, :, 1], label = "ODE", color = "C2")
ax[2].scatter([], [], color = "C0", alpha = scatter_alpha, label="GS")
ax[2].scatter([], [], color = "C1", alpha = scatter_alpha, label="SAD")
for jacob in gs_jacobians
    ax[2].scatter(t_range, jacob[:, 2], color = "C0", alpha = scatter_alpha)
end
for jacob in sad_jacobians
    ax[2].scatter(t_range, jacob[:, 2], color = "C1", alpha = scatter_alpha)
end

ax[3].plot(t_range, analytical_gradient[6, :], label = "Analytical", color = "black")
ax[3].plot(t_range, ode_gradient[3, :, 2], label = "ODE", color = "C2")
ax[3].scatter([], [], color = "C0", alpha = scatter_alpha, label="GS")
ax[3].scatter([], [], color = "C1", alpha = scatter_alpha, label="SAD")
for jacob in gs_jacobians
    ax[3].scatter(t_range, jacob[:, 3], color = "C0", alpha = scatter_alpha)
end
for jacob in sad_jacobians
    ax[3].scatter(t_range, jacob[:, 3], color = "C1", alpha = scatter_alpha)
end
for i in 1:3
	ax[i].set_xlabel("Time-step")
    ax[i].set_ylabel("Gradient")
	ax[i].legend()
end
#f.savefig("./figures/sir_gradients.png", bbox_inches = "tight")
f

##
