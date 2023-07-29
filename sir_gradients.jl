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
beta = 0.1
gamma = 0.05
diff_mode = GS()
delta_t = 1.0
graph = complete_graph(n_agents)

function gradient_I_aux_beta(; beta, graph = graph, gamma = gamma, initial_fraction_infected = initial_fraction_infected, n_timesteps = n_timesteps, delta_t = delta_t, diff_mode = diff_mode)
	S_t, I_t, R_t = SIR.run(graph, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, n_agents, diff_mode)
	return S_t, I_t, R_t
end

function run_ode_sir(p)
	sol = SIR.run_ode(p[1], p[2], initial_fraction_infected, n_agents, n_timesteps, delta_t)
	return stack(sol.u)[:,2:end]
end

function get_agent_jacobian(diff_mode)
	dS_dBeta = Zygote.jacobian(x -> gradient_I_aux_beta(beta = x, diff_mode = diff_mode)[1], beta)[1]
	dI_dBeta = Zygote.jacobian(x -> gradient_I_aux_beta(beta = x, diff_mode = diff_mode)[2], beta)[1]
	dR_dGamma = Zygote.jacobian(x -> gradient_I_aux_beta(beta = beta, gamma = x, diff_mode = diff_mode)[3], beta)[1]
	return stack([dS_dBeta, dI_dBeta, dR_dGamma])
end

function get_agent_jacobian_mean_and_std(diff_mode, n_samples)
    jacobians = [get_agent_jacobian(diff_mode) for i = 1:n_samples]
    return mean(jacobians), std(jacobians)
end

S_t, I_t, R_t = gradient_I_aux_beta(beta = beta)

analytical_gradient = compute_analytical_gradient(beta, gamma, I_t, S_t, delta_t, n_timesteps)

n_samples = 10
println("Gumbel softmax...")
gs_mean, gs_std = get_agent_jacobian_mean_and_std(GS(), 10)
sad_mean, sad_std = get_agent_jacobian_mean_and_std(SAD(), 10)
println("ODE gradients...")
@time ode_gradient = Zygote.jacobian(run_ode_sir, [beta, gamma])[1]
ode_gradient = reshape(ode_gradient, (3, n_timesteps, 2))



# make a plot with the analytical gradient for dS_dBeta, dI_dBeta, and dR_dGamma
f, ax = subplots(1, 3, figsize = (15, 5))
t_range = 1:n_timesteps
ax[1].set_title(L"\frac{\partial S}{\partial \beta}")
ax[2].set_title(L"\frac{\partial I}{\partial \beta}")
ax[3].set_title(L"\frac{\partial R}{\partial \gamma}")

ax[1].plot(t_range, analytical_gradient[1, :], label = "Analytical", color = "black")
ax[1].plot(t_range, gs_mean[:, 1], label = "GS", color = "C0")
ax[1].fill_between(t_range, gs_mean[:, 1] - gs_std[:, 1], gs_mean[:, 1] + gs_std[:, 1], alpha = 0.2, color = "C0")
ax[1].plot(t_range, sad_mean[:, 1], label = "SAD", color = "C1")
ax[1].fill_between(t_range, sad_mean[:, 1] - sad_std[:, 1], sad_mean[:, 1] + sad_std[:, 1], alpha = 0.2, color = "C1")
ax[1].plot(t_range, ode_gradient[1, :, 1], label = "ODE", color = "C2")

ax[2].plot(t_range, analytical_gradient[2, :], label = "Analytical", color = "black")
ax[2].plot(t_range, ode_gradient[2, :, 1], label = "ODE", color = "C2")
ax[2].plot(t_range, gs_mean[:, 2], label = "GS", color = "C0")
ax[2].fill_between(t_range, gs_mean[:, 2] - gs_std[:, 2], gs_mean[:, 2] + gs_std[:, 2], alpha = 0.2, color = "C0")
ax[2].plot(t_range, sad_mean[:, 2], label = "SAD", color = "C1")
ax[2].fill_between(t_range, sad_mean[:, 2] - sad_std[:, 2], sad_mean[:, 2] + sad_std[:, 2], alpha = 0.2, color = "C1")

ax[3].plot(t_range, analytical_gradient[6, :], label = "Analytical", color = "black")
ax[3].plot(t_range, ode_gradient[3, :, 2], label = "ODE", color = "C2")
ax[3].plot(t_range, gs_mean[:, 3], label = "GS", color = "C0")
ax[3].fill_between(t_range, gs_mean[:, 3] - gs_std[:, 3], gs_mean[:, 3] + gs_std[:, 3], alpha = 0.2, color = "C0")
ax[3].plot(t_range, sad_mean[:, 3], label = "SAD", color = "C1")
ax[3].fill_between(t_range, sad_mean[:, 3] - sad_std[:, 3], sad_mean[:, 3] + sad_std[:, 3], alpha = 0.2, color = "C1")
for i in 1:3
	ax[i].set_xlabel("Time-step")
    ax[i].set_ylabel("Gradient")
	ax[i].legend()
end
f
