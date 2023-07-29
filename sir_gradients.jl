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

Random.seed!(2)

n_timesteps = 50
n_agents = 1000
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
	return stack(sol.u)
end

function get_jacobian(diff_mode)
	dS_dBeta = Zygote.jacobian(x -> gradient_I_aux_beta(beta = x, diff_mode = diff_mode)[1], beta)[1]
	dI_dBeta = Zygote.jacobian(x -> gradient_I_aux_beta(beta = x, diff_mode = diff_mode)[2], beta)[1]
	dR_dGamma = Zygote.jacobian(x -> gradient_I_aux_beta(beta = beta, gamma = x, diff_mode = diff_mode)[3], beta)[1]
	return stack([dS_dBeta, dI_dBeta, dR_dGamma])
end


S_t, I_t, R_t = gradient_I_aux_beta(beta = beta)

analytical_gradient = compute_analytical_gradient(beta, gamma, I_t, S_t, delta_t, n_timesteps)

println("Gumbel softmax...")
@time ad_gradient_gs = get_jacobian(GS())
println("Stochastic AD ...")
@time ad_gradient_sad = get_jacobian(SAD())
println("ODE gradients...")
@time ode_gradient = Zygote.jacobian(run_ode_sir, [beta, gamma])[1]
ode_gradient = reshape(ode_gradient, (3, n_timesteps + 1, 2))



# make a plot with the analytical gradient for dS_dBeta, dI_dBeta, and dR_dGamma
f, ax = subplots(1, 3, figsize = (15, 5))
ax[1].set_title(L"\frac{\partial S}{\partial \beta}")
ax[2].set_title(L"\frac{\partial I}{\partial \beta}")
ax[3].set_title(L"\frac{\partial R}{\partial \gamma}")
ax[1].plot(analytical_gradient[1, :], label = "Analytical")
ax[1].plot(ad_gradient_gs[:, 1], label = "GS")
ax[1].plot(ad_gradient_sad[:, 1], label = "SAD")
ax[1].plot(ode_gradient[1, :, 1], label = "ODE")
ax[2].plot(analytical_gradient[2, :], label = "Analytical")
ax[2].plot(ad_gradient_gs[:, 2], label = "GS")
ax[2].plot(ad_gradient_sad[:, 2], label = "SAD")
ax[2].plot(ode_gradient[2, :, 1], label = "ODE")
ax[3].plot(analytical_gradient[6, :], label = "Analytical")
ax[3].plot(ad_gradient_gs[:, 3], label = "GS")
ax[3].plot(ad_gradient_sad[:, 3], label = "SAD")
ax[3].plot(ode_gradient[3, :, 2], label = "ODE")
for i in 1:3
	ax[i].set_xlabel("Time-step")
    ax[i].set_ylabel("Gradient")
	ax[i].legend()
end
f
