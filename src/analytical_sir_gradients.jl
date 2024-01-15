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
		#dI_dBeta_previous
		I_previous * S_previous * delta_t
		+ beta * dI_dBeta_previous * S_previous * delta_t
		+ beta * I_previous * dS_dBeta_previous * delta_t
		#-
		#gamma * dI_dBeta_previous * delta_t
	)
end


function _get_dR_dBeta(gamma, dR_dBeta_previous, dI_dBeta_previous, delta_t)
	#return dR_dBeta_previous + gamma * dI_dBeta_previous * delta_t
	return gamma * dI_dBeta_previous * delta_t
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
		#dR_dGamma_previous + I_previous * delta_t + gamma * dI_dGamma_previous * delta_t
		I_previous * delta_t + gamma * dI_dGamma_previous * delta_t
	)
end

function compute_analytical_gradient(beta, gamma, I_t, S_t, delta_t, n_timesteps)
	dS_dBeta_previous = 0.0
	dI_dBeta_previous = 0.0
	dR_dBeta_previous = 0.0
	dS_dGamma_previous = 0.0
	dI_dGamma_previous = 0.0
	dR_dGamma_previous = 0.0
	gradients = [[dS_dBeta_previous, dI_dBeta_previous, dR_dBeta_previous, dS_dGamma_previous, dI_dGamma_previous, dR_dGamma_previous]]

	for i ∈ 1:n_timesteps-1
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
	ret = stack(gradients)
    return [[ret[1,:], ret[2,:], ret[3,:]] [ret[4,:], ret[5,:], ret[6,:]]]
end

function analytical_si_model(beta, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    t = 0:delta_t:(n_timesteps-1)*delta_t 
    I_0 = initial_fraction_infected * n_agents
    S_0 = n_agents - I_0
    I = @. (n_agents * I_0) / (I_0 + S_0 * exp(-beta * t))
    S = @. (n_agents - I)
    return S ./ n_agents, I ./ n_agents
end