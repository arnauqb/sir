export run_si, analytical_si_gradients

function run_si(beta, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    t = 0.0:delta_t:(n_timesteps-1)*delta_t
    I0 = round(Int, initial_fraction_infected * n_agents)
    S0 = n_agents - I0
    S = @. S0 / I0 * exp(-beta * t) / (1 + S0/I0 * exp(-beta * t))
    I = @. 1.0 - S
    return hcat(S, I)
end

function analytical_si_gradients(beta, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    t = 0.0:delta_t:(n_timesteps-1)*delta_t
    I0 = round(Int, initial_fraction_infected * n_agents)
    S0 = n_agents - I0
    dSdBeta = @. - S0 / I0 * (t * exp(-beta * t)) / (1 + S0/I0 * exp(-beta * t))^2
    dIdBeta = @. - dSdBeta
    ret = Matrix{Any}(undef, 2, 1)
    ret[1,1] = dSdBeta
    ret[2,1] = dIdBeta
    return ret
end
