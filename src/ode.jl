export run_ode

function sir_static(u, p, t)
    @inbounds begin
        S, I, R = u
        dS = -p[1] * S * I
        dI = p[1] * S * I - p[2] * I
        dR = p[2] * I
    end
    @SVector [dS, dI, dR]
end

function run_ode(β, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    u0 = @SVector [1 - initial_fraction_infected, initial_fraction_infected, 0.0]
    p = @SVector [β, γ]
    tspan = (0.0, n_timesteps * delta_t)
    prob = ODEProblem(sir_static, u0, tspan, p)
    return solve(prob, Tsit5(), sensealg=ForwardDiffSensitivity(), saveat=delta_t)
end