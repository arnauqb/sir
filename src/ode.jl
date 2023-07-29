export run_ode

function sir_ode!(du, u, p, t)
    S, I, R = u
    β, γ = p
    
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

function run_ode(β, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    # Initial conditions
    infected_0 = Int(round(n_agents * initial_fraction_infected))
    u0 = [n_agents - infected_0, infected_0, 0.0]

    # Parameters
    p = [β, γ]

    # Time range
    tspan = (0.0, n_timesteps * delta_t)

    # Create an ODEProblem
    prob = ODEProblem(sir_ode!, u0, tspan, p)

    # Solve the ODE
    sol = solve(prob, Tsit5())
    return sol
end