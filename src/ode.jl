export run_ode, ODEParameters

struct ODEParameters
    beta::Float64
    gamma::Float64
    initial_fraction_infected::Float64
    n_timesteps::Int64
    delta_t::Float64
end

function sir_static(u, p, t)
    @inbounds begin
        S, I, R = u
        dS = -p[1] * S * I
        dI = p[1] * S * I - p[2] * I
        dR = p[2] * I
    end
    @SVector [dS, dI, dR]
end

function run_ode(β, γ, initial_fraction_infected, n_timesteps, delta_t)
    u0 = @SVector [1 - initial_fraction_infected, initial_fraction_infected, 0.0]
    p = @SVector [β, γ]
    tspan = (0.0, n_timesteps * delta_t)
    prob = ODEProblem(sir_static, u0, tspan, p)
    return stack(solve(prob, Tsit5(), sensealg=ForwardDiffSensitivity(), saveat=delta_t).u)[:, 1:end-1]
end

function run_ode(p::ODEParameters)
    run_ode(p.beta, p.gamma, p.initial_fraction_infected, p.n_timesteps, p.delta_t)
end