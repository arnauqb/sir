using SIR, Random, Graphs, StochasticAD, Zygote, CairoMakie, ForwardDiff

Random.seed!(1)

n_timesteps = 10
n_agents = 1000
initial_fraction_infected = 0.1
beta = 0.1
gamma = 0.05
diff_mode = SAD()
delta_t = 1.0
graph = complete_graph(n_agents)

function gradient_I_aux_beta(beta, graph=graph, gamma=gamma, initial_fraction_infected=initial_fraction_infected, n_timesteps=n_timesteps, delta_t=delta_t, diff_mode=diff_mode)
    S_t, I_t, R_t = SIR.run(graph, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, diff_mode)
    return S_t, I_t, R_t
end

function compute_analytical_gradient(S_t, I_t, R_t, n_agents, delta_t)
    return @. I_t * S_t / n_agents  * delta_t
end

S_t, I_t, R_t = gradient_I_aux_beta(beta)

analytical_gradient = compute_analytical_gradient(S_t, I_t, R_t, n_agents, delta_t)

autodiff_gradient_sad = Zygote.jacobian(x -> gradient_I_aux_beta(x)[2], beta)


figure, axis, lineplot = lines(analytical_gradient, label="Analytical gradient")
lines!(autodiff_gradient[1], label="Autodiff gradient")
lines!(autodiff_gradient_sad[1], label="Autodiff gradient SAD")
axislegend()
figure