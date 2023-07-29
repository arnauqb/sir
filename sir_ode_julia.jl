using SIR, PyPlot, SciMLSensitivity, Zygote

β = 0.1
γ = 0.05
initial_fraction_infected = 0.1
n_agents = 10000
n_timesteps = 100
delta_t = 1.0

sol_ode = SIR.run_ode(β, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t);

t = range(0, n_timesteps * delta_t, length=n_timesteps + 1);
u = sol_ode(t);

# plot u
f, ax = subplots()
ax.plot(t, u[1, :], label = "S")
ax.plot(t, u[2, :], label = "I")
ax.plot(t, u[3, :], label = "R")
f

# compute derivative respect to beta 
function f_aux(beta, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    sol_ode = SIR.run_ode(beta, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t);
    t = range(0, n_timesteps * delta_t, length=n_timesteps + 1);
    u = sol_ode(t);
    return u[2,:]
end

asd = f_aux(β, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t);

grad = Zygote.jacobian(x -> f_aux(x, γ, initial_fraction_infected, n_agents, n_timesteps, delta_t), 0.1);