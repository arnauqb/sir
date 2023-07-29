using SIR, PyPlot, SciMLSensitivity, Zygote

beta = 0.1
gamma = 0.05
initial_fraction_infected = 0.1
n_agents = 1000
n_timesteps = 10
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

function fsir(beta, gamma)
    sol = SIR.run_ode(beta, gamma, initial_fraction_infected, n_agents, n_timesteps, delta_t)
    return stack(sol.u)
    #return stack((sol[1, :], sol[2, :], sol[3, :]))
end
a = fsir(beta, gamma)

dS_dBeta = Zygote.jacobian(x -> fsir(x, gamma)[1,:], beta)[1]
dI_dBeta = Zygote.jacobian(x -> fsir(x, gamma)[2,:], beta)[1]
dR_dGamma = Zygote.jacobian(x -> fsir(beta, x)[3,:], gamma)[1]

