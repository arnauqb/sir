using SIR, Graphs, PyPlot, Random

Random.seed!(0)

n_timesteps = 100
n_agents = 1000
initial_fraction_infected = 0.1
beta = 0.1
gamma = 0.05
diff_mode = Main.SIR.SAD()
delta_t = 1.0
graph = complete_graph(n_agents)

S_t, I_t, R_t = SIR.run(graph, beta, gamma, initial_fraction_infected, n_timesteps, delta_t, diff_mode)

# plot the results normalized with labels, title and save to file.
f, ax = subplots()
ax.plot(1:n_timesteps, S_t ./ n_agents, label = "S")
ax.plot(1:n_timesteps, I_t ./ n_agents, label = "I")
ax.plot(1:n_timesteps, R_t ./ n_agents, label = "R")
ax.set_title("SIR Julia")
ax.set_xlabel("Time")
ax.set_ylabel("Fraction of population")
ax.legend()
f.savefig("./figures/sir_julia.png", dpi=150)