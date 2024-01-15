import torch
import networkx
import torch_geometric


class SIR(torch.nn.Module):
    def __init__(
        self,
        graph: networkx.Graph,
        n_timesteps: int,
        n_agents: int,
        device: str = "cpu",
        delta_t: float = 1.0,
    ):
        """
        Implements a differentiable SIR model on a graph.

        **Arguments:**

        - `graph`: a networkx graph
        - `n_timesteps`: the number of timesteps to run the model for
        - `device` : device to use (eg. "cpu" or "cuda:0")
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_agents = n_agents
        self.delta_t = delta_t
        # convert graph from networkx to pytorch geometric
        self.graph = torch_geometric.utils.convert.from_networkx(graph).to(device)
        self.mp = SIRMessagePassing(aggr="add", node_dim=-1)
        self.aux = torch.ones(self.n_agents, device=device)
        self.device = device

    def sample_bernoulli_gs(self, probs: torch.Tensor, tau: float = 0.1):
        """
        Samples from a Bernoulli distribution in a diferentiable way using Gumble-Softmax

        **Arguments:**

        - probs: a tensor of shape (n,) containing the probabilities of success for each trial
        - tau: the temperature of the Gumble-Softmax distribution
        """
        logits = torch.vstack((probs, 1 - probs)).T.log()
        gs_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        return gs_samples[:, 0]

    def initialize(self, initial_fraction_infected):
        """
        Initializes the model setting the adequate number of initial infections.

        **Arguments**:

        - `initial_fraction_infected`: the fraction of infected agents at the beginning of the simulation
        """
        n_agents = self.graph.num_nodes
        # sample the initial infected nodes
        probs = initial_fraction_infected * torch.ones(n_agents, device=self.device)
        new_infected = self.sample_bernoulli_gs(probs)
        # set the initial state
        infected = new_infected
        susceptible = 1 - new_infected
        recovered = torch.zeros(n_agents, device=self.device)
        x = torch.vstack((infected, susceptible, recovered))
        return x.reshape(1, 3, n_agents)

    def step(self, beta, gamma, x: torch.Tensor):
        """
        Runs the model forward for one timestep.

        **Arguments**:

        - `beta`: the infection probability
        - `gamma`: the recovery probability
        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        infected, susceptible, recovered = x[-1]
        # Get number of infected neighbors per node, return 0 if node is not susceptible.
        n_infected_neighbors = self.mp(self.graph.edge_index, infected, susceptible)
        n_neighbors = self.mp(
            self.graph.edge_index,
            self.aux,
            self.aux
        )
        # each contact has a beta chance of infecting a susceptible node
        prob_infection = 1.0 - torch.exp(
            -beta * n_infected_neighbors / n_neighbors * self.delta_t
        )
        prob_infection = torch.clip(prob_infection, min=1e-10, max=1.0)
        # sample the infected nodes
        new_infected = self.sample_bernoulli_gs(prob_infection)
        # sample recoverd people
        prob_recovery = gamma * infected
        prob_recovery = torch.clip(prob_recovery, min=1e-10, max=1.0)
        new_recovered = self.sample_bernoulli_gs(prob_recovery)
        # update the state of the agents
        infected = infected + new_infected - new_recovered
        susceptible = susceptible - new_infected
        recovered = recovered + new_recovered
        x = torch.vstack((infected, susceptible, recovered)).reshape(1, 3, -1)
        return x

    def observe(self, x: torch.Tensor):
        """
        Returns the total number of infected and recovered agents per time-step

        **Arguments**:

        - x: a tensor of shape (3, n_agents) containing the infected, susceptible, and recovered counts.
        """
        return [
            x[:, 0, :].sum(1) / self.n_agents,
            x[:, 1, :].sum(1) / self.n_agents,
            x[:, 2, :].sum(1) / self.n_agents,
        ]

    def forward(self, params):
        """
        Runs the model for the specified number of timesteps.

        **Arguments**:

        - params: a tensor of shape (2,) containing the beta and gamma parameters
        """
        beta, gamma, initial_fraction_infected = params
        x = self.initialize(initial_fraction_infected)
        infected_per_day, susceptible_per_day, recovered_per_day = self.observe(x)
        for i in range(self.n_timesteps):
            x = self.step(beta, gamma, x)
            # get the observations
            infected, susceptible, recovered = self.observe(x)
            infected_per_day = torch.cat((infected_per_day, infected))
            susceptible_per_day = torch.cat((susceptible_per_day, susceptible))
            recovered_per_day = torch.cat((recovered_per_day, recovered))
        return susceptible_per_day, infected_per_day, recovered_per_day


class SIRMessagePassing(torch_geometric.nn.conv.MessagePassing):
    """
    Class used to pass messages between agents about their infected status.
    """

    def forward(
        self,
        edge_index: torch.Tensor,
        infected: torch.Tensor,
        susceptible: torch.Tensor,
    ):
        """
        Computes the sum of the product between the node's susceptibility and the neighbors' infected status.

        **Arguments**:

        - edge_index: a tensor of shape (2, n_edges) containing the edge indices
        - infected: a tensor of shape (n_nodes,) containing the infected status of each node
        - susceptible: a tensor of shape (n_nodes,) containing the susceptible status of each node
        """
        return self.propagate(edge_index, x=infected, y=susceptible)

    def message(self, x_j, y_i):
        return x_j * y_i


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=1000)
    parser.add_argument("--initial_fraction_infected", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--delta_t", type=float, default=1.0)

    params = parser.parse_args()

    # create a random graph
    # graph = networkx.erdos_renyi_graph(params.n_agents, 0.1)
    # graph = networkx.complete_graph(params.n_agents)
    graph = networkx.watts_strogatz_graph(params.n_agents, 10, 0.01)

    # create the model
    model = SIR(
        graph, params.n_timesteps, params.n_agents, params.device, params.delta_t
    )
    t1 = time()
    S, I, R = model(
        torch.tensor([params.beta, params.gamma, params.initial_fraction_infected])
    )
    t2 = time()
    print(f"Time elapsed: {t2 - t1:.2f} seconds")

    # plot the results
    plt.plot(S.cpu(), label="S")
    plt.plot(I.cpu(), label="I")
    plt.plot(R.cpu(), label="R")
    plt.xlabel("Time")
    plt.ylabel("Fraction of agents")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Agents PyTorch")
    plt.savefig("./figures/sir_agents_torch.png", dpi=150)
    # plt.show()
