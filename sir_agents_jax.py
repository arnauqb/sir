import jax
import numpy as np
import jax.numpy as jnp
import jraph
import networkx
import random

USE_GS  = False

def generate_key():
    return jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))


def jax_gumbel_softmax(probs, temp=0.1, min_probs=1e-10, max_probs=1.0):
    # clip probss which are incompatible
    probs = jnp.minimum(jnp.maximum(probs, min_probs), max_probs)
    # stack probss for p and q
    logits = jnp.log(jnp.stack([1 - probs, probs]))
    # sample gumbel noise
    gumbel = jax.random.gumbel(generate_key(), logits.shape)
    # softmax -> probabilities for success and failure
    r = jax.nn.softmax((logits + gumbel) / temp, axis=0)
    y = jnp.argmax(r, axis=0)
    # attach the r success gradient
    y = jax.lax.stop_gradient(y - r[0]) + r[0]
    return y


def sample_bernoulli(p):
    ret = jax.random.bernoulli(generate_key(), p).astype(jnp.float32)
    return ret


def convert_networkx_graph_to_jraph(graph_nx: networkx.Graph):
    """
    Converts a networkx graph to a jraph graph.
    """
    graph_nx = graph_nx.to_directed()
    mapping = dict(zip(graph_nx.nodes(), range(graph_nx.number_of_nodes())))
    senders = np.empty(graph_nx.number_of_edges(), dtype=int)
    receivers = np.empty(graph_nx.number_of_edges(), dtype=int)
    for i, (src, dst) in enumerate(graph_nx.edges()):
        senders[i] = mapping[src]
        receivers[i] = mapping[dst]
    n_nodes = np.array([len(graph_nx.nodes)])
    n_edges = np.array([len(graph_nx.edges)])
    node_features = np.ones((n_nodes[0], 4))  # infected, susceptible, recovered, total
    edge_features = np.ones((n_edges[0], 1))  # transmission probability.
    global_features = np.ones((1, 3))  # parameters beta and gamma, delta_t
    return jraph.GraphsTuple(
        n_node=n_nodes,
        n_edge=n_edges,
        nodes=node_features,
        edges=edge_features,
        globals=global_features,
        senders=senders,
        receivers=receivers,
    )


def make_network(graph):
    def update_edge_fn(
        edge_features, sender_node_features, receiver_node_features, globals_
    ):
        """Returns the update edge features."""
        ret = sender_node_features[:, [0, 3]] * receiver_node_features[:, [1, 3]]
        return ret

    def update_node_fn(
        node_features,
        aggregated_sender_edge_features,
        aggregated_receiver_edge_features,
        globals_,
    ):
        beta = globals_[:, 0]
        gamma = globals_[:, 1]
        delta_t = globals_[:, 2]
        exponent = (
            aggregated_receiver_edge_features[:, 0]
            / aggregated_receiver_edge_features[:, 1]
            * beta * delta_t
        )
        probs_infected = 1.0 - jnp.exp(-exponent)
        probs_recovered = gamma * node_features[:, 0]
        if USE_GS:
            new_infected = jax_gumbel_softmax(probs_infected)
            new_recovered = jax_gumbel_softmax(probs_recovered)
        else:
            new_infected = sample_bernoulli(probs_infected)
            new_recovered = sample_bernoulli(probs_recovered)
        infected = node_features[:, 0] + new_infected - new_recovered
        susceptible = node_features[:, 1] - new_infected
        recovered = node_features[:, 2] + new_recovered
        ret = jnp.stack((infected, susceptible, recovered, node_features[:, 3]), axis=1)
        return ret

    def update_globals_fn(aggregated_node_features, aggregated_edge_features, globals_):
        """Returns the global features."""
        return globals_

    aggregate_edges_for_nodes_fn = jraph.segment_sum
    aggregate_nodes_for_globals_fn = jraph.segment_sum
    aggregate_edges_for_globals_fn = jraph.segment_sum

    attention_logit_fn = None
    attention_reduce_fn = None

    network = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_globals_fn,
        attention_logit_fn=attention_logit_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
        attention_reduce_fn=attention_reduce_fn,
    )
    return network


class SIR:
    def __init__(self, graph: networkx.Graph, n_timesteps: int):
        self.graph_nx = graph
        self.graph = convert_networkx_graph_to_jraph(graph)
        self.n_timesteps = n_timesteps
        self.network = make_network(graph)

    def initialize(self, initial_fraction_infected):
        """
        Infects a fraction of the nodes and returns the initial state.
        """
        n_agents = self.graph.n_node
        probs = initial_fraction_infected * jnp.ones(n_agents)
        if USE_GS:
            new_infected = jax_gumbel_softmax(probs)
        else:
            new_infected = sample_bernoulli(probs)
        self.graph.nodes[:, 0] = new_infected
        self.graph.nodes[:, 1] = 1 - new_infected
        self.graph.nodes[:, 2] = jnp.zeros_like(new_infected)
        self.graph.nodes[:, 3] = jnp.ones_like(new_infected)

    def step(self, beta, gamma, delta_t):
        self.graph.globals[:] = jnp.array([[beta, gamma, delta_t]])
        self.graph = self.network(self.graph)

    def observe(self):
        infected = self.graph.nodes[:, 0].sum()
        susceptible = self.graph.nodes[:, 1].sum()
        recovered = self.graph.nodes[:, 2].sum()
        return infected, susceptible, recovered

    def run(self, beta, gamma, initial_fraction_infected, delta_t):
        self.initialize(initial_fraction_infected)
        for _ in range(self.n_timesteps):
            self.step(beta, gamma, delta_t)
            yield self.observe()


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=1000)
    parser.add_argument("--initial_fraction_infected", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--delta_t", type=float, default=1.0)
    parser.add_argument("--gs", type=bool, default=False)

    params = parser.parse_args()
    USE_GS = params.gs

    graph = networkx.complete_graph(params.n_agents)
    sir = SIR(graph, params.n_timesteps)

    infected_per_day = []
    susceptible_per_day = []
    recovered_per_day = []
    for ts in sir.run(
        params.beta, params.gamma, params.initial_fraction_infected, params.delta_t
    ):
        infected_t, susceptible_t, recovered_t = ts
        infected_per_day.append(infected_t)
        susceptible_per_day.append(susceptible_t)
        recovered_per_day.append(recovered_t)
    infected_per_day = jnp.stack(infected_per_day)
    susceptible_per_day = jnp.stack(susceptible_per_day)
    recovered_per_day = jnp.stack(recovered_per_day)

    # plot normalized sir
    f, ax = plt.subplots()
    ax.plot(susceptible_per_day / params.n_agents, label="susceptible")
    ax.plot(infected_per_day / params.n_agents, label="infected")
    ax.plot(recovered_per_day / params.n_agents, label="recovered")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("fraction of population")
    f.savefig("sir_jax.png", dpi=150)
    #plt.show()
