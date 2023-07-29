# append current path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import networkx

from sir_ode_torch import SIRODE
from sir_agents_torch import SIR


def _get_dS_dBeta(
    beta, dS_dBeta_previous, dI_dBeta_previous, I_previous, S_previous, delta_t
):
    return (
        dS_dBeta_previous
        - I_previous * S_previous * delta_t
        - beta * dI_dBeta_previous * S_previous * delta_t
        - beta * I_previous * dS_dBeta_previous * delta_t
    )


def _get_dI_dBeta(
    beta, gamma, dS_dBeta_previous, dI_dBeta_previous, I_previous, S_previous, delta_t
):
    return (
        dI_dBeta_previous
        + I_previous * S_previous * delta_t
        + beta * dI_dBeta_previous * S_previous * delta_t
        + beta * I_previous * dS_dBeta_previous * delta_t
        - gamma * dI_dBeta_previous * delta_t
    )


def _get_dR_dBeta(gamma, dR_dBeta_previous, dI_dBeta_previous, delta_t):
    return dR_dBeta_previous + gamma * dI_dBeta_previous * delta_t


def _get_dS_dGamma(
    beta, dS_dGamma_previous, dI_dGamma_previous, I_previous, S_previous, delta_t
):
    return (
        dS_dGamma_previous
        - beta * dI_dGamma_previous * S_previous * delta_t
        - beta * I_previous * dS_dGamma_previous * delta_t
    )


def _get_dI_dGamma(
    beta, gamma, dS_dGamma_previous, dI_dGamma_previous, I_previous, S_previous, delta_t
):
    return (
        dI_dGamma_previous
        + beta * dI_dGamma_previous * S_previous * delta_t
        + beta * I_previous * dS_dGamma_previous * delta_t
        - I_previous * delta_t
        - gamma * dI_dGamma_previous * delta_t
    )


def _get_dR_dGamma(gamma, dR_dGamma_previous, dI_dGamma_previous, I_previous, delta_t):
    return (
        dR_dGamma_previous + I_previous * delta_t + gamma * dI_dGamma_previous * delta_t
    )


def compute_analytical_gradient(beta, gamma, I_t, S_t, R_t, delta_t):
    dS_dBeta_previous = 0.0
    dI_dBeta_previous = 0.0
    dR_dBeta_previous = 0.0
    dS_dGamma_previous = 0.0
    dI_dGamma_previous = 0.0
    dR_dGamma_previous = 0.0
    gradients = []
    for i in range(len(I_t)):
        dS_dBeta = _get_dS_dBeta(
            beta,
            dS_dBeta_previous,
            dI_dBeta_previous,
            I_t[i],
            S_t[i],
            delta_t,
        )
        dI_dBeta = _get_dI_dBeta(
            beta,
            gamma,
            dS_dBeta_previous,
            dI_dBeta_previous,
            I_t[i],
            S_t[i],
            delta_t,
        )
        dR_dBeta = _get_dR_dBeta(
            gamma,
            dR_dBeta_previous,
            dI_dBeta_previous,
            delta_t,
        )
        dS_dGamma = _get_dS_dGamma(
            beta,
            dS_dGamma_previous,
            dI_dGamma_previous,
            I_t[i],
            S_t[i],
            delta_t,
        )
        dI_dGamma = _get_dI_dGamma(
            beta,
            gamma,
            dS_dGamma_previous,
            dI_dGamma_previous,
            I_t[i],
            S_t[i],
            delta_t,
        )
        dR_dGamma = _get_dR_dGamma(
            gamma,
            dR_dGamma_previous,
            dI_dGamma_previous,
            I_t[i],
            delta_t,
        )
        dS_dBeta_previous = dS_dBeta
        dI_dBeta_previous = dI_dBeta
        dR_dBeta_previous = dR_dBeta
        dS_dGamma_previous = dS_dGamma
        dI_dGamma_previous = dI_dGamma
        dR_dGamma_previous = dR_dGamma
        gradients.append(torch.tensor([dS_dBeta, dI_dBeta, dR_dBeta, dS_dGamma, dI_dGamma, dR_dGamma]))
    return torch.stack(gradients)
    


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_fraction_infected", type=float, default=0.1)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--n_agents", type=int, default=10000)
    parser.add_argument("--delta_t", type=int, default=1.0)
    parser.add_argument("--n_timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    t1 = args.t0 + args.n_timesteps * args.delta_t
    t = torch.linspace(args.t0, t1, int((t1 - args.t0) / args.delta_t) + 1)
    y0 = torch.tensor(
        [1 - args.initial_fraction_infected, args.initial_fraction_infected, 0.0],
        dtype=torch.float32,
    )
    ode_model = SIRODE(y0, t)
    ode_params = torch.tensor([args.beta, args.gamma], dtype=torch.float32)

    # agents
    graph = networkx.complete_graph(args.n_agents)
    agents_model = SIR(
        graph, args.n_timesteps, args.n_agents, args.device, delta_t=args.delta_t
    )
    agent_params = torch.tensor(
        [args.beta, args.gamma, args.initial_fraction_infected], dtype=torch.float32
    )

    # check models agree by plotting
    y = ode_model(ode_params)
    S = y[:, 0]
    I = y[:, 1]
    R = y[:, 2]
    S_agents, I_agents, R_agents = agents_model(agent_params)
    f, ax = plt.subplots()
    # plot three curves with agents one as a dashed line
    ax.plot(S, label="S", color="C0")
    ax.plot(I, label="I", color="C1")
    ax.plot(R, label="R", color="C2")
    ax.plot(S_agents, "--", label="S agents", color="C0")
    ax.plot(I_agents, "--", label="I agents", color="C1")
    ax.plot(R_agents, "--", label="R agents", color="C2")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of agents")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.show()

    # gradients
    # compute analytical gradient
    analytical_gradient = compute_analytical_gradient(
        args.beta, args.gamma, I, S, R, args.delta_t
    )
    # compute ode jacobian
    ode_jacobian = torch.autograd.functional.jacobian(ode_model, ode_params)
    # compute agents jacobian
    agents_jacobian = torch.autograd.functional.jacobian(agents_model, agent_params)

    # plot comparing jacobians
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(ode_jacobian[:, 0, 0], label="ode")
    ax[0].plot(agents_jacobian[0][:, 0], label="agents")
    ax[0].plot(analytical_gradient[:, 0], label="analytical")
    ax[0].set_title(r"$dS / d \beta$")
    ax[0].legend()
    ax[1].plot(ode_jacobian[:, 1, 0], label="ode")
    ax[1].plot(agents_jacobian[1][:, 0], label="agents")
    ax[1].plot(analytical_gradient[:, 1], label="analytical")
    ax[1].set_title(r"$dI / d \beta$")
    ax[1].legend()
    ax[2].plot(ode_jacobian[:, 2, 1], label="ode")
    ax[2].plot(agents_jacobian[2][:, 1], label="agents")
    ax[2].plot(analytical_gradient[:, -1], label="analytical")
    ax[2].set_title(r"$dR / d \gamma$")
    ax[2].legend()
    ax[0].set_ylabel("Gradient")
    for i in range(3):
        ax[i].set_xlabel("Time")
    f.savefig("figures/gradient_comparison_torch.png", dpi=150, bbox_inches="tight")
    # plt.show()
