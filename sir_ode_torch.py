import torch

from torchdiffeq import odeint

class ODE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, y, beta, gamma):
        S, I, _ = y
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR])

class SIRODE(torch.nn.Module):
    def __init__(self, y0, t):
        super().__init__()
        self.y0 = y0
        self.t = t
        self.ode = ODE()

    def forward(self, params):
        beta, gamma = params
        def f_ode(t, y):
            return self.ode(t, y, beta, gamma)
        return odeint(f_ode, self.y0, self.t)

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--y0", type=float, nargs=3, default=[0.90, 0.10, 0.0])
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)

    args = parser.parse_args()

    t = torch.linspace(args.t0, args.t1, int((args.t1 - args.t0) / args.dt) + 1)
    y0 = torch.tensor(args.y0, dtype=torch.float32)

    model = SIRODE(y0, t)
    params = torch.tensor([args.beta, args.gamma], dtype=torch.float32)
    y = model(params)

    # plot results
    plt.plot(t, y[:, 0], label="S")
    plt.plot(t, y[:, 1], label="I")
    plt.plot(t, y[:, 2], label="R")
    plt.ylim(0,1)
    plt.xlabel("Time")
    plt.ylabel("Fraction of agents")
    plt.legend()
    plt.title("ODE PyTorch")
    plt.savefig("./figures/sir_ode_torch.png", dpi=150)
    #plt.show()


