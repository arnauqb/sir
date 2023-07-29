module SIR
export run, SAD, GS

using GraphNeuralNetworks, Distributions, GumbelSoftmax, Graphs, Zygote, DifferentialEquations

include("./agents.jl")
include("./ode.jl")

end # module