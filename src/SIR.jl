module SIR
export run, SAD, GS

using DifferentialEquations
using Distributions
using GraphNeuralNetworks
using Graphs
using GumbelSoftmax
using StaticArrays
using SciMLSensitivity
using Zygote

include("./agents.jl")
include("./ode.jl")

end # module