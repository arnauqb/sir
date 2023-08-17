module SIR
export run, SAD, GS, BernoulliGrad, AgentParameters, ODEParameters

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
include("./si.jl")


end # module