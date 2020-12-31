#=

=#

module Duran

import JuMP
import Compat
using JuMP: @variable, @constraint, @objective, Model, optimizer_with_attributes
using JSON
using LinearAlgebra
using Random
using Distributed
using Statistics
using SparseArrays

using MathOptInterface, Ipopt, Cbc
const MOI = MathOptInterface
const MOIU = MOI.Utilities

# functions
const SVF = MOI.SingleVariable
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const VECTOR = MOI.VectorOfVariables

include("storage_types.jl")
include("util.jl")
include("oa_util.jl")
include("OriginalProblem.jl")
include("solver_attributes.jl")
include("moi_wrapper.jl")

function Base.show(io::IO, opts::SolverOptions)
    longest_field_name = maximum([length(string(fname)) for fname in fieldnames(SolverOptions)])+2
    for name in fieldnames(SolverOptions)
        sname = string(name)
        pname = sname*repeat(" ", longest_field_name-length(sname))
        if getfield(opts,name) === nothing
            println(io, pname, ": NA")
        else
            println(io, pname, ": ", getfield(opts,name))
        end
    end
end



end # module