#=
    Used from https://github.com/lanl-ansi/Juniper.jl
=#

"""
moi_wrapper for variables start
"""

"""
MOI variables
"""

"""
Getters for variables
"""
MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variable_info)

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(model.variable_info)]
end

"""
``MOI.add_variable()`` overloads and safety functions
"""
function MOI.add_variable(model::Optimizer)
    push!(model.variable_info, VariableInfo())
    return MOI.VariableIndex(length(model.variable_info))
end

function MOI.add_variables(model::Optimizer, n::Int)
    return [MOI.add_variable(model) for i in 1:n]
end

function check_inbounds(model::Optimizer, vi::VI)
	num_variables = length(model.variable_info)
	if !(1 <= vi.value <= num_variables)
	    @error "Invalid variable index $vi. ($num_variables variables in the model.)"
	end
	return
end

check_inbounds(model::Optimizer, var::SVF) = check_inbounds(model, var.variable)

function check_inbounds(model::Optimizer, aff::SAF)
	for term in aff.terms
	    check_inbounds(model, term.variable_index)
	end
	return
end

function check_inbounds(model::Optimizer, quad::SQF)
	for term in quad.affine_terms
	    check_inbounds(model, term.variable_index)
	end
	for term in quad.quadratic_terms
	    check_inbounds(model, term.variable_index_1)
	    check_inbounds(model, term.variable_index_2)
	end
	return
end

has_upper_bound(model::Optimizer, vi::VI) =
    model.variable_info[vi.value].has_upper_bound

has_lower_bound(model::Optimizer, vi::VI) =
    model.variable_info[vi.value].has_lower_bound

is_fixed(model::Optimizer, vi::VI) =
    model.variable_info[vi.value].is_fixed

"""
Primal-start support for HybridMINLPSolver
"""
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{VI}) = true

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Union{Real, Nothing})
    check_inbounds(model, vi)
    if value === nothing
        value = 0.0
    end
    model.variable_info[vi.value].start = value
    return
end

"""
MOI.VariableName attribute support
"""
MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{VI}) = true

function MOI.set(model::Optimizer, ::MOI.VariableName, vi::VI, name::String)
	model.variable_info[vi.value].name = name
	return
end

function MOI.get(model::Optimizer, ::MOI.VariableName, vi::VI)
	return model.variable_info[vi.value].name
end

"""
Populating Variable bounds
"""
function MOI.add_constraint(model::Optimizer, v::SVF, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(lt.upper)
        @error "Invalid upper bound value $(lt.upper)."
    end
    if has_upper_bound(model, vi)
        @error "Upper bound on variable $vi already exists."
    end
    if is_fixed(model, vi)
        @error "Variable $vi is fixed. Cannot also set upper bound."
    end
    model.variable_info[vi.value].upper_bound = lt.upper
    model.variable_info[vi.value].has_upper_bound = true
    return MOI.ConstraintIndex{SVF, MOI.LessThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::SVF, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(gt.lower)
        @error "Invalid lower bound value $(gt.lower)."
    end
    if has_lower_bound(model, vi)
        @error "Lower bound on variable $vi already exists."
    end
    if is_fixed(model, vi)
        @error "Variable $vi is fixed. Cannot also set lower bound."
    end
    model.variable_info[vi.value].lower_bound = gt.lower
    model.variable_info[vi.value].has_lower_bound = true
    return MOI.ConstraintIndex{SVF, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::SVF, eq::MOI.EqualTo{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(eq.value)
        @error "Invalid fixed value $(eq.value)."
    end
    if has_lower_bound(model, vi)
        @error "Variable $vi has a lower bound. Cannot be fixed."
    end
    if has_upper_bound(model, vi)
        @error "Variable $vi has an upper bound. Cannot be fixed."
    end
    if is_fixed(model, vi)
        @error "Variable $vi is already fixed."
    end
    model.variable_info[vi.value].lower_bound = eq.value
    model.variable_info[vi.value].upper_bound = eq.value
    model.variable_info[vi.value].is_fixed = true
    return MOI.ConstraintIndex{SVF, MOI.EqualTo{Float64}}(vi.value)
end

"""
moi_wrapper for variables end
"""



"""
moi_wrapper for constraints start
"""

"""
MOI constraints
"""

"""
Single variable bound constraints
"""
MOI.supports_constraint(::Optimizer, ::Type{SVF}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{SVF}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{SVF}, ::Type{MOI.EqualTo{Float64}}) = true

"""
Binary/Integer variable support
"""
MOI.supports_constraint(::Optimizer, ::Type{SVF}, ::Type{<:VAR_TYPES}) = true

"""
Linear constraints
"""
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:BOUNDS}) = true

"""
Quadratic constraints (scalar i.e., vectorized constraints are not supported)
"""
MOI.supports_constraint(::Optimizer, ::Type{SQF}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{SQF}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{SQF}, ::Type{MOI.EqualTo{Float64}}) = true


"""
``define_add_constraint`` macro - from Ipopt.jl
"""
macro define_add_constraint(function_type, set_type, array_name)
    quote
        function MOI.add_constraint(model::Optimizer, func::$function_type, set::$set_type)
            check_inbounds(model, func)
            push!(model.$(array_name), (func, set))
            return MOI.ConstraintIndex{$function_type, $set_type}(length(model.$(array_name)))
        end
    end
end

"""
``MOI.add_constraint()`` overloads for all the supported constraint types
"""
@define_add_constraint(SAF, MOI.LessThan{Float64}, linear_le_constraints)
@define_add_constraint(SAF, MOI.GreaterThan{Float64}, linear_ge_constraints)
@define_add_constraint(SAF, MOI.EqualTo{Float64}, linear_eq_constraints)
@define_add_constraint(SQF, MOI.LessThan{Float64}, quadratic_le_constraints)
@define_add_constraint(SQF, MOI.GreaterThan{Float64}, quadratic_ge_constraints)
@define_add_constraint(SQF, MOI.EqualTo{Float64}, quadratic_eq_constraints)

"""
Binary variable support
"""
function MOI.add_constraint(model::Optimizer, v::SVF, ::MOI.ZeroOne)
    vi = v.variable
    check_inbounds(model, vi)
    # the bounds are set using info_array_of_variables
    # according to mlubin the bounds should not be set here as the bounds should stay when
    # the binary constraint is deleted
    model.variable_info[vi.value].is_binary = true

    return MOI.ConstraintIndex{SVF, MOI.ZeroOne}(vi.value)
end

"""
Integer variable support
"""
function MOI.add_constraint(model::Optimizer, v::SVF, ::MOI.Integer)
    vi = v.variable
	check_inbounds(model, vi)
    model.variable_info[vi.value].is_integer = true

    return MOI.ConstraintIndex{SVF, MOI.Integer}(vi.value)
end

"""
ConstraintIndex support
"""
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{CI}) = true


"""
moi_wrapper for constraints end
"""

"""
moi_wrapper for objective start
"""
"""
MOI objective
"""

"""
Supported objective types and sense
"""
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{SVF}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{SAF}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{SQF}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

"""
set and get function overloads
"""
MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction, func::Union{SVF, SAF, SQF})
    check_inbounds(model, func)
    model.objective = func
    return
end

"""
moi_wrapper for objective end
"""


"""
moi_wrapper for nlp start
"""
"""
MOI NLPBlock
"""

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

"""
moi_wrapper for nlp end
"""

"""
moi_wrapper for results start
"""
# MathOptInterface results

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
	return model.inner.status
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return string(model.inner.status)
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if state_is_optimal(model.inner.status; allow_almost=model.inner.options.allow_almost_solved)
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end

function MOI.get(model::Optimizer, ::MOI.DualStatus)
    if state_is_optimal(model.inner.status; allow_almost=model.inner.options.allow_almost_solved)
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end


function MOI.get(model::Optimizer, ::MOI.ObjectiveValue)
    if model.inner.status == MOI.OPTIMIZE_NOT_CALLED
        @error "optimize! not called"
    end
    return model.inner.objval
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    if model.inner.status == MOI.OPTIMIZE_NOT_CALLED
        @error "optimize! not called"
    end
    return model.inner.best_bound
end

function MOI.get(model::Optimizer, ::MOI.RelativeGap)
    if model.inner.status == MOI.OPTIMIZE_NOT_CALLED
        @error "optimize! not called"
    end
    if isnan(model.inner.objval) || isnan(model.inner.best_bound)
        return NaN
    end
    return abs(model.inner.best_bound-model.inner.objval)/abs(model.inner.objval)
end

function MOI.get(model::Optimizer, ::MOI.SolveTime)
    if model.inner.status == MOI.OPTIMIZE_NOT_CALLED
        @error "optimize! not called"
    end
    return model.inner.soltime
end

function MOI.get(model::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    if model.inner.status == MOI.OPTIMIZE_NOT_CALLED
        @error "optimize! not called"
    end
    check_inbounds(model, vi)
    return model.inner.solution[vi.value]
end

"""
moi_wrapper for results end
"""
