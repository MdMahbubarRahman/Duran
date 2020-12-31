

"""
MOI functions, sets and, other type definitions
"""
# indices
const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex

# sets
const BOUNDS = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
}

const VAR_TYPES = Union{
    MOI.ZeroOne,
    MOI.Integer
}

# other MOI types
const AFF_TERM = MOI.ScalarAffineTerm{Float64}
const QUAD_TERM = MOI.ScalarQuadraticTerm{Float64}

"""
Variable information struct definition
"""
mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # false implies lower_bound == -Inf
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # false implies upper_bound == Inf
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound
    is_binary::Bool       # Implies lower_bound == 0, upper_bound == 1 and is MOI.ZeroOne
    is_integer::Bool      # Implies variable is MOI.Integer
    start::Real           # Primal start
    name::String
end
VariableInfo() = VariableInfo(-Inf, false, Inf, false, false, false, false, 0.0, "")

"""
Optimizer struct
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{OriginalProblem, Nothing}
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{SVF, SAF, SQF, Nothing}
    linear_le_constraints::Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{Tuple{SAF, MOI.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{Tuple{SQF, MOI.LessThan{Float64}}}
    quadratic_ge_constraints::Vector{Tuple{SQF, MOI.GreaterThan{Float64}}}
    quadratic_eq_constraints::Vector{Tuple{SQF, MOI.EqualTo{Float64}}}
    #one field needs to be added to keep track of nlp constraint types
    options::SolverOptions
end

MOI.get(::Optimizer, ::MOI.SolverName) = "DuranSolver"

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::Optimizer, ::MOI.RawParameter) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    if value
        model.options.log_levels = []
    end
    model.options.silent = value
    return
end

function MOI.set(model::Optimizer, ::MOI.NumberOfThreads, value::Union{Nothing,Int})
    if value === nothing
        model.options.processors = 1
    else
        model.options.processors = value
    end
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Union{Nothing,Float64})
    if value === nothing
        model.options.time_limit = Inf
    else
        model.options.time_limit = value
    end
    return
end
#this function needs to be updated
function MOI.set(model::Optimizer, p::MOI.RawParameter, value)
    p_symbol = Symbol(p.name)
    if in(p_symbol, fieldnames(SolverOptions))
        type_of_param = fieldtype(SolverOptions, p_symbol)
        if hasmethod(convert, (Type{type_of_param}, typeof(value)))
            passed_checks = true
            if p_symbol == :traverse_strategy
                if !(value in [:BFS, :DFS, :DBFS])
                    passed_checks = false
                    @error "Traverse strategy $(value) is not supported. Use one of `[:BFS, :DFS, :DBFS]`."
                end
            end
            if p_symbol == :branch_strategy
                if !(value in [:StrongPseudoCost, :PseudoCost, :Reliability, :MostInfeasible])
                    passed_checks = false
                    @error "Branch strategy $(value) is not supported. Use one of `[:StrongPseudoCost, :PseudoCost, :Reliability, :MostInfeasible]`."
                end
            end
            passed_checks && setfield!(model.options, p_symbol, convert(type_of_param, value))
        else
            @error "The option $(p.name) has a different type ($(type_of_param))"
        end
    else
        @error "The option $(p.name) doesn't exist."
    end
    return
end

# Returns the number of solutions that can be retrieved
MOI.get(model::Optimizer, ::MOI.ResultCount) = length(model.inner.solutions)

MOI.get(model::Optimizer, ::MOI.NumberOfThreads) = model.options.processors

MOI.get(model::Optimizer, ::MOI.TimeLimitSec) = model.options.time_limit

MOI.get(model::Optimizer, ::MOI.Silent) = model.options.silent

function MOI.get(model::Optimizer, p::MOI.RawParameter)
    if in(p.name, fieldnames(SolverOptions))
        return getfield(model.options, p.name)
    end
    @error "The option $(p.name) doesn't exist."
end

"""
EmptyNLPEvaluator struct and associated functions
"""
struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::EmptyNLPEvaluator, features) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN
function MOI.eval_constraint(::EmptyNLPEvaluator, g, x)
    @assert length(g) == 0
    return
end
function MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x)
    fill!(g, 0.0)
    return
end
MOI.jacobian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
function MOI.eval_constraint_jacobian(::EmptyNLPEvaluator, J, x)
    @assert length(J) == 0
    return
end
function MOI.eval_hessian_lagrangian(::EmptyNLPEvaluator, H, x, σ, μ)
    @assert length(H) == 0
    return
end
empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

function register(s::Symbol, dimension::Integer, f::Function; autodiff::Bool=false)
    return RegisteredFunction(s, dimension, f, nothing, nothing, autodiff)
end

function register(s::Symbol, dimension::Integer, f::Function, gradf::Function; autodiff::Bool=false)
    return RegisteredFunction(s, dimension, f, gradf, nothing, autodiff)
end

function register(s::Symbol, dimension::Integer, f::Function, gradf::Function, grad2f::Function; autodiff::Bool=false)
    return RegisteredFunction(s, dimension, f, gradf, grad2f, autodiff)
end

"""
Optimizer struct constructor
"""
function Optimizer(;options...)

    solver_options = combine_options(options)

    return Optimizer(
    nothing,
    [],
    empty_nlp_data(),
    MOI.FEASIBILITY_SENSE,
    nothing,
    [], [], [], # linear constraints
    [], [], [], # quadratic constraints
    solver_options)
end

function Optimizer(options::Vector{Pair{String,Any}})
    symbol_options = Dict{Symbol, Any}()
    for option in options
        symbol_options[Symbol(option.first)] = option.second
    end
    return Optimizer(symbol_options)
end

function Optimizer(options::Dict{Symbol,Any})

    solver_options = combine_options(options)

    return Optimizer(
    nothing,
    [],
    empty_nlp_data(),
    MOI.FEASIBILITY_SENSE,
    nothing,
    [], [], [], # linear constraints
    [], [], [], # quadratic constraints
    solver_options)
end

"""
Printing the optimizer
"""
function Base.show(io::IO, model::Optimizer)
    println("A HybridMINLPSolver MathOptInterface model with backend")
    return
end

"""
Copy constructor for the optimizer
"""
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = true
function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; kws...)
    return MOI.Utilities.automatic_copy_to(model, src; kws...)
end

"""
``MOI.is_empty(model::Optimizer)`` overload for Alpine.Optimizer
"""
function MOI.is_empty(model::Optimizer)
    return isempty(model.variable_info) &&
        model.nlp_data.evaluator isa EmptyNLPEvaluator &&
        model.sense == MOI.FEASIBILITY_SENSE &&
        isempty(model.linear_le_constraints) &&
        isempty(model.linear_ge_constraints) &&
        isempty(model.linear_eq_constraints) &&
        isempty(model.quadratic_le_constraints) &&
        isempty(model.quadratic_ge_constraints) &&
        isempty(model.quadratic_eq_constraints)
end

"""
``MOI.empty!(model::Optimizer)`` overload for Alpine.Optimizer
"""
function MOI.empty!(model::Optimizer)
    model.inner = nothing
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_ge_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_ge_constraints)
    empty!(model.quadratic_eq_constraints)
end

"""
ordering of constraints provided to HybridMINLPSolver.jl
"""
linear_le_offset(model::Optimizer) = 0
linear_ge_offset(model::Optimizer) = length(model.linear_le_constraints)
linear_eq_offset(model::Optimizer) = linear_ge_offset(model) + length(model.linear_ge_constraints)
quadratic_le_offset(model::Optimizer) = linear_eq_offset(model) + length(model.linear_eq_constraints)
quadratic_ge_offset(model::Optimizer) = quadratic_le_offset(model) + length(model.quadratic_le_constraints)
quadratic_eq_offset(model::Optimizer) = quadratic_ge_offset(model) + length(model.quadratic_ge_constraints)
nlp_constraint_offset(model::Optimizer) = quadratic_eq_offset(model) + length(model.quadratic_eq_constraints)


function info_array_of_variables(variable_info::Vector{VariableInfo}, attr::Symbol)
    len_var_info = length(variable_info)
    type_dict = get_type_dict(variable_info[1])
    result = Array{type_dict[attr], 1}(undef, len_var_info)
    for i = 1:len_var_info
        result[i] = getfield(variable_info[i], attr)
        # if type is binary then set bounds correctly
        if result[i] < 0 && attr == :lower_bound && getfield(variable_info[i], :is_binary)
            result[i] = 0
        end
        if result[i] > 1 && attr == :upper_bound && getfield(variable_info[i], :is_binary)
            result[i] = 1
        end
    end
    return result
end


"""
``MOI.optimize!()`` for HybridMINLPSolver
"""
function MOI.optimize!(model::Optimizer)
    print(model)
    print("How are you doing now as you got what you wanted\n")
    MOI.initialize(model.nlp_data.evaluator, [:Grad, :Jac, :ExprGraph])
    if ~isa(model.nlp_data.evaluator, EmptyNLPEvaluator)
    else
        @info "no explicit NLP constraints or objective provided using @NLconstraint or @NLobjective macros"
    end
    model.inner = OriginalProblem()
    @views op = model.inner
    init_hybrid_problem!(op, model)
    incumbent = nothing
    create_root_model!(model, op)
    fix_primal_start!(op)
    print(op.model)
    #incumbent = solve_root_incumbent_model(op)
    #unfix_primal_start!(op)
    #backend                = JuMP.backend(op.model)
    #print(backend)
    print("\n")
    print(model.nlp_data.constraint_bounds)
    print("\n")
    IJ_jac = MOI.jacobian_structure(model.nlp_data.evaluator)
    print("\n")
    print(IJ_jac)
    print("\n")
    incum=[0.0, 0.0, 0.0, 0.0, 0.0]
    val=zeros(1, length(IJ_jac))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, val, incum)
    print("\n")
    print(val)
    print("\n")
    print(op.num_nl_constr, "\n")
    print("\n")
    print(op.num_constr, "\n")

    #this part creates linear model
    oa_data=OAdata()
    init_oa_data(op::OriginalProblem, oa_data)
    construct_linear_model(model, op, oa_data)
    add_oa_cut(model, op, oa_data)
    print("\n The mip model is the following \n")
    print(oa_data.mip_model)
    #JuMP.optimize!(oa_data.mip_model)
    #println(JuMP.value.(oa_data.mip_x))
    construct_nlp_model(model, op, oa_data)
    print(oa_data.nlp_model)

    reformulated_nlp_model(model, op, oa_data)
    print("The reformulated nlp problem is the following: \n")
    print(oa_data.ref_nlp_model)
    """
    constr_expr = MOI.constraint_expr(model.nlp_data.evaluator, 1)
    print(constr_expr, "\n")

    obj_expr = MOI.objective_expr(model.nlp_data.evaluator)
    print(obj_expr, "\n")

    as, bs = get_binvar_and_objterms(obj_expr, oa_data.nlp_model, op)
    cs=[]
    b = length(as)
    for i in 1:b
        #expr_dereferencing(as[i], oa_data.nlp_model)
        expbs = expr_dereferencing(bs[i], oa_data.nlp_model)
        push!(cs, expbs)
    end
    JuMP.@variable(oa_data.nlp_model, y[1:b])
    for i in 1:b
        exp = :((y[i])-(cs[i]) >= 0)
        JuMP.add_NL_constraint(oa_data.nlp_model, exp)
    end
    #JuMP.@NLexpression(oa_data.nlp_model, exp2, y-(as[1]-1))
    #JuMP.@NLexpression(oa_data.nlp_model, exp1, (bs[1]))
    #JuMP.add_NL_constraint(oa_data.nlp_model, exp)
    print(oa_data.nlp_model)
    """
end

getnsolutions(m::OriginalProblem) = m.nsolutions
getsolutions(m::OriginalProblem) = m.solutions


include("moi_wrapper.jl")
