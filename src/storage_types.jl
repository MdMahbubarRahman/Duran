#=
    many of the types are
    Used from https://github.com/lanl-ansi/Juniper.jl
=#

mutable struct RegisteredFunction
    s         :: Symbol
    dimension :: Integer
    f         :: Function
    gradf     :: Union{Nothing, Function}
    grad2f    :: Union{Nothing, Function}
    autodiff  :: Bool
end


# Options for the solver (more details like defaults in solver.jl)
mutable struct SolverOptions
    nl_solver                           :: Any# needs to be set
    log_levels                          :: Vector{Symbol}
    silent                              :: Bool
    atol                                :: Float64
    incumbent_constr                    :: Bool
    obj_epsilon                         :: Float64

    time_limit                          :: Float64
    mip_gap                             :: Float64

    best_obj_stop                       :: Float64
    solution_limit                      :: Int64
    all_solutions                       :: Bool

    list_of_solutions                   :: Bool
    mip_solver                          :: Any
    allow_almost_solved                 :: Bool
    allow_almost_solved_integral        :: Bool
    registered_functions                :: Union{Nothing, Vector{RegisteredFunction}}
    processors                          :: Int64
end

mutable struct SolutionObj
    solution    :: Vector{Float64}
    objval      :: Float64
end


#OriginalProblem
mutable struct OriginalProblem
    nl_solver           :: Any
    nl_solver_options   :: Vector{Pair}
    model               :: JuMP.Model
    relaxation_status   :: MOI.TerminationStatusCode
    relaxation_objval   :: Float64
    relaxation_solution :: Vector{Float64}
    status              :: MOI.TerminationStatusCode
    objval              :: Float64
    best_bound          :: Float64
    x                   :: Vector{JuMP.VariableRef}
    primal_start        :: Vector{Real}
    num_constr          :: Int64
    num_nl_constr       :: Int64
    num_q_constr        :: Int64
    num_l_constr        :: Int64
    num_var             :: Int64
    l_var               :: Vector{Float64}
    u_var               :: Vector{Float64}
    has_nl_objective    :: Bool
    nlp_evaluator       :: MOI.AbstractNLPEvaluator
    objective           :: Union{SVF, SAF, SQF, Nothing}
    disc2var_idx        :: Vector{Int64}
    var2disc_idx        :: Vector{Int64}
    var_type            :: Vector{Symbol}
    obj_sense           :: Symbol
    num_disc_var        :: Int64
    solution            :: Vector{Float64}
    soltime             :: Float64
    solutions           :: Vector{SolutionObj}
    nsolutions          :: Int64
    mip_solver          :: Any
    mip_solver_options  :: Vector{Pair}
    relaxation_time     :: Float64
    start_time          :: Float64
    # Info
    nintvars            :: Int64
    nbinvars            :: Int64
    options             :: SolverOptions
    nnodes              :: Int64

    nlp_constr_type     :: Vector{Symbol}

    OriginalProblem() = new()
end


#OriginalProblem
mutable struct OAdata
    mip_model               :: JuMP.Model
    nlp_model               :: JuMP.Model
    ref_nlp_model           :: JuMP.Model
    ref_mip_model           :: JuMP.Model

    mip_x                   :: Vector{JuMP.VariableRef}
    nlp_x                   :: Vector{JuMP.VariableRef}
    ref_nlp_x               :: Vector{JuMP.VariableRef}
    ref_mip_x               :: Vector{JuMP.VariableRef}

    ref_l_var               :: Vector{Float64}
    ref_u_var               :: Vector{Float64}
    ref_var_type            :: Vector{Symbol}
    ref_num_var             :: Int
    ref_num_nl_constr       :: Int
    ref_nlp_constr_type     :: Vector{Symbol}

    obj_sense               :: Symbol

    oa_status               :: Symbol
    oa_started              :: Bool
    incumbent               :: Vector{Float64}
    new_incumbent           :: Bool
    total_time              :: Float64
    obj_val                 :: Float64
    obj_bound               :: Float64
    obj_gap                 :: Float64
    oa_iter                 :: Int64

    OAdata() = new()
end



"""
    is_type_correct(x, var_type, atol)
Check whether a variable x has the correct type
"""
function is_type_correct(x, var_type, atol)
    if var_type != :Cont
        if !isapprox(round(x)-x, 0; atol=atol)
            return false
        end
    end
    return true
end

"""
    are_type_correct(sol, types, disc2var_idx, atol)
Check whether all variables have the correct type
"""
function are_type_correct(sol, types, disc2var_idx, atol)
    for i in disc2var_idx
        if !isapprox(round(sol[i])-sol[i], 0; atol=atol)
            return false
        end
    end
    return true
end

"""
    all_reasonable_type_correct(sol, disc2var_idx, reasonable_idx, atol)
Check whether all reasonable variable have are discrete alreadyy
"""
function all_reasonable_type_correct(sol, disc2var_idx, reasonable_idx, atol)
    for i in reasonable_idx
        idx = disc2var_idx[i]
        if !isapprox(round(sol[idx])-sol[idx], 0; atol=atol)
            return false
        end
    end
    return true
end
