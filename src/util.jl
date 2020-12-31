#=
    Used from https://github.com/lanl-ansi/Juniper.jl
=#

#=
    Used from https://github.com/lanl-ansi/Alpine.jl
=#
function expr_dereferencing!(expr, m)
    for i in 2:length(expr.args)
        if isa(expr.args[i], Union{Float64,Int64,JuMP.VariableRef})
            k = 0
        elseif expr.args[i].head == :ref
            @assert isa(expr.args[i].args[2], MOI.VariableIndex)
            expr.args[i] = JuMP.VariableRef(m, expr.args[i].args[2])
        elseif expr.args[i].head == :call
            expr_dereferencing!(expr.args[i], m)
        else
            error("expr_dereferencing :: Unexpected term in expression tree.")
        end
    end
end

function expr_dereferencing(expr, m)
    c_expr = copy(expr)
    expr_dereferencing!(c_expr, m)
    return c_expr
end


function get_type_dict(obj)
    T = typeof(obj)
    type_dict = Dict{Symbol,Type}()
    for (name, typ) in zip(fieldnames(T), T.types)
        type_dict[name] = typ
    end
    return type_dict
end

"""
    is_global_status(state::MOI.TerminationStatusCode)
Returns true if either ALMOST_OPTIMAL, OPTIMAL or INFEASIBLE and false otherwise
"""
function is_global_status(state::MOI.TerminationStatusCode)
    return state == MOI.ALMOST_OPTIMAL || state == MOI.OPTIMAL || state == MOI.INFEASIBLE
end

"""
    only_almost_solved(state::MOI.TerminationStatusCode)
Returns true if either ALMOST_OPTIMAL or ALMOST_LOCALLY_SOLVED
"""
function only_almost_solved(state::MOI.TerminationStatusCode)
    return state == MOI.ALMOST_OPTIMAL || state == MOI.ALMOST_LOCALLY_SOLVED
end

"""
    state_is_optimal(state::MOI.TerminationStatusCode; allow_almost=false)
Returns true if either optimal or locally solved. If allow_almost then check for `ALMOST_LOCALLY_SOLVED` and `ALMOST_OPTIMAL`
"""
function state_is_optimal(state::MOI.TerminationStatusCode; allow_almost=false)
    return state == MOI.OPTIMAL || state == MOI.LOCALLY_SOLVED ||
            (allow_almost && state == MOI.ALMOST_LOCALLY_SOLVED) || (allow_almost && state == MOI.ALMOST_OPTIMAL)
end

"""
    state_is_infeasible(state::MOI.TerminationStatusCode)
Returns true if either infeasible or locally infeasible
"""
function state_is_infeasible(state::MOI.TerminationStatusCode)
    return state == MOI.INFEASIBLE || state == MOI.LOCALLY_INFEASIBLE
end

"""
    add_obj_constraint(jp::HybridProblem, rhs::Float64)
Add a constraint for the objective based on whether the objective is linear/quadratic or non linear.
If the objective sense is :MIN than add objective <= rhs else objective >= rhs
"""
function add_obj_constraint(jp::HybridProblem, rhs::Float64)
    if jp.has_nl_objective
        obj_expr = MOI.objective_expr(jp.nlp_evaluator)
        if jp.obj_sense == :Min
            obj_constr = Expr(:call, :<=, obj_expr, rhs)
        else
            obj_constr = Expr(:call, :>=, obj_expr, rhs)
        end
        HybridMINLPSolver.expr_dereferencing!(obj_constr, jp.model)
        JuMP.add_NL_constraint(jp.model, obj_constr)
    else # linear or quadratic
        backend = JuMP.backend(jp.model);
        if isa(jp.objective, MOI.SingleVariable)
            if jp.obj_sense == :Min
                JuMP.set_upper_bound(jp.x[jp.objective.variable.value], rhs)
            else
                JuMP.set_lower_bound(jp.x[jp.objective.variable.value], rhs)
            end
        else
            if jp.obj_sense == :Min
                MOI.add_constraint(backend, jp.objective, MOI.LessThan(rhs))
            else
                MOI.add_constraint(backend, jp.objective, MOI.GreaterThan(rhs))
            end
        end
    end
end


"""
    evaluate_objective(optimizer::MOI.AbstractOptimizer, jp::HybridProblem, xs::Vector{Float64})
If no objective exists => return 0
Evaluate the objective whether it is non linear, linear or quadratic
"""
function evaluate_objective(optimizer::MOI.AbstractOptimizer, jp::HybridProblem, xs::Vector{Float64})
    if optimizer.nlp_data.has_objective
        return MOI.eval_objective(optimizer.nlp_data.evaluator, xs)
    elseif optimizer.objective !== nothing
        return MOIU.eval_variables(vi -> xs[vi.value], optimizer.objective)
    else
        return 0.0
    end
end


function set_time_limit!(optimizer, time_limit::Union{Nothing,Float64})
    old_time_limit = Inf
    if MOI.supports(optimizer, MOI.TimeLimitSec())
        old_time_limit = MOI.get(optimizer, MOI.TimeLimitSec())
        MOI.set(optimizer, MOI.TimeLimitSec(), time_limit)
    end
    return old_time_limit
end

"""
    optimize_get_status_backend(model::JuMP.Model; solver=nothing)
Run optimize! and get the status and the backend
"""
function optimize_get_status_backend(model::JuMP.Model; solver=nothing)
    if solver === nothing
        JuMP.optimize!(model)
    else
        JuMP.set_optimizer(model, solver)
        JuMP.optimize!(model)
    end
    backend = JuMP.backend(model)
    status = MOI.get(backend, MOI.TerminationStatus())
    return status, backend
end

function register_functions!(model, registered_functions)
    if registered_functions !== nothing
        for reg_f in registered_functions
            if reg_f.gradf === nothing
                JuMP.register(model, reg_f.s, reg_f.dimension, reg_f.f; autodiff=reg_f.autodiff)
            elseif reg_f.grad2f === nothing
                JuMP.register(model, reg_f.s, reg_f.dimension, reg_f.f, reg_f.gradf)
            else
                JuMP.register(model, reg_f.s, reg_f.dimension, reg_f.f, reg_f.gradf, reg_f.grad2f)
            end
        end
    end
end
