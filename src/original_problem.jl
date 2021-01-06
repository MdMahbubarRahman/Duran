
#=
    Used from https://github.com/lanl-ansi/Juniper.jl
=#

function init_original_problem!(op::OriginalProblem, model::MOI.AbstractOptimizer)
    num_variables = length(model.variable_info)
    num_linear_le_constraints = length(model.linear_le_constraints)
    num_linear_ge_constraints = length(model.linear_ge_constraints)
    num_linear_eq_constraints = length(model.linear_eq_constraints)
    num_quadratic_le_constraints = length(model.quadratic_le_constraints)
    num_quadratic_ge_constraints = length(model.quadratic_ge_constraints)
    num_quadratic_eq_constraints = length(model.quadratic_eq_constraints)

    op.status = MOI.OPTIMIZE_NOT_CALLED
    op.relaxation_status = MOI.OPTIMIZE_NOT_CALLED
    op.has_nl_objective = model.nlp_data.has_objective
    op.nlp_evaluator = model.nlp_data.evaluator
    op.objective = model.objective

    op.objval = NaN
    op.best_bound = NaN
    op.solution = fill(NaN, num_variables)
    op.nsolutions = 0
    op.solutions = []
    op.num_disc_var = 0
    op.nintvars = 0
    op.nbinvars = 0
    op.nnodes = 1 # is set to one for the root node

    op.nlp_constr_type = []

    op.start_time = time()

    op.nl_solver = model.options.nl_solver
    nl_vec_opts = Vector{Pair}()
    if isa(op.nl_solver, MOI.OptimizerWithAttributes)
        for arg in model.options.nl_solver.params
            push!(nl_vec_opts, arg)
        end
    else
        op.nl_solver = JuMP.optimizer_with_attributes(op.nl_solver)
    end
    op.nl_solver_options = nl_vec_opts

    if model.options.mip_solver !== nothing
        op.mip_solver = model.options.mip_solver
        mip_vec_opts = Vector{Pair}()
        if isa(op.mip_solver, MOI.OptimizerWithAttributes)
            for arg in model.options.mip_solver.params
                push!(mip_vec_opts, arg)
            end
        else
            op.mip_solver = JuMP.optimizer_with_attributes(op.mip_solver)
        end
        op.mip_solver_options = mip_vec_opts
    end
    op.options = model.options
    if model.sense == MOI.MIN_SENSE
        op.obj_sense = :Min
    else
        op.obj_sense = :Max
    end
    op.l_var = info_array_of_variables(model.variable_info, :lower_bound)
    op.u_var = info_array_of_variables(model.variable_info, :upper_bound)
    integer_bool_arr = info_array_of_variables(model.variable_info, :is_integer)
    binary_bool_arr = info_array_of_variables(model.variable_info, :is_binary)
    primal_start_arr = info_array_of_variables(model.variable_info, :start)
    op.primal_start = primal_start_arr
    op.nintvars = sum(integer_bool_arr)
    op.nbinvars = sum(binary_bool_arr)
    op.num_disc_var = sum(integer_bool_arr)+sum(binary_bool_arr)
    op.num_var = length(model.variable_info)
    op.var_type = [:Cont for i in 1:op.num_var]
    op.var_type[integer_bool_arr .== true] .= :Int
    op.var_type[binary_bool_arr .== true] .= :Bin
    op.disc2var_idx = zeros(op.num_disc_var)
    op.var2disc_idx = zeros(op.num_var)
    int_i = 1
    for i=1:op.num_var
        if op.var_type[i] != :Cont
            op.disc2var_idx[int_i] = i
            op.var2disc_idx[i] = int_i
            int_i += 1
        end
    end
    op.num_l_constr = num_linear_le_constraints+num_linear_ge_constraints+num_linear_eq_constraints
    op.num_q_constr = num_quadratic_le_constraints+num_quadratic_ge_constraints+num_quadratic_eq_constraints
    op.num_nl_constr = length(model.nlp_data.constraint_bounds)
    op.num_constr = op.num_l_constr+op.num_q_constr+op.num_nl_constr
end




function create_original_model!(optimizer::MOI.AbstractOptimizer, op::OriginalProblem; fix_start=false)
    ps = op.options.log_levels

    op.model = Model()
    lb = op.l_var
    ub = op.u_var
    # all continuous we solve relaxation first
    @variable(op.model, lb[i] <= x[i=1:op.num_var] <= ub[i])

    for i=1:op.num_var
        JuMP.set_start_value(x[i], op.primal_start[i])
    end

    register_functions!(op.model, op.options.registered_functions)

    # TODO check whether it is supported
    if optimizer.nlp_data.has_objective
        obj_expr = MOI.objective_expr(optimizer.nlp_data.evaluator)
        obj_expr = expr_dereferencing(obj_expr, op.model)
        try
            JuMP.set_NL_objective(op.model, optimizer.sense, obj_expr)
        catch
            error("Have you registered a function? Then please register the function also for Juniper see: \n
            https://lanl-ansi.github.io/Juniper.jl/stable/options/#registered_functions%3A%3AUnion%7BNothing%2CVector%7BRegisteredFunction%7D%7D-%5Bnothing%5D-1")
        end
    elseif optimizer.objective !== nothing
        MOI.set(op.model, MOI.ObjectiveFunction{typeof(optimizer.objective)}(), optimizer.objective)
        MOI.set(op.model, MOI.ObjectiveSense(), optimizer.sense)
    end

    backend = JuMP.backend(op.model);
    llc = optimizer.linear_le_constraints
    lgc = optimizer.linear_ge_constraints
    lec = optimizer.linear_eq_constraints
    qlc = optimizer.quadratic_le_constraints
    qgc = optimizer.quadratic_ge_constraints
    qec = optimizer.quadratic_eq_constraints
    for constr_type in [llc, lgc, lec, qlc, qgc, qec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end
    for i in 1:op.num_nl_constr
        constr_expr = MOI.constraint_expr(optimizer.nlp_data.evaluator, i)
        constr_expr = expr_dereferencing(constr_expr, op.model)
        push!(op.nlp_constr_type, constr_expr.args[1])
        try
            JuMP.add_NL_constraint(op.model, constr_expr)
        catch
            error("Have you registered a function? Then please register the function also for Juniper see: \n
            https://lanl-ansi.github.io/Juniper.jl/stable/options/#registered_functions%3A%3AUnion%7BNothing%2CVector%7BRegisteredFunction%7D%7D-%5Bnothing%5D-1")
        end
    end

    op.x = x
end

function fix_primal_start!(op::OriginalProblem)
    lb = op.l_var
    ub = op.u_var
    x = op.x
    for i=1:op.num_var
        if op.var_type[i] != :Cont && lb[i] <= op.primal_start[i] <= ub[i]
            JuMP.set_lower_bound(op.x[i], op.primal_start[i])
            JuMP.set_upper_bound(op.x[i], op.primal_start[i])
        else
            JuMP.set_start_value(x[i], op.primal_start[i])
        end
    end
end

function unfix_primal_start!(op::OriginalProblem)
    lb = op.l_var
    ub = op.u_var
    x = op.x
    for i=1:op.num_var
        JuMP.set_lower_bound(op.x[i], lb[i])
        JuMP.set_upper_bound(op.x[i], ub[i])
        JuMP.set_start_value(x[i], op.primal_start[i])
    end
end

function solve_root_incumbent_model(op::OriginalProblem)
    status, backend = optimize_get_status_backend(op.model; solver=op.nl_solver)
    incumbent = nothing
    ps = op.options.log_levels
    if state_is_optimal(status; allow_almost=op.options.allow_almost_solved)
        # set incumbent
        objval = MOI.get(backend, MOI.ObjectiveValue())
        solution = JuMP.value.(op.x)
        if are_type_correct(solution, op.var_type, op.disc2var_idx, op.options.atol)
            if only_almost_solved(status) && op.options.allow_almost_solved_integral
                @warn "Start value incumbent only almost locally solved. Disable with `allow_almost_solved_integral=false`"
            end
        end
        #incumbent = Incumbent(objval, solution, only_almost_solved(status))
    end
    return incumbent
end
