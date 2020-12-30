
function init_original_problem!(op::HybridProblem, model::MOI.AbstractOptimizer)
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
    op.var_type = [:Cont for i in 1:jp.num_var]
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
