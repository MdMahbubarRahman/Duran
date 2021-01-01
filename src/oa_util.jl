"""
necessary functions for outer approximation framework
"""

"""
function init_oa_data(op::OriginalProblem, oa_data::OAdata)
    initiates OAdata structure using information of the OriginalProblem.
"""
function init_oa_data(op::OriginalProblem, oa_data::OAdata)
    oa_data.ref_l_var               = Float64[]
    oa_data.ref_u_var               = Float64[]
    oa_data.ref_num_var             = 0
    oa_data.ref_num_nl_constr       = 0

    oa_data.obj_sense               = op.obj_sense

    oa_data.oa_status               = :Unknown
    oa_data.oa_started              = false
    oa_data.incumbent               = Float64[]
    oa_data.new_incumbent           = false
    oa_data.total_time              = 0
    oa_data.obj_val                 = Inf
    oa_data.obj_bound               = -Inf
    oa_data.obj_gap                 = Inf
    oa_data.oa_iter                 = 0
end

"""
function construct_linear_model(optimizer::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    this function generates a new mip_model with the linear constraints and linear objective function
    of the original model OriginalProblem.
"""
function construct_linear_model(optimizer::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    oa_data.mip_model = Model()
    lb = op.l_var
    ub = op.u_var
    push!(lb, -Inf)
    push!(ub, Inf)
    #vartype = op.var_type
    #push!(vartype, :Cont)
    # all continuous we solve relaxation first
    @variable(oa_data.mip_model, lb[i] <= x[i=1:(op.num_var+1)] <= ub[i])
    for i in 1:length(op.var_type)
        op.var_type[i] == :Int && JuMP.set_integer(x[i])
        op.var_type[i] == :Bin && JuMP.set_binary(x[i])
    end
    # register function into the mip model
    register_functions!(oa_data.mip_model, op.options.registered_functions)
    # attach linear constraints to the mip model
    backend1 = JuMP.backend(oa_data.mip_model);
    llc = optimizer.linear_le_constraints
    lgc = optimizer.linear_ge_constraints
    lec = optimizer.linear_eq_constraints
    #add linear constraints to the mip model
    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend1, constr[1], constr[2])
        end
    end
    #set objective
    if op.has_nl_objective
        @info "the objective function is nonlinear.\n"
        op.obj_sense == :Min && @objective(oa_data.mip_model, Min, x[op.num_var+1])
        op.obj_sense == :Max && @objective(oa_data.mip_model, Max, x[op.num_var+1])
    else
        @info "the objective function is linear or quadratic.\n"
        MOI.set(oa_data.mip_model, MOI.ObjectiveFunction{typeof(optimizer.objective)}(), optimizer.objective)
        MOI.set(oa_data.mip_model, MOI.ObjectiveSense(), optimizer.sense)
    end
    oa_data.mip_x = x
end

"""
function add_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    this function generates oa cuts for nonlinear constraints and objective function and adds the cuts
    to the mip_model which is a milp model.
"""
function add_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    jac_IJ = MOI.jacobian_structure(model.nlp_data.evaluator)
    nlp_solution=ones(op.num_var)

    g_val = zeros(op.num_nl_constr)
    g_jac = zeros(length(jac_IJ))
    MOI.eval_constraint(model.nlp_data.evaluator, g_val, nlp_solution)
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, g_jac, nlp_solution)

    #constraints cut
    varidx_new = [zeros(Int, 0) for i in 1:op.num_nl_constr]
    coef_new = [zeros(0) for i in 1:op.num_nl_constr]

    for k in 1:length(jac_IJ)
        row = jac_IJ[k][1]
        push!(varidx_new[row], jac_IJ[k][2])
        push!(coef_new[row], g_jac[k])
    end

    for i in 1:op.num_nl_constr
        new_rhs = -g_val[i]
        for j in 1:length(varidx_new[i])
            new_rhs += coef_new[i][j] * nlp_solution[Int(varidx_new[i][j])]
        end
        if op.nlp_constr_type[i] == :(<=)
            @constraint(oa_data.mip_model, dot(coef_new[i], oa_data.mip_x[varidx_new[i]]) <= new_rhs)
        else
            @constraint(oa_data.mip_model, dot(coef_new[i], oa_data.mip_x[varidx_new[i]]) >= new_rhs)
        end
    end

    #objective cut
    if op.has_nl_objective
        f_nab = zeros(op.num_var+1)
        f_val = MOI.eval_objective(model.nlp_data.evaluator, nlp_solution)
        MOI.eval_objective_gradient(model.nlp_data.evaluator, f_nab, nlp_solution)

        new_rhs = -f_val
        varidx = zeros(Int, op.num_var+1)
        for j in 1:op.num_var
            varidx[j] = j
            new_rhs += f_nab[j] * nlp_solution[j]
        end

        varidx[op.num_var+1] = op.num_var + 1
        f_nab[op.num_var+1] = -1.0

        if op.obj_sense == :Max
            @constraint(oa_data.mip_model, dot(f_nab, oa_data.mip_x[varidx]) >= new_rhs)
        else
            @constraint(oa_data.mip_model, dot(f_nab, oa_data.mip_x[varidx]) <= new_rhs)
        end
    end
end


"""
function construct_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    this function generates a nlp_model from the original model provided as OriginalProblem.
"""
function construct_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    oa_data.nlp_model = Model()
    lb = op.l_var
    ub = op.u_var
    @variable(oa_data.nlp_model, lb[i] <= x[i=1:op.num_var] <= ub[i])
    register_functions!(oa_data.nlp_model, op.options.registered_functions)

    backend2 = JuMP.backend(oa_data.nlp_model);
    llc = model.linear_le_constraints
    lgc = model.linear_ge_constraints
    lec = model.linear_eq_constraints

    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend2, constr[1], constr[2])
        end
    end

    for i in 1:op.num_nl_constr
        constr_expr = MOI.constraint_expr(model.nlp_data.evaluator, i)
        constr_expr = expr_dereferencing(constr_expr, oa_data.nlp_model)
        try
            JuMP.add_NL_constraint(oa_data.nlp_model, constr_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    end

    if model.nlp_data.has_objective
        obj_expr = MOI.objective_expr(model.nlp_data.evaluator)
        obj_expr = expr_dereferencing(obj_expr, oa_data.nlp_model)
        try
            JuMP.set_NL_objective(oa_data.nlp_model, model.sense, obj_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    elseif model.objective !== nothing
        MOI.set(oa_data.nlp_model, MOI.ObjectiveFunction{typeof(model.objective)}(), model.objective)
        MOI.set(oa_data.nlp_model, MOI.ObjectiveSense(), model.sense)
    end

    oa_data.nlp_x = x

    return nothing
end


"""
function solve_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    this function solves mip_model with mixed interger solver.
"""

function solve_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    JuMP.set_optimizer(oa_data.mip_model, model.options.mip_solver)
    JuMP.optimize!(oa_data.mip_model)

    if termination_status(oa_data.mip_model) == :OPTIMAL || termination_status(oa_data.mip_model) == :LOCALLY_SOLVED
        @info "There is a feasible solution in the mip_model.\n"
        value = JuMP.value.(oa_data.mip_x)
        return value
    else
        @error "There is no feasible solution from mip_model. \n"
    end
    #status = JuMP.primal_status(oa_data.solve_mip_model)
end

function solve_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    JuMP.set_optimizer(oa_data.nlp_model, model.options.nlp_solver)
    JuMP.optimize!(oa_data.nlp_model)

    if termination_status(oa_data.nlp_model) == :OPTIMAL || termination_status(oa_data.nlp_model) == :LOCALLY_SOLVED
        @info "There is a feasible solution in the mip_model.\n"
        value = JuMP.value.(oa_data.nlp_x)
        return value
    else
        @error "There is no feasible solution from mip_model. \n"
    end
    #status = JuMP.primal_status(oa_data.solve_mip_model)
end



"""
function construct_ref_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    this function generates a ref_nlp_model from the original model.
"""
function construct_ref_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oa_data::OAdata)
    oa_data.ref_nlp_model = Model()
    lb = op.l_var
    ub = op.u_var
    vartype = op.var_type

    for i in 1:(op.nbinvars)
        push!(lb, -Inf)
        push!(ub, Inf)
        push!(vartype, :Cont)
    end

    oa_data.ref_l_var = lb
    oa_data.ref_u_var = ub
    oa_data.ref_var_type = vartype

    @variable(oa_data.ref_nlp_model, lb[i] <= x[i=1:(op.num_var+op.nbinvars)] <= ub[i])
    register_functions!(oa_data.ref_nlp_model, op.options.registered_functions)

    oa_data.ref_num_var = length(x)

    backend = JuMP.backend(oa_data.ref_nlp_model);
    llc = model.linear_le_constraints
    lgc = model.linear_ge_constraints
    lec = model.linear_eq_constraints

    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end

    for i in 1:op.num_nl_constr
        constr_expr = MOI.constraint_expr(model.nlp_data.evaluator, i)
        constr_expr = expr_dereferencing(constr_expr, oa_data.ref_nlp_model)
        try
            JuMP.add_NL_constraint(oa_data.ref_nlp_model, constr_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    end
    num_nl_constr = op.num_nl_constr
    #additional constraints from the nlobj
    if model.nlp_data.has_objective
        if op.obj_sense == :Max
            @info "The objective sense is Max. \n"
            @objective(oa_data.ref_nlp_model, Max, sum(x[i] for i in (op.num_var+1):(op.num_var+op.nbinvars)))
        elseif op.obj_sense == :Min
            @info "The objective sense is Min. \n"
            @objective(oa_data.ref_nlp_model, Min, sum(x[i] for i in (op.num_var+1):(op.num_var+op.nbinvars)))
        end

        obj_expr = MOI.objective_expr(model.nlp_data.evaluator)
        moi_bin_vars, moi_obj_terms = get_binvar_and_objterms(obj_expr, oa_data.ref_nlp_model, op)

        siz = length(moi_bin_vars)
        jump_bin_vars = []
        jump_obj_terms = []
        for i in 1:siz
            jump_bin = JuMP.VariableRef(oa_data.ref_nlp_model, moi_bin_vars[i].args[2])
            push!(jump_bin_vars, jump_bin)
            jump_exp = expr_dereferencing(moi_obj_terms[i], oa_data.ref_nlp_model)
            push!(jump_obj_terms, jump_exp)
        end
        num_nl_constr = num_nl_constr+siz
        ref_nlpcontr_type = op.nlp_constr_type
        for i in 1:siz
            L=-100000
            U=+100000
            JuMP.@constraint(oa_data.ref_nlp_model, x[i+op.num_var]-jump_bin_vars[i]*L >= 0)
            exp = :($(x[i+op.num_var])-($(jump_bin_vars[i])-$(1))*$(U)-$(jump_obj_terms[i]) >= 0)
            JuMP.add_NL_constraint(oa_data.ref_nlp_model, exp)
            push!(ref_nlpcontr_type, :>=)
        end
        oa_data.ref_nlp_constr_type = ref_nlpcontr_type

    elseif model.objective !== nothing
        MOI.set(oa_data.ref_nlp_model, MOI.ObjectiveFunction{typeof(model.objective)}(), model.objective)
        MOI.set(oa_data.ref_nlp_model, MOI.ObjectiveSense(), model.sense)
    end
    oa_data.ref_num_nl_constr = num_nl_constr
    oa_data.ref_nlp_x = x

    return nothing
end


function _expr_deref!(expr, exprd, m, op, dict, bin_vars, bin_obj_terms)
    for i in 2:length(expr.args)
        if isa(expr.args[i], Union{Float64,Int64,JuMP.VariableRef})
            k = 0
        elseif expr.args[i].head == :ref
            @assert isa(expr.args[i].args[2], MOI.VariableIndex)
            exprd.args[i] = JuMP.VariableRef(m, expr.args[i].args[2])
            if op.var_type[dict[exprd.args[i]]] == :Bin
                push!(bin_vars, expr.args[i])
                push!(bin_obj_terms, expr.args[i+1])   #assuming bin variable appears first
            end
        elseif expr.args[i].head == :call
            _expr_deref!(expr.args[i], exprd.args[i], m, op, dict, bin_vars, bin_obj_terms)
        else
            error("expr_dereferencing :: Unexpected term in expression tree.")
        end
    end
    return bin_vars, bin_obj_terms
end

function get_binvar_and_objterms(obj_expr, m, op)
    list = JuMP.all_variables(m)
    dict = Dict{}(list[i] => i for i in 1:op.num_var)
    bin_vars=[]
    bin_obj_terms=[]
    c_expr = copy(obj_expr)
    d_expr = copy(obj_expr)
    return _expr_deref!(c_expr, d_expr, m, op, dict, bin_vars, bin_obj_terms)
end




function construct_ref_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    oad.ref_mip_model = Model()
    lb = oad.ref_l_var
    ub = oad.ref_u_var
    vartype = oad.ref_var_type
    # all continuous we solve relaxation first
    @variable(oad.ref_mip_model, lb[i] <= x[i=1:oad.ref_num_var] <= ub[i])
    for i in 1:length(oad.ref_var_type)
        oad.ref_var_type[i] == :Int && JuMP.set_integer(x[i])
        oad.ref_var_type[i] == :Bin && JuMP.set_binary(x[i])
    end
    # register function into the mip model
    register_functions!(oad.ref_mip_model, op.options.registered_functions)
    # attach linear constraints to the mip model
    backend = JuMP.backend(oad.ref_mip_model);
    llc = model.linear_le_constraints
    lgc = model.linear_ge_constraints
    lec = model.linear_eq_constraints
    #add linear constraints to the mip model
    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end

    #set objective
    func = JuMP.objective_function(oad.ref_nlp_model)
    func = JuMP.moi_function(func)
    MOI.set(oad.ref_mip_model, MOI.ObjectiveFunction{typeof(func)}(), func)
    MOI.set(oad.ref_mip_model, MOI.ObjectiveSense(), model.sense)
    oad.ref_mip_x = x
end

function add_ref_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    d=JuMP.NLPEvaluator(oad.ref_nlp_model)
    features = MOI.features_available(d)
    MOI.initialize(d, features)

    jac_IJ = MOI.jacobian_structure(d)
    nlp_solution=ones(oad.ref_num_var)

    g_val = zeros(oad.ref_num_nl_constr)
    g_jac = zeros(length(jac_IJ))
    MOI.eval_constraint(d, g_val, nlp_solution)
    MOI.eval_constraint_jacobian(d, g_jac, nlp_solution)

    #constraints cut
    varidx_new = [zeros(Int, 0) for i in 1:oad.ref_num_nl_constr]
    coef_new = [zeros(0) for i in 1:oad.ref_num_nl_constr]

    for k in 1:length(jac_IJ)
        row = jac_IJ[k][1]
        push!(varidx_new[row], jac_IJ[k][2])
        push!(coef_new[row], g_jac[k])
    end

    for i in 1:oad.ref_num_nl_constr
        new_rhs = -g_val[i]
        for j in 1:length(varidx_new[i])
            new_rhs += coef_new[i][j] * nlp_solution[Int(varidx_new[i][j])]
        end
        if oad.ref_nlp_constr_type[i] == :(<=)
            @constraint(oad.ref_mip_model, dot(coef_new[i], oad.ref_mip_x[varidx_new[i]]) <= new_rhs)
        else
            @constraint(oad.ref_mip_model, dot(coef_new[i], oad.ref_mip_x[varidx_new[i]]) >= new_rhs)
        end
    end
    #no need to derive obj cut as the ref_nlp obj is linear.
end
