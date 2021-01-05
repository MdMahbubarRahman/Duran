"""
necessary functions for outer approximation framework
"""

"""
function init_oad(op::OriginalProblem, oad::OAdata)
    initiates OAdata structure using information of the OriginalProblem.
"""
function init_oad(op::OriginalProblem, oad::OAdata)
    oad.num_l_constr              = op.num_l_constr
    oad.ref_l_var                 = Float64[]
    oad.ref_u_var                 = Float64[]
    oad.ref_num_var               = 0
    oad.ref_num_nl_constr         = 0

    oad.obj_sense                 = op.obj_sense

    oad.oa_status                 = :Unknown
    oad.oa_started                = false
    oad.incumbent                 = Float64[]
    oad.new_incumbent             = false
    oad.total_time                = 0
    oad.obj_val                   = Inf
    oad.obj_bound                 = -Inf
    oad.obj_gap                   = Inf
    oad.oa_iter                   = 0

    oad.milp_sol_available        = false
    oad.ref_mip_solution          = Float64[]
    oad.mip_infeasible            = false
    oad.ref_nlp_solution          = Float64[]
    oad.nlp_infeasible            = false
    oad.ref_feasibility_solution  = Float64[]

    oa_dat.int_idx                = filter(i -> (op.var_type[i] in (:Int, :Bin)), 1:op.num_var)
    oad.prev_ref_mip_solution     = Float64[]

    oad.ref_linear_le_constraints = []
    oad.ref_linear_ge_constraints = []
    oad.ref_linear_eq_constraints = []
end

"""
function construct_linear_model(optimizer::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    this function generates a new mip_model with the linear constraints and linear objective function
    of the original model OriginalProblem.
"""
function construct_linear_model(optimizer::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    oad.mip_model = Model()
    lb = op.l_var
    ub = op.u_var
    push!(lb, -Inf)
    push!(ub, Inf)
    #vartype = op.var_type
    #push!(vartype, :Cont)
    # all continuous we solve relaxation first
    @variable(oad.mip_model, lb[i] <= x[i=1:(op.num_var+1)] <= ub[i])
    for i in 1:length(op.var_type)
        op.var_type[i] == :Int && JuMP.set_integer(x[i])
        op.var_type[i] == :Bin && JuMP.set_binary(x[i])
    end
    # register function into the mip model
    register_functions!(oad.mip_model, op.options.registered_functions)
    # attach linear constraints to the mip model
    backend1 = JuMP.backend(oad.mip_model);
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
        op.obj_sense == :Min && @objective(oad.mip_model, Min, x[op.num_var+1])
        op.obj_sense == :Max && @objective(oad.mip_model, Max, x[op.num_var+1])
    else
        @info "the objective function is linear or quadratic.\n"
        MOI.set(oad.mip_model, MOI.ObjectiveFunction{typeof(optimizer.objective)}(), optimizer.objective)
        MOI.set(oad.mip_model, MOI.ObjectiveSense(), optimizer.sense)
    end
    oad.mip_x = x
end

"""
function add_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    this function generates oa cuts for nonlinear constraints and objective function and adds the cuts
    to the mip_model which is a milp model.
"""
function add_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
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
            @constraint(oad.mip_model, dot(coef_new[i], oad.mip_x[varidx_new[i]]) <= new_rhs)
        else
            @constraint(oad.mip_model, dot(coef_new[i], oad.mip_x[varidx_new[i]]) >= new_rhs)
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
            @constraint(oad.mip_model, dot(f_nab, oad.mip_x[varidx]) >= new_rhs)
        else
            @constraint(oad.mip_model, dot(f_nab, oad.mip_x[varidx]) <= new_rhs)
        end
    end
end


"""
function construct_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    this function generates a nlp_model from the original model provided as OriginalProblem.
"""
function construct_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    oad.nlp_model = Model()
    lb = op.l_var
    ub = op.u_var
    @variable(oad.nlp_model, lb[i] <= x[i=1:op.num_var] <= ub[i])
    register_functions!(oad.nlp_model, op.options.registered_functions)

    backend2 = JuMP.backend(oad.nlp_model);
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
        constr_expr = expr_dereferencing(constr_expr, oad.nlp_model)
        try
            JuMP.add_NL_constraint(oad.nlp_model, constr_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    end

    if model.nlp_data.has_objective
        obj_expr = MOI.objective_expr(model.nlp_data.evaluator)
        obj_expr = expr_dereferencing(obj_expr, oad.nlp_model)
        try
            JuMP.set_NL_objective(oad.nlp_model, model.sense, obj_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    elseif model.objective !== nothing
        MOI.set(oad.nlp_model, MOI.ObjectiveFunction{typeof(model.objective)}(), model.objective)
        MOI.set(oad.nlp_model, MOI.ObjectiveSense(), model.sense)
    end

    oad.nlp_x = x

    return nothing
end


"""
function solve_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    this function solves mip_model with mixed interger solver.
"""

function solve_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    JuMP.set_optimizer(oad.mip_model, model.options.mip_solver)
    JuMP.optimize!(oad.mip_model)

    if termination_status(oad.mip_model) == :OPTIMAL || termination_status(oad.mip_model) == :LOCALLY_SOLVED
        @info "There is a feasible solution in the mip_model.\n"
        value = JuMP.value.(oad.mip_x)
        return value
    else
        @error "There is no feasible solution from mip_model. \n"
    end
    #status = JuMP.primal_status(oad.solve_mip_model)
end

function solve_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    JuMP.set_optimizer(oad.nlp_model, model.options.nlp_solver)
    JuMP.optimize!(oad.nlp_model)

    if termination_status(oad.nlp_model) == :OPTIMAL || termination_status(oad.nlp_model) == :LOCALLY_SOLVED
        @info "There is a feasible solution in the mip_model.\n"
        value = JuMP.value.(oad.nlp_x)
        return value
    else
        @error "There is no feasible solution from mip_model. \n"
    end
    #status = JuMP.primal_status(oad.solve_mip_model)
end



"""
function construct_ref_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    this function generates a ref_nlp_model from the original model.
"""
function construct_ref_nlp_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    oad.ref_nlp_model = Model()
    lb = op.l_var
    ub = op.u_var
    vartype = op.var_type

    for i in 1:(op.nbinvars)
        push!(lb, -Inf)
        push!(ub, Inf)
        push!(vartype, :Cont)
    end

    oad.ref_l_var = lb
    oad.ref_u_var = ub
    oad.ref_var_type = vartype

    @variable(oad.ref_nlp_model, lb[i] <= x[i=1:(op.num_var+op.nbinvars)] <= ub[i])
    register_functions!(oad.ref_nlp_model, op.options.registered_functions)

    oad.ref_num_var = length(x)

    backend = JuMP.backend(oad.ref_nlp_model);
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
        constr_expr = expr_dereferencing(constr_expr, oad.ref_nlp_model)
        try
            JuMP.add_NL_constraint(oad.ref_nlp_model, constr_expr)
        catch
            error("Have you registered a function? Then please register the function.\n")
        end
    end
    num_nl_constr = op.num_nl_constr
    #additional constraints from the nlobj
    if model.nlp_data.has_objective
        if op.obj_sense == :Max
            @info "The objective sense is Max. \n"
            @objective(oad.ref_nlp_model, Max, sum(x[i] for i in (op.num_var+1):(op.num_var+op.nbinvars)))
        elseif op.obj_sense == :Min
            @info "The objective sense is Min. \n"
            @objective(oad.ref_nlp_model, Min, sum(x[i] for i in (op.num_var+1):(op.num_var+op.nbinvars)))
        end

        obj_expr = MOI.objective_expr(model.nlp_data.evaluator)
        moi_bin_vars, moi_obj_terms = get_binvar_and_objterms(obj_expr, oad.ref_nlp_model, op)

        siz = length(moi_bin_vars)
        jump_bin_vars = []
        jump_obj_terms = []
        for i in 1:siz
            jump_bin = JuMP.VariableRef(oad.ref_nlp_model, moi_bin_vars[i].args[2])
            push!(jump_bin_vars, jump_bin)
            jump_exp = expr_dereferencing(moi_obj_terms[i], oad.ref_nlp_model)
            push!(jump_obj_terms, jump_exp)
        end
        num_nl_constr = num_nl_constr+siz
        ref_nlpcontr_type = op.nlp_constr_type
        for i in 1:siz
            L=-100000
            U=+100000
            JuMP.@constraint(oad.ref_nlp_model, x[i+op.num_var]-jump_bin_vars[i]*L >= 0)
            exp = :($(x[i+op.num_var])-($(jump_bin_vars[i])-$(1))*$(U)-$(jump_obj_terms[i]) >= 0)
            JuMP.add_NL_constraint(oad.ref_nlp_model, exp)
            push!(ref_nlpcontr_type, :>=)
        end
        oad.ref_nlp_constr_type = ref_nlpcontr_type

    elseif model.objective !== nothing
        MOI.set(oad.ref_nlp_model, MOI.ObjectiveFunction{typeof(model.objective)}(), model.objective)
        MOI.set(oad.ref_nlp_model, MOI.ObjectiveSense(), model.sense)
    end
    oad.ref_num_nl_constr = num_nl_constr
    oad.ref_nlp_x = x

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


"""
function construct_ref_mip_model(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    This function genearates reformulated mip model.
"""
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

"""
function add_ref_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    This function facilates to add linear approximation cut to the reformulated mip model.
"""
function add_ref_oa_cut(model::MOI.AbstractOptimizer, op::OriginalProblem, oad::OAdata)
    d=JuMP.NLPEvaluator(oad.ref_nlp_model)
    features = MOI.features_available(d)
    MOI.initialize(d, features)

    jac_IJ = MOI.jacobian_structure(d)

    !(oad.nlp_infeasible) && (nlp_solution = oad.ref_nlp_solution)
    oad.nlp_infeasible && (nlp_solution = oad.ref_feasibility_solution[1:oad.ref_num_var])

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


"""
function construct_ref_feasibility_model(oad::OAdata)
    This function generates a feasibility model in case the nlp problem is infeasible.
"""
function construct_ref_feasibility_model(oad::OAdata)
    # in case of infeasible nlp problem, we need to solve a feasibility problem
    oad.ref_feasibility_model = Model()
    @variable(oad.ref_feasibility_model,  x[1:(oad.ref_num_var+oad.ref_num_nl_constr)] >= 0.0)
    #linear constraints are automatically satisfied
    #we only need to focus on nonlinear constraints
    d = JuMP.NLPEvaluator(oad.ref_nlp_model)
    features = MOI.features_available(d)
    MOI.initialize(d, features)
    #define constraints
    for i in 1:jp.ref_num_nl_constr
        constr_expr = MOI.constraint_expr(d, i)
        constr_expr = expr_dereferencing(constr_expr[2], oad.ref_feasibility_model)
        if constr_expr.args[1] == :(<=)
            con_ex = :($(constr_expr) <= $(x[i+oad.ref_num_var]))
            JuMP.add_NL_constraint(oad.ref_feasibility_model, con_ex)
        elseif constr_expr.args[1] == :(>=)
            con_ex = :(-$(constr_expr) <= -$(x[i+oad.ref_num_var]))
            JuMP.add_NL_constraint(oad.ref_feasibility_model, con_ex)
        else
            @error "\n Nonlinear constarint should not be equality constarint. \n"
        end
    end
    #define obj function
    oad.obj_sense == :Min && JuMP.@objective(oad.ref_feasibility_model, Min, sum(x[i] for i in (oad.ref_num_var+1):(oad.ref_num_var+oad.ref_num_nl_constr)))
    oad.obj_sense == :Max && JuMP.@objective(oad.ref_feasibility_model, Max, sum(x[i] for i in (oad.ref_num_var+1):(oad.ref_num_var+oad.ref_num_nl_constr)))

    oad.ref_feasibility_x = x
end



"""
function reserve_ref_linear_constraints(oad::OAdata)
    This function saves linear constraints of the reformulated nlp model to the outer approximation data structure.
    This constraints are in MOI format and later will be used to generate reformulated mip model.
"""
function reserve_ref_linear_constraints(oad::OAdata)
    #list of different types of constraints
    constr_types = JuMP.list_of_constraint_types(oad.ref_nlp_model)
    #number of different constraint types
    siz = length(constr_types)
    #reserve constraints for each type
    for i in 1:siz
        num_constr = JuMP.num_constraints(oad.ref_nlp_model, constr_types[i][1], constr_types[i][2])
        constrs    = JuMP.all_constraints(oad.ref_nlp_model, constr_types[i][1], constr_types[i][2])
        for j in 1:num_constr
            con_obj = JuMP.constraint_object(constrs[j])
            func    = JuMP.moi_function(con_obj.func)
            set     = con_obj.set
            (constr_types[i][2] == MOI.EqualTo{Float64}) && push!(oad.ref_linear_eq_constraints, (func, set))
            (constr_types[i][2] == MOI.LessThan{Float64}) && push!(oad.ref_linear_le_constraints, (func, set))
            (constr_types[i][2] == MOI.GreaterThan{Float64}) && push!(oad.ref_linear_ge_constraints, (func, set))
        end
    end
end
