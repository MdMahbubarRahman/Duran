"""
This file have the basic outer approximation algoirthm
"""

"""
function solve_ref_nlp_model(model::MOI.AbstractOptimizer, oad::OAdata)
    This function solves reformulated nlp problem.
"""
function solve_ref_nlp_model(model::MOI.AbstractOptimizer, oad::OAdata)
    if oad.milp_sol_available
        vars = JuMP.all_variables(oad.ref_nlp_model)
        for i in 1:oad.ref_num_var
            if oad.ref_var_type[i] == :Int || oad.ref_var_type[i] ==:Bin
                JuMP.delete_lower_bound(vars[i])
                JuMP.delete_upper_bound(vars[i])
                JuMP.set_lower_bound(vars[i], oad.ref_mip_solution[i])
                JuMP.set_upper_bound(vars[i], oad.ref_mip_solution[i])
            end
        end
    end

    JuMP.set_optimizer(oad.ref_nlp_model, model.options.nlp_solver)
    JuMP.optimize!(oad.ref_nlp_model);
    #the problem is here
    if (JuMP.termination_status(oad.ref_nlp_model) == MOI.OPTIMAL) || (JuMP.termination_status(oad.ref_nlp_model) == MOI.LOCALLY_SOLVED)
        @info "\n The reformulated nlp problem has a feasible solution. \n"
        oad.milp_sol_available && (oad.ref_nlp_solution  = JuMP.value.(vars))
        !oad.milp_sol_available && (oad.ref_nlp_solution  = JuMP.value.(oad.ref_nlp_x))
        nlp_objval            = JuMP.objective_value(oad.ref_nlp_model)
        #update objval
        if (nlp_objval < oad.obj_val) && oad.milp_sol_available            #considering the problem is a min sense problem
            oad.obj_val       = nlp_objval
            oad.incumbent     = oad.ref_nlp_solution
            oad.new_incumbent = true
        end
        #update objgap
        if isfinite(oad.obj_val) && isfinite(oad.obj_bound)
            oad.obj_gap = (oad.obj_val - oad.obj_bound)/(abs(oad.obj_val) + 1e-5)
        end
    elseif (JuMP.termination_status(oad.ref_nlp_model) == MOI.INFEASIBLE) || (JuMP.termination_status(oad.ref_nlp_model) == MOI.DUAL_INFEASIBLE) || (JuMP.termination_status(oad.ref_nlp_model) == MOI.LOCALLY_INFEASIBLE) || (JuMP.termination_status(oad.ref_nlp_model) == MOI.INFEASIBLE_OR_UNBOUNDED)
        @info "\n The reformulated nlp problem is infeasible. \n"
        oad.nlp_infeasible = true
    end
end


"""
function solve_ref_mip_model(model::MOI.AbstractOptimizer, oad::OAdata)
    This function solves reformulatd mip problem.
"""
function solve_ref_mip_model(model::MOI.AbstractOptimizer, oad::OAdata)
    JuMP.set_optimizer(oad.ref_mip_model, model.options.mip_solver)
    JuMP.optimize!(oad.ref_mip_model);

    if (JuMP.termination_status(oad.ref_mip_model) ==  MOI.OPTIMAL) || (JuMP.termination_status(oad.ref_mip_model) ==  MOI.LOCALLY_SOLVED)
        @info "\n The reformulated mip problem has a feasible solution. \n"
        (oad.milp_sol_available == true) && (oad.prev_ref_mip_solution = oad.ref_mip_solution)
        (oad.milp_sol_available == false) && (oad.milp_sol_available = true)
        oad.ref_mip_solution = JuMP.value.(oad.ref_mip_x)
        # update bestbound
        mipobjbound = JuMP.objective_bound(oad.ref_mip_model)
        if isfinite(mipobjbound) && (mipobjbound > oad.obj_bound)
            oad.obj_bound = mipobjbound
        end
    else (JuMP.termination_status(oad.ref_mip_model) == MOI.INFEASIBLE) || (JuMP.termination_status(oad.ref_mip_model) == MOI.DUAL_INFEASIBLE) || (JuMP.termination_status(oad.ref_mip_model) == MOI.LOCALLY_INFEASIBLE) || (JuMP.termination_status(oad.ref_mip_model) == MOI.INFEASIBLE_OR_UNBOUNDED)
        @info "\n The reformulated mip problem is infeasible. \n"
        oad.mip_infeasible = true
    end
end


"""
function solve_ref_feasibility_model(model::MOI.AbstractOptimizer, oa_data::OAdata)
    This function solves reformulated feasibility problem.
"""
function solve_ref_feasibility_model(model::MOI.AbstractOptimizer, oad::OAdata)
    #fix int variables
    if oad.milp_sol_available
        for i in 1:oad.ref_num_var
            if (oad.ref_var_type[i] == :Int) || (oad.ref_var_type[i] ==:Bin)
                JuMP.set_lower_bound(oad.ref_feasibility_x[i], oad.ref_mip_solution[i])
                JuMP.set_upper_bound(oad.ref_feasibility_x[i], oad.ref_mip_solution[i])
            end
        end
    end
    #set optimizer and solve
    JuMP.set_optimizer(oad.ref_feasibility_model, model.options.nlp_solver)
    JuMP.optimize!(oad.ref_feasibility_model);

    if (JuMP.termination_status(oad.ref_feasibility_model) == MOI.OPTIMAL) || (JuMP.termination_status(oad.ref_feasibility_model) == MOI.LOCALLY_SOLVED)
        @info "\n The reformulated feasibility problem has a feasible solution. \n"
        oad.ref_feasibility_solution = JuMP.value.(oad.ref_feasibility_x)
    else
        @error "\n The reformulated feasibility problem is infeasible. \n"
    end
end


"""
function check_termination_condition(oad::OAdata)
    this function checks termination status of the oa algorithm.
"""
function check_termination_condition(oad::OAdata)
    # finish if optimal or cycling integer solutions
    if oad.obj_gap <= 0.00001
        oad.oa_status = :Optimal
    elseif round.(oad.prev_ref_mip_solution[oad.int_idx]) == round.(oad.ref_mip_solution[oad.int_idx])
        @warn "mixed-integer cycling detected, terminating outer approximation algorithm"
        if isfinite(oad.obj_gap)
            oad.oa_status = :Suboptimal
        else
            oad.oa_status = :FailedOA
        end
    elseif oad.mip_infeasible && isfinite(oad.obj_val)
        oad.oa_status = :Optimal
    end
end




"""
function oa_algorithm(model::MOI.AbstractOptimizer, op::HybridProblem)
    This function performs outer approximation algorithm.
"""
function oa_algorithm(model::MOI.AbstractOptimizer, op::OriginalProblem)
    #create linear approximation (milp) model from the original problem
    oad = OAdata()
    init_oa_data(op, oad)
    construct_linear_model(model, op, oad)
    add_oa_cut(model, op, oad)
    #create nlp model from the original problem
    construct_nlp_model(model, op, oad)
    #create reformulated nlp model from the original problem
    construct_ref_nlp_model(model, op, oad)
    reserve_ref_linear_constraints(oad)
    #create linear approximation (milp) model from the reformulated nlp model
    construct_ref_mip_model(model, op, oad)
    #create reformulated feasibility problem
    construct_ref_feasibility_model(oad::OAdata)
    #start of outer approximation algoirthm
    start_time = time()
    oad.oa_started = true
    duration = 10000 #this could be provided as parameter
    while (time() - start_time) < duration
        oad.oa_iter += 1
        #solve reformulated nlp problem
        solve_ref_nlp_model(model, oad);
        oad.nlp_infeasible && solve_ref_feasibility_model(model, oad)
        #update milp master problem
        add_ref_oa_cut(model, oad)
        #solve milp problem
        solve_ref_mip_model(model, oad);
        #check termination conditions
        check_termination_condition(oad)
        if (oad.oa_status == :Optimal) || (oad.oa_status == :Suboptimal) || (oad.oa_status == :FailedOA)
            break
        end
    end
    end_time = time()
    oad.total_time = (end_time-start_time)
    print("\n The outer approximation algorithm has been terminated. \n")
    print("\n The solution status :", oad.oa_status, "\n")
    print("\n The number of oa iterations : ", oad.oa_iter, "\n")
    print("\n The solution time : ", oad.total_time, " secs \n")
    print("\n The objective value :", oad.obj_val, "\n")
    print("\n The objective gap :", oad.obj_gap, "\n")
    print("\n The solution :", oad.incumbent, "\n")
    print("\n The mip solution :", oad.ref_mip_solution, "\n")
end
