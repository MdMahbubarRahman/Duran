"""
This file have the basic outer approximation algoirthm
"""

function solve_ref_nlp_model(model::MOI.AbstractOptimizer, oa_data::OAdata)
    if oa_data.milp_sol_available
        for i in 1:oa_data.ref_num_var
            if oa_data.ref_var_type[i] == :Int || oa_data.ref_var_type[i] ==:Bin
                JuMP.set_lower_bound(oa_data.ref_nlp_x[i], oa_data.ref_mip_solution[i])
                JuMP.set_upper_bound(oa_data.ref_nlp_x[i], oa_data.ref_mip_solution[i])
            end
        end
    end

    JuMP.set_optimizer(oa_data.ref_nlp_model, model.options.nlp_solver)
    JuMP.optimize!(oa_data.ref_nlp_model)

    if JuMP.termination_status(oa_data.ref_nlp_model) == :OPTIMAL || JuMP.termination_status(oa_data.ref_nlp_model) == :LOCALLY_SOLVED
        @info "\n The reformulated nlp problem has a feasible solution. \n"
        oa_data.ref_nlp_solution = JuMP.value.(oa_data.ref_nlp_x)
    else
        @info "\n The reformulated nlp problem is infeasible. \n"
        oa_data.nlp_infeasible = true
    end
end



function solve_ref_mip_model(model::MOI.AbstractOptimizer, oa_data::OAdata)
    JuMP.set_optimizer(oa_data.ref_mip_model, model.options.mip_solver)
    JuMP.optimize!(oa_data.ref_mip_model)

    if JuMP.termination_status(oa_data.ref_mip_model) == :OPTIMAL || JuMP.termination_status(oa_data.ref_mip_model) == :LOCALLY_SOLVED
        @info "\n The reformulated mip problem has a feasible solution. \n"
        oa_data.milp_sol_available = true
        oa_data.ref_mip_solution = JuMP.value.(oa_data.ref_mip_x)
    else
        @info "\n The reformulated mip problem is infeasible. \n"
        oa_data.mip_infeasible = true
    end
end



function solve_ref_feasibility_model(model::MOI.AbstractOptimizer, oa_data::OAdata)
    #fix int variables
    if oa_data.milp_sol_available
        for i in 1:oa_data.ref_num_var
            if (oa_data.ref_var_type[i] == :Int) || (oa_data.ref_var_type[i] ==:Bin)
                JuMP.set_lower_bound(oa_data.ref_feasibility_x[i], oa_data.ref_mip_solution[i])
                JuMP.set_upper_bound(oa_data.ref_feasibility_x[i], oa_data.ref_mip_solution[i])
            end
        end
    end
    #set optimizer and solve
    JuMP.set_optimizer(oa_data.ref_feasibility_model, model.options.nlp_solver)
    JuMP.optimize!(oa_data.ref_feasibility_model)

    if JuMP.termination_status(oa_data.ref_feasibility_model) == :OPTIMAL || JuMP.termination_status(oa_data.ref_feasibility_model) == :LOCALLY_SOLVED
        @info "\n The reformulated feasibility problem has a feasible solution. \n"
        oa_data.ref_feasibility_solution = JuMP.value.(oa_data.ref_feasibility_x)

    else
        @error "\n The reformulated feasibility problem is infeasible. \n"
    end
end




function oa_algorithm(model::MOI.AbstractOptimizer, op::OriginalProblem)
    #create linear approximation (milp) model from the original problem
    oa_data=OAdata()
    init_oa_data(op::OriginalProblem, oa_data)
    construct_linear_model(model, op, oa_data)
    add_oa_cut(model, op, oa_data)
    #create nlp model from the original problem
    construct_nlp_model(model, op, oa_data)
    #create reformulated nlp model from the original problem
    construct_ref_nlp_model(model, op, oa_data)
    #create linear approximation (milp) model from the reformulated nlp model
    construct_ref_mip_model(model, op, oa_data)

    oa_data.oa_started == false && oa_data.oa_started = true
    #oa_data.incumbent = solve_ref_nlp_model(model, oa_data)

    while oa_data.oa_started
        oa_data.oa_iter += 1
        #solve reformulated nlp problem
        solve_ref_nlp_model(model, oa_data)
        oa_data.nlp_infeasible && solve_ref_feasibility_model(model, oa_data)
        #update milp master problem
        add_ref_oa_cut(model, oa_data)
        #solve milp problem
        solve_ref_mip_model(model, oa_data)
        oa_data.mip_infeasible && oa_data.oa_started = false
    end
end
