"""
this function solves Dantzig-Wolfe ralaxed main model.
"""
function solve_dw_main_model(cgd::CGdata, optimizer::MOI.AbstractOptimizer)
    λ_counter       = cgd.λ_counter
    μ_counter       = cgd.μ_counter
    extr_ptn_sol    = cgd.extr_ptn_sol[λ_counter][1]
    extr_dir_sol    = cgd.extr_dir_sol[μ_counter][1]
    disc2var_idx    = cgd.pricing_sub_disc2var_idx
    cons1           = cgd.dw_main_constr_ref[1]
    cons2           = cgd.dw_main_constr_ref[2]
    cons3           = cgd.dw_main_constr_ref[3]
    nitr            = cgd.num_oa_itr
    nbin            = length(cgd.jump_bin_var)
    constr_varidx   = cgd.dw_main_constr_varidx
    constr_coef     = cgd.dw_main_constr_coef
    #update coefficient of constr set "z[1] >= sum(u_p[1,i]*L[1]*λ[i] for i in Κ) + sum(u_d[1,j]*L[1]*μ[j] for j in Τ)"
    if cgd.psm_feasible == true
        for i in 1:nbin
            val = cgd.obj_terms_lower_bound[i]*extr_ptr_sol[disc2var_idx[i]]
            JuMP.set_normalized_coefficient(cons1[i], λ[λ_counter], val)
        end
    elseif cgd.psm_extr_dir_feasible == true
        for i in 1:nbin
            val = cgd.obj_terms_lower_bound[i]*extr_dir_sol[disc2var_idx[i]]
            JuMP.set_normalized_coefficient(cons1[i], μ[μ_counter], val)
        end
    end
    #update coefficient of constr set "z[1]+∇f1(̄x)*̄x+U[1] >= sum((u_p[1,i]*U[1]+∇f1(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[1,j]*U[1]+∇f1(̄x)*v_d[j])*μ[j] for j in Τ)"
    number = 0
    if cgd.psm_feasible == true
        for i in 1:nitr
            for j in 1:nbin
                number += 1
                val = dot(constr_coef[i][j],extr_ptr_sol[constr_varidx[i][j]])
                JuMP.set_normalized_coefficient(cons2[number], λ[λ_counter], val)
            end
        end
    elseif cgd.psm_extr_dir_feasible == true
        for i in 1:nitr
            for j in 1:nbin
                number += 1
                val = dot(constr_coef[i][j],extr_dir_sol[constr_varidx[i][j]])
                JuMP.set_normalized_coefficient(cons2[idx], μ[μ_counter], val)
            end
        end
    end
    #update for the third set of constraints
    if cgd.psm_feasible == true
        JuMP.set_normalized_coefficient(cons3[1], λ[λ_counter], 1.0)
    end
    #solve the dw_main_model
    JuMP.set_optimizer(cgd.dw_main_model, optimizer.options.lp_solver)
    JuMP.optimize!(cgd.dw_main_model)
    if (JuMP.termination_status(cgd.dw_main_model) ==  MOI.OPTIMAL) || (JuMP.termination_status(cgd.dw_main_model) ==  MOI.LOCALLY_SOLVED)
        obj_val              = JuMP.objective_value(cgd.dw_main_model)
        dw_main_x            = JuMP.all_variables(cgd.dw_main_model)
        cgd.dw_main_solution = JuMP.value.(dw_main_x)
        dw_sol_obj           = dwmSolutionObj(cgd.dw_main_solution, obj_val)
        #λ_val   = JuMP.value.(cgd.dw_main_x[1])
        #μ_val   = JuMP.value.(cgd.dw_main_x[2])
        #t_val   = JuMP.value.(cgd.dw_main_x[3])
    else
        @warn "The dw_main_model is infeasible.\n"
    end
    #get dual solutions
    dual_sol_exists = JuMP.has_duals(cgd.dw_main_model)
    dual_solution = []
    if dual_sol_exists == true
        w1 = JuMP.dual.(cons1)
        w2 = JuMP.dual.(cons2)
        w3 = JuMP.dual.(cons3)
        w = [w1, w2, w3]
        for d in w
            for d_val in d
                push!(dual_solution, d_val)
            end
        end
    end
    cgd.dw_main_dual_solution = dual_solution
    #return dual solution
    return nothing
end


"""
This function solves the dw pricing sub model.
"""
function solve_pricing_sub_model(cgd::CGdata, optimizer::MOI.AbstractOptimizer)
    dwmd_sol     = cgd.dw_main_dual_solution
    model        = cgd.pricing_sub_model
    x            = JuMP.all_variables(model)
    obj_terms_lb = cgd.obj_terms_lower_bound
    varid        = cgd.dw_main_constr_varidx
    coef         = cgd.dw_main_constr_coef
    bin_vars     = cgd.jump_bin_var
    num_constr   = cgd.dw_main_num_constr
    #set obj func
    siz = length(bin_vars)
    (cgd.pricing_sub_obj_sense == MOI.MAX_SENSE) && JuMP.@objective(model, Max, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)+dwmd_sol[num_constr])
    (cgd.pricing_sub_obj_sense == MOI.MIN_SENSE) && JuMP.@objective(model, Min, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)+dwmd_sol[num_constr])
    #set lp solver
    JuMP.set_optimizer(model, optimizer.options.lp_solver)
    JuMP.optimize!(model)
    if (JuMP.termination_status(model) ==  MOI.OPTIMAL) || (JuMP.termination_status(model) ==  MOI.LOCALLY_SOLVED)
        cgd.pricing_sub_solution   = JuMP.value.(x)
        objective_value            = JuMP.objective_value(model)
        psm_sol_obj                = psmSolutionObj(cgd.pricing_sub_solution, objective_value)
        cgd.psm_feasible           = true
        cgd.psm_extr_dir_feasible  = false
        cgd.λ_counter             += 1
        push!(cgd.extr_ptn_sol, psm_sol_obj)
    else
        @info "\n The pricing sub problem is infeasible. \n"
    end
end


"""
This function solves the pricing sub problem for extreme directions
"""
function solve_pricing_sub_extm_dir_model(cgd::CGdata, optimizer::MOI.AbstractOptimizer)
    dwmd_sol     = cgd.dw_main_dual_solution
    obj_terms_lb = cgd.obj_terms_lower_bound
    varid        = cgd.dw_main_constr_varidx
    coef         = cgd.dw_main_constr_coef
    num_constr   = cgd.dw_main_num_constr
    siz          = length(cgd.pricing_sub_disc2var_idx)
    model        = cgd.pricing_sub_extm_dir_model
    x            = cgd.pricing_sub_extm_dir_x[1]
    #define obj function
    (cgd.pricing_sub_obj_sense == MOI.MAX_SENSE) && JuMP.@objective(model, Max, sum(x[cgd.pricing_sub_disc2var_idx[i]]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr))
    (cgd.pricing_sub_obj_sense == MOI.MIN_SENSE) && JuMP.@objective(model, Min, sum(x[cgd.pricing_sub_disc2var_idx[i]]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr))
    #set lp solver
    JuMP.set_optimizer(model, optimizer.options.lp_solver)
    JuMP.optimize!(model)
    if (JuMP.termination_status(model) ==  MOI.OPTIMAL) || (JuMP.termination_status(model) ==  MOI.LOCALLY_SOLVED)
        cgd.pricing_sub_extr_dir_solution   = JuMP.value.(x)
        objective_value                     = JuMP.objective_value(model)
        psm_extm_dir_sol_obj                = psmExtrmDirSolutionObj(pricing_sub_extr_dir_solution, objective_value)
        cgd.psm_feasible                    = false
        cgd.psm_extr_dir_feasible           = true
        cgd.μ_counter                      += 1
        push!(cgd.extr_dir_sol, psm_extm_dir_sol_obj)
    else
        @info "\n The pricing sub problem for extreme direction is infeasible. \n"
    end
end



"""
This function checks whether the optimality of the cg algorithm has been achieved.
"""
function check_cg_optimality(cgd::CGdata)
    #when and how to get out of cg algorihm is need to be added here
    #terminating condition
    if obj2>-0.0
        print("Optimal Solution achieved\n")
        break
    else
        continue
    end
end


"""
The column generation algorithm is stated here
"""
function cg_algorithm(model::JuMP.Model, oad::OAdata, op::HybridProblem, optimizer::MOI.AbstractOptimizer)
    #initiate cg data storage
    cgd = CGdata()
    #populate cgd with oad data
    init_cg_data(cgd, oad)
    #decompose the model into main and sub problems
    decompose_lp_model(model, oad, op, optimizer, cgd)
    while(cg_not_optimize)
        #solve dw pricing sub problem
        solve_pricing_sub_model(cgd, optimizer)
        #solve dw main problem
        solve_dw_main_model(cgd, optimizer)
        #check optimality condition
        check_cg_optimality(cgd)
    end
    #return solution of the input model
    return cgd.best_solution
end
