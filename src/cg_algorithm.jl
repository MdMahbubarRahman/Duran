"""
this function solves Dantzig-Wolfe ralaxed main model.
"""
#solve_dw_main_model(cgd, mip_d, optimizer)
function solve_dw_main_model(cgd::CGdata, mip_d::MIPModelInfo, optimizer::MOI.AbstractOptimizer)
    λ_counter       = cgd.λ_counter
    μ_counter       = cgd.μ_counter
    (λ_counter > 0) && (extr_ptn_sol = cgd.extr_ptn_sol[λ_counter].solution)
    (μ_counter > 0) && (extr_dir_sol = cgd.extr_dir_sol[μ_counter].solution)
    cons1           = cgd.dw_main.constr_ref[1]
    cons2           = cgd.dw_main.constr_ref[2]
    cons3           = cgd.dw_main.constr_ref[3]
    cons4           = cgd.dw_main.constr_ref[4]
    λ               = cgd.dw_main.λ
    μ               = cgd.dw_main.μ
    nitr            = mip_d.num_oa_iter
    nbin            = length(cgd.dw_main.t)
    constr_varidx   = mip_d.dw_main_constr_varidx
    constr_coef     = mip_d.dw_main_constr_coef
    (λ_counter > 0) && print("\n", extr_ptn_sol, "\n")
    (μ_counter > 0) && print("\n", extr_dir_sol, "\n")
    #update coefficient of constr set "z[1] >= sum(u_p[1,i]*L[1]*λ[i] for i in Κ) + sum(u_d[1,j]*L[1]*μ[j] for j in Τ)"
    if cgd.psm_feasible == true
        for i in 1:nbin
            #val = cgd.obj_terms_lower_bound[i]*extr_ptn_sol[disc2var_idx[i]]
            val = -1*cgd.obj_terms_lower_bound[i]*extr_ptn_sol[i]
            JuMP.set_normalized_coefficient(cons1[i], λ[λ_counter], val)
        end
    elseif cgd.psm_extr_dir_feasible == true
        for i in 1:nbin
            #val = cgd.obj_terms_lower_bound[i]*extr_dir_sol[disc2var_idx[i]]
            val = -1*cgd.obj_terms_lower_bound[i]*extr_dir_sol[i]
            JuMP.set_normalized_coefficient(cons1[i], μ[μ_counter], val)
        end
    end
    #update coefficient of constr set "z[1]+∇f1(̄x)*̄x+U[1] >= sum((u_p[1,i]*U[1]+∇f1(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[1,j]*U[1]+∇f1(̄x)*v_d[j])*μ[j] for j in Τ)"
    number = 0
    if cgd.psm_feasible == true
        for i in 1:nitr
            for j in 1:nbin
                number += 1
                val = dot(constr_coef[i][j],extr_ptn_sol[constr_varidx[i][j]])
                JuMP.set_normalized_coefficient(cons2[number], λ[λ_counter], val)
            end
        end
    elseif cgd.psm_extr_dir_feasible == true
        for i in 1:nitr
            for j in 1:nbin
                number += 1
                val = dot(constr_coef[i][j],extr_dir_sol[constr_varidx[i][j]])
                JuMP.set_normalized_coefficient(cons2[number], μ[μ_counter], val)
            end
        end
    end
    #update for the third set of constraints
    if cgd.psm_feasible == true
        JuMP.set_normalized_coefficient(cons3[1], λ[λ_counter], 1.0)
    end
    #update for cut constraints
    len = length(cons4)
    if cgd.psm_feasible == true
        for i in 1:len
            val = extr_ptn_sol[cgd.cut_var_indices[i]]
            JuMP.set_normalized_coefficient(cons4[i], λ[λ_counter], val)
        end
    elseif cgd.psm_extr_dir_feasible == true
        for i in 1:len
            val = extr_dir_sol[cgd.cut_var_indices[i]]
            JuMP.set_normalized_coefficient(cons4[i], μ[μ_counter], val)
        end
    end
    #solve the dw_main_model
    JuMP.set_optimizer(cgd.dw_main.model, optimizer.options.lp_solver)
    JuMP.optimize!(cgd.dw_main.model)
    if (JuMP.termination_status(cgd.dw_main.model) ==  MOI.OPTIMAL) || (JuMP.termination_status(cgd.dw_main.model) ==  MOI.LOCALLY_SOLVED)
        obj_val              = JuMP.objective_value(cgd.dw_main.model)
        dw_main_vars         = JuMP.all_variables(cgd.dw_main.model)
        dw_main_solution     = JuMP.value.(dw_main_vars)
        dw_sol_obj           = dwmSolutionObj(dw_main_solution, obj_val)
        push!(cgd.dw_main_sol, dw_sol_obj)
        cgd.λ_val = JuMP.value.(cgd.dw_main.λ)
        cgd.μ_val = JuMP.value.(cgd.dw_main.μ)
    else
        @warn "The dw_main_model is infeasible.\n"
        #TO DO: Need to incorporate methods to handle infeasibility
    end
    #get dual solutions
    dual_sol_exists = JuMP.has_duals(cgd.dw_main.model)
    dual_solution = []
    if dual_sol_exists == true
        w1 = JuMP.dual.(cons1)
        w2 = JuMP.dual.(cons2)
        w3 = JuMP.dual.(cons3)
        if len > 0
            w4 = JuMP.dual.(cons4)
            w = [w1, w2, w3, w4]
        else
            w = [w1, w2, w3]
        end
        for d in w
            for d_val in d
                push!(dual_solution, d_val)
            end
        end
    end
    cgd.dw_main_dual_solution = dual_solution
    return nothing
end


"""
This function solves the dw pricing sub model.
"""
function solve_pricing_sub_model(cgd::CGdata, mip_d::MIPModelInfo, optimizer::MOI.AbstractOptimizer)
    dwmd_sol     = cgd.dw_main_dual_solution
    model        = cgd.pricing_sub.model
    x            = cgd.pricing_sub.x
    bin_vars     = cgd.pricing_sub.jump_bin_var
    obj_terms_lb = mip_d.obj_terms_lower_bound
    varid        = mip_d.dw_main_constr_varidx
    coef         = mip_d.dw_main_constr_coef
    #num_constr   = cgd.dw_main_num_constr
    nitr         = mip_d.num_oa_iter
    nbin         = length(bin_vars)
    num_constr   = nbin+nbin*nitr+1
    len          = length(dwmd_sol)
    #set obj func
    if len > num_constr
        (mip_d.obj_sense == :MAX_SENSE) && JuMP.@objective(model, Max, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:nbin)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[nbin+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)-dwmd_sol[num_constr]-sum(dwmd_sol[i]*x[cgd.dw_main.cut_var_idx[i-num_constr]] for i in (num_constr+1):len))
        (mip_d.obj_sense == :MIN_SENSE) && JuMP.@objective(model, Min, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:nbin)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[nbin+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)-dwmd_sol[num_constr]-sum(dwmd_sol[i]*x[cgd.dw_main.cut_var_idx[i-num_constr]] for i in (num_constr+1):len))
    else
        (mip_d.obj_sense == :MAX_SENSE) && JuMP.@objective(model, Max, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:nbin)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[nbin+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)-dwmd_sol[num_constr])
        (mip_d.obj_sense == :MIN_SENSE) && JuMP.@objective(model, Min, sum(bin_vars[i]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:nbin)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[nbin+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr)-dwmd_sol[num_constr])
    end
    #set lp solver
    JuMP.set_optimizer(model, optimizer.options.lp_solver)
    JuMP.optimize!(model)
    if (JuMP.termination_status(model) ==  MOI.OPTIMAL) || (JuMP.termination_status(model) ==  MOI.LOCALLY_SOLVED)
        pricing_sub_solution       = JuMP.value.(x)
        objective_value            = JuMP.objective_value(model)
        psm_sol_obj                = psmSolutionObj(pricing_sub_solution, objective_value)
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
function solve_pricing_sub_extm_dir_model(cgd::CGdata, mip_d::MIPModelInfo, optimizer::MOI.AbstractOptimizer)
    dwmd_sol     = cgd.dw_main_dual_solution
    obj_terms_lb = mip_d.obj_terms_lower_bound
    varid        = mip_d.dw_main_constr_varidx
    coef         = mip_d.dw_main_constr_coef
    num_constr   = mip_d.num_constr
    nitr         = mip_d.num_oa_iter
    siz          = length(mip_d.moi_bin_var)
    model        = cgd.pricing_sub_extrm_dir.model
    x            = cgd.pricing_sub_extrm_dir.d

    #define obj function
    (cgd.pricing_sub_obj_sense == :MAX_SENSE) && JuMP.@objective(model, Max, sum(x[mip_d.disc2var_idx[i]]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr))
    (cgd.pricing_sub_obj_sense == :MIN_SENSE) && JuMP.@objective(model, Min, sum(x[mip_d.disc2var_idx[i]]*obj_terms_lb[i]*dwmd_sol[i] for i in 1:siz)+sum(sum(dot(coef[i][j],x[varid[i][j]])*dwmd_sol[siz+(i-1)*nbin+j] for j in 1:nbin) for i in 1:nitr))
    #set lp solver
    JuMP.set_optimizer(model, optimizer.options.lp_solver)
    JuMP.optimize!(model)
    if (JuMP.termination_status(model) ==  MOI.OPTIMAL) || (JuMP.termination_status(model) ==  MOI.LOCALLY_SOLVED)
        pricing_sub_extr_dir_solution       = JuMP.value.(x)
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
The column generation algorithm is stated here.
"""
function cg_algorithm(node::Node, mip_d::MIPModelInfo, optimizer::MOI.AbstractOptimizer)
    #add cuts to the node if any
    len  = length(node.cut_var_indices)
    λlen = length(node.dwmainObj.λ)
    μlen = length(node.dwmainObj.μ)
    for i in 1:len
        ep = JuMP.@expression(node.dwmainObj.model, 0)
        for k in 1:λlen
            cp = 0
            ep = ep+JuMP.@expression(node.dwmainObj.model, cp*node.dwmainObj.λ[k])
        end
        ed = JuMP.@expression(node.dwmainObj.model, 0)
        for l in 1:μlen
            cd = 0
            ed = ed+JuMP.@expression(node.dwmainObj.model, cd*node.dwmainObj.μ[l])
        end
        if node.cut_inequality_types[i] == :(<=)
            con = JuMP.@constraint(node.dwmainObj.model, ep+ed <= node.cut_rhs[i])
            push!(node.dwmainObj.constr_ref[4], con)
        elseif node.cut_inequality_types[i] == :(>=)
            con = JuMP.@constraint(node.dwmainObj.model, ep+ed >= node.cut_rhs[i])
            push!(node.dwmainObj.constr_ref[4], con)
        end
    end
    #init cg data
    cgd = CGdata()
    init_cg_data(cgd, node)
    duration = 100
    start = time()
    #start cg loop
    while((time()-start) < duration)
        cgd.cg_iter += 1
        #solve dw main problem
        if cgd.cg_iter <= 1
            #solve pricing sub problem for a feasible solution
            ps_model = cgd.pricing_sub.model
            x        = cgd.pricing_sub.x
            #set obj func
            (mip_d.obj_sense == :MAX_SENSE) && JuMP.@objective(ps_model, Max, 0)
            (mip_d.obj_sense == :MIN_SENSE) && JuMP.@objective(ps_model, Min, 0)
            #set lp solver
            JuMP.set_optimizer(ps_model, optimizer.options.lp_solver)
            #print(ps_model)
            JuMP.optimize!(ps_model)
            if (JuMP.termination_status(ps_model) ==  MOI.OPTIMAL) || (JuMP.termination_status(ps_model) ==  MOI.LOCALLY_SOLVED)
                pricing_sub_solution       = JuMP.value.(x)
                objective_value            = JuMP.objective_value(ps_model)
                psm_sol_obj                = psmSolutionObj(pricing_sub_solution, objective_value)
                cgd.psm_feasible           = true
                cgd.psm_extr_dir_feasible  = false
                cgd.λ_counter             += 1
                push!(cgd.extr_ptn_sol, psm_sol_obj)
            else
                @info "\n The pricing sub problem is infeasible. \n"
                #solve pricing sub problem for feasible extreme directions
                ps_dir_model   = cgd.pricing_sub_extrm_dir.model
                x              = cgd.pricing_sub_extrm_dir.d
                #define obj function
                (mip_d.obj_sense == :MAX_SENSE) && JuMP.@objective(ps_dir_model, Max, 0)
                (mip_d.obj_sense == :MIN_SENSE) && JuMP.@objective(ps_dir_model, Min, 0)
                #set lp solver
                JuMP.set_optimizer(ps_dir_model, optimizer.options.lp_solver)
                JuMP.optimize!(ps_dir_model)
                if (JuMP.termination_status(ps_dir_model) ==  MOI.OPTIMAL) || (JuMP.termination_status(ps_dir_model) ==  MOI.LOCALLY_SOLVED)
                    pricing_sub_extr_dir_solution       = JuMP.value.(x)
                    objective_value                     = JuMP.objective_value(ps_dir_model)
                    psm_extm_dir_sol_obj                = psmExtrmDirSolutionObj(pricing_sub_extr_dir_solution, objective_value)
                    cgd.psm_feasible                    = false
                    cgd.psm_extr_dir_feasible           = true
                    cgd.μ_counter                      += 1
                    push!(cgd.extr_dir_sol, psm_extm_dir_sol_obj)
                else
                    @info "\n The pricing sub problem for extreme direction is infeasible. \n"
                end
            end
            #solve dw_main_model
            solve_dw_main_model(cgd, mip_d, optimizer)
        elseif (cgd.cg_iter >1)
            solve_dw_main_model(cgd, mip_d, optimizer)
        #elseif TO DO: for n>50 need to update basis of the dw_model instead of adding columns
        elseif cgd.cg_iter > 50
            @info "maximum number of column generation iteration reached. \n"
            break
        end
        #solve dw pricing sub problem
        solve_pricing_sub_model(cgd, mip_d, optimizer)
        #check cg optimality
        if cgd.psm_feasible && (cgd.cg_iter > 3)
            if (cgd.extr_ptn_sol[cgd.λ_counter].obj_value >= 0)
                cgd.cg_status = :Optimal
                @info "The optimality condition has been satisfied. \n"
                break
            end
        end
        #solve pricing sub for extreme direction in case psm infeasible
        !cgd.psm_feasible && solve_pricing_sub_extm_dir_model(cgd, mip_d, optimizer)
    end
    ExtremePoints = cgd.extr_ptn_sol
    ExtremeDirections = cgd.extr_dir_sol
    dw_solution = cgd.dw_main_sol[cgd.num_cg_iter-1].solution
    (cgd.λ_counter > 0) && (lp_ext_sol = sum(ExtremePoints[i].solution*cgd.λ_val[i] for i in 1:cgd.λ_counter))
    (cgd.μ_counter > 0) && (lp_ext_dir_sol = sum(ExtremeDirections[i].solution*cgd.μ_val[i] for i in 1:cgd.μ_counter))
    (cgd.λ_counter > 0) && (cgd.μ_counter > 0) && (lp_solution = lp_ext_sol+lp_ext_dir_sol)
    (cgd.λ_counter > 0) && !(cgd.μ_counter > 0) && (lp_solution = lp_ext_sol)
    #return solution of the input model
    print("The optimal lp solution using column generation method is :\n", lp_solution, "\n Thank you for having patience.\n")
    cgd.cg_solution  = lp_solution
    cgd.cg_objval    = cgd.dw_main_sol[cgd.num_cg_iter-1].obj_value
    return (cgd.cg_solution, cgd.cg_objval, cgd.cg_status)
end
