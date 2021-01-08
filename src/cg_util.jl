"""
This file contains necessary utility functions required for column generation algorithm.
"""
"""
our problem or model structure is as follows:
min z[1]+z[2]+z[3]
st.
    Ax+By <= 0

    z[1] >= y[1]*L[1]
    z[2] >= y[2]*L[2]
    z[3] >= y[3]*L[3]

    z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
    z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
    z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)

    x = [x[1], x[2], x[3]], :Cont
    y = [y[1], y[2], y[3]], :Bin
    z = [z[1], z[2], z[3]], :Cont
    A = (mxn) matrix
    B = (mxn) matrix

First, we need to do dantzig-wolfe decomposition of the above problem .

The constraint set
"Ax+By <= 0"
acts as feasibility constraint and these constraints indirectly influence the objective function.

The constraint set
"z[1] >= y[1]*L[1]
 z[2] >= y[2]*L[2]
 z[3] >= y[3]*L[3]
 z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
 z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
 z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)"
directly influence the objective function.

Now we can do Dantzig-Wolfe decomposition of the model as follows:

main problem:
min z[1]+z[2]+z[3]
(x,y) ∈ Q
st.
    z[1] >= y[1]*L[1]
    z[2] >= y[2]*L[2]
    z[3] >= y[3]*L[3]

    z[1] >= (y[1]-1)*U[1]+∇f1(̄x)(x-̄x)
    z[2] >= (y[2]-1)*U[2]+∇f2(̄x)(x-̄x)
    z[3] >= (y[3]-1)*U[3]+∇f3(̄x)(x-̄x)

where Q = {(x,y): Ax+By <= 0, x :Cont, y :Cont}.

The Dantzig-Wolfe relaxation of the main problem:

min z[1]+z[2]+z[3]
(λ,μ)
st.
    z[1] >= sum(u_p[1,i]*L[1]*λ[i] for i in Κ) + sum(u_d[1,j]*L[1]*μ[j] for j in Τ)
    z[2] >= sum(u_p[2,i]*L[2]*λ[i] for i in Κ) + sum(u_d[2,j]*L[2]*μ[j] for j in Τ)
    z[3] >= sum(u_p[3,i]*L[3]*λ[i] for i in Κ) + sum(u_d[3,j]*L[3]*μ[j] for j in Τ)

    z[1]+∇f1(̄x)*̄x+U[1] >= sum((u_p[1,i]*U[1]+∇f1(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[1,j]*U[1]+∇f1(̄x)*v_d[j])*μ[j] for j in Τ)
    z[2]+∇f2(̄x)*̄x+U[2] >= sum((u_p[2,i]*U[2]+∇f2(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[2,j]*U[2]+∇f2(̄x)*v_d[j])*μ[j] for j in Τ)
    z[3]+∇f3(̄x)*̄x+U[3] >= sum((u_p[3,i]*U[3]+∇f3(̄x)*v_p[i])*λ[i] for i in Κ)+sum((u_d[3,j]*U[3]+∇f3(̄x)*v_d[j])*μ[j] for j in Τ)

    sum(λ[i] for i in K) = 1
    λ[i] >= 0, ∀i ∈ K
    μ[j] >= 0, ∀j ∈ T

where (v_p[i], u_p[i]) ≡ (x,y) is the i^th extreme point of Q, and (v_d[j], u_d[j]) is the j^th extreme direction of Q.

Suppose the dual variable solution of the above main relaxed problem is (̄γ,̄π,̄σ).
The pricing subproblem:

max sum(L[i]*y[i]*̄γ[i] for i in 1:3)+sum((U[j]*y[j]+∇f[j](̄x)*x)*̄π[j] for j in 1:3)+̄σ
(x,y)
st. Ax+By <= 0
    x :Cont
    y :Cont
"""




"we need a data structure for recording data from column generation algorithm"
mutable struct CGdata
    lp_model                         :: JuMP.Model
    dw_main_model                    :: JuMP.Model
    pricing_sub_model                :: JuMP.Model

    lp_x                             :: Vector{JuMP.VariableRef}
    dw_main_x                        :: Vector{JuMP.VariableRef}
    pricing_sub_x                    :: Vector{JuMP.VariableRef}

    lp_num_var                       :: Int64
    dw_main_num_var                  :: Int64
    pricing_sub_num_var              :: Int64

    lp_num_constr                    :: Int64
    dw_main_num_constr               :: Int64
    pricing_sub_num_constr           :: Int64

    lp_le_constr                     :: Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    lp_ge_constr                     :: Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    lp_eq_constr                     :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}
    dw_main_le_constr                :: Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    dw_main_ge_constr                :: Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    dw_main_eq_constr                :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}
    pricing_sub_le_constr            :: Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    pricing_sub_ge_constr            :: Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    pricing_sub_ge_constr            :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}

    lp_obj_sense                     :: Symbol
    dw_main_obj_sense                :: Symbol
    pricing_sub_obj_sense            :: Symbol

    lp_l_var                         :: Vector{Float64}
    lp_u_var                         :: Vector{Float64}
    dw_main_l_var                    :: Vector{Float64}
    dw_main_u_var                    :: Vector{Float64}
    pricing_sub_l_var                :: Vector{Float64}
    pricing_sub_u_var                :: Vector{Float64}

    lp_solution                      :: Vector{Float64}
    dw_main_solution                 :: Vector{Float64}
    pricing_sub_solution             :: Vector{Float64}
    dw_main_dual_solution            :: Vector{Float64}

    cg_status                        :: Symbol
    cg_started                       :: Bool
    incumbent                        :: Vector{Float64}
    new_incumbent                    :: Bool
    total_time                       :: Float64
    obj_val                          :: Float64
    obj_bound                        :: Float64
    obj_gap                          :: Float64
    oa_iter                          :: Int64

    moi_bin_var                      :: Vector{Any}
    obj_terms_lower_bound            :: Vector{Float64}
    obj_terms_upper_bound            :: Vector{Float64}

    varidx_content                    #no specific type definition, close to list in other language
    coef_content                      #no specific type definition, close to list in other language

    CGdata() = new()
end



function init_cg_data(cgd::CGdata, oad::OAdata)
    cgd.lp_num_var                       = 0
    cgd.dw_main_num_var                  = 0
    cgd.pricing_sub_num_var              = 0

    cgd.lp_num_constr                    = 0
    cgd.dw_main_num_constr               = 0
    cgd.pricing_sub_num_constr           = 0

    cgd.lp_le_constr                     = []
    cgd.lp_ge_constr                     = []
    cgd.lp_eq_constr                     = []
    cgd.dw_main_le_constr                = []
    cgd.dw_main_ge_constr                = []
    cgd.dw_main_eq_constr                = []
    cgd.pricing_sub_le_constr            = []
    cgd.pricing_sub_ge_constr            = []
    cgd.pricing_sub_eq_constr            = []

    cgd.lp_obj_sense                     = :Unknown
    cgd.dw_main_obj_sense                = :Unknown
    cgd.pricing_sub_obj_sense            = :Unknown

    cgd.lp_l_var                         = Float64[]
    cgd.lp_u_var                         = Float64[]
    cgd.dw_main_l_var                    = Float64[]
    cgd.dw_main_u_var                    = Float64[]
    cgd.pricing_sub_l_var                = Float64[]
    cgd.pricing_sub_u_var                = Float64[]

    cgd.lp_solution                      = Float64[]
    cgd.dw_main_solution                 = Float64[]
    cgd.pricing_sub_solution             = Float64[]
    cgd.dw_main_dual_solution            = Float64[]

    cgd.cg_status                        = :Unknown
    cgd.cg_started                       = false
    cgd.incumbent                        = Float64[]
    cgd.new_incumbent                    = false
    cgd.total_time                       = 0
    cgd.obj_val                          = Inf
    cgd.obj_bound                        = -Inf
    cgd.obj_gap                          = Inf
    cgd.cg_iter                          = 0

    cgd.moi_bin_var                      = oad.moi_bin_var
    cgd.obj_terms_lower_bound            = oad.obj_terms_lower_bound
    cgd.obj_terms_upper_bound            = oad.obj_terms_upper_bound

    cgd.varidx_content                   = oad.varidx_new_content
    cgd.coef_content                     = oad.coef_new_content
end


function decompose_lp_model(model::JuMP.Model, optimizer::MOI.AbstractOptimizer, op::HybridProblem, oad::OAdata)
    cgd = CGdata()
    init_cg_data(cgd, oad)
    #extract all the constraints from the lp model and save it as cgd lb constraints
    extract_lp_constraints(model, cgd)
    #set user defined lp constraints as the constraints for pricing subproblem
    cgd.pricing_sub_le_constr = optimizer.linear_le_constraints
    cgd.pricing_sub_ge_constr = optimizer.linear_ge_constraints
    cgd.pricing_sub_eq_constr = optimizer.linear_eq_constraints
    #set obj sense
    cgd.lp_obj_sense       = JuMP.objective_sense(mod)
    cgd.dw_main_obj_sense  = cgd.lp_obj_sense
    (cgd.dw_main_obj_sense == MOI.MIN_SENSE) && (cgd.pricing_sub_obj_sense = MOI.MAX_SENSE)
    (cgd.dw_main_obj_sense == MOI.MAX_SENSE) && (cgd.pricing_sub_obj_sense = MOI.MIN_SENSE)
    #set num var
    cgd.lp_num_var          = JuMP.num_variables(model)
    cgd.pricing_sub_num_var = length(optimizer.variable_info)
    #set num constr
    cgd.lp_num_constr           = length(cgd.lp_le_constr)+length(cgd.lp_ge_constr)+length(cgd.lp_eq_constr)
    cgd.pricing_sub_num_constr  = length(cgd.pricing_sub_le_constr)+length(cgd.pricing_sub_ge_constr)+length(cgd.pricing_sub_eq_constr)
    #set var bound
    lp_var = JuMP.all_variables(model)
    for i in 1:cgd.lp_num_var
        JuMP.has_lower_bound(lp_var[i]) && push!(cgd.lp_l_var, JuMP.lower_bound(lp_var[i]))
        !JuMP.has_lower_bound(lp_var[i]) && (op.var_type[i] == :Bin) && push!(cgd.lp_l_var, 0.0)
        JuMP.has_upper_bound(lp_var[i]) && push!(cgd.lp_u_var, JuMP.upper_bound(lp_var[i]))
        !JuMP.has_upper_bound(lp_var[i]) && (op.var_type[i] == :Bin) && push!(cgd.lp_u_var, 1.0)
    end
    cgd.pricing_sub_l_var = op.l_var
    cgd.pricing_sub_u_var = op.u_var
    construct_dw_main_model(cgd)
    construct_pricing_sub_model(cgd)
end


function construct_dw_main_model(cgd)
    dwm = Model()

end



function construct_pricing_sub_model(cgd)
    lb = cgd.pricing_sub_l_var
    ub = cgd.pricing_sub_u_var
    psm = Model()
    @variable(psm, lb <= x[1:cgd.pricing_sub_num_var] <= ub)
    backend = JuMP.backend(psm)
    llc = cgd.pricing_sub_le_constr
    lgc = cgd.pricing_sub_ge_constr
    lec = cgd.pricing_sub_eq_constr
    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end
    cgd.pricing_sub_model = psm
    cgd.pricing_sub_x = x
    #set obj function
    #max sum(L[i]*y[i]*̄γ[i] for i in 1:3)+sum((U[j]*y[j]+∇f[j](̄x)*x)*̄π[j] for j in 1:3)+̄σ
end


function solve_pricing_sub_model(cgd::CGdata, dw_main_dual_solution, oad::OAdata, optimizer::MOI.AbstractOptimizer)
    model = cgd.pricing_sub_model
    x = JuMP.all_variables(model)
    obj_terms_lb = cgd.obj_terms_lower_bound
    varid = cgd.varidx_content
    coef = cgd.coef_content

    bin_bars = []
    siz = length(cgd.moi_bin_var)
    for i in 1:siz
        push!(bin_vars, JuMP.VariableRef(model, cgd.moi_bin_var[i].args[2]))
    end
    (cgd.pricing_sub_obj_sense == MOI.MAX_SENSE) && JuMP.@objective(model, Max, sum(bin_vars[i]*obj_terms_lb[i] for i in 1:siz)+sum(dot(coef[i],x[varid[i]]) for i in 1:oad.oa_iter))
    (cgd.pricing_sub_obj_sense == MOI.MIN_SENSE) && JuMP.@objective(model, Min, sum(bin_vars[i]*obj_terms_lb[i] for i in 1:siz)+sum(dot(coef[i],x[varid[i]]) for i in 1:oad.oa_iter))

    JuMP.set_optimizer(model, optimizer.options.lp_solver)
    JuMP.optimize!(model)
    if (JuMP.termination_status(model) ==  MOI.OPTIMAL) || (JuMP.termination_status(model) ==  MOI.LOCALLY_SOLVED)
        cgd.pricing_sub_solution = JuMP.value.(x)
    else
        @ifo "\n The pricing sub problem is infeasible. \n"
    end
end



function extract_lp_constraints(model::JuMP.Model, cgd::CGdata)
    #list of different types of constraints
    constr_types = JuMP.list_of_constraint_types(model)
    #number of different constraint types
    siz = length(constr_types)
    #reserve constraints for each type
    for i in 1:siz
        num_constr = JuMP.num_constraints(model, constr_types[i][1], constr_types[i][2])
        constrs    = JuMP.all_constraints(model, constr_types[i][1], constr_types[i][2])
        for j in 1:num_constr
            con_obj = JuMP.constraint_object(constrs[j])
            func    = JuMP.moi_function(con_obj.func)
            set     = con_obj.set
            (constr_types[i][2] == MOI.EqualTo{Float64}) && push!(cgd.lp_eq_constr, (func, set))
            (constr_types[i][2] == MOI.LessThan{Float64}) && push!(cgd.lp_le_constr, (func, set))
            (constr_types[i][2] == MOI.GreaterThan{Float64}) && push!(cgd.lp_ge_constr, (func, set))
        end
    end
end
