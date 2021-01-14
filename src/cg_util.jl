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

"""
#What do i need to do
1. decompose milp model
2. build main problem and subproblem
3. how to keep track of constraint sets
4. how to reapetedly update main problem
5. we might include the set of constraints
"z[1] >= y[1]*L[1]
z[2] >= y[2]*L[2]
z[3] >= y[3]*L[3]"
in the subproblem to minimimze the size of the main problem.
6. and of course, every model would use JuMP environment
"""


"""
now consider building the pricing subproblem.
what we need?
1. the constraint set "Ax+By <= 0"
2. the obj function
"sum(L[i]*y[i]*̄γ[i] for i in 1:3)+sum((U[j]*y[j]+∇f[j](̄x)*x)*̄π[j] for j in 1:3)+̄σ"
we can update this function as JuMP expression and define obj function.
"""



"""
we need a data structure for recording data from column generation algorithm
"""

mutable struct lpSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end

mutable struct dwmSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end

mutable struct psmSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end

mutable struct psmExtrmDirSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end


#define cg data structure
mutable struct CGdata
    lp_model                         :: JuMP.Model
    dw_main_model                    :: JuMP.Model
    pricing_sub_model                :: JuMP.Model
    pricing_sub_extm_dir_model       :: JuMP.Model

    lp_x                             :: Vector{JuMP.VariableRef}
    dw_main_x                        #:: Vector{JuMP.VariableRef}
    pricing_sub_x                    :: Vector{JuMP.VariableRef}
    pricing_sub_extm_dir_x

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
    pricing_sub_eq_constr            :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}

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
    dw_main_dual_solution            :: Vector{Float64}
    pricing_sub_solution             :: Vector{Float64}
    pricing_sub_extr_dir_solution    :: Vector{Float64}


    cg_status                        :: Symbol
    cg_started                       :: Bool
    incumbent                        :: Vector{Float64}
    new_incumbent                    :: Bool
    total_time                       :: Float64
    obj_val                          :: Float64
    obj_bound                        :: Float64
    obj_gap                          :: Float64

    num_oa_iter                      :: Int64
    num_cg_iter                      :: Int64

    moi_bin_var                      :: Vector{Any}
    jump_bin_var                     :: Vector{Any}
    obj_terms_lower_bound            :: Vector{Float64}
    obj_terms_upper_bound            :: Vector{Float64}

    dw_main_constr_varidx
    dw_main_constr_coef
    dw_main_constr_type              :: Vector{Symbol}
    dw_main_constr_rhs               :: Vector{Float64}
    dw_main_constr_ref

    pricing_sub_disc2var_idx         :: Vector{Int64}
    pricing_sub_var_type             :: Vector{Symbol}

    λ_counter                        :: Int64
    μ_counter                        :: Int64
    extr_ptn_sol                     :: Vector{psmSolutionObj}
    extr_dir_sol                     :: Vector{psmExtrmDirSolutionObj}
    dw_main_sol                      :: Vector{dwmSolutionObj}

    psm_feasible                     :: Bool
    psm_extr_dir_feasible            :: Bool
    best_solution                    :: Vector{Float64}

    λ_val                            :: Vector{Float64}
    μ_val                            :: Vector{Float64}

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
    cgd.dw_main_dual_solution            = Float64[]
    cgd.pricing_sub_solution             = Float64[]
    cgd.pricing_sub_extr_dir_solution    = Float64[]

    cgd.cg_status                        = :Unknown
    cgd.cg_started                       = false
    cgd.incumbent                        = Float64[]
    cgd.new_incumbent                    = false
    cgd.total_time                       = 0
    cgd.obj_val                          = Inf
    cgd.obj_bound                        = -Inf
    cgd.obj_gap                          = Inf

    cgd.num_oa_iter                      = oad.oa_iter
    cgd.num_cg_iter                      = 0

    cgd.moi_bin_var                      = oad.moi_bin_var
    cgd.jump_bin_var                     = []
    cgd.obj_terms_lower_bound            = oad.obj_terms_lower_bound
    cgd.obj_terms_upper_bound            = oad.obj_terms_upper_bound

    cgd.dw_main_constr_varidx            = oad.ref_linearized_constr_varidx
    cgd.dw_main_constr_coef              = oad.ref_linearized_constr_coef
    cgd.dw_main_constr_type              = oad.ref_linearized_constr_type
    cgd.dw_main_constr_rhs               = oad.ref_linearized_constr_rhs
    cgd.dw_main_constr_ref               = []

    cgd.pricing_sub_disc2var_idx         = []
    cgd.pricing_sub_var_type             = []

    cgd.λ_counter                        = 0
    cgd.μ_counter                        = 0
    cgd.extr_ptn_sol                     = []
    cgd.extr_dir_sol                     = []
    cgd.dw_main_sol                      = []
    cgd.dw_main_x                        = []

    cgd.psm_feasible                     = false
    cgd.psm_extr_dir_feasible            = false
    cgd.best_solution                    = []

    cgd.λ_val                            = []
    cgd.μ_val                            = []
end


function decompose_lp_model(model::JuMP.Model, optimizer::MOI.AbstractOptimizer, op::HybridProblem, oad::OAdata, cgd::CGdata)
    cgd.lp_model = model
    #extract all the constraints from the lp model and save it as cgd lb constraints
    extract_lp_constraints(model, cgd)
    #set user defined lp constraints as the constraints for pricing subproblem
    cgd.pricing_sub_le_constr = optimizer.linear_le_constraints
    cgd.pricing_sub_ge_constr = optimizer.linear_ge_constraints
    cgd.pricing_sub_eq_constr = optimizer.linear_eq_constraints
    #set obj sense
    #obj_sense       = JuMP.objective_sense(model)
    (JuMP.objective_sense(model) == MOI.MAX_SENSE) && (cgd.lp_obj_sense = :MAX_SENSE)
    (JuMP.objective_sense(model) == MOI.MIN_SENSE) && (cgd.lp_obj_sense = :MIN_SENSE)
    cgd.dw_main_obj_sense  = cgd.lp_obj_sense
    (cgd.dw_main_obj_sense == :MIN_SENSE) && (cgd.pricing_sub_obj_sense = :MIN_SENSE)
    (cgd.dw_main_obj_sense == :MAX_SENSE) && (cgd.pricing_sub_obj_sense = :MAX_SENSE)
    #set num var
    cgd.lp_num_var              = JuMP.num_variables(model)
    cgd.pricing_sub_num_var     = length(optimizer.variable_info)
    cgd.pricing_sub_var_type    = oad.ref_var_type[1:cgd.pricing_sub_num_var]
    #set num constr
    cgd.lp_num_constr           = length(cgd.lp_le_constr)+length(cgd.lp_ge_constr)+length(cgd.lp_eq_constr)
    cgd.pricing_sub_num_constr  = length(cgd.pricing_sub_le_constr)+length(cgd.pricing_sub_ge_constr)+length(cgd.pricing_sub_eq_constr)
    cgd.dw_main_num_constr      = cgd.lp_num_constr-cgd.pricing_sub_num_constr+1
    #set var bound
    lp_var = JuMP.all_variables(model)
    for i in 1:cgd.lp_num_var
        JuMP.has_lower_bound(lp_var[i]) && push!(cgd.lp_l_var, JuMP.lower_bound(lp_var[i]))
        !JuMP.has_lower_bound(lp_var[i]) && (op.var_type[i] == :Bin) && push!(cgd.lp_l_var, 0.0)
        JuMP.has_upper_bound(lp_var[i]) && push!(cgd.lp_u_var, JuMP.upper_bound(lp_var[i]))
        !JuMP.has_upper_bound(lp_var[i]) && (op.var_type[i] == :Bin) && push!(cgd.lp_u_var, 1.0)
    end
    #get data from original problem
    cgd.pricing_sub_l_var = op.l_var
    cgd.pricing_sub_u_var = op.u_var
    cgd.pricing_sub_disc2var_idx = op.disc2var_idx
    #decompose the lp model
    construct_pricing_sub_model(cgd)
    construct_dw_main_model(cgd)
end

"""
This function generates initial Dantzig-Wolfe relaxed main problem.
"""
function construct_dw_main_model(cgd::CGdata)
    #initial vals
    nλ = 100 #max 1000 extreme points
    nμ = 100 #max 1000 extreme directions
    nbin         = length(cgd.jump_bin_var)
    #num_constr   = cgd.dw_main_num_constr
    constr_type  = cgd.dw_main_constr_type
    rhs          = cgd.dw_main_constr_rhs
    nitr         = cgd.num_oa_iter
    num_constr   = nbin+nbin*nitr+1
    #the dw relaxed main problem
    dwm = Model()
    @variable(dwm, λ[1:nλ] >= 0)
    @variable(dwm, μ[1:nμ] >= 0)
    @variable(dwm, t[1:nbin])
    #define constr containers
    constr_set1 = []
    constr_set2 = []
    constr_set3 = []
    #define dwd constr for "t >= yL" set of constr
    for i in 1:nbin
        ep = JuMP.@expression(dwm, 0)
        for k in 1:nλ
            cp = 0
            ep = ep+JuMP.@expression(dwm, cp*λ[k])
        end
        ed = JuMP.@expression(dwm, 0)
        for l in 1:nμ
            cd = 0
            ed = ed+JuMP.@expression(dwm, cd*μ[l])
        end
        con=@constraint(dwm, -ep-ed+t[i] >= 0.0)
        push!(constr_set1, con)
    end
    #define dwd constr for "t >= (y-1)U+∇f(̄x)(x-̄x)" set of constr
    indx = 0
    for i in 1:nitr
        for j in 1:nbin
            indx += 1
            ep = JuMP.@expression(dwm, 0)
            for k in 1:nλ
                cp = 0
                ep = ep+JuMP.@expression(dwm, cp*λ[k])
            end
            ed = JuMP.@expression(dwm, 0)
            for l in 1:nμ
                cd = 0
                ed = ed+JuMP.@expression(dwm, cd*μ[l])
            end
            if constr_type[indx] == :(<=)
                con=@constraint(dwm, ep+ed-t[j] <= rhs[indx])
                push!(constr_set2, con)
            elseif constr_type[indx] == :(>=)
                con=@constraint(dwm, ep+ed+t[j] >= rhs[indx])
                push!(constr_set2, con)
            end
        end
    end
    #define convexity constr for "sum(λ[i]) == 1" constr
    con=@constraint(dwm, sum(λ[k]*0 for k in 1:nλ) == 1.0)
    push!(constr_set3, con)
    constr_ref = [constr_set1, constr_set2, constr_set3]
    #define obj func
    (cgd.dw_main_obj_sense == :MIN_SENSE) && @objective(dwm, Min, sum(t[i] for i in 1:nbin))
    (cgd.dw_main_obj_sense == :MAX_SENSE) && @objective(dwm, Max, sum(t[i] for i in 1:nbin))
    cgd.dw_main_constr_ref = constr_ref
    cgd.dw_main_model = dwm
    cgd.dw_main_x = [λ, μ, t]
end



"""
This function constructs the dw pricing sub model.
"""
function construct_pricing_sub_model(cgd)
    lb = cgd.pricing_sub_l_var
    ub = cgd.pricing_sub_u_var
    psm = Model()
    @variable(psm, lb[i] <= x[i=1:cgd.pricing_sub_num_var] <= ub[i])
    backend = JuMP.backend(psm)
    llc = cgd.pricing_sub_le_constr
    lgc = cgd.pricing_sub_ge_constr
    lec = cgd.pricing_sub_eq_constr
    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end
    #convert moi variable references to jump variables
    bin_vars = []
    siz = length(cgd.moi_bin_var)
    for i in 1:siz
        push!(bin_vars, JuMP.VariableRef(psm, cgd.moi_bin_var[i].args[2]))
    end
    cgd.jump_bin_var = bin_vars
    cgd.pricing_sub_model = psm
    cgd.pricing_sub_x = x
    #set obj function
    #max sum(L[i]*y[i]*̄γ[i] for i in 1:3)+sum((U[j]*y[j]+∇f[j](̄x)*x)*̄π[j] for j in 1:3)+̄σ
end


"""
This function extracts constraints from the input model of the cg algorithm and saves them in the cg data structure
"""
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


"""
This function constructs pricing sub model for extreme directions.
"""
function construct_pricing_sub_model_for_extreme_directions(cgd::CGdata)
    #define model
    psmed = Model()
    @variable(psmed, x[1:cgd.pricing_sub_num_var]) #moi references are created according to the order of variable creation
    @variable(psmed, 0 <= y[1:cgd.pricing_sub_num_var] <= 1)
    @variable(psmed, 0 <= z[1:cgd.pricing_sub_num_var] <= 1)
    for i in 1:cgd.pricing_sub_num_var
        if cgd.pricing_sub_var_type[i] == :Bin
            JuMP.set_lower_bound(x[i], 0)
            JuMP.set_upper_bound(x[i], 1)
        else
            JuMP.set_lower_bound(x[i], -1)
            JuMP.set_upper_bound(x[i], 1)
        end
    end
    #need a function to modify the input set of constraints to the set required for ext direction
    constr_types = JuMP.list_of_constraint_types(cgd.pricing_sub_model)
    #number of different constraint types
    len = length(constr_types)
    #define psmed constraints
    for i in 1:len
        num_constr = JuMP.num_constraints(cgd.pricing_sub_model, constr_types[i][1], constr_types[i][2])
        constrs    = JuMP.all_constraints(cgd.pricing_sub_model, constr_types[i][1], constr_types[i][2])
        for j in 1:num_constr
            con_obj = JuMP.constraint_object(constrs[j])
            func    = JuMP.moi_function(con_obj.func)
            exp = JuMP.@expression(psmed, 0)
            for term in func.terms
                exp = exp+JuMP.@expression(psmed, term.coefficient*(y[term.variable_index.value]-z[term.variable_index.value]))
            end
            (constr_types[i][2] == MOI.LessThan{Float64}) && @constraint(psmed, exp <= 0)
            (constr_types[i][2] == MOI.GreaterThan{Float64}) && @constraint(psmed, -1*exp <= 0)
            if (constr_types[i][2] == MOI.EqualTo{Float64})
                @constraint(psmed, exp <= 0)
                @constraint(psmed, -1*exp <= 0)#need some clarification here, cuz this could be con_obj.set.value
            end
        end
    end
    for i in 1:cgd.pricing_sub_num_var
        @constraint(psmed, x[i] == y[i]-z[i])
    end
    cgd.pricing_sub_extm_dir_model = psmed
    cgd.pricing_sub_extm_dir_x     = [x, y, z]
end
