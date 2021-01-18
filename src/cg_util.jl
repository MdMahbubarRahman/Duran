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
mutable struct dwmSolutionObj
    solution        :: Vector{Float64}
    obj_value       :: Float64
end

mutable struct psmSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end

mutable struct psmExtrmDirSolutionObj
    solution  :: Vector{Float64}
    obj_value :: Float64
end

#Datzig-Wolfe main data structure
mutable struct DWMain
    model                   :: JuMP.Model
    λ                       :: Vector{JuMP.VariableRef}
    μ                       :: Vector{JuMP.VariableRef}
    t                       :: Vector{JuMP.VariableRef}
    num_constr              :: Int64
    constr_ref
end

#pricing sub problem data structure
mutable struct pricingSub
    model                   :: JuMP.Model
    x                       :: Vector{JuMP.VariableRef}
    var_type                :: Vector{Symbol}
    jump_bin_var            :: Vector{Any}
end

#pricing sub problem for extreme directions
mutable struct pricingSubExtrmDir
    model                   :: JuMP.Model
    d                       :: Vector{JuMP.VariableRef}
    d1                      :: Vector{JuMP.VariableRef}
    d2                      :: Vector{JuMP.VariableRef}
end


#CG prerequisite data structure
mutable struct MIPModelInfo
    model                       :: JuMP.Model
    x                           :: Vector{JuMP.VariableRef}
    num_var                     :: Int64
    num_constr                  :: Int64
    le_constr                   :: Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    ge_constr                   :: Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    eq_constr                   :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}
    obj_sense                   :: Symbol
    l_var                       :: Vector{Float64}
    u_var                       :: Vector{Float64}

    num_oa_iter                 :: Int64
    moi_bin_var                 :: Vector{Any}
    obj_terms_lower_bound       :: Vector{Float64}
    obj_terms_upper_bound       :: Vector{Float64}
    dw_main_constr_varidx
    dw_main_constr_coef
    dw_main_constr_type         :: Vector{Symbol}
    dw_main_constr_rhs          :: Vector{Float64}
    var2disc_idx                :: Vector{Int64}
    disc2var_idx                :: Vector{Int64}
    var_type                    :: Vector{Symbol}
    #nice set of constrs
    pricing_sub_le_constr       :: Vector{Tuple{SAF, MOI.LessThan{Float64}}}
    pricing_sub_ge_constr       :: Vector{Tuple{SAF, MOI.GreaterThan{Float64}}}
    pricing_sub_eq_constr       :: Vector{Tuple{SAF, MOI.EqualTo{Float64}}}

    MIPModelInfo() = new()
end


function init_model_info(m::MIPModelInfo, model::JuMP.Model, oad::OAdata, op::HybridProblem, optimizer::MOI.AbstractOptimizer)
    m.model                    = model
    m.x                        = JuMP.all_variables(model)
    m.num_var                  = op.num_var
    m.num_oa_iter              = oad.oa_iter
    m.moi_bin_var              = oad.moi_bin_var
    m.obj_terms_lower_bound    = oad.obj_terms_lower_bound
    m.obj_terms_upper_bound    = oad.obj_terms_upper_bound
    m.dw_main_constr_varidx    = oad.ref_linearized_constr_varidx
    m.dw_main_constr_coef      = oad.ref_linearized_constr_coef
    m.dw_main_constr_type      = oad.ref_linearized_constr_type
    m.dw_main_constr_rhs       = oad.ref_linearized_constr_rhs
    m.var2disc_idx             = op.var2disc_idx
    m.disc2var_idx             = op.disc2var_idx
    m.var_type                 = op.var_type
    m.obj_sense                = JuMP.objective_sense(model) == MOI.MIN_SENSE ? :MIN_SENSE : :MAX_SENSE
    m.le_constr                = []
    m.ge_constr                = []
    m.eq_constr                = []
    m.l_var                    = []
    m.u_var                    = []
    #extract model constraints
    constr_types = JuMP.list_of_constraint_types(model)
    siz = length(constr_types)
    for i in 1:siz
        num_constr = JuMP.num_constraints(model, constr_types[i][1], constr_types[i][2])
        constrs    = JuMP.all_constraints(model, constr_types[i][1], constr_types[i][2])
        for j in 1:num_constr
            con_obj = JuMP.constraint_object(constrs[j])
            func    = JuMP.moi_function(con_obj.func)
            set     = con_obj.set
            (constr_types[i][2] == MOI.EqualTo{Float64}) && push!(m.eq_constr, (func, set))
            (constr_types[i][2] == MOI.LessThan{Float64}) && push!(m.le_constr, (func, set))
            (constr_types[i][2] == MOI.GreaterThan{Float64}) && push!(m.ge_constr , (func, set))
        end
    end
    m.num_constr               = length(m.le_constr)+length(m.ge_constr)+length(m.eq_constr)
    #update variable bounds
    for i in 1:m.num_var
        JuMP.has_lower_bound(m.x[i]) && push!(m.l_var, JuMP.lower_bound(m.x[i]))
        !JuMP.has_lower_bound(m.x[i]) && (op.var_type[i] == :Bin) && push!(m.l_var, 0.0)
        !JuMP.has_lower_bound(m.x[i]) && !(op.var_type[i] == :Bin) && push!(m.l_var, -Inf)
        JuMP.has_upper_bound(m.x[i]) && push!(m.u_var, JuMP.upper_bound(m.x[i]))
        !JuMP.has_upper_bound(m.x[i]) && (op.var_type[i] == :Bin) && push!(m.u_var, 1.0)
        !JuMP.has_upper_bound(m.x[i]) && !(op.var_type[i] == :Bin) && push!(m.u_var, Inf)
    end
    #nice set of constr(s)
    m.pricing_sub_le_constr = optimizer.linear_le_constraints
    m.pricing_sub_ge_constr = optimizer.linear_ge_constraints
    m.pricing_sub_eq_constr = optimizer.linear_eq_constraints
end


#define cg data structure
mutable struct CGdata
    dw_main                          :: DWMain
    dw_main_sol                      :: Vector{dwmSolutionObj}
    dw_main_dual_solution            :: Vector{Float64}
    pricing_sub                      :: pricingSub
    extr_ptn_sol                     :: Vector{psmSolutionObj}
    pricing_sub_extrm_dir            :: pricingSubExtrmDir
    extr_dir_sol                     :: Vector{psmExtrmDirSolutionObj}

    λ_val                            :: Vector{Float64}
    μ_val                            :: Vector{Float64}
    #TO DO: dw_infeasible
    psm_feasible                     :: Bool
    psm_extr_dir_feasible            :: Bool
    λ_counter                        :: Int64
    μ_counter                        :: Int64

    cg_status                        :: Symbol
    cg_iter                          :: Int64
    cg_total_time                    :: Float64
    cg_solution                      :: Vector{Float64}
    cg_objval                        :: Float64

    cut_var_indices                  :: Vector{Int64}

    CGdata() = new()
end


function init_cg_data(cgd::CGdata, node)
    cgd.dw_main                      = node.dwmainObj
    cgd.pricing_sub                  = node.psmObj
    cgd.pricing_sub_extrm_dir        = node.psmedObj
    cgd.dw_main_sol                  = []
    cgd.dw_main_dual_solution        = []
    cgd.λ_val                        = []
    cgd.μ_val                        = []
    #TO DO: dw_infeasible
    cgd.psm_feasible                 = false
    cgd.psm_extr_dir_feasible        = false
    cgd.λ_counter                    = 0
    cgd.μ_counter                    = 0
    cgd.cg_status                    = :Unknown
    cgd.cg_iter                      = 0
    cgd.cg_total_time                = 0
    cgd.cg_solution                  = []
    cgd.cg_objval                    = Inf
    cgd.cut_var_indices              = node.cut_var_indices
    cgd.extr_ptn_sol                 = []
    cgd.extr_dir_sol                 = []
end



function decompose_model(mip_d::MIPModelInfo)
    #construct dw_main model
    dwmObj   = construct_dw_main_model(mip_d)
    #construct pricing sub model
    psmObj   = construct_pricing_sub_model(mip_d)
    #construct pricing sub model for extreme direction
    psmedObj = construct_pricing_sub_model_for_extreme_directions(mip_d, psmObj)
    #return decomposed model objects
    return dwmObj, psmObj, psmedObj
end

"""
This function generates initial Dantzig-Wolfe relaxed main problem.
"""
function construct_dw_main_model(mip_d::MIPModelInfo)
    nλ = 100 #max num of extreme points
    nμ = 100
    nbin         = length(mip_d.moi_bin_var)
    constr_type  = mip_d.dw_main_constr_type
    rhs          = mip_d.dw_main_constr_rhs
    nitr         = mip_d.num_oa_iter
    #define Dantzig-Wolfe main model
    dwm = Model()
    @variable(dwm, λ[1:nλ] >= 0)
    @variable(dwm, μ[1:nμ] >= 0)
    @variable(dwm, t[1:nbin])
    #define constr ref container
    constr_set1 = []
    constr_set2 = []
    constr_set3 = []
    constr_set4 = []
    #define "t >= yL" type constr
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
    #define "t >= (y-1)U+∇f(̄x)(x-̄x)" type constr
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
    #define convexity constr "sum(λ[i]) == 1"
    con=@constraint(dwm, sum(λ[k]*0 for k in 1:nλ) == 1.0)
    push!(constr_set3, con)
    constr_ref = [constr_set1, constr_set2, constr_set3, constr_set4]
    #define obj func
    (mip_d.obj_sense == :MIN_SENSE) && @objective(dwm, Min, sum(t[i] for i in 1:nbin))
    (mip_d.obj_sense == :MAX_SENSE) && @objective(dwm, Max, sum(t[i] for i in 1:nbin))
    num_constr      = length(constr_set1)+length(constr_set2)+length(constr_set3)+length(constr_set4)
    #return DW main object
    return DWMain(dwm, λ, μ, t, num_constr, constr_ref)
end



"""
This function constructs the dw pricing sub model.
"""
function construct_pricing_sub_model(mip_d::MIPModelInfo)
    lb = mip_d.l_var
    ub = mip_d.u_var
    psm = Model()
    @variable(psm, lb[i] <= x[i=1:mip_d.num_var] <= ub[i])
    backend = JuMP.backend(psm)
    llc = mip_d.pricing_sub_le_constr
    lgc = mip_d.pricing_sub_ge_constr
    lec = mip_d.pricing_sub_eq_constr
    for constr_type in [llc, lgc, lec]
        for constr in constr_type
            MOI.add_constraint(backend, constr[1], constr[2])
        end
    end
    #convert moi variable references to jump variables
    bin_vars = []
    siz = length(mip_d.moi_bin_var)
    for i in 1:siz
        push!(bin_vars, JuMP.VariableRef(psm, mip_d.moi_bin_var[i].args[2]))
    end
    var_type           = mip_d.var_type
    #return pricing sub problem obj
    return pricingSub(psm, x, var_type, bin_vars)
end



"""
This function constructs pricing sub model for extreme directions.
"""
function construct_pricing_sub_model_for_extreme_directions(mip_d::MIPModelInfo, psmobj::pricingSub)
    psmed = Model()
    @variable(psmed, x[1:mip_d.num_var])
    @variable(psmed, 0 <= y[1:mip_d.num_var] <= 1)
    @variable(psmed, 0 <= z[1:mip_d.num_var] <= 1)
    for i in 1:mip_d.num_var
        if mip_d.var_type[i] == :Bin
            JuMP.set_lower_bound(x[i], 0)
            JuMP.set_upper_bound(x[i], 1)
        else
            JuMP.set_lower_bound(x[i], -1)
            JuMP.set_upper_bound(x[i], 1)
        end
    end
    #need a function to modify the input set of constraints to the set required for ext direction
    constr_types = JuMP.list_of_constraint_types(psmobj.model)
    #number of different constraint types
    len = length(constr_types)
    #define psmed constraints
    for i in 1:len
        num_constr = JuMP.num_constraints(psmobj.model, constr_types[i][1], constr_types[i][2])
        constrs    = JuMP.all_constraints(psmobj.model, constr_types[i][1], constr_types[i][2])
        for j in 1:num_constr
            con_obj = JuMP.constraint_object(constrs[j])
            func    = JuMP.moi_function(con_obj.func)
            exp = JuMP.@expression(psmed, 0)
            if typeof(func) ==  SAF
                for term in func.terms
                    exp = exp+JuMP.@expression(psmed, term.coefficient*(y[term.variable_index.value]-z[term.variable_index.value]))
                end
                (constr_types[i][2] == MOI.LessThan{Float64}) && @constraint(psmed, exp <= 0)
                (constr_types[i][2] == MOI.GreaterThan{Float64}) && @constraint(psmed, -1*exp <= 0)
                if (constr_types[i][2] == MOI.EqualTo{Float64})
                    @constraint(psmed, exp <= 0)
                    @constraint(psmed, -1*exp <= 0)
                end
            end
        end
    end
    for i in 1:mip_d.num_var
        @constraint(psmed, x[i] == y[i]-z[i])
    end
    #return pricing sub model obj for extreme directions
    return pricingSubExtrmDir(psmed, x, y, z)
end
