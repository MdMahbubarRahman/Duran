"""
Node for branch and bound tree
"""
mutable struct Node
    indx                 :: Int64
    dwmainObj            :: DWMain
    psmObj               :: pricingSub
    psmedObj             :: pricingSubExtrmDir
    cut_var_indices      :: Vector{Int64}
    cut_inequality_types :: Vector{Symbol}
    cut_rhs              :: Vector{Float64}
    parent               :: Int64 #parent node index
    left_child           :: Union{Node, Nothing}
    right_child          :: Union{Node, Nothing}
end


"""
Incumbent solution strorage
"""
mutable struct Incumbent
    solution     :: Vector{Float64}
    obj_val      :: Float64
end



"""
Branch and Price tree object
"""
mutable struct BPTreeObj
    total_time                         :: Float64
    incumbent                          :: Incumbent
    num_solutions                      :: Int64
    int_solutions                      :: Vector{dwmSolutionObj}
    num_node_pruned_by_bound           :: Int64
    num_node_pruned_by_infeasibility   :: Int64
    num_node_pruned_by_integrality     :: Int64
    best_bound                         :: Float64
    branch_nodes                       :: Vector{Node}
    num_nodes                          :: Int64
    status                             :: Symbol
    exit_by_time_limit                 :: Bool

    BPTreeObj() = new()
end

function init_bp_tree(bp_tree::BPTreeObj)
    bp_tree.total_time                         = 0
    bp_tree.num_solutions                      = 0
    bp_tree.int_solutions                      = []
    bp_tree.num_node_pruned_by_bound           = 0
    bp_tree.num_node_pruned_by_infeasibility   = 0
    bp_tree.num_node_pruned_by_integrality     = 0
    bp_tree.best_bound                         = Inf
    bp_tree.branch_nodes                       = []
    bp_tree.num_nodes                          = 0
    bp_tree.status                             = :Unknown
    bp_tree.exit_by_time_limit                 = false
end

"""
The branch and price algorithm is implemented here
"""
function branch_and_price(model::JuMP.Model, oad::OAdata, op::HybridProblem, optimizer::MOI.AbstractOptimizer)
    #initiate input model data struct
    mip_d = MIPModelInfo()
    #populate model data
    init_model_info(mip_d, model, oad, op, optimizer)
    #decompose model
    dwmObj, psmObj, psmedObj = decompose_model(mip_d)
    dw_obj = deepcopy(dwmObj)
    ps_obj = deepcopy(psmObj)
    psed_obj = deepcopy(psmedObj)
    root_node = Node(1, dw_obj, ps_obj, psed_obj, [], [], [], 0, nothing, nothing)
    #start branch and price storage
    bp_tree = BPTreeObj()
    init_bp_tree(bp_tree)
    #insert root node to the tree
    push!(bp_tree.branch_nodes, root_node)
    #filter integer/or binary variable indexes
    int_ind = filter(i -> (mip_d.var_type[i] in (:Int, :Bin)), 1:mip_d.num_var)
    #start bb search
    start_time = time()
    node_counter = 1
    duration = 1000 #max duration the bp algorithm will run
    while !isempty(bp_tree.branch_nodes)
        if (time() - start_time) >= duration
            @info "The branch and price algorithm has hit the time limit.\n"
            bp_tree.exit_by_time_limit = true
            break
        end
        #select a node from the nodes pool
        node = select_node(bp_tree)
        #solve the node using cg algorithm
        solution, objval, status = cg_algorithm(node, mip_d, optimizer)
        #check for status of cg output
        if (status == :Optimal) || (status == :Suboptimal)
            if objval < bp_tree.best_bound
                #update sol to int equivalent sol
                soln = integer_equivalent(solution, int_ind)
                #check if the sol is mip solution
                is_int_solution = is_solution_integer(soln, int_ind)
                if is_int_solution
                    @info "The node is pruned by integralilty. \n"
                    push!(bp_tree.int_solutions, dwmSolutionObj(soln, objval))
                    bp_tree.num_solutions += 1
                    bp_tree.num_node_pruned_by_integrality += 1
                    bp_tree.incumbent = Incumbent(soln, objval)
                    bp_tree.best_bound = objval
                else
                    var_idx     = most_int_infeasible_branching_rule(soln, int_ind)
                    dwm_obj     = deepcopy(dwmObj)
                    psm_obj     = deepcopy(psmObj)
                    psmed_obj   = deepcopy(psmedObj)
                    left_cvi    = node.cut_var_indices
                    right_cvi   = node.cut_var_indices
                    push!(left_cvi, var_idx)
                    push!(right_cvi, var_idx)
                    left_cct    = node.cut_inequality_types
                    right_cct   = node.cut_inequality_types
                    push!(left_cct, :(<=))
                    push!(right_cct, :(>=))
                    left_crhs   = node.cut_rhs
                    left_crhs   = node.cut_rhs
                    push!(left_crhs, 0)
                    push!(right_crhs, 1)
                    node_counter += 1
                    left_child  = Node(node_counter, dwm_obj, psm_obj, psmed_obj, left_cvi, left_cct, left_crhs, node.indx, nothing, nothing)
                    node_counter += 1
                    right_child = Node(node_counter, dwm_obj, psm_obj, psmed_obj, right_cvi, right_cct, right_crhs, node.indx, nothing, nothing)
                    push!(bp_tree.branch_nodes, left_child)
                    push!(bp_tree.branch_nodes, right_child)
                end
            else
                @info "The node is pruned by bound. \n"
                bp_tree.num_node_pruned_by_bound += 1
            end
        else
            @info "The node is pruned by infeasibility.\n"
            bp_tree.num_node_pruned_by_infeasibility += 1
        end
    end
    (bp_tree.num_solutions == 0) && (bp_tree.status = :Infeasible)
    (bp_tree.num_solutions == 0) && (bp_tree.incumbent = Incumbent([0,0], 0))
    !(bp_tree.num_solutions == 0) && (bp_tree.status = :Optimal)
    (bp_tree.exit_by_time_limit == true) && !(bp_tree.num_solutions == 0) && (bp_tree.status = :Suboptimal)
    bp_tree.num_nodes = node_counter
    bp_tree.total_time = time() - start_time
    #return incumbent solution along with status and total time
    return bp_tree.status, bp_tree.total_time, bp_tree.incumbent
end
