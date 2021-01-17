"""
This function implements node selection method: Depth-first search
"""
function select_node(bp_tree::BPTreeObj)
    #the dept-first search algorithm is used in selecting new node
    #TO Do: A^* search or other search method could be used to check performance
    node = pop!(bp_tree.branch_nodes)
    return node
end


"""
This function checks whether the solution is mip solution
"""
function is_solution_integer(solution, int_ind)
    flag = true
    for i in int_ind
        if !isinteger(solution[i])
            flag = false
            break
        end
    end
    return flag
end


"""
This function implements branching rule: most integer infeasible variable index is used for generating two new branches
"""
function most_int_infeasible_branching_rule(solution, int_ind)
    #TO DO: other brancing rules could be used to check performance
    floor_gap = 0
    var_idx   = 0
    value = 0
    for i in int_ind
        floor_gap   = ceil(solution[i])-solution[i]
        solution[i] = floor_gap
        if value < solution[i]
            var_idx = i
            value   = solution[i]
        end
    end
    return var_idx
end


"""
This fucntion converts a value to integer if it falls within a tolerance
"""
function integer_equivalent(solution, int_ind)
    for i in int_ind
        if solution[i]-floor(solution[i]) <= 0.0001
            solution[i]=floor(solution[i])
        elseif ceil(solution[i])-solution[i] <= 0.0001
            solution[i]=ceil(solution[i])
        end
    end
    return solution
end
