#=
    Used from https://github.com/lanl-ansi/Juniper.jl
=#

function get_default_options()
    nl_solver                           = nothing
    log_levels                          = [:Options,:Table,:Info]
    silent                              = false
    atol                                = 1e-6
    # Obj cuts
    incumbent_constr                    = false
    obj_epsilon                         = 0
    # :UserLimit
    time_limit                          = Inf
    mip_gap                             = 1e-4
    best_obj_stop                       = NaN
    solution_limit                      = 0
    all_solutions                       = false
    list_of_solutions                   = false

    mip_solver                          = nothing
    allow_almost_solved                 = true
    allow_almost_solved_integral        = true
    registered_functions                = nothing
    # Parallel
    processors                          = 1

    return SolverOptions(nl_solver,log_levels,silent,atol,incumbent_constr,obj_epsilon,time_limit,mip_gap,best_obj_stop,solution_limit,all_solutions,
        list_of_solutions, mip_solver, allow_almost_solved, allow_almost_solved_integral, registered_functions, processors)
end

function combine_options(options)
    options_dict = Dict{Symbol,Any}()
    for kv in options
        if !in(kv[1], fieldnames(SolverOptions))

        else
            options_dict[kv[1]] = kv[2]
        end
    end
    if haskey(options_dict, :log_levels)
        if length(options_dict[:log_levels]) == 0
            options_dict[:log_levels] = Symbol[]
        end
    end

    defaults = get_default_options()

    return defaults
end
