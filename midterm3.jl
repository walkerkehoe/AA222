

rosenbrocks(x; a=1, b=5) = (a-x[1])^2 + b*(x[2]-x[1]^2)^2

data_options = ["0","1","2","3","4","5","6","7","8","9","x1","x2","+","-","*","/"]
terminal_data = ["0","1","2","3","4","5","6","7","8","9","x1","x2"]
#constructing a node

function generateRandomNode(node)
    if node[1] in terminal_data
        return
    else
        r = rand(1:length(data_options))
        d = data_options[r]
        #println("left child: $d")
        push!(node,[])
        push!(node[2],d)
        generateRandomNode(node[2])

        r = rand(1:length(data_options))
        d = data_options[r]
        #println("right child: $d")
        push!(node,[])
        push!(node[3],d)
        generateRandomNode(node[3])

        return
    end
end

function generateRandomTree()
    r = rand(1:length(data_options))
    d = data_options[r]
    #println("root: $d")
    root = []
    push!(root, d)
    generateRandomNode(root)
    return root
end


function readTree(tree)
    symbols = tree[1]
    if length(tree) == 1
        return symbols
    else
        dl = tree[2][1]
        sl = ""
        sl = dl * sl
        sl = readTree(tree[2])
        symbols = "(" * sl * symbols

        dr = tree[3][1]
        sr = ""
        sr *= dr
        sr = readTree(tree[3])
        symbols *= sr * ")"

        return symbols
    end
end

function objective(individual,rosenbrocks)
    n = 20
    x1_vec = linspace(-2,2,n)
    x2_vec = linspace(-2,2,n)

    x2grid = repmat(x2_vec',n,1)
    x1grid = repmat(x1_vec,1,n)

    z, r = zeros(n,n), zeros(n,n)
    x1 = 1
    x2 = 1
    L1 = 0
    for i in 1:n
        for j in 1:n
            x1, x2 = x1_vec[j], x2_vec[i]
            indiv = eval(parse(individual))
            rosenb = rosenbrocks([x1_vec[j];x2_vec[i]])
            z[i:i,j:j] = indiv;
            r[i:i,j:j] = rosenb;
            L1 += abs(rosenb-indiv)
        end
    end
    L1
end

function mutateTree(t,p_mutation)
    tree = deepcopy(t)
    replacement_node = generateRandomTree()
    if rand() <= p_mutation
        return replacement_node
    elseif length(tree) == 1
        return tree
    else
        for i in 2:length(tree)
            if rand() <= p_mutation
                tree[i] = replacement_node
                return tree
            else
                tree[i] = mutateTree(tree[i],p_mutation)
            end
        end
    end
    return tree
end

#crossover
function pickGene(t)
    node = deepcopy(t)
    if rand() <= 0.5
        return node
    else
        for i in 2:length(node)
            if rand() <= 0.5
                return node[i]
            else node[i] = pickGene(node[i])
            end
        end
    end
    return node
end

#replacement_node = pickGene(replacement_tree)
function crossoverTree(p1, p2)
    child_tree = deepcopy(p1)
    replacement_node = deepcopy(p2)
    if rand() <= 0.5 && length(child_tree) == 1
        return replacement_node
    else
        for i in 2:length(child_tree)
            if rand() <= 0.5
                child_tree[i] = replacement_node
                return child_tree
            else
                child_tree[i] = crossoverTree(child_tree[i],replacement_node)
            end
        end
    end
    return child_tree
end

function generateNewPopulation(pop,best_obj,mean_obj,best_ind)
    #population is an array of individuals and their symbols
    #truncation selection retaining the top 60%
    population = deepcopy(pop)
    x1, x2, = 1,1
    sorted = []
    best = Inf
    obj_list = []
    for i in 1:length(population)
        dist = objective(population[i][2],rosenbrocks)
        push!(obj_list,dist)
        if dist <= best
            insert!(sorted,1,population[i])
            best = dist
        else
            push!(sorted,population[i])
        end
    end
    best_ind = sorted[1]
    best_obj = best
    sum_obj = 0
    for o in obj_list
        #println(o)
        if o != Inf
            sum_obj += o
        end
    end
    mean_obj = sum_obj/length(obj_list)

    new_pop = sorted[1:Integer(0.6*length(sorted))]

    for j in 1:Integer(0.4*length(population))
        c = []
        parent1 = new_pop[rand(1:end)]
        parent2 = new_pop[rand(1:end)]
        replacement_node = pickGene(parent2[1])
        child = crossoverTree(parent1[1],replacement_node)
        push!(c,child)
        push!(c,readTree(child))
        push!(new_pop,c)
    end
    for k in 1:length(new_pop)
        if rand() <= 0.2
            #println(k)
            t = []
            val = mutateTree(new_pop[k][1],0.2)
            push!(t,val)
            push!(t,readTree(val))
            new_pop[k] = t
        end
    end
    return new_pop, best_obj, mean_obj, best_ind
end

pop = []
for i in 1:100
    p = []
    t = generateRandomTree()
    push!(p,t)
    push!(p,readTree(t))
    push!(pop,p)
end
best_obj = Inf
mean_obj = Inf
best_ind = pop[1]
plot_best = []
plot_mean = []
for i in 1:20
    pop, best_obj, mean_obj, best_ind = generateNewPopulation(pop,best_obj,mean_obj,best_ind)
    push!(plot_best,best_obj)
    push!(plot_mean,mean_obj)
end
bi = best_ind[2]
println("best individual: $bi")

using PyPlot
plot(collect(1:1:20),plot_best, color="blue")
xlabel("Iterations")
ylabel("Objective Function")
title("Best Individual")

plot(collect(1:1:20),plot_mean, color="red")
xlabel("Iterations")
ylabel("Objective Function")
title("Best Individual")
