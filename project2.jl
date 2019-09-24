#The code used in the submission is
#simple_1, 2, 3, secret_1: Newton's method with finite difference gradient
#and Hessian and quadratic penalty
#secret_2: ADAM with quadratic penalty
#Additional code is included at the end for simulated annealing and gradient desc

#path = []

function grad_findiff(f, x, calls; h=0.00001)
    grad=zeros(length(x))
    for i in 1:length(x)
        step = zeros(length(x))
        step[i] = h
        grad[i] = (f(x + step) - f(x - step))/(2.0*h)
        calls+= 2
    end
    return grad, calls
end

function hess_findiff(f, x, calls; h=0.00001)
    hess = zeros(length(x),length(x))
    for i in 1:length(x)
        for j in 1:length(x)
            if i == j
                step = zeros(length(x))
                step[i] = h
                hess[i,j] = (f(x + step)-2.0*f(x)+f(x-step))/(h^2)
                calls += 3
            else
                step1, step2 = zeros(length(x)), zeros(length(x))
                step1[i] = h
                step2[j] = h
                hess[i,j]=(f(x+step1+step2)-f(x+step1-step2)-f(x-step1+step2)+f(x-step1-step2))/(4.0*h^2)
                calls += 4
            end
        end
    end
    return hess, calls
end

function optimize(f, g, c, x0, n, prob)
    println("commence optimizationnnnnNNNNNNN")
    calls = 0
    calls_c = 0
    x, x_best = x0, x0
    p_quad(x) = sum(max.(c(x),0).^2)
    ρ1 = 1.0
    P(x) = f(x) + ρ1*p_quad(x)
    #push!(path,x0)
    y_best = Inf
    feas = []
################### secret_2 ###################
    if prob == "secret_2" #adam
        m = 1000
        ϵ = 1e-8 #small value
        x_new = Inf*ones(length(x))
        α = .5 #learning rate
        γv = 0.9 #decay
        γs = 0.999 #decay
        k = 0 #step counter
        v = zeros(length(x)) #first moment est
        s = zeros(length(x)) #second moment est
        while calls < n -m && calls_c < n - m #&& abs((norm(x_new)-norm(x)) / norm(x)) >= 0.05
            grad, calls = grad_findiff(P, x, calls)
            calls_c += 2
            println("grad calls: $calls")
            #println(calls)
            #grad, calls = 2*sum(max.(c(x),0)).*g(x)
            v[:] = γv*v + (1-γv)*grad
            s[:] = γs*s + (1-γs)*grad.*grad
            k += 1
            v_hat = v ./ (1 - γv^k)
            s_hat = s ./ (1 - γs^k)
            x_new = x - α*v_hat ./ (ϵ + sqrt.(s_hat))
            calls_c += 1
            if all(c(x_new) .<= 0)
                println("C CALLS: $calls_c")
                println("found a feasible: $x_new")
                push!(feas,x_new)
            end
            x = x_new
            #push!(path,[x,f(x)])
            ρ1 *= 10
        end
            #println("next iteration")
        if length(feas) > 0
            for i in feas
                y, calls = f(i), calls + 1
                if y < y_best

                    x_best, y_best = i, y
                    println("xbest: $x_best")
                    println(calls)
                end
                if calls == n-1
                    println("ran out of calls")
                    break
                end
            #println("x_best: $i, y_best: $y_best")
            end
            return x_best
        else
            println("no feasible points")
        end
        return
###################### non secret_2 ################
    else
        if prob in ["simple_1","simple_2"]
            m = 2
        elseif prob == "simple_3"
            m = 28
        else m = 16
        end
        while calls < n-m
            grad, calls = grad_findiff(P, x, calls)
            hess, calls = hess_findiff(P, x, calls)
            try
                x = x - inv(hess)*grad
            catch
                break
            end
            ρ1*=2.0
            #push!(path,x)
        end
    end
    return x #, calls, feasible_xs
end

################################################################
############## simulated annealing #############################
################################################################

# function conrana_update(v,a,c,ns)
#     for i in 1:length(v)
#         ai, ci = a[i], c[i]
#         if ai > 0.6ns
#             v[i] *= (1+ ci*(ai/ns - 0.6)/0.4)
#         elseif ai < 0.4ns
#             v[i] /= (1 + ci*(0.4-ai/ns)/0.4)
#         end
#     end
#     return v
# end
#
# if prob == "secret_2" #simulated annealing
#     k_max = n
#     t = #starting temp
#     v = ones(length(x)) #starting step vector
#     a = zeros(length(x)) #vector of number of accepted steps in each coordinate direction
#     csa = 2*ones(length(x)) #vector of step scaling factors for each coordinate direction
#     ns = 20 #number of cycles before running the step size adjustment
#     y = P(x)
#     x_best, y_best = x,y
#     for k in 1:k_max
#         r = (2*rand()-1) #uniform random [-1,1]
#         xp = x + r*corana_update(v,a,csa,ns) #*e??
#         y = P(xp)
#         Δy = yp - y
#         if Δy <= 0 || rand() < exp(-Δy/t(k))
#             x, y = xp, yp
#         end
#         if yp < y_best
#             x_best, y_best = xp, yp
#             a = a + ones(length(x))
#         end
#     end
#     return x_best #, y_best, calls

################################################################
##############find a feasible point first using gradient descent
################################################################

# if prob == "secret_2"
#     x, x_best = x0, x0
#     push!(path_c,x0)
#     m = 24
#     while calls_c < n-m
#         c_eval, calls_c = constraint(x), calls_c+1
#         println("function eval c: $calls_c")
#         c_grad, calls_c = grad_findiff(constraint, x, calls_c)
#         println("gradient eval c: $calls_c")
#         try
#             c_hess, calls_c = hess_findiff(constraint, x, calls_c)
#             println("hess eval c: $calls_c")
#         catch
#             println("error with c_hessian calc")
#             break
#         end
#         try
#             if abs((norm(x-inv(c_hess)*c_grad) - norm(x)) / norm(x)) >= 0.05
#                 x = x - inv(c_hess)*c_grad
#                 y, calls_c = constraint(x), calls_c+1
#                 push!(path_c, x)
#                 println("main c: $calls_c")
#             else
#                 #println("x: $x")
#                 break
#             end
#         catch
#             println("error with inverting c_hessian")
#             break
#         end
#             #println("$x, $y")
#     end
#     println("finding feasible points")
#     xs = linspace(0.5*x[1],1.5*x[1],500)
#     ys = linspace(0.5*x[2],1.5*x[2],500)
#     for i in xs
#         cs, calls_c = c([i,x[2]]), calls_c + 1
#         println("random search x c: $calls_c")
#         if all(cs .<= 0)
#             push!(feasible_xs,[i,x[2]])
#         end
#     end
#     for j in ys
#         cs, calls_c = c([x[1],j]), calls_c+ 1
#         println("random search y c: $calls_c")
#         if all(cs .<= 0)
#             push!(feasible_xs,[x[1],j])
#         end
#     end
#     println("feasible x's")
#     for k in feasible_xs
#         println(k)
#     end

############################ testing and plotting ######################
# f_one(x) = -x[1]*x[2]
# c_one(x) = [x[1]+x[2]^2-1,-x[1]-x[2]]
# g_one(x) = [-x[2],-x[1]]
#
# f_two(x) = 100(x[1]-x[2]^2)^2 + (1-x[2])^2
# c_two(x) = [(x[1]-1)^3-x[2]+1,x[1]+x[2]-2]
#
# f_three(x) = x[1] - 2*x[2] + x[3]
# c_three(x) = [x[1]^2+x[2]^2+x[3]^2-1]

#x_best = optimize(f_three, g_one, c_three, [2,0,0], 2000, "simple_3")

# #-1,2
# #1,1 (-2,-2)
# #2,0
# x1=[]
# x2=[]
# y=[]
# for i in 1:length(path)
#     push!(x1,path[i][1])
#     push!(x2,path[i][2])
#     push!(y,f_one([path[i][1],path[i][2]]))
# end
#
# using PyPlot
# plot(collect(1:1:length(y)),y, color="blue", label="(2,0,0)")
# ylim(-10,10)
# xlabel("steps")
# ylabel("f(x)")
# xlim(0,20)
# title("Simple_3 Convergence")
# legend()
# using Distributions
#
# n = 100
# x = linspace(-4, 4, n)
# y = linspace(-4,4,n)
#
# c1x1 = []
# c1x2 = []
# c2x1 = []
# c2x2 = []
# for i in 1:n
#     push!(c1x2,x[i])#push!(c1x2,1+(x[i]-1)^3)
#     push!(c1x1,1-x[i]^2)#push!(c1x1,x[i])
#     push!(c2x2,x[i])#push!(c2x2,2-x[i])
#     push!(c2x1,-x[i])#push!(c2x1,x[i])
# end
#
# xgrid = repmat(x',n,1)
# ygrid = repmat(y,1,n)
#
# z = zeros(n,n)
#
# for i in 1:n
#     for j in 1:n
#         z[i:i,j:j] = f_one([x[i];y[j]]);
#     end
# end
# fig = figure("pyplot_contour",figsize=(10,10))
# subplot(1,1,1)
# ax = fig[:add_subplot](1,1,1)
# cp = ax[:contour](xgrid, ygrid, z, collect([1:4:16]))
# ax[:clabel](cp, inline=1, fontsize=10)
# ax[:plot](2/3,1.0/sqrt(3.0),"bx")
# ax[:plot](x1,x2, color="blue")
# ax[:plot](c1x1,c1x2, color="black",linestyle=":")
# ax[:plot](c2x1,c2x2, color="black",linestyle="--")
# ylim(-4,4)
# xlim(-4,4)
# xlabel("X 1")
# ylabel("X 2")
# title("Function 1 Contour Plot")
# tight_layout()
