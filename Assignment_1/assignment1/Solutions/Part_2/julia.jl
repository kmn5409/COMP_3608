using JuMP, GLPK
m = Model(GLPK.Optimizer)
 

#Declaring
@variable(m, x1, Bin)
@variable(m, x2, Bin)
@variable(m, x3, Bin)
@variable(m, x4, Bin)
@variable(m, x5, Bin)
@variable(m, x6, Bin)
@variable(m, x7, Bin)

#Setting objective
@objective(m, Min, x1+x2+x3+x4+x5+x6+x7)

@constraint(m, constraint1, x1+x3 >=1)
@constraint(m, constraint2, x1+x2 >=1)
@constraint(m, constraint3, x2 >=1)
@constraint(m, constraint4, x4 >=1)
@constraint(m, constraint5, x2+x6 >=1)
@constraint(m, constraint6, x4+x5 >=1)
@constraint(m, constraint7, x3+x5+x6 >=1)
@constraint(m, constraint8, x4 >=1)
@constraint(m, constraint9, x3+x4+x5 >=1)
@constraint(m, constraint10, x3+x6 >=1)
@constraint(m, constraint11, x5 >=1)
@constraint(m, constraint12, x6+x7 >=1)
@constraint(m, constraint13, x7 >=1)
@constraint(m, constraint14, x6+x7 >=1)
@constraint(m, constraint15, x7 >=1)

JuMP.optimize!(m)

# Printing the optimal solutions obtained
println("Optimal Solutions:")
println("Tower 1 = ", value(x1))
println("Tower 2 = ", value(x2))
println("Tower 3 = ", value(x3))
println("Tower 4 = ", value(x4))
println("Tower 5 = ", value(x5))
println("Tower 6 = ", value(x6))
println("Tower 7 = ", value(x7))

