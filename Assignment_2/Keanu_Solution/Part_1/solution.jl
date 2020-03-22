using JuMP, GLPK
m = Model(GLPK.Optimizer)

@variable(m, x, Int)
@variable(m, y, Int)

@objective(m, Min, (1-x)^2 + 100(y-x^2)^2)

@constraint(m, -1.5 <= x <= 1.5)
@constraint(m, -1.5 <= y <= 1.5)
@constraint(m, x^2 + y^2 <= 2)

JuMP.optimize!(m)

println("x = ", value(x))
println("y = ", value(y))
