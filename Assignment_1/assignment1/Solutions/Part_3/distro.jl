using JuMP, GLPK
m = Model(GLPK.Optimizer)

@variable(m, x_1_1, Int)
@variable(m, x_1_2, Int)
@variable(m, x_1_3, Int)
@variable(m, x_1_4, Int)
@variable(m, x_2_1, Int)
@variable(m, x_2_2, Int)
@variable(m, x_2_3, Int)
@variable(m, x_2_4, Int)
@variable(m, x_3_1, Int)
@variable(m, x_3_2, Int)
@variable(m, x_3_3, Int)
@variable(m, x_3_4, Int)


@objective(m, Min, x_1_1 * 464 + x_2_1 * 352 + x_3_1 * 995 +
               x_1_2 * 513 + x_2_2 * 416 + x_3_2 * 682 +
               x_1_3 * 654 + x_2_3 * 690 + x_3_3 * 388 +
               x_1_4 * 867 + x_2_4 * 791 + x_3_4 * 685)
#Supply
@constraint(m, constraint1, x_1_1 + x_1_2 + x_1_3 + x_1_4 == 75)
@constraint(m, constraint2, x_2_1 + x_2_2 + x_2_3 + x_2_4 == 125)
@constraint(m, constraint3, x_3_1 + x_3_2 + x_3_3 + x_3_4 == 100)


#Demand
@constraint(m, constraint4, x_1_1 + x_2_1 + x_3_1 >= 80)
@constraint(m, constraint5, x_1_2 + x_2_2 + x_3_2 >= 65)
@constraint(m, constraint6, x_1_3 + x_2_3 + x_3_3 >= 70)
@constraint(m, constraint7, x_1_4 + x_2_4 + x_3_4 >= 85)

@constraint(m, constraint8, x_1_1 >= 0)
@constraint(m, constraint9, x_1_2 >= 0)
@constraint(m, constraint10, x_1_3 >= 0)
@constraint(m, constraint11, x_1_4 >= 0)

@constraint(m, constraint12, x_2_1 >= 0)
@constraint(m, constraint13, x_2_2 >= 0)
@constraint(m, constraint14, x_2_3 >= 0)
@constraint(m, constraint15, x_2_4 >= 0)

@constraint(m, constraint16, x_3_1 >= 0)
@constraint(m, constraint17, x_3_2 >= 0)
@constraint(m, constraint18, x_3_3 >= 0)
@constraint(m, constraint19, x_3_4 >= 0)

JuMP.optimize!(m)

println("Cannery 1 Warehouse 1 = ",value(x_1_1));
println("Cannery 1 Warehouse 2 = ",value(x_1_2));
println("Cannery 1 Warehouse 3 = ",value(x_1_3));
println("Cannery 1 Warehouse 4 = ",value(x_1_4));
println("Cannery 2 Warehouse 1 = ",value(x_2_1));
println("Cannery 2 Warehouse 2 = ",value(x_2_2));
println("Cannery 2 Warehouse 3 = ",value(x_2_3));
println("Cannery 2 Warehouse 4 = ",value(x_2_4));
println("Cannery 3 Warehouse 1 = ",value(x_3_1));
println("Cannery 3 Warehouse 2 = ",value(x_3_2));
println("Cannery 3 Warehouse 3 = ",value(x_3_3));
println("Cannery 3 Warehouse 4 = ",value(x_3_4));

#=
Cannery 1 Warehouse 1 = 0.0
Cannery 1 Warehouse 2 = 20.0
Cannery 1 Warehouse 3 = 0.0
Cannery 1 Warehouse 4 = 55.0
Cannery 2 Warehouse 1 = 80.0
Cannery 2 Warehouse 2 = 45.0
Cannery 2 Warehouse 3 = 0.0
Cannery 2 Warehouse 4 = 0.0
Cannery 3 Warehouse 1 = 0.0
Cannery 3 Warehouse 2 = 0.0
Cannery 3 Warehouse 3 = 70.0
Cannery 3 Warehouse 4 = 30.0
=#

