%Hello

var 0..1: x_1;
var 0..1: x_2;
var 0..1: x_3;
var 0..1: x_4;
var 0..1: x_5;
var 0..1: x_6;
var 0..1: x_7;

constraint x_1 +       x_3                         >= 1; %1
constraint x_1 + x_2                               >= 1; %2
constraint       x_2                               >= 1; %3
constraint                   x_4                   >= 1; %4
constraint       x_2                   + x_6       >= 1; %5
constraint                   x_4 + x_5             >= 1; %6
constraint             x_3 +       x_5 + x_6       >= 1; %7
constraint                   x_4                   >= 1; %8
constraint             x_3 + x_4 + x_5             >= 1; %9
constraint             x_3             + x_6       >= 1; %10
constraint                         x_5             >= 1; %11
constraint                               x_6 + x_7 >= 1; %12
constraint                                     x_7 >= 1; %13
constraint                               x_6 + x_7 >= 1; %14
constraint                                     x_7 >= 1; %15

solve minimize x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7;
