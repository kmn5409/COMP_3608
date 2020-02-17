%Hello

var 0..1: x_1;
var 0..1: x_2;
var 0..1: x_3;
var 0..1: x_4;
var 0..1: x_5;
var 0..1: x_6;
var 0..1: x_7;
constraint x_1 + x_2 + x_3                           >= 1;
constraint x_1 + x_2 + x_3                           >= 1;
constraint             x_3                           >= 1;
constraint                              x_6 + x_7    >= 1; 
constraint                  x_4                      >= 1;
constraint x_1 +            x_4 + x_5 + x_6          >=1;
constraint x_1 + x_2 +      x_4 + x_5                >=1;
solve minimize x_1 + x_2 + x_3 + x_4 + x_5 + x_6;
