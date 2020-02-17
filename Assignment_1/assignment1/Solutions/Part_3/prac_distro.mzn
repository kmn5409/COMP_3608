var int: x_1_1;
var int: x_1_2;
var int: x_1_3;
var int: x_2_1;
var int: x_2_2;
var int: x_2_3;

%supply
constraint x_1_1 + x_1_2 + x_1_3 = 40;
constraint x_2_1 + x_2_2 + x_2_3 = 20;

%demand
constraint x_1_1 + x_2_1 = 25;
constraint x_1_2 + x_2_2 = 10;
constraint x_1_3 + x_2_3 = 25;

constraint x_1_1 >= 0;
constraint x_1_2 >= 0;
constraint x_1_3 >= 0;
constraint x_2_1 >= 0;
constraint x_2_2 >= 0;


%constraint w_1 * 995 + w_2 * 682 + w_3 * 388 + w_4 * 685 >= 100; 


solve minimize x_1_1 * 550 + x_2_1 * 350 + x_1_2 * 300 + x_2_2 * 300 + x_1_3 * 400 + x_2_3 * 100;
