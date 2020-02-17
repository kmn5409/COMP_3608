var int: x_1_1;
var int: x_1_2;
var int: x_1_3;
var int: x_1_4;
var int: x_2_1;
var int: x_2_2;
var int: x_2_3;
var int: x_2_4;
var int: x_3_1;
var int: x_3_2;
var int: x_3_3;
var int: x_3_4;

%supply/output
constraint x_1_1 + x_1_2 + x_1_3 + x_1_4 = 75;
constraint x_2_1 + x_2_2 + x_2_3 + x_2_4 = 125;
constraint x_3_1 + x_3_2 + x_3_3 + x_3_4 = 100;


%demand
constraint x_1_1 + x_2_1 + x_3_1 = 80;
constraint x_1_2 + x_2_2 + x_3_2 = 65;
constraint x_1_3 + x_2_3 + x_3_3 = 70;
constraint x_1_4 + x_2_4 + x_3_4 = 85;

constraint x_1_1 >= 0;
constraint x_1_2 >= 0;
constraint x_1_3 >= 0;
constraint x_1_4 >= 0;
constraint x_2_1 >= 0;
constraint x_2_2 >= 0;
constraint x_2_3 >= 0;
constraint x_2_4 >= 0;
constraint x_3_1 >= 0;
constraint x_3_2 >= 0;
constraint x_3_3 >= 0;
constraint x_3_4 >= 0;


%constraint w_1 * 995 + w_2 * 682 + w_3 * 388 + w_4 * 685 >= 100;


solve minimize x_1_1 * 464 + x_2_1 * 352 + x_3_1 * 995 +
               x_1_2 * 513 + x_2_2 * 416 + x_3_2 * 682 +
               x_1_3 * 654 + x_2_3 * 690 + x_3_3 * 388 +
               x_1_4 * 867 + x_2_4 * 791 + x_3_4 * 685;

