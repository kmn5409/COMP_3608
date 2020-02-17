var int: c1w1; %cannery 1 to warehouse 1
var int: c1w2; %cannery 1 to warehouse 2
var int: c1w3; %cannery 1 to warehouse 3
var int: c1w4; %cannery 1 to warehouse 4
var int: c2w1; %cannery 2 to warehouse 1
var int: c2w2; %cannery 2 to warehouse 2
var int: c2w3; %cannery 2 to warehouse 3
var int: c2w4; %cannery 2 to warehouse 4
var int: c3w1; %cannery 3 to warehouse 1
var int: c3w2; %cannery 3 to warehouse 2
var int: c3w3; %cannery 3 to warehouse 3
var int: c3w4; %cannery 3 to warehouse 4


constraint c1w1+c1w2+c1w3+c1w4<=75;
constraint c2w1+c2w2+c2w3+c2w4<=125;
constraint c3w1+c3w2+c3w3+c3w4<=100;

constraint c1w1+c2w1+c3w1>=80;
constraint c1w2+c2w2+c3w2>=65;
constraint c1w3+c2w3+c3w3>=70;
constraint c1w4+c2w4+c3w4>=85;

constraint c1w1>=0;
constraint c1w2>=0;
constraint c1w3>=0;
constraint c1w4>=0;

constraint c2w1>=0;
constraint c2w2>=0;
constraint c2w3>=0;
constraint c2w4>=0;


constraint c3w1>=0;
constraint c3w2>=0;
constraint c3w3>=0;
constraint c3w4>=0;


solve minimize 464*c1w1 + 513*c1w2 + 654*c1w3 + 867*c1w4 
+ 352*c2w1 + 416*c2w2 + 690*c2w3 + 791*c2w4 
+ 995*c3w1 + 682*c3w2 + 388*c3w3 + 685*c3w4;

