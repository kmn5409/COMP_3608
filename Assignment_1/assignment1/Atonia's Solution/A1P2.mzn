% We have 7 towers covering the following neighborhoods
% Tower  Neighborhood
% 1      {1,2}
% 2      {2,3,5}
% 3      {1,7,9,10}
% 4      {4,6,8,9}
% 5      {6,7,9,11}
% 6      {5,7,10,12,14}
% 7      {12,13,14,15}
% We aim to minimize the number of towers used

var 0..1: x1; %tower location 1
var 0..1: x2; %tower location 2
var 0..1: x3; %tower location 3
var 0..1: x4; %tower location 4
var 0..1: x5; %tower location 5
var 0..1: x6; %tower location 6
var 0..1: x7; %tower location 7

constraint x1+x3>0; %neighborhood 1
constraint x1+x2>0; %neighborhood 2
constraint x2>0;    %neighborhood 3
constraint x4>0;    %neighborhood 4 and 8
constraint x2+x6>0; %neighborhood 5
constraint x4+x5>0; %neighborhood 6
constraint x3+x5+x6>0; %neighborhood 7
constraint x3+x4+x5>0; %neighborhood 9
constraint x3+x6>0; %neighborhood 10
constraint x5>0; %neighborhood 11
constraint x6+x7>0; %neighborhood 12 and 14
constraint x7>0; %neighborhood 13 and 15

solve minimize x1+x2+x3+x4+x5+x6+x7;

