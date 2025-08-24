/*直接建立列連表，算McNemar’s
為前後比較或配對資料比較不一致*/
data case
input CVD smoking count;
datalines;
1 1 43 
1 0 23
0 1 56
0 0 262);
proc freq data = case;
table smoking * CVD / agree;
weight count;
run;
/*加上count、weight count*/
