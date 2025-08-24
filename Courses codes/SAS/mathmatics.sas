proc means data = df mean std median q1 p50 q3 qrange maxdec = 2;
class CVD; 
var Age Waist SBP DBP;
run;
/*IQR = qrange
Maxdec = 小數點幾位
Var要顯示的變量，不寫就全部
也可以用univariate，直接改掉means地方*/
proc summary data = df mean std median q1 p50 q3 qrange maxdec = 2 print;
class CVD; 
var Age Waist SBP DBP;
output out = summary;
run;
/*也可以summary，後面要加print*/
/*class為以CVD分組，但class分組不會呈現缺失值，By分組才會呈現缺失值，用by前一定要有sort先排序*/
proc sort data = df;
By CVD;
Run;
proc means data = df mean std median q1 p50 q3 qrange maxdec = 2;
by CVD; 
var Age Waist SBP DBP;
run;
/*然後在proc means中class改為by*/

