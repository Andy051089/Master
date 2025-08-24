/*常態檢定及畫圖*/
proc univariate data = cvd normal plot;
class CVD;
var age;
run;
/*兩者要搭配一起看*/
