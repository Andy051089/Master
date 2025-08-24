/*H0 :無差異 HA :有差異*/
/*獨立樣本，無母數Wilcoxon rank sum test
男生與與女生的SBP中位數有無差別*/
proc npar1way data = cvd Wilcoxon;
class CVD;
var age;
run;
/*配對樣本，無母數Wilcoxon signed rank test
SBP及DBP兩者相差中位數60*/
proc univariate data=cvd mu0 = 60;
var diff;
run;
