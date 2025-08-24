/*H0 :無差異 HA :有差異*/
/*獨立樣本，有母數t-test(看Satterthwaite的結果)
男生與與女生的平均SBP有無差別*/
proc ttest data = case;
class gender;
var SBP;
run;
/*配對樣本有母數paired t-test
SBP及DBP兩者相差平均60*/
proc ttest data = case H0 = 60;
paired SBP * DBP;
run;
