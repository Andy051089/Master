proc reg data = cvd1;
model sbp = agegroup ;
run;

/*設定Dummy variable*/
proc glm data=cvd1;
class agegroup(ref="年輕人");
model sbp= agegroup / solution;
run;

proc genmod data=cvd1;
class agegroup(ref="年輕人");
model sbp = agegroup ;
run;
