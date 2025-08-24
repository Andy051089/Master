/*建立列連表後做卡方
H0 : 無關(獨立)
HA : 有關
所有格子都不能有小於1、80%的格子期望值要大於5，不然用fisher檢定，方式一樣會在裡面顯示*/
proc freq data = cvd2;
table agegroup * CVD  / chisq expected;
label CVD = '有無心血管疾病' agegroup = '年齡群';
format CVD CVDfmt.;
run;
/*expected顯示期望值
當期望與實際差愈大，拒絕虛無
P>0.05不拒絕虛無假設:在顯著水準為0.05下，我們沒有充分證據顯示(暴露)與(疾病)有相關性
P<0.05拒絕虛無假設:在顯著水準為0.05下，我們有充分證據顯示(暴露)與(疾病)有相關性*/
