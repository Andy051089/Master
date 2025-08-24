/*改變colname或表格內顯示的內容名子*/
/*在進行format前邀先設定好format*/
proc format;
value CVDfmt 0 = '無心血管疾病' 1 = '有心血管疾病';
value BPfmt 0 = '無高血壓' 1 = '有高血壓';
run;

/*直接改變資料在data中的顯示*/
data case; 
input Sub_Num $ 1-2 Sex $ 4 mean_sd median_IQR;
label mean_sd = 'mean±sd' median_IQR = 'median(IQR)'; 
format agegroup agegroupfmt.;
list;
datalines; 
07 M 85 89 A;

/*不改變資料，只在列連表中改*/
proc freq data = cvd2;
table agegroup*smoking; 
label smoking = '有無抽菸' agegroup = '年齡分組';
format smoking smokingfmt. agegroup agegroupfmt.;
run;
/*table為建立列連表，Row*Col
Table只有一個變數時為人數比例*/
