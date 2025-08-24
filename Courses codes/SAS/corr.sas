ods graphics on;
proc corr pearson spearman data=cvd nomiss plots(maxpoints=700000)=matrix(histogram);
var waist sbp;
run;
ods graphics off;
