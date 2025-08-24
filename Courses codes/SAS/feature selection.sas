proc glmselect data = cvd;
model waist = cvd age / selection = backward showpvalues stats = all sle=0.05 sls=0.05;
run;
/*selection = forward stepwise*/

proc reg data = cvd;
model waist = cvd age / selection = adjrsq mse aic bic cp rmse sbc;
run;

proc logistic data = cvd descending;
model cvd = waist age/selection = backward;
run;
