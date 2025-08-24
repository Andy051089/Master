/*相對風險RR
暴露有病P1/沒暴露有病P2
暴露在ROW(上有下無)、得病COL(左有右無)
一定要照這格式*/
proc freq data = cvd order = formatted;
tables gender* CVD / chisq expected or;
run; 
/*order = formatted(照format筆畫排，多到少)
order = data(照KEY順序)
假設檢定P1 = P2或P1-P2 = 0 VS RR =1 差別在如果兩個P都很小那相減還是很小會沒感覺有差別，但算比例可能差很大
RR < 1 暴露不容易得病(保護因子)
RR = 1 暴不暴露都沒差
RR > 1 暴露因子
CI區間包含H0的 = 值，在RR中是1，就不拒絕虛無假設。*/

