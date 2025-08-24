WITH
TEMP_TMU AS (
	SELECT * FROM v_exp_item_T
	WHERE R_ITEM_NAME LIKE 'alb%' 
	)
select DISTINCT
	t3.ID_no, t.CHR_NO, t.FEE_NO, t1.fee_code, t2.tube_no,
	T2.R_DATE, T2.R_TIME, t4.r_item_name, t2.value 
INTO T_OPD_LAB_ALB
from v_opd_exper_T t
inner join v_opd_fee_T t1
on t.FEE_NO = t1.fee_no
and t.seq_no = t1.SEQ_NO
inner join v_labresult_T t2
on t.FEE_NO = t2.FEE_NO
and t.TUBE_NO = t2.tube_no
and t.ITEM_NO = t2.O_ITEM
inner join v_chr_basic_T as t3
on t.chr_no = t3.CHR_NO
inner join temp_TMU t4 
on t2.R_item = t4.r_item
where t1.FEE_CODE in (
select fee_code
from v_fee_basic_T
where INS_FEE_CODE in ('09038C')
)

WITH
TEMP_TMU AS (
	SELECT * FROM v_exp_item_T
	WHERE R_ITEM_NAME LIKE 'alb%' 
	)
select DISTINCT
	t3.ID_no, t.CHR_NO, t.FEE_NO, t1.fee_code, t2.tube_no,
	T2.R_DATE, T2.R_TIME, t4.r_item_name, t2.value 
INTO T_IPD_LAB_ALB
from v_opd_exper_T t
inner join v_Ipd_fee_T t1
on t.FEE_NO = t1.fee_no
and t.seq_no = t1.SEQ_NO2
inner join v_labresult_T t2
on t.FEE_NO = t2.FEE_NO
and t.TUBE_NO = t2.tube_no
and t.ITEM_NO = t2.O_ITEM
inner join v_chr_basic_T as t3
on t.chr_no = t3.CHR_NO
inner join temp_TMU t4 
on t2.R_item = t4.r_item
where t1.FEE_CODE in (
select fee_code
from v_fee_basic_T
where INS_FEE_CODE in ('09038C')
)

WITH COMBINED_ALB AS (
    SELECT ID_no, CHR_NO, FEE_NO, fee_code, tube_no, R_DATE, R_TIME, r_item_name, value
    FROM T_OPD_LAB_ALB
    UNION ALL
    SELECT ID_no, CHR_NO, FEE_NO, fee_code, tube_no, R_DATE, R_TIME, r_item_name, value
    FROM T_IPD_LAB_ALB
),
RANKED_ALB AS (
    SELECT DISTINCT *,
        ROW_NUMBER() OVER (
            PARTITION BY FEE_NO 
            ORDER BY R_DATE, R_TIME
        ) as rn
    FROM COMBINED_ALB
    WHERE ISNUMERIC(LEFT(value, CHARINDEX(' ', value + ' ') - 1)) = 1
)
SELECT DISTINCT
    FEE_NO, ID_NO,
	LEFT(value, CHARINDEX(' ', value + ' ') - 1) AS ALB
INTO FINAL_T_LAB_ALB
FROM RANKED_ALB
WHERE rn = 1