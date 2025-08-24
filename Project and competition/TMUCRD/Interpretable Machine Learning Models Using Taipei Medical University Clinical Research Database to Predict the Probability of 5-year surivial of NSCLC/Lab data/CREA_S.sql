WITH
TEMP_SHH AS (
	SELECT * FROM v_exp_item_S
	WHERE R_ITEM_NAME LIKE 'CRE%' 
	)
select DISTINCT
	t3.ID_no, t.CHR_NO, t.FEE_NO, t1.fee_code, t2.tube_no,
	T2.R_DATE, T2.R_TIME, t4.r_item_name, t2.value 
INTO S_OPD_LAB_CREA
from v_opd_exper_s t
inner join v_opd_fee_s t1
on t.FEE_NO = t1.fee_no
and t.seq_no = t1.SEQ_NO
inner join v_labresult_s t2
on t.FEE_NO = t2.FEE_NO
and t.TUBE_NO = t2.tube_no
and t.ITEM_NO = t2.O_ITEM
inner join v_chr_basic_s as t3
on t.chr_no = t3.CHR_NO
inner join temp_shh t4 
on t2.R_item = t4.r_item
where t1.FEE_CODE in (
select fee_code
from v_fee_basic_s
where INS_FEE_CODE in ('09015C')
)

WITH
TEMP_SHH AS (
	SELECT * FROM v_exp_item_S
	WHERE R_ITEM_NAME LIKE 'CRE%' 
	)
select DISTINCT
	t3.ID_no, t.CHR_NO, t.FEE_NO, t1.fee_code, t2.tube_no,
	T2.R_DATE, T2.R_TIME, t4.r_item_name, t2.value 
INTO S_IPD_LAB_CREA
from v_opd_exper_s t
inner join v_Ipd_fee_s t1
on t.FEE_NO = t1.fee_no
and t.seq_no = t1.SEQ_NO2
inner join v_labresult_s t2
on t.FEE_NO = t2.FEE_NO
and t.TUBE_NO = t2.tube_no
and t.ITEM_NO = t2.O_ITEM
inner join v_chr_basic_s as t3
on t.chr_no = t3.CHR_NO
inner join temp_shh t4 
on t2.R_item = t4.r_item
where t1.FEE_CODE in (
select fee_code
from v_fee_basic_s
where INS_FEE_CODE in ('09015C')
)

WITH COMBINED_CREA AS (
    SELECT ID_no, CHR_NO, FEE_NO, fee_code, tube_no, R_DATE, R_TIME, r_item_name, value
    FROM S_OPD_LAB_CREA
    UNION ALL
    SELECT ID_no, CHR_NO, FEE_NO, fee_code, tube_no, R_DATE, R_TIME, r_item_name, value
    FROM S_IPD_LAB_CREA
),
RANKED_CREA AS (
    SELECT DISTINCT *,
        ROW_NUMBER() OVER (
            PARTITION BY FEE_NO 
            ORDER BY R_DATE, R_TIME
        ) as rn
    FROM COMBINED_CREA
    WHERE ISNUMERIC(LEFT(value, CHARINDEX(' ', value + ' ') - 1)) = 1
)
SELECT DISTINCT
    FEE_NO, ID_NO,
	LEFT(value, CHARINDEX(' ', value + ' ') - 1) AS CREA
INTO FINAL_S_LAB_CREA
FROM RANKED_CREA
WHERE rn = 1