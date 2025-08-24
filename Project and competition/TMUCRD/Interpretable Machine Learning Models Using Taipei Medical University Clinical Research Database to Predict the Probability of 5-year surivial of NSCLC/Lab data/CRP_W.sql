--W crp門診單
select t3.ID_NO, t.chr_no, t.FEE_NO,  t2.REPORT_DATE,
		T2.REPORT_TIME, t.GROUP_CODE, t.TUBE_NUMBER, t2.EXPER_CODE,
		t2.EXPER_DATA2
--INTO W_OPD_LAB_CRP
from v_exper_order_w t
inner join v_opd_fee_w t1
on t.FEE_NO = t1.FEE_NO and t.GROUP_CODE = t1.FEE_CODE
inner join v_exper_sign_w t2
on t.FEE_no = t2.FEE_NO
and t.TUBE_NUMBER = t2.TUBE_NUMBER
and t.GROUP_CODE = t2.GROUP_CODE
inner join v_chr_basic_w t3
on t.CHR_NO = t3.CHR_NO
where t1.fee_code in (
select fee_code 
from v_fee_basic_w
where INS_FEE_CODE in ('12015C')
) 
and t2.EXPER_CLASS like '01'
and t2.exper_code like 'crp%'

select *
from v_exper_sign_w
where exper_code like 'crp%'

--住院單
select t3.ID_NO, t.chr_no, t.FEE_NO, t.EXPER_DATE,
		t.GROUP_CODE, t.TUBE_NUMBER, t2.EXPER_CODE,
		t2.EXPER_DATA2
INTO OPD_CRP
from v_exper_order_w t
inner join v_ipd_fee_w t1
on t.FEE_NO = t1.FEE_NO and t.GROUP_CODE = t1.FEE_CODE
inner join v_exper_sign_w t2
on t.FEE_no = t2.FEE_NO
and t.TUBE_NUMBER = t2.TUBE_NUMBER
and t.GROUP_CODE = t2.GROUP_CODE
inner join v_chr_basic_w t3
on t.CHR_NO = t3.CHR_NO
where t1.fee_code in (
select fee_code 
from v_fee_basic_w
where INS_FEE_CODE in ('12015C')
) 
and t2.EXPER_CLASS like '01'
and t2.exper_code like 'CRP'

SELECT *
--INTO LUNGCA_FINAL_W_CRP
FROM(
	SELECT *
	FROM OPD_CRP
	UNION ALL
	SELECT *
	FROM IPD_CRP
) AS T


SELECT FEE_NO, EXPER_DATA2, MIN(EXPER_DATE) as HB_DATE
INTO MIN_CRP
FROM (
    SELECT *
    FROM OPD_CRP
    UNION ALL
    SELECT *
    FROM IPD_CRP
) AS T
GROUP BY FEE_NO, EXPER_DATA2

SELECT FEE_NO,
      ROUND(AVG(CASE WHEN ISNUMERIC(EXPER_DATA2) = 1 
               THEN CAST(EXPER_DATA2 AS FLOAT) 
               ELSE NULL END), 2) AS AVG_CRP,
       MAX(CASE WHEN ISNUMERIC(EXPER_DATA2) = 1 
                THEN CAST(EXPER_DATA2 AS FLOAT) 
                ELSE NULL END) AS MAX_CRP,
       MIN(CASE WHEN ISNUMERIC(EXPER_DATA2) = 1 
                THEN CAST(EXPER_DATA2 AS FLOAT) 
                ELSE NULL END) AS MIN_CRP
INTO FINAL_CRP
FROM (
    SELECT *
    FROM OPD_CRP
    UNION ALL
    SELECT *
    FROM IPD_CRP
) AS T
GROUP BY FEE_NO