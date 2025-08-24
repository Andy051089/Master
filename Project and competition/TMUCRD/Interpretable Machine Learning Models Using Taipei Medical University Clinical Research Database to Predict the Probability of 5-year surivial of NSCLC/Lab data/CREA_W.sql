--W_CREA門診單
select DISTINCT t3.ID_NO, t.chr_no, t.FEE_NO, t2.REPORT_DATE,
		T2.REPORT_TIME,	t.GROUP_CODE, t.TUBE_NUMBER, t2.EXPER_CODE,
		t2.EXPER_DATA2
INTO W_OPD_LAB_CREA
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
where INS_FEE_CODE in ('09015C')
) 
and t2.EXPER_CLASS like '01'
and t2.exper_code like 'CREB'

select *
from v_exper_sign_w
where exper_code like 'CRE%'


--住院單
select DISTINCT t3.ID_NO, t.chr_no, t.FEE_NO, t2.REPORT_DATE,
		T2.REPORT_TIME,	t.GROUP_CODE, t.TUBE_NUMBER, t2.EXPER_CODE,
		t2.EXPER_DATA2
INTO W_IPD_LAB_CREA
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
where INS_FEE_CODE in ('09015C')
) 
and t2.EXPER_CLASS like '01'
and t2.exper_code like 'CREB'

WITH COMBINED_CREA AS (
    SELECT FEE_NO, EXPER_DATA2, REPORT_DATE, REPORT_TIME,ID_NO
    FROM W_OPD_LAB_CREA
    UNION ALL
    SELECT FEE_NO, EXPER_DATA2, REPORT_DATE, REPORT_TIME, ID_NO
    FROM W_IPD_LAB_CREA
),
RANKED_CREA AS (
    SELECT DISTINCT*,
        ROW_NUMBER() OVER (
            PARTITION BY FEE_NO 
            ORDER BY REPORT_DATE, REPORT_TIME
        ) as rn
    FROM COMBINED_CREA
    WHERE ISNUMERIC(LEFT(EXPER_DATA2, CHARINDEX(' ', EXPER_DATA2 + ' ') - 1)) = 1
)
SELECT DISTINCT
    FEE_NO,ID_NO,
    EXPER_DATA2 AS CREA
INTO FINAL_W_LAB_CREA
FROM RANKED_CREA
WHERE rn = 1