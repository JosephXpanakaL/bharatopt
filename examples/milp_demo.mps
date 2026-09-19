NAME          BHARATOPT-MILP
ROWS
 N  COST
 L  BUDGET
 G  COVER
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    UNIT_A    COST       -7.0   BUDGET      1.0
    UNIT_A    COVER       1.0
    UNIT_B    COST       -6.0   BUDGET      1.0
    UNIT_B    COVER       1.0
    UNIT_C    COST       -5.0   BUDGET      1.0
    UNIT_C
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1      BUDGET      2.0   COVER       1.0
BOUNDS
 BV BND1      UNIT_A
 BV BND1      UNIT_B
 BV BND1      UNIT_C
ENDATA
