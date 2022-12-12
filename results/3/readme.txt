!!!!!!!
NO INFECTEDS IN THRESHOLD
!!!!!!!
TICK_COUNT = 10

# Mobility parameters
POPSIZE = 20  # 7.5k/km**3
TOTAL_AREA_SIZES = (100, 100)
AREA_SIDES = (50, 50)

# Network parameters (contact threshold not used in current version)
RISK_THRESHOLD = 10  # more than RISK_THRESHOLD contact => at risk
CONTACT_THRESHOLD = 2  # less than CONTACT_THRESHOLD distance away => in contact
MO_COUNT = 2

# Encryption parameters
POLYNOMIAL_DEGREE = 256
CIPHERTEXT_MODULUS = 1 << 744
BIG_MODULUS = 1 << 930
SCALING_FACTOR = 1 << 49

# Infection param
INFECTED_COUNT = 5