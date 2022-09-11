import time

from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

# Hyperparam initialisation
DEGREE = 2
CIPH_MODULUS = 1 << 600
BIG_MODULUS = 1 << 1200
SCALING_FACTOR = 1 << 30
NUM_TAYLOR_EXP_ITERATIONS = 12

# Context setup
pre_setup = time.time()
params = CKKSParameters(poly_degree=DEGREE,
                        ciph_modulus=CIPH_MODULUS,
                        big_modulus=BIG_MODULUS,
                        scaling_factor=SCALING_FACTOR)
key_generator = CKKSKeyGenerator(params)
public_key = key_generator.public_key
secret_key = key_generator.secret_key
relin_key = key_generator.relin_key

skey_generator = CKKSKeyGenerator(params)
spublic_key = skey_generator.public_key
ssecret_key = skey_generator.secret_key
srelin_key = skey_generator.relin_key

print(public_key == spublic_key)
print(secret_key == ssecret_key)
print(relin_key == srelin_key)