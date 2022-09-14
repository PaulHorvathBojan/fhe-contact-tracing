import time
import numpy as np

from pymobility.models.mobility import gauss_markov

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
encoder = CKKSEncoder(params)
encryptor = CKKSEncryptor(params, public_key, secret_key)
decryptor = CKKSDecryptor(params, secret_key)
evaluator = CKKSEvaluator(params)
post_setup = time.time() - pre_setup
print("Took " + str(post_setup) + " seconds to set up the context.")

gm1 = gauss_markov(nr_nodes=10,
                   dimensions=(10,10))

locs = next(gm1)
print(locs)

print((int(np.rint(locs[0][0])), int(np.rint(locs[0][1]))))