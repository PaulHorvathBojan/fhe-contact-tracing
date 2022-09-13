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
encoder = CKKSEncoder(params)
encryptor = CKKSEncryptor(params, public_key, secret_key)
decryptor = CKKSDecryptor(params, secret_key)
evaluator = CKKSEvaluator(params)
post_setup = time.time() - pre_setup
print("Took " + str(post_setup) + " seconds to set up the context.")


# 0.194 is the contact threshold for integer distances given current scoring formula

def enco_encr(plain_val, encoder, encryptor, scaling_factor):
    list_val = [complex(plain_val, 0)]

    encoded = encoder.encode(values=list_val, scaling_factor=scaling_factor)

    encrypted = encryptor.encrypt(plain=encoded)

    return encrypted


def decr_deco(encr_val, encoder, decryptor):
    decrypted = decryptor.decrypt(ciphertext=encr_val)

    decoded = encoder.decode(decrypted)

    return decoded


pre_test_time = time.time()
max_diff = 0
for i in range(1000000):
    encenc = enco_encr(plain_val=i,
                       encoder=encoder,
                       encryptor=encryptor,
                       scaling_factor=SCALING_FACTOR)
    decdec = decr_deco(encr_val=encenc,
                       encoder=encoder,
                       decryptor=decryptor)
    eedd_val = decdec[0].real
    curr_diff = abs(i - eedd_val)
    if curr_diff > max_diff:
        max_diff = curr_diff
post_test_time = time.time()
test_time = post_test_time - pre_test_time
print("It took " + str(test_time) + " to encode-encrypt-decrypt-decode a million numbers.")
print("diff: " + str(max_diff))
evaluator.mul
