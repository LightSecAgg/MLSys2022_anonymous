import numpy as np
import torch
import copy


def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)


def divmod(_num, _den, _p):
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = np.int64(1)
    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)    
    return accum


def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    m, d = np.shape(X)
    U = gen_Lagrange_coeffs(beta_s, alpha_s, p).astype("int64")
    X_LCC = np.zeros((len(beta_s), d), dtype="int64")
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    return np.mod(X_LCC, p)


def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points
    beta_s = target_points
    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
    f_recon = np.mod((U_dec).dot(f_eval), p)

    return f_recon


def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=0):
    if is_K1 == 1:
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    U = np.zeros((num_alpha, len(beta_s)), dtype="int64")

    w = np.zeros((len(beta_s)), dtype="int64")
    for j in range(len(beta_s)):
        cur_beta = beta_s[j]
        den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
        w[j] = den

    l = np.zeros((num_alpha), dtype="int64")
    for i in range(num_alpha):
        l[i] = PI([alpha_s[i] - o for o in beta_s], p)

    for j in range(len(beta_s)):
        for i in range(num_alpha):
            den = np.mod(np.mod(alpha_s[i] - beta_s[j], p) * w[j], p)
            U[i][j] = divmod(l[i], den, p)
    return U.astype("int64")


def model_masking(weights_finite, dimensions, local_mask, prime_number):
    pos = 0
    for i, k in enumerate(weights_finite):
        tmp = weights_finite[k]
        cur_shape = tmp.shape
        d = dimensions[i]
        cur_mask = local_mask[pos : pos + d, :]
        cur_mask = np.reshape(cur_mask, cur_shape)
        weights_finite[k] += cur_mask
        weights_finite[k] = np.mod(weights_finite[k], prime_number)
        pos += d
    return weights_finite


def mask_encoding(
    total_dimension, num_clients, targeted_number_active_clients, privacy_guarantee, prime_number, local_mask
):
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = prime_number

    beta_s = np.array(range(N)) + 1
    alpha_s = np.array(range(U)) + (N+1)

    n_i = np.random.randint(p, size=(T * d // (U - T), 1))
    LCC_in = np.concatenate([local_mask, n_i], axis=0)
    LCC_in = np.reshape(LCC_in, (U, d // (U - T)))
    encoded_mask_set = LCC_encoding_with_points(LCC_in, alpha_s, beta_s, p).astype("int64")

    return encoded_mask_set


def compute_aggregate_encoded_mask(encoded_mask_dict, p, active_clients):
    aggregate_encoded_mask = np.zeros((np.shape(encoded_mask_dict[0])))
    for client_id in active_clients:
        aggregate_encoded_mask += encoded_mask_dict[client_id]
    aggregate_encoded_mask = np.mod(aggregate_encoded_mask, p).astype("int")
    return aggregate_encoded_mask


def aggregate_models_in_finite(weights_finite, prime_number):
    '''
    weights_finite : array of state_dict()
    prime_number   : size of the finite field
    '''
    w_sum = copy.deepcopy(weights_finite[0])

    for key in w_sum.keys():

        for i in range(1, len(weights_finite)):
            w_sum[key] += weights_finite[i][key]
            w_sum[key] = np.mod(w_sum[key], prime_number)

    return w_sum