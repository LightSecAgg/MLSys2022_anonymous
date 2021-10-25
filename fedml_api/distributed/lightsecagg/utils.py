import logging
import os
import copy

import numpy as np
import torch


def model_dimension(weights):
    logging.info("Get model dimension")
    dimensions = []
    for k in weights.keys():
        tmp = weights[k].cpu().detach().numpy()
        cur_shape = tmp.shape
        _d = int(np.prod(cur_shape))
        dimensions.append(_d)
    total_dimension = sum(dimensions)
    logging.info("Dimension of model d is %d." % total_dimension)
    return dimensions, total_dimension


def transform_tensor_to_finite(model_params, p, q_bits):
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        tmp_finite = my_q(tmp, q_bits, p)
        model_params[k] = tmp_finite
    return model_params


def transform_finite_to_tensor(model_params, p, q_bits):
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        tmp_real = my_q_inv(tmp, q_bits, p)
        """
        Handle two types of numpy variables, array and float
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] 
        DEBUG tmp_real = [4.20669909e+07 3.95378213e+08 3.20326493e+08 2.69357568e+08
         1.50241777e+08 4.17396229e+08 4.70202170e+07 2.97959914e+08
         3.03893524e+08 5.43255322e+07 4.22816876e+08 3.57158791e+06
         1.56766933e+08 2.36449153e+08 6.66361822e+07 1.66616215e+08]
        0 - Wed, 13 Oct 2021 07:50:59 utils.py[line:33] DEBUG tmp_real = 256812209.4375
        """
        # logging.debug("tmp_real = {}".format(tmp_real))
        tmp_real = torch.Tensor([tmp_real]) if isinstance(tmp_real, np.floating) else torch.Tensor(tmp_real)
        model_params[k] = tmp_real
    return model_params


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))


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
    accum = 1
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


def my_q(X, q_bit, p):
    X_int = np.round(X * (2 ** q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int)) / 2
    out = X_int + p * is_negative
    return out.astype("int64")


def my_q_inv(X_q, q_bit, p):
    flag = X_q - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q = X_q - p * is_negative
    return X_q.astype(float) / (2 ** q_bit)