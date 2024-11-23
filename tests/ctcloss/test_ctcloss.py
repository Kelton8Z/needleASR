import sys
sys.path.append('./python')
import numpy as np
import pytest
import os
import pickle
from needle import backend_ndarray as nd
import needle as ndl
from needle.ops import CTC, CTCLoss, CTCLossGradient
from needle.nn import CTCLoss as CTCLossModule

data_path = "tests/ctcloss/data"
ref_data_path = "tests/ctcloss/data/ctc_ref_data"
ndl_device = ndl.cuda() # ndl.cuda()

def test_ctc_extend_seq():
    # Get curr data
    probs = np.load(os.path.join(data_path, "X.npy"))
    targets = np.load(os.path.join(data_path, "Y.npy"))
    input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
    out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

    CTC_user = ndl.ops.CTC(blank=0)

    f_ref_S_ext = open(os.path.join(ref_data_path, "ref_S_ext.pkl"), "rb")
    f_ref_Skip_Connect = open(
        os.path.join(ref_data_path, "ref_Skip_Connect.pkl"), "rb"
    )

    ref_S_ext_ls = pickle.load(f_ref_S_ext)
    ref_Skip_Connect_ls = pickle.load(f_ref_Skip_Connect)

    print("load complete")
    _, B, _ = probs.shape
    for b in range(B):
        target = targets[b, : out_lens[b]]

        print("Begin user compute")
        user_S_ext, user_Skip_Connect = CTC_user.extend_target_with_blank(ndl.NDArray(target, device=ndl_device))
        print("End user compute")
        user_S_ext, user_Skip_Connect = (
            user_S_ext.numpy(),
            user_Skip_Connect.numpy(),
        )

        ref_S_ext = ref_S_ext_ls[b]
        ref_Skip_Connect = ref_Skip_Connect_ls[b]

        diff_S_ext = np.linalg.norm(user_S_ext - ref_S_ext)
        diff_Skip_Connect = np.linalg.norm(user_Skip_Connect - ref_Skip_Connect)

        if diff_S_ext < 1e-5:
            print(f"+++++++++S_ext pass+++++++++: {diff_S_ext}")
        else:
            print(f"ref_S_ext: {ref_S_ext}")
            print(f"user_S_ext: {user_S_ext}")
            print(f"---------S_ext fail---------: {diff_S_ext}")
        if diff_Skip_Connect < 1e-5:
            print(f"+++++++++Skip_Connect pass+++++++++: {diff_Skip_Connect}")
        else:
            print(f"ref_Skip_Connect: {ref_Skip_Connect}")
            print(f"user_Skip_Connect: {user_Skip_Connect}")
            print(f"---------Skip_Connect fail---------: {diff_Skip_Connect}")

    f_ref_S_ext.close()
    f_ref_Skip_Connect.close()

def test_ctc_forward():
    probs = np.load(os.path.join(data_path, "X.npy"))
    targets = np.load(os.path.join(data_path, "Y.npy"))
    input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
    out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))
    print("load complete")

    CTC_user = CTCLoss(blank=0)
    user_loss = CTC_user.compute(ndl.NDArray(probs, device=ndl_device), ndl.NDArray(targets, device=ndl_device), ndl.NDArray(input_lens, device=ndl_device), ndl.NDArray(out_lens, device=ndl_device))

    ref_loss = np.load(os.path.join(ref_data_path, "ref_loss.npy"))

    diff = np.linalg.norm(user_loss.numpy() - ref_loss)

    if diff < 1e-5:
        print("=========Pass CTC forward===============")
    else:
        print("=========Fail CTC forward===============")

    return True

def test_ctc_backward():
    # Get curr data
    probs = np.load(os.path.join(data_path, "X.npy"))
    targets = np.load(os.path.join(data_path, "Y.npy"))
    input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
    out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

    probs = ndl.Tensor(np.array(probs), device=ndl_device)
    targets = ndl.Tensor(np.array(targets), device=ndl_device, requires_grad=False)
    input_lens = ndl.Tensor(np.array(input_lens), device=ndl_device, requires_grad=False)
    out_lens = ndl.Tensor(np.array(out_lens), device=ndl_device, requires_grad=False)

    CTC_user = CTCLossModule()
    user_loss = CTC_user(probs, targets, input_lens, out_lens)
    user_loss.backward()
    user_dy = user_loss.inputs[0].grad.data.numpy()

    ref_dy = np.load(os.path.join(ref_data_path, "ref_dy.npy"))

    diff = np.linalg.norm(user_dy - ref_dy)
    if diff < 1e-5:
        print("=========Pass CTC backward===============")
    else:
        print("=========Fail CTC backward===============")

    return True

if __name__ == "__main__":
    test_ctc_forward()
    test_ctc_backward()
