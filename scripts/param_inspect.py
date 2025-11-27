import numpy as np

p = np.load("../fit_egfr/fitted_params.npz", allow_pickle=True)

print("k_act   =", p["k_act"])
print("k_deact =", p["k_deact"])
print("beta_g  =", p["beta_g"])
print("beta_l  =", p["beta_l"])

residues = p["residues"]
k_on  = p["k_on"]
k_off = p["k_off"]

for res, kon, koff in zip(residues, k_on, k_off):
    print(f"{res:6s}  k_on={kon:.4f}  k_off={koff:.4f}")
