# baseline
python3 baseline.py --lr 0.03 --noise_mode asym --noise_ratio 0.4 --run 0 > checkpoint/ce_asym04_run0_lr003_log

# baseline with forward loss correction
python3 baseline.py --lr 0.03 --noise_mode asym --noise_ratio 0.4 --run 0 --use_tm > checkpoint/ce_tm_asym04_run0_lr003_log

# original MLNT
python3 main.py --num_fast 4 --num_ssl 0 --noise_mode asym --noise_ratio 0.4 --run 0 --pretrain_ckpt ce_asym04_run0_lr003 > checkpoint/mlnt_asym04_run0_M4S0n10rho05_log

# MLNT with weighted consistency loss
python3 main.py --num_fast 4 --num_ssl 0 --noise_mode asym --noise_ratio 0.4 --run 0 --pretrain_ckpt ce_asym04_run0_lr003 > checkpoint/wcl_asym04_run0_M4S0n10rho05_w05_log

# set-level self-supervised learning
python3 main.py --num_fast 0 --num_ssl 4 --noise_mode asym --noise_ratio 0.4 --run 0 --pretrain_ckpt ce_asym04_run0_lr003 > checkpoint/slssl_asym04_run0_M0S4n10rho05_log

# set-level self-supervised learning with forward loss correction by a dynamically-updated class transition matrix
python3 main.py --num_fast 0 --num_ssl 4 --noise_mode asym --noise_ratio 0.4 --run 0 --pretrain_ckpt ce_asym04_run0_lr003 --use_tm > checkpoint/slssl_tm_asym04_run0_M0S4n10rho05_log
