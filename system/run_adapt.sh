# rpc: ratio_poisoned_clients: 0.5
# nt: noise_type: label_flipping/ gaussian_noise
# psr: poisoned_sample_ratio: 0.5
# ns: noise_std: 0.001
# ans: attack_noise_std: 0.1
# at: aggregation_type: attention/ average/ kmean/ split
# csa: coef_shrink_ae: 10, 20, 30, ...


# ctu13_08
nohup python -u main.py -data ctu13_08 -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data ctu13_08 -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# ctu13_09
nohup python -u main.py -data ctu13_09 -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data ctu13_09 -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# ctu13_10
nohup python -u main.py -data ctu13_10 -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data ctu13_10 -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# ctu13_13
nohup python -u main.py -data ctu13_13 -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data ctu13_13 -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# internet_ad
nohup python -u main.py -data internet_ad -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data internet_ad -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# cic_ids
nohup python -u main.py -data cic_ids -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data cic_ids -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# nslkdd
nohup python -u main.py -data nslkdd -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data nslkdd -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# spambase
nohup python -u main.py -data spambase -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data spambase -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &

# unsw
nohup python -u main.py -data unsw -m sae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
nohup python -u main.py -data unsw -m sdae -rpc 0.5  -nt gaussian_noise  -psr 0.5 -ns 0.001 -ans 0.1 -at attention -csa 10 -lbs 8 -lr 0.005 -gr 200 -ls 5 -algo adapt &
