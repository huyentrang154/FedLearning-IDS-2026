import argparse
import os
import time
import warnings
import numpy as np
import random
import torch

from flcore.servers.serverAvg import FedAvg
from flcore.servers.serverProx import FedProx
from flcore.servers.serverMOON import MOON
from flcore.servers.serverScaffold import Scaffold
from flcore.servers.serverPer import FedPer
from flcore.servers.serverRep import FedRep
from flcore.servers.serverLG import LG_FedAvg
from flcore.servers.serverGH import FedGH
from flcore.servers.serverALA import FedALA
from flcore.servers.serverAPFL import APFL
from flcore.servers.serverApple import APPLE
from flcore.servers.serverFomo import FedFomo
from flcore.servers.serverUS import FedUS
from flcore.servers.serverHomeLG import FedHomeLG
from flcore.servers.serverHome import FedHome
from flcore.servers.serverMedian import FedMedian
from flcore.servers.serverTrimmedMean import FedTrimmedMean
from flcore.servers.serverFLTrust import FLTrust
from flcore.servers.serverAE import FedAE
from flcore.trainmodel.models import *
from flcore.servers.serverNoRo import FedNoRo # LyVT
from flcore.servers.serverFedCorr import FedCorr
from flcore.centralized.centralizedAE import CentralizedAE
from flcore.servers.model_saver import normalize_noise_type
from utils.data_utils import resolve_data_type

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
warnings.simplefilter("ignore")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    time_list = []
    algorithm = args.algorithm
    for i in range(args.prev, args.times):
        print(f"\n============= Running {i + 1}/{args.times} =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if args.dataset == 'cic_ids':
            n_features, num_classes = 80, 12
        elif args.dataset == 'ctu13_13' or args.dataset == 'ctu13_08':
            n_features, num_classes = 40, 2
        elif args.dataset == 'ctu13_09':
            n_features = 41
        elif args.dataset == 'ctu13_10':
            n_features = 38
        elif args.dataset == 'internet_ad':
            n_features = 1558
        elif args.dataset == 'nslkdd':
            n_features, num_classes = 122, 5
        elif args.dataset == 'spambase':
            n_features = 57
        elif args.dataset == 'unsw':
            n_features, num_classes = 196, 10
        elif args.dataset == 'cifar10':
            # 32x32x3 = 3072
            input_channels, image_size, num_classes = 3, 32, 10
        elif args.dataset == 'bot_iot':
            n_features, num_classes = 34, 5
        elif args.dataset == 'crawl':
            n_features, num_classes = 15, 4
        elif args.dataset == 'toniot':
            n_features, num_classes = 42, 10
        else:
            raise NotImplementedError

        args.num_classes = num_classes
        if algorithm == 'fednoro':
            args.model = MLP(n_features=n_features, num_classes=num_classes).to(args.device)
        elif algorithm == 'fedcorr':
            args.model = MLP(n_features=n_features, num_classes=num_classes).to(args.device)
        elif args.model_type == 'mlp':
            args.model = MLP(n_features=n_features, num_classes=num_classes).to(args.device)
        elif args.model_type == 'ae':
            args.model = AE(n_features=n_features, num_classes=num_classes).to(args.device)
        elif args.model_type == 'cnn':
            # Bây giờ đã có n_features làm tham số
            args.model = CNN(input_channels=input_channels, image_size=image_size, num_classes=num_classes).to(args.device)
        elif args.model_type == 'cnnae':
            args.model = CNNAE(input_channels=input_channels, image_size=image_size, num_classes=num_classes).to(args.device)

        if algorithm == 'fedavg':
            server = FedAvg(args, i)
        elif algorithm == 'fedae':
            server = FedAE(args, i)
        elif algorithm == 'fedprox':
            server = FedProx(args, i)
        elif algorithm == 'moon':
            server = MOON(args, i)
        elif algorithm == 'scaffold':
            server = Scaffold(args, i)
        elif algorithm == 'fedper':
            server = FedPer(args, i)
        elif algorithm == 'fedrep':
            server = FedRep(args, i)
        elif algorithm == 'lg_fedavg':
            server = LG_FedAvg(args, i)
        elif algorithm == 'fedgh':
            server = FedGH(args, i)
        elif algorithm == 'fedala':
            server = FedALA(args, i)
        elif algorithm == 'apfl':
            server = APFL(args, i)
        elif algorithm == 'apple':
            server = APPLE(args, i)
        elif algorithm == 'fedfomo':
            server = FedFomo(args, i)
        elif algorithm == 'fedus':
            server = FedUS(args, i)
        elif algorithm == 'fedhomelg':
            server = FedHomeLG(args, i)
        elif algorithm == 'fedhome':
            server = FedHome(args, i)
        elif algorithm == 'fedmedian':
            server = FedMedian(args, i)
        elif algorithm == 'fedtrimmedmean':
            server = FedTrimmedMean(args, i)
        elif algorithm == 'fedtrust':
            server = FLTrust(args, i)
        elif algorithm == 'fednoro': #Ly
            server = FedNoRo(args, i)
        elif algorithm == 'fedcorr':
            server = FedCorr(args, i)
        elif algorithm == 'centralized':
            server = CentralizedAE(args, i)
        else:
            raise NotImplementedError

        print('Algorithm', algorithm)
        server.train()
        server.save_results()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-algo', "--algorithm", type=str, default='fedus')

    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cic_ids")
    parser.add_argument('-is', "--image_size", type=int, default=0,
                    help="Kích thước cạnh của ảnh (ví dụ: 32 cho CIFAR-10)")
    parser.add_argument('-ic', "--input_channels", type=int, default=0,
                    help="Số kênh màu của dữ liệu (3 cho RGB, 1 cho Grayscale/Tabular)")
    parser.add_argument('-nb', "--num_classes", type=int, default=12)
    parser.add_argument('-m', "--model_type", type=str, default="ae")
    parser.add_argument('-lbs', "--batch_size", type=int, default=4)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('--round_to_add_clients', type=int, default=0,
                    help="Round number to start including the last 5 clients in training")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
 
    # FedProx
    parser.add_argument('-mu', "--mu", type=float, default=0.1)

    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)

    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)

    parser.add_argument('-dt', "--data_type", type=str, default='noniid')

    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")

    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)

    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)

    # FedCorr
    parser.add_argument('--iteration1', type=int, default=5,
                        help="Stage 1: number of noise-detection outer iterations")
    parser.add_argument('--rounds1', type=int, default=10,
                        help="Stage 2: fine-tuning rounds on clean clients")
    parser.add_argument('--rounds2', type=int, default=40,
                        help="Stage 3: final FedAvg rounds (all clients)")
    parser.add_argument('--frac1', type=float, default=1.0,
                        help="Client fraction per sub-round in Stage 1")
    parser.add_argument('--frac2', type=float, default=1.0,
                        help="Client fraction in Stage 2 / Stage 3")
    parser.add_argument('--beta_fedcorr', type=float, default=0.0,
                        help="Proximal term coefficient (original default=0, i.e. disabled)")
    parser.add_argument('--relabel_ratio', type=float, default=0.5,
                        help="Fraction of estimated noisy samples to relabel")
    parser.add_argument('--confidence_thres', type=float, default=0.5,
                        help="Min softmax confidence required to accept a relabelling")
    parser.add_argument('--clean_set_thres', type=float, default=0.1,
                        help="Max noise level to classify a client as clean in Stage 2")
    parser.add_argument('--correction', type=int, default=0,
                        help="1 = enable label correction; 0 = disable (original default=0)")
    parser.add_argument('--fine_tuning', type=int, default=0,
                        help="1 = enable Stage 2 fine-tuning; 0 = skip (original default=0)")

    # FedNoRo data / result path args
    parser.add_argument('--patition', type=str, default='noniid',
                        help="Data partition type (used in result path)")
    parser.add_argument('--alpha_dir', type=float, default=0.5,
                        help="Dirichlet alpha (used in result path)")
    parser.add_argument('--poison_attack', type=str, default=None,
                        help="Poison attack label for result path")
    parser.add_argument('--binary', type=int, default=1,
                        help="1 = use binary result path prefix")

    # FedNoRo - LyVT
    parser.add_argument('-warmup', "--warmup_rounds", type=int, default=10,
                        help="Stage-1 warm-up rounds before noisy-client identification")
    parser.add_argument('-kd_begin', "--kd_begin", type=int, default=10,
                        help="Round to start ramping up the KD weight w_kd")
    parser.add_argument('-kd_end', "--kd_end", type=int, default=49,
                        help="Round at which w_kd reaches its maximum")
    parser.add_argument('-kd_a', "--kd_a", type=float, default=0.8,
                        help="Maximum KD weight multiplier (paper default 0.8)")

    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # Label flipping / Noise
    parser.add_argument('-nt', '--noise_type', type=str, default="label_flipping",
                    help="Type of noise: None, 'gaussian_noise', 'label_flipping', 'Asymmetric', 'Symmetric'.")
    parser.add_argument('-npw', '--num_of_poisoned_workers', type=float, default=0.3,
                        help="Ratio of clients to be poisoned (e.g., 0, 0.3, 0.5).")
    parser.add_argument('-psr', '--poisoned_sample_ratio', type=float, default=1.0,
                        help="Ratio of data samples to be poisoned (0 to 1, e.g., 1.0, 0.4).")
    parser.add_argument('-ans', '--attack_noise_std', type=float, default=0,
                        help="Standard deviation for Gaussian noise.")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    args = parser.parse_args()

    # Chuẩn hóa algorithm, data_type (noniid -> noniid20, exdir -> noniidexdir20) và noise_type
    args.algorithm = args.algorithm.strip().lower().replace('-', '_')
    args.data_type = resolve_data_type(args.data_type, args.num_clients)
    args.noise_type = normalize_noise_type(args.noise_type)
    if args.noise_type is None or args.num_of_poisoned_workers <= 0 or args.poisoned_sample_ratio <= 0:
        args.noise_type = None
        args.num_of_poisoned_workers = 0.0
        args.poisoned_sample_ratio = 0.0

    # Cast int flags to bool for FedCorr
    args.correction = bool(args.correction)
    args.fine_tuning = bool(args.fine_tuning)
    set_random_seed(args.seed)

    print("=" * 60)
    print(f"Dataset                : {args.dataset} ({args.data_type})")
    print(f"Algorithm              : {args.algorithm}")
    print(f"Random Seed            : {args.seed}")
    print(f"Noise Type             : {args.noise_type}")
    print(f"Poisoned Workers Ratio : {args.num_of_poisoned_workers}")
    print(f"Poisoned Sample Ratio  : {args.poisoned_sample_ratio}")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)
#K0VitxF3C#8E
