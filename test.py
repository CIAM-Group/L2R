##########################################################################################
# Machine Environment Config
import os
import argparse
import torch
import logging
from datetime import datetime
import pytz
from utils.utils import *
from Tester import Tester as Tester_Synthetic
from Tester_Bench import Tester as Tester_Bench
from args import obtain_all_hyperparameters   
from test_config import TEST_CONFIG 
##########################################################################################

# parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing")
    obtain_all_hyperparameters(parser)
    args = parser.parse_args()
    
    '''
    For testing, we will test on all combinations of distributions and problem sizes, 
    and the results will be organized in different folders according to the distribution and problem size.
    
    all distributions: ["uniform", "clustered1", "explosion", "implosion", "tsplib", "dimacs_e", "cvrplib"]
    '''
    
    enabled_distributions_dict = {
        "tsp": ["clustered1", "explosion", "implosion",  "tsplib"],
        "cvrp": ["clustered1", "explosion", "implosion", "cvrplib"],
        }
    
    for problem, problem_cfg in TEST_CONFIG.items():
        args.model_path = problem_cfg["model_path"]
        args.problem = problem
        args.lower_neighbors_num = problem_cfg["lower_neighbors_num"]
        enabled_distributions = enabled_distributions_dict[problem]   
        for distribution in enabled_distributions:
            args.distribution = distribution
            distribution_cfg = problem_cfg["distributions"][distribution]
            
            if distribution_cfg["type"] == "synthetic":
                problem_size_items = distribution_cfg["cases"].items()
            else:
                problem_size_items = [(distribution_cfg["problem_sizes"], None)]
                
            for problem_size, case_cfg in problem_size_items:
                args.problem_size = problem_size

                if distribution_cfg["type"] == "synthetic":
                    args.test_episodes = case_cfg["episodes"]
                    args.test_batch_size = case_cfg["batch_size"]
                    args.data_dir = os.path.join(distribution_cfg["data_root"], case_cfg["filename"])
                    args.max_problem_size = problem_size
                    
                else:
                    args.test_episodes = None
                    args.test_batch_size = None
                    args.data_dir = distribution_cfg["data_dir"]
                    args.max_problem_size = distribution_cfg["max_problem_size"]
                
                
                env_params = {
                    'problem': args.problem,
                    'problem_size': args.problem_size,
                    'lower_neighbors_num': args.lower_neighbors_num,
                    'reduction_percentage': args.reduction_percentage,
                    'distribution': args.distribution,
                    'max_problem_size': args.max_problem_size
                }

                model_params = {
                    'embedding_dim': args.embedding_dim,
                    'sqrt_embedding_dim': args.embedding_dim**(1/2),
                    'logit_clipping': args.logit_clipping,
                    'decoder_layer_num': args.decoder_layer_num,
                    'ff_hidden_dim': args.ff_hidden_dim,
                    'eval_type': args.eval_type,
                }

                tester_params = {
                    'use_cuda': args.cuda >= 0 and torch.cuda.is_available(),
                    'cuda_device_num': args.cuda,
                    'model_load': {
                        'path': args.model_path, # directory path of pre-trained model and log files saved.
                    },
                    'test_episodes': args.test_episodes,
                    'test_batch_size': args.test_batch_size,
                    'test_data_load': {
                        'filename': args.data_dir,
                        'solution_filename': None,
                    },
                    "detailed_log": False, # whether to log the detailed results of each instance. If False, only the summary results will be logged.
                }

                process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
                file_str = f"L2R_test_{args.problem}{args.problem_size}_{distribution}_K{args.lower_neighbors_num}"

                logger_params = {
                    'log_file': {
                        'desc':  file_str,
                        'filename': 'run.log',
                        'filepath': f'./result_test/{args.problem}/{distribution}_{args.problem_size}_{args.max_problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
                    }
                }
            
                seed_everything(args.seed)
                create_logger(**logger_params)
                    
                logger = logging.getLogger('root')
                print_startup(args=args, 
                            logger=logger, 
                            result_folder=get_result_folder(), 
                            log_filename=logger_params['log_file']['filename'],
                            phase="testing")
                logger.info("=" * 81)
                logger.info(f"Testing on distribution: {args.distribution}, problem size: {args.problem_size}, \
                            test episodes: {args.test_episodes}, test batch size: {args.test_batch_size}")    
                logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(tester_params['use_cuda'], args.cuda))
                [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

                if args.distribution in ['uniform', 'clustered1', 'explosion', 'implosion']:
                    tester = Tester_Synthetic(env_params=env_params,
                                    model_params=model_params,
                                    tester_params=tester_params)
                else:
                    tester = Tester_Bench(env_params=env_params,
                                        model_params=model_params,
                                        tester_params=tester_params)
                copy_all_src(tester.result_folder)
                tester.run()
                logger.info("=" * 81)
            
