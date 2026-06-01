##########################################################################################
# Machine Environment Config
import argparse
import torch
import logging
import pytz
from datetime import datetime
from utils.utils import *
from Trainer import Trainer
from args import obtain_all_hyperparameters
##########################################################################################

# parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    obtain_all_hyperparameters(parser)
    args = parser.parse_args()

    env_params = {
        'problem': args.problem,
        'problem_size': args.problem_size,
        'lower_neighbors_num': args.lower_neighbors_num,
        'reduction_percentage': args.reduction_percentage,
    }

    model_params = {
        'embedding_dim': args.embedding_dim,
        'sqrt_embedding_dim': args.embedding_dim**(1/2),
        'logit_clipping': args.logit_clipping,
        'decoder_layer_num': args.decoder_layer_num,
        'ff_hidden_dim': args.ff_hidden_dim,
        'eval_type': args.eval_type,
    }

    optimizer_params = {
        'optimizer': {
            'lr': args.lr,
        },
        'scheduler': {
            'gamma': args.gamma,
        },
    }

    trainer_params = {
        'use_cuda': args.cuda >= 0 and torch.cuda.is_available(),
        'cuda_device_num': args.cuda,
        'epochs': args.training_epochs,
        'train_episodes': args.train_batch_size * args.batches_per_epoch,
        'train_batch_size': args.train_batch_size,
        'logging': {
            'model_save_interval': args.model_save_interval,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_score.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'
            },
        },
        'model_load': {
            'enable': args.model_path is not None and 'pretrained' not in args.model_path,  # only load checkpoint for continuing training, not for testing
            'path': args.model_path,  # path to the checkpoint file to load
        },
        'bl_warmup': not args.disable_warmup,
        'eval_episodes': args.eval_episodes,
        'eval_problem_size': env_params['problem_size'],
        'bl_alpha': args.bl_alpha,
        'exp_beta': args.exp_beta,
        'max_norm': args.max_norm,

    }


    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    
    file_str = f"L2R_train_{args.problem}{args.problem_size}_k{args.lower_neighbors_num}_bs{args.train_batch_size}_epoch{args.training_epochs}"
    
    logger_params = {
        'log_file': {
            'desc': file_str,
            'filename': 'run.log',
            'filepath': f'./result_train/{args.problem}/{process_start_time.strftime("%Y-%m-%d")}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }

    # main
    seed_everything(args.seed)
    create_logger(**logger_params)
    
    logger = logging.getLogger('root')
    
    print_startup(args=args, 
                  logger=logger, 
                  result_folder=get_result_folder(), 
                  log_filename=logger_params['log_file']['filename'],
                  phase="training")
    
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(trainer_params['use_cuda'], args.cuda))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()



