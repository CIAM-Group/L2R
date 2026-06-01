'''
This module defines common argument parsing functionality for the project.

Note that detailed settings for testing will be defined in test.py, 
and the parameters defined here are common for both training and testing


'''

def obtain_all_hyperparameters(parser):
    # machine parameters
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--problem", type=str, choices=["tsp", "cvrp"], default="tsp", help="Problem type to train")

    # common environment parameters
    parser.add_argument("--problem_size", type=int, default=100, help="Problem size for training")
    parser.add_argument("--lower_neighbors_num", type=int, default=20, 
                        help="Number of candidate nodes for the lower model, which is probelm-specific: 20 for TSP, 50 for CVRP, and 15 for CVRPTW by default.")
    parser.add_argument("--reduction_percentage", type=float, default=0.1, help="Static reduction percentage for the model")
    
    # format: 'result_trained_models/checkpoint-{epoch}.pt' (training) or 'pretrained/{problem}_best.pt' (testing)
    parser.add_argument("--model_path", type=str, 
                        default="./pretrained/cvrp_best.pt", 
                        help="Path to saved checkpoint for continuing training. If None, training will start from scratch.")
    
    # training parameters 
    parser.add_argument("--training_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=180, 
                        help="Training batch size: 180 for TSP, 60 for CVRP, and 128 for CVRPTW in our experiments by default.")
    parser.add_argument("--batches_per_epoch", type=int, default=2500, help="Steps per epoch")
    parser.add_argument("--model_save_interval", type=int, default=1, help="Model save interval in epochs")

    # model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--decoder_layer_num", type=int, default=6, help="Number of decoder layers in the model")
    parser.add_argument("--ff_hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward layer in the model")
    parser.add_argument("--logit_clipping", type=float, default=10, help="Logit clipping value for the model")
    # training: sampling, testing: greedy
    parser.add_argument("--eval_type", type=str, choices=["sampling", "greedy"], default="greedy", 
                        help="Decoding method for model evaluation. During training, we use sampling for better exploration. During testing, we use greedy for more stable evaluation.")
    
    # optimizer_params
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.98, help="Gamma for the learning rate scheduler")

    # other trainer parameters
    parser.add_argument("--disable_warmup", action='store_true', help="Disable baseline warmup")
    parser.add_argument("--eval_episodes", type=int, default=10000, help="Number of evaluation episodes for baseline evaluation")
    parser.add_argument("--bl_alpha", type=float, default=0.05, help="Significance level for baseline evaluation")
    parser.add_argument("--exp_beta", type=float, default=0.8, help="Beta for exponential moving average baseline update")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Max norm for gradient clipping")
