import random
import time
import torch
from logging import getLogger
from tqdm import tqdm

from utils.utils import *
import env
import model

from problem.DataReader import get_saved_data

class Tester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        
        self.problem = self.env_params['problem'].upper()

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
            
        self.device = device
        self.env_params['device'] = device
        self.model_params['device'] = device
        
        # ENV and MODEL
        self.upper_model = getattr(model, f"{self.problem}UpperModel")(**self.model_params)
        self.lower_model = getattr(model, f"{self.problem}LowerModel")(**self.model_params)
        self.env = getattr(env, f"{self.problem}Env")(**self.env_params)

        # Restore
        checkpoint_fullname = tester_params['model_load']['path'] 
        self.logger.info("Load model from: {0}".format(checkpoint_fullname))

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.upper_model.load_state_dict(checkpoint['upper_model_state_dict'])
        self.lower_model.load_state_dict(checkpoint['lower_model_state_dict'])
        total_params = list(self.upper_model.parameters()) + list(self.lower_model.parameters())

        total = sum([param.nelement() for param in total_params])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        AvgScore = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']

        optimal = 1.0
        file_name = self.tester_params['test_data_load']['filename']
        solution_file = self.tester_params['test_data_load']['solution_filename']
        self.dataset_dict, optimal = get_saved_data(self.problem, file_name, test_num_episode, self.device,solution_file=solution_file)
        self.env.input_saved_data(self.dataset_dict,self.device)
        self.logger.info("Saved dataset loaded successfully!!!")
        self.logger.info("Data loaded from: {0}".format(file_name))
        self.logger.info('problem:{0:4s}, problem_size: {1:5d} , test_episodes: {2:5d}, optimal : {3:.4f}'.format(
                        self.problem, self.env.saved_node_xy.shape[1], test_num_episode, optimal))
        
        assert self.env_params['problem_size'] == self.env.problem_size, "Problem size mismatch between env_params and env"


        episode = 0
        self.start_time = time.time()
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score = self._test_one_batch(batch_size)
            AvgScore.update(score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(" Episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], greedy score:{:.3f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score))
            self.logger.info("Current average score with greedy decoding: {:.4f}".format(AvgScore.avg))

            all_done = (episode == test_num_episode)

            if all_done:
                end_time = time.time()
                self.logger.info(" *** Test Done *** ")
                gap = (AvgScore.avg - optimal) * 100 / optimal
                
                distribution = self.env_params['distribution']
                self.logger.info("=" * 80)
                self.logger.info(" optimal score: {0:.4f} ".format(optimal))
                self.logger.info(" problem size: {0}, distribution: {1}".format(self.env.problem_size, distribution))

                self.logger.info("=" * 80)
                self.logger.info("SCORE:{:.4f}, 2 decimal places:{:.2f}".format(AvgScore.avg, AvgScore.avg))
                self.logger.info("Model gap:{0:.4f}%, 2 decimal places:{1:.2f}%".format(gap, gap))

                self.logger.info("=" * 80)
                self.logger.info(" total time: {:.2f} sec".format(end_time - self.start_time))
                self.logger.info(" total time: {:.2f} min".format((end_time - self.start_time) / 60))
                self.logger.info(" avg time: {:.2f} sec".format((end_time - self.start_time) / test_num_episode))
                self.logger.info(" avg time: {:.2f} mins".format((end_time - self.start_time) / test_num_episode / 60))

    def _test_one_batch(self, batch_size):

            # Ready
            ###############################################
            self.upper_model.eval()
            self.lower_model.eval()
            self.upper_model.set_decoder_method('greedy')
            self.lower_model.set_decoder_method('greedy')
            
            self.env.load_problems(batch_size, 
                                   problem_size=self.env_params['problem_size'])
            
            # reset peak memory stats to get correct memory usage for each batch
            torch.cuda.reset_peak_memory_stats(device=self.device)
            
            reset_state, _, _ = self.env.reset()
            
            with torch.no_grad():
                self.upper_model.pre_forward(reset_state=reset_state)
                
                # Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                with tqdm(total=0) as pbar:
                    while not done:
                        if state.current_node is not None:
                            state = self.env.get_upper_input()
                            upper_scores,_,_ = self.upper_model(state)
                            self.env.update_cur_scores(upper_scores=upper_scores)
                            
                        state = self.env.get_lower_transformed_neighbors()
                        low_selected, _ = self.lower_model(state)
                        # shape: (batch,)
                        state, reward, done = self.env.step(low_selected)
                        # shape: (batch,)
                        pbar.total += 1
                        pbar.update(1)
            batch_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
            self.logger.info("batch_memory: {:.2f}MB, {:.2f}GB, avg_memory:{:.2f}MB".format(
                batch_memory, batch_memory / 1024, batch_memory / batch_size))

            # Return
            ###############################################
            avg_score = -reward.float().mean().item()  # negative sign to make positive value
            
            return avg_score

