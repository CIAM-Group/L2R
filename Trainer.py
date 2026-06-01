import copy
import math

import torch
from logging import getLogger

import env
import model

from scipy.stats import ttest_rel

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from utils.utils import *

from problem.DataReader import get_saved_data
from problem.ProblemDef import get_random_instances


class Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        self.problem = self.env_params['problem'].upper()

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # evaluation dataset
        self.eval_size = self.trainer_params['eval_episodes']
        self.eval_problem_size = self.trainer_params['eval_problem_size']
        self.bl_alpha = self.trainer_params['bl_alpha']
        self.warmup_baseline = ExponentialBaseline(exp_beta=self.trainer_params['exp_beta'])  # only applied to the first epoch

        self.saved_problems_eval_baseline = None

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device
        self.env_params['device'] = device
        self.model_params['device'] = device

        # Main Components
        self.upper_model = getattr(model, f"{self.problem}UpperModel")(**self.model_params)
        self.lower_model = getattr(model, f"{self.problem}LowerModel")(**self.model_params)
        self.env = getattr(env, f"{self.problem}Env")(**self.env_params)
        self.baseline_env = getattr(env, f"{self.problem}Env")(**self.env_params)

        self.baseline_upper = copy.deepcopy(self.upper_model)
        self.baseline_lower = copy.deepcopy(self.lower_model)

        self.optimized_params = list(self.upper_model.parameters()) + list(self.lower_model.parameters())
        self.optimizer = Adam(self.optimized_params, **self.optimizer_params['optimizer'])
        self.scheduler = ExponentialLR(self.optimizer, **self.optimizer_params['scheduler'])


        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = model_load['path']
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.upper_model.load_state_dict(checkpoint['upper_model_state_dict'])
            self.lower_model.load_state_dict(checkpoint['lower_model_state_dict'])
            self.baseline_upper.load_state_dict(checkpoint['baseline_upper_state_dict'])
            self.baseline_lower.load_state_dict(checkpoint['baseline_lower_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')
            self.logger.info('Model Path: {0}'.format(checkpoint_fullname))

        saved_data_dict = {
            'TSP': {
                1000: 'MCTS_tsp1000_test_concorde.txt',
                10000: 'MCTS_tsp10000_test_concorde.txt',
            },
            'CVRP': {
                1000: 'vrp1000_train_hgs_resolve_GLOP_n100.txt',
                10000: 'vrp10000_test_n16_solution_flag.txt',
            }
        }
        filename_1000 = f'./dataset/synthetic/{self.problem}/{saved_data_dict[self.problem][1000]}'
        self.saved_dataset_1000,self.optimal_1000 = get_saved_data(self.problem, filename_1000, 128, self.device)
        self.logger.info('Successfully load {0:5d} {1}{2:5d} instances, optimal: {3}'.format(
            self.saved_dataset_1000["node_xy"].size(0), self.problem, self.saved_dataset_1000["node_xy"].size(1), self.optimal_1000))

        filename_10000 = f'./dataset/synthetic/{self.problem}/{saved_data_dict[self.problem][10000]}'
        self.saved_dataset_10000, self.optimal_10000 = get_saved_data(self.problem, filename_10000, 16, self.device)
        self.logger.info('Successfully load {0:5d} {1}{2:5d} instances, optimal: {3}'.format(
            self.saved_dataset_10000["node_xy"].size(0), self.problem, self.saved_dataset_10000["node_xy"].size(1), self.optimal_10000))

        self.best_gap = 100.0

        self.eval_reward_candidate = None
        self.eval_reward_baseline = None
        self.eval_reward_baseline_mean = None

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        
        self.curves_saved_folder = os.path.join(self.result_folder, "curves")
        os.makedirs(self.curves_saved_folder, exist_ok=True)

        if self.trainer_params['model_load']['enable']:
            self._update_baseline(initial_baseline=False)  # initial baseline
        else:
            self._update_baseline(initial_baseline=True)

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            #########################################################################
            # Train
            #########################################################################
            self.logger.info('Epoch {:4d}: Current learning rate (Upper & Lower): {}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            # Train
            train_score, train_loss = self._train_one_epoch(epoch)

            # LR Decay
            self.scheduler.step()

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            #########################################################################
            # Validation on evaluation dataset, every epoch
            #########################################################################
            self._validation_each_epoch(self.saved_dataset_1000, self.optimal_1000, epoch=epoch)
            self._validation_each_epoch(self.saved_dataset_10000, self.optimal_10000, epoch=epoch)

            #########################################################################
            # Logs & Checkpoint
            #########################################################################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:4d}/{:4d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            #########################################################################
            # Save latest images, every epoch
            #########################################################################
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.curves_saved_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'upper_model_state_dict': self.upper_model.state_dict(),
                    'lower_model_state_dict': self.lower_model.state_dict(),
                    'baseline_upper_state_dict': self.baseline_upper.state_dict(),
                    'baseline_lower_state_dict': self.baseline_lower.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        if epoch == 1 and self.trainer_params['bl_warmup']:
            self.logger.info("Epoch {:4d}, the baseline is exponential baseline (β = 0.8).".format(epoch))
        else:
            self.logger.info("Epoch {:4d}, the baseline is greedyrollout baseline.".format(epoch))

        score = AverageMeter()
        loss = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        batches_per_epoch = math.ceil(train_num_episode / self.trainer_params['train_batch_size'])
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            capacity = 50.0  # the capacity is fixed to 50, only used for CVRP instances
            avg_score,avg_loss = self._train_one_batch(batch_size,epoch,capacity)
            score.update(avg_score, batch_size)
            loss.update(avg_loss, batch_size)

            episode += batch_size

            loop_cnt += 1
            if loop_cnt <= 5 or loop_cnt % 250 == 0:
                self.logger.info(
                    'Epoch {:4d}: Trained batches {:4d}/{:4d}({:5.1f}%)   Score: {:.4f},  Loss: {:.4f}'
                    .format(epoch, loop_cnt, batches_per_epoch, 100. * loop_cnt / batches_per_epoch,
                            score.avg, loss.avg))
                self.logger.info(
                    'Epoch {:4d}: Trained batches {:4d}/{:4d}({:5.1f}%)   Current Score: {:.4f}, Current Loss: {:.4f}'
                    .format(epoch, loop_cnt, batches_per_epoch, 100. * loop_cnt / batches_per_epoch,
                            avg_score, avg_loss))

        #######################################################################################################
        #  Challenges the current baseline with the model and replaces the baseline model if it is improved.
        #######################################################################################################
        self.logger.info("Evaluating baseline model on evaluation dataset ({:4d} instances with {:4d} nodes)".
                         format(self.eval_size, self.eval_problem_size))
        self._epoch_callback(epoch)
        
        #########################################################################
        # Log Once, for each epoch
        #########################################################################
        self.logger.info('Epoch {:4d}: Train ({:3.0f}%) Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * loop_cnt / batches_per_epoch, score.avg, loss.avg))

        return score.avg, loss.avg

    def _train_one_batch(self, batch_size,epoch,capacity):

        self.upper_model.train()
        self.lower_model.train()
        self.upper_model.set_decoder_method("sampling")
        self.lower_model.set_decoder_method("sampling")

        # Prep
        ###############################################
        self.env.load_problems(batch_size,
                               problem_size=self.env_params['problem_size'],
                               device=self.device,
                               capacity=capacity)
        reset_state, _, _ = self.env.reset()
        self.upper_model.pre_forward(reset_state=reset_state)

        prob_list = torch.zeros(size=(batch_size, 0))
        # shape: (batch, 0~problem)
        upper_selected_score = None
        upper_score_list = torch.zeros(size=(batch_size, 0))

        # AM Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            if state.current_node is not None:
                state = self.env.get_upper_input()
                upper_scores, upper_selected, upper_selected_score = self.upper_model(state)
                self.env.update_cur_scores(upper_scores=upper_scores)
            state = self.env.get_lower_transformed_neighbors()
            low_selected, lower_prob = self.lower_model(state)
            # shape: (batch,)
            if state.current_node is None:
                upper_selected_score = torch.ones(size=(batch_size, ), device=self.device)
            state, reward, done = self.env.step(low_selected)
            # shape: (batch,)
            prob_list = torch.cat((prob_list, lower_prob[:, None]), dim=1)
            upper_score_list = torch.cat((upper_score_list, upper_selected_score[:, None]), dim=1)

        # Loss
        ###############################################
        """
        baseline is not in gradient computation.
        Following Kool et.al, an exponential baseline (β = 0.8) is used in the first epoch.
        This operation is optional and is only for stabilizing the initial learning.   
        In other epochs(>1), the baseline is GreedyRollout baseline.
        """
        if epoch == 1 and self.trainer_params['bl_warmup']:
            reward_bl = self.warmup_baseline.eval(reward)
            # shape: (1,)
        else:
            reward_bl = rollout(self.baseline_upper, self.baseline_lower,
                                self.baseline_env,self.env.dataset_dict)

        advantage = reward - reward_bl # reward < 0
        # shape: (batch, )
        log_prob = prob_list.log().sum(dim=1)
        # size = (batch, )
        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, )
        loss_lower = loss.mean()

        log_score_upper = upper_score_list.log().sum(dim=1)
        # shape: (batch, )
        loss_upper = (- advantage * log_score_upper).mean()
        loss_mean = loss_lower + loss_upper
        # Score
        ###############################################
        score_mean = -reward.float().mean() # instances' costs

        # Step & Return
        ###############################################
        max_norm = self.trainer_params['max_norm']

        self.optimizer.zero_grad()
        loss_mean.backward()  
        torch.nn.utils.clip_grad_norm_(parameters=self.optimized_params,
                                       max_norm=max_norm)
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()

    def _validation_each_epoch(self, dataset_dict, optimal_score, epoch):

        problem_size = dataset_dict['node_xy'].shape[1]
        reward = rollout(self.upper_model, self.lower_model, self.env, dataset_dict)

        # Return
        ###############################################
        score_mean = -reward.float().mean()
        gap = (score_mean - optimal_score) * 100 / optimal_score

        # logger
        ###############################################
        self.result_log.append(f'eval_{problem_size}', epoch, score_mean.item())
        self.result_log.append(f'gap_{problem_size}', epoch, gap.item())


        self.logger.info(
            'Epoch {:4d}: Eval score on instances with {:5d} nodes: {:7.4f}, Gap: {:7.4f}%'.format(
                epoch, problem_size, score_mean.item(), gap.item()))
        if epoch > 1:
            image_prefix = '{}/latest'.format(self.result_folder)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'eval_{problem_size}'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'gap_{problem_size}'])

        # save best model
        ###############################################
        if problem_size == 1000:
            if gap < self.best_gap:
                self.best_gap = gap
                self.logger.info('Best model updated, gap: {0:.4f}% on {1}{2}'.format(
                    self.best_gap,self.problem,problem_size))
                checkpoint_dict = {
                    'epoch': epoch,
                    'upper_model_state_dict': self.upper_model.state_dict(),
                    'lower_model_state_dict': self.lower_model.state_dict(),
                    'baseline_upper_state_dict': self.baseline_upper.state_dict(),
                    'baseline_lower_state_dict': self.baseline_lower.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{0}/{1}{2}_best_model.pt'.format(self.result_folder, self.problem, problem_size))

        return score_mean.item(), gap.item()

    # Evaluating baseline model on evaluation dataset
    # https://github.com/wouterkool/attention-learn-to-route/blob/master/reinforce_baselines.py#L143-L227
    def _epoch_callback(self, epoch):

        eval_reward_candidate = -1 * rollout(self.upper_model, self.lower_model,self.env, self.saved_problems_eval_baseline)
        eval_reward_candidate_mean = eval_reward_candidate.mean().item()

        self.logger.info( "Evaluation result: Epoch {:4d} ==> candidate (greedy) mean {:3.4f}, baseline (greedy) mean {:3.4f}, difference {:3.4f}".format(
                epoch, eval_reward_candidate_mean, self.eval_reward_baseline_mean, eval_reward_candidate_mean - self.eval_reward_baseline_mean))

        if eval_reward_candidate_mean - self.eval_reward_baseline_mean < 0:
            self.logger.info('The candidate model is better than the baseline model, so we evaluate the baseline policy.')
            # Calc p value
            t, p = ttest_rel(eval_reward_candidate.cpu().numpy(),self.eval_reward_baseline.cpu().numpy())
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            self.logger.info("OneSidedPairedTTest  bl_alpha is {0:.4f}%, p-value: {1:.4f}%".format(self.bl_alpha * 100,p_val * 100))
            if p_val < self.bl_alpha:
                self._update_baseline(initial_baseline=True)

        else:
            self.logger.info('The baseline policy is better than the candidate policy, so we keep the baseline policy.')

    def _update_baseline(self,initial_baseline=True):
        self.logger.info('Updating baseline and generating new {0} {1}{2} instances for evaluation'.format(
                            self.eval_size, self.problem, self.eval_problem_size))
        if initial_baseline:
            self.baseline_upper = copy.deepcopy(self.upper_model)
            self.baseline_lower = copy.deepcopy(self.lower_model)
            self.logger.info('Initial baseline model is updated.')
        else:
            self.logger.info('Baseline model is obtained from the saved model parameters.')
            
        # If the baseline policy is updated, we sample new evaluation instances to prevent overfitting
        self.saved_problems_eval_baseline = get_random_instances(self.problem, self.eval_size, self.eval_problem_size, self.device, capacity=50.0)
        self.eval_reward_baseline = -1 * rollout(self.baseline_upper, 
                                                 self.baseline_lower,
                                                 self.baseline_env, 
                                                 self.saved_problems_eval_baseline)
        self.eval_reward_baseline_mean = self.eval_reward_baseline.mean().item()

def rollout(upper_model: torch.nn.Module, lower_model: torch.nn.Module,env,dataset_dict):

    upper_model.eval()
    lower_model.eval()
    upper_model.set_decoder_method("greedy")
    lower_model.set_decoder_method("greedy")

    batch_size = dataset_dict['node_xy'].size(0)
    problem_size = dataset_dict['node_xy'].size(1)
    env.load_problems(batch_size,
                      problem_size,
                      validation_data=dataset_dict,
                      device=dataset_dict['node_xy'].device)
    reset_state, _, _ = env.reset()
    with torch.no_grad():
        upper_model.pre_forward(reset_state=reset_state)
        state, reward, done = env.pre_step()
        while not done:
            if state.current_node is not None:
                state = env.get_upper_input()
                upper_scores, _, _ = upper_model(state)
                env.update_cur_scores(upper_scores=upper_scores)
            state = env.get_lower_transformed_neighbors()
            low_selected, _ = lower_model(state)
            # shape: (batch,)
            state, reward, done = env.step(low_selected)
            # shape: (batch,)

    return reward

# https://github.com/wouterkool/attention-learn-to-route/blob/master/reinforce_baselines.py#L86-L110
class ExponentialBaseline:
    def __init__(self,exp_beta):

        self.warmup_exp_beta = exp_beta  # beta is 0.8, following the original AM code
        self.reward = None

    def eval(self, cost):  # cost of instances(batch)

        if self.reward is None:
            reward = cost.mean()
        else:
            reward = self.warmup_exp_beta * self.reward + (1. - self.warmup_exp_beta) * cost.mean()

        self.reward = reward.detach()  # Detach since we never want to backprop
        return self.reward





