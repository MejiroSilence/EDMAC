from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from utils.th_utils import print_flops_data


# This multi-agent controller shares parameters between agents
class EDMACMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(EDMACMAC, self).__init__(scheme, groups, args)
        self.alpha_start = args.alpha_start
        self.alpha_finish = args.alpha_finish
        self.alpha_delta =  (self.alpha_finish - self.alpha_start) / args.alpha_anneal_time
        self.alpha = self.alpha_start





    def update_alpha(self, t_env):
        self.alpha = min(self.alpha_finish, self.alpha_start + self.alpha_delta * t_env)

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.alpha = other_mac.alpha
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, calculate_loss=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        # print_flops_data(self.agent, (agent_inputs, self.alpha, ep_batch["state"][:, t], self.hidden_states, True, False))
        agent_outs, self.hidden_states, extra_losses = self.agent(
            inputs = agent_inputs,
            alpha = self.alpha,
            state = ep_batch["state"][:, t],
            hidden_state = self.hidden_states,
            test_mode = test_mode,
            calculate_loss = calculate_loss
        )

        return agent_outs, extra_losses