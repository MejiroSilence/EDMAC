from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
# This multi-agent controller shares parameters between agents
class BS1MAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(BS1MAC, self).__init__(scheme, groups, args)
        
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
        agent_outs, self.hidden_states, extra_losses = self.agent(
            inputs = agent_inputs,
            hidden_state = self.hidden_states,
            need_loss = calculate_loss,
            state = ep_batch["state"][:, t],
            test_mode = test_mode
        )

        return agent_outs, extra_losses