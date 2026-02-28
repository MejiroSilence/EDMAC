REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .edmac_agent import EDMACAgent
from .masia_agent import MASIAAgent
from .tarmac_agent import TarMACAgent
from .tmac_p2p_comm_rnn_msg_agent import RnnMsgAgent
from .maic_agent import MAICAgent
from .edmac_agent_ablation import EDMACAblationAgent
from .edmac_agent_ablation_lately_update import EDMACAblationLateAgent
from .edmac_agent_vae import EDMACVAEAgent
from .bs1_agent import BS1Agent
from .bs1_agent_target_dec import BS1TardecAgent
from .edmac_agent_club import EDMACClubAgent
from .edmac_agent_club_ck import EDMACClubAgent_ck
from .edmac_agent_club_all_env import EDMACClubAgent_all_env
from .edmac_agent_club_all_dec import EDMACClubAgent_all_dec
from .edmac_agent_club_all_envp import EDMACClubAgent_all_envp
from .edmac_agent_club_all_decp import EDMACClubAgent_all_decp
from .edmac_agent_club_head import EDMACClubAgent_head

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["edmac"] = EDMACAgent
REGISTRY["new_edmac"] = EDMACVAEAgent
REGISTRY["masia"] = MASIAAgent
REGISTRY["tarmac"] = TarMACAgent
REGISTRY["tmac_p2p_comm_rnn_msg"] = RnnMsgAgent
REGISTRY["maic"] = MAICAgent
REGISTRY["edmac_ablation"] = EDMACAblationAgent
REGISTRY["edmac_ablation_late"] = EDMACAblationLateAgent
REGISTRY["bs1"] = BS1Agent
REGISTRY["bs1t"] = BS1TardecAgent
REGISTRY["dedmac"] = EDMACClubAgent
REGISTRY["dedmac_ck"] = EDMACClubAgent_ck
REGISTRY["dedmac_all_env"] = EDMACClubAgent_all_env
REGISTRY["dedmac_all_envp"] = EDMACClubAgent_all_envp
REGISTRY["dedmac_all_dec"] = EDMACClubAgent_all_dec
REGISTRY["dedmac_all_decp"] = EDMACClubAgent_all_decp
REGISTRY["dedmac_head"] = EDMACClubAgent_head