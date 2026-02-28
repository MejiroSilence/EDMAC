from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner import NQLearner
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .fmac_learner import FMACLearner
from .edmac_learner import EDMACLearner
from .masia_learner import MASIALearner
from .categorical_q_learner import CateQLearner
from .tmac_p2p_comm_learner import QLearner as T2MACLearner
from .maic_learner import MAICLearner
from .edmac_learner_ablation import EDMACAblationLearner
from .edmac_learner_vae import EDMACLearnerVAE
from .bs1_learner import BS1Learner
from .edmac_learner_club import EDMACCLUBLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["edmac_learner"] = EDMACLearner
REGISTRY["masia_learner"] = MASIALearner
REGISTRY["cate_q_learner"] = CateQLearner
REGISTRY["tmac_p2p_comm_learner"] = T2MACLearner
REGISTRY["maic_learner"] = MAICLearner
REGISTRY["edmac_ablation_learner"] = EDMACAblationLearner
REGISTRY["new_edmac_learner"] = EDMACLearnerVAE
REGISTRY["bs1_learner"] = BS1Learner
REGISTRY["dedmac_learner"] = EDMACCLUBLearner