REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .edmac_controller import EDMACMAC
from .masia_controller import MASIAMAC
from .tar_comm_controller import TarCommMAC
from .tmac_p2p_comm_controller import VffacMAC
from .maic_controller import MAICMAC
from .edmac_controller_ablation_lately_update import EDMACLATEMAC
from .bs1_controller import BS1MAC
from .edmac_club_controller import EDMACCLUBMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY["edmac_mac"] = EDMACMAC
REGISTRY["masia_mac"] = MASIAMAC
REGISTRY["tar_comm_mac"] = TarCommMAC
REGISTRY["tmac_p2p_comm_mac"] = VffacMAC
REGISTRY["maic_mac"] = MAICMAC
REGISTRY["edmac_ablation_late_mac"] = EDMACLATEMAC
REGISTRY["bs1_mac"] = BS1MAC
REGISTRY["dedmac"] = EDMACCLUBMAC