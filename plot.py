from rl_plotter.logger import Logger
import json
"""
rl_plotter --show --save --avg_group --shaded_err --shaded_std --resample=128
"""

map = "1o_10b_vs_1r"
#algo = "edmac"
algo = "edmac_0.01_0.2_0.02"
results = [6,7,8,9,10]

for i in results:
    with open(f"./results/sacred/{map}/{algo}/{i}/info.json", "r") as f:
        content = json.load(f)
        wrs = content["test_battle_won_mean"]
        ts = content["test_battle_won_mean_T"]
        logger = Logger(exp_name="4", env_name=map)
        for wr, t in zip(wrs,ts):
            logger.update([wr],t)


