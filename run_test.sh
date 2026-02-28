CUDA_VISIBLE_DEVICES=0 python src/main.py --config=edmac --env-config=lbf 
CUDA_VISIBLE_DEVICES=0 python src/main.py --config=bs1 --env-config=sc2 with env_args.map_name=1o_10b_vs_1r

cd EDMAC/ ; 
conda activate pymarl

python3 src/main.py --config=edmac_vae --env-config=sc2 with env_args.map_name=1o_10b_vs_1r

4090
bs1
bash run.sh dedmac_all_env lbf 11x11-6p-4f-s1 name=dedmac_all_env 5 0,1 5
bs2
bash run.sh dedmac_all_dec lbf 11x11-6p-4f-s1 name=dedmac_all_dec 5 1,0 5
bs4
bash run.sh dedmac_all_envp lbf 11x11-6p-4f-s1 name=dedmac_all_envp 5 0,1 5
bs5
bash run.sh dedmac_all_decp lbf 11x11-6p-4f-s1 name=dedmac_all_decp 5 1,0 5
bs6
bash run.sh dedmac_head lbf 11x11-6p-4f-s1 name=dedmac_head 5 0,1 5
bs3
bash run.sh bs1 sc2 1o_10b_vs_1r name=bs1_ablation_loss_s,loss_weight_state=0.0 5 0,1 5
bs7
bash run.sh bs1 sc2 1o_10b_vs_1r name=bs1_ablation_loss_q,loss_weight_q=0.0 5 1,0 5

3090
bs1
bash run.sh bs1 sc2 1o_10b_vs_1r name=bs1 5 0,1,2,3 5
bs2
bash run.sh bs1 sc2 1o_2r_vs_4r name=bs1 5 1,2,3,0 5
bs3
bash run.sh bs1 lbf 20x20-10p-6f-s1 name=bs1 5 2,3,0,1 5
bs4
bash run.sh bs1 lbf11 11x11-6p-4f-s1 name=bs1 5 3,0,1,2 5
bs5
bash run.sh bs1 hallway_group 354610 name=bs1 5 0,1,2,3 5
bs6
bash run.sh bs1 hallway 4610 name=bs1 5 1,2,3,0 5


|||||||||||||||||
plan


|||||||||||||||||||||||||
finish
4090
bs2
bash run.sh dedmac_para hallway_group 354610 name=dedmac 5 0,1 5
bs1
bash run.sh dedmac_para hallway 4610 name=dedmac 5 1,0 5
4090
bs1
bash run.sh dedmac lbf 11x11-6p-4f-s1 name=dedmac_ablation_loss_s,loss_weight_s=0.0 5 0,1 5
bs2
bash run.sh dedmac lbf 11x11-6p-4f-s1 name=dedmac_ablation_loss_a,loss_weight_a=0.0 5 0,1 5
bs3
bash run.sh dedmac lbf 11x11-6p-4f-s1 name=dedmac_ablation_loss_m,loss_weight_msg=0.0,loss_weight_est=0.0 5 1,0 5
3090
bs5
bash run.sh dedmac sc2 1o_10b_vs_1r name=dedmac 5 2,3 5
bs6
bash run.sh dedmac sc2 1o_2r_vs_4r name=dedmac 5 3,2 5
bs4
bash run.sh dedmac lbf 20x20-10p-6f-s1 name=dedmac 5 3,2 5
bs3
bash run.sh dedmac lbf11 11x11-6p-4f-s1 name=dedmac 5 2,3 5









|||||||others
bash run.sh bs1 hallway_group 354610 loss_weight_state=0.5,name=bs1_5 5 3,0 5

bash run.sh edmac lbf new_20x20-10p-6f-s1 epsilon_anneal_time=50000 5 0,1,2,3 5

bash run.sh edmac lbf new_20x20-10p-6f-s1 epsilon_anneal_time=50000 5 0,1,2,3 5
bash run.sh qmix sc2 6h_vs_8z epsilon_anneal_time=500000,td_lambda=0.3 5 0,1,2,3 5
bash run.sh actionmix sc2 6h_vs_8z epsilon_anneal_time=500000,td_lambda=0.3 5 1,2,3,0 5

bash run.sh edmac hallway_group 354610 epsilon_anneal_time=50000 1 0 1

bash run.sh edmac_ablation hallway_group 354610 name=env-only,enable_dec=False 5 0,1 5
bash run.sh edmac_ablation hallway_group 354610 name=dec-only,enable_env=False 5 1,0 5
bash run.sh edmac_ablation hallway_group 354610 name=no-split,enable_split=False 5 0,1 5
bash run.sh edmac_ablation hallway_group 354610 name=no-weight,enable_weight=False 5 1,0 5

bash run.sh edmac_ablation hallway_group 354610 name=ablation000,loss_weight_cos=0.00,loss_weight_state=0.0,loss_weight_kl=0.00 5 0,1,2,3 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation001,loss_weight_cos=0.00,loss_weight_state=0.0,loss_weight_kl=0.01 5 1,2,3,0 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation010,loss_weight_cos=0.00,loss_weight_state=0.5,loss_weight_kl=0.00 5 2,3,0,1 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation011,loss_weight_cos=0.00,loss_weight_state=0.5,loss_weight_kl=0.01 5 3,0,1,2 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation100,loss_weight_cos=0.01,loss_weight_state=0.0,loss_weight_kl=0.00 5 0,1 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation101,loss_weight_cos=0.01,loss_weight_state=0.0,loss_weight_kl=0.01 5 1,0 5
bash run.sh edmac_ablation hallway_group 354610 name=ablation110,loss_weight_cos=0.01,loss_weight_state=0.5,loss_weight_kl=0.00 5 0,1 5

bash run.sh edmac_ablation hallway_group 354610 name=no-denoising,denoising=False 5 1,2,3,0 5
bash run.sh edmac_ablation_lately_update hallway_group 354610 name=late 5 2,3,0,1 5