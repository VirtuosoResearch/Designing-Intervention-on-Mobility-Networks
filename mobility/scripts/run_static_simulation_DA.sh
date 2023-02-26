msa_name=DA
beta_base=0.206160508358388
poi_psi=0.000130940700529142
p_zero=0.0002
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 10000 \
    --lp_budget $budget --lp_components 44 --lp_epochs 21\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
