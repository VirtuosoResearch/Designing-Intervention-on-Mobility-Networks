msa_name=PH
beta_base=0.136110588507162
poi_psi=0.00428064406178194
p_zero=0.0005
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 100 \
    --lp_budget $budget --lp_components 22 --lp_epochs 49\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
