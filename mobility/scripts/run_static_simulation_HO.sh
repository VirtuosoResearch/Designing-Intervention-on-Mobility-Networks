msa_name=HO
beta_base=0.17748305453155
poi_psi=0.00298783355930082
p_zero=0.0002
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 100 \
    --lp_budget $budget --lp_components 24 --lp_epochs 50\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
