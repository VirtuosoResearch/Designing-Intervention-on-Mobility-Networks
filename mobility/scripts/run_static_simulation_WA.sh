msa_name=WA
beta_base=0.155299241365275
poi_psi=0.00133618171788944
p_zero=0.0005
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 1000 \
    --lp_budget $budget --lp_components 11 --lp_epochs 11\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
