msa_name=AT
beta_base=0.1394646921559
poi_psi=0.00085636978668962
p_zero=0.0005
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 10000 \
    --lp_budget $budget --lp_components 1 --lp_epochs 3\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
