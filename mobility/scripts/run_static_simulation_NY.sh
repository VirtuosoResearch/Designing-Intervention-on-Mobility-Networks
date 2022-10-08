msa_name=NY
beta_base=0.0855188357042064
poi_psi=0.00293994466873546
p_zero=0.0001
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy none \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero 

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy uniform --budget $budget \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy edge_weight --budget $budget \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy capped --budget $budget \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy category --budget $budget \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero

python run_simulation_static.py --MSA $msa_name --epochs 100 --strategy edge_centrality_delete --budget $budget --top_k 1\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 1 \
    --lp_budget $budget --lp_components 42 --lp_epochs 7\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
