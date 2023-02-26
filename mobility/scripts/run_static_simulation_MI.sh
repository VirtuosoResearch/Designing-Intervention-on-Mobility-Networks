msa_name=MI
beta_base=0.0169249008573913
poi_psi=0.0347149337071007
p_zero=0.0002
budget=0.05

python run_simulation_static.py --MSA $msa_name --epochs 100 \
    --strategy global --budget $budget --top_k 10000 \
    --lp_budget $budget --lp_components 20 --lp_epochs 49\
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero
done
