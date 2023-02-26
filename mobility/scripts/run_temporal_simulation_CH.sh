budget=0.05

msa_name=CH
beta_base=0.193010340451304
poi_psi=0.00711154831553788
p_zero=0.0002

python run_simulation_temporal.py --MSA $msa_name --epochs 70 \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero \
    --budget $budget \
    --temporal_strategy global_uniform

python run_simulation_temporal.py --MSA $msa_name --epochs 70 \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero \
    --budget $budget \
    --temporal_strategy global_weighted

for epochs in 0 5 10 15 20
do
python run_simulation_temporal.py --MSA $msa_name --epochs 70 \
    --beta_base $beta_base \
    --poi_psi $poi_psi \
    --p_zero $p_zero \
    --budget $budget --components $components \
    --temporal_strategy global --lp_budget $budget --lp_components 20 --lp_epochs $epochs
done