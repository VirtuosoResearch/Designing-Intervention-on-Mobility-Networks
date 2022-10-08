net_name=advogato
budget=0.20
epochs=50

python run_simulation.py --net_name $net_name --strategy none --epochs $epochs
python run_simulation.py --net_name $net_name --budget $budget\
    --strategy uniform --epochs $epochs

python run_simulation.py --net_name $net_name --budget $budget\
    --strategy edge_weight --epochs $epochs

python run_simulation.py --net_name $net_name --budget $budget\
    --strategy edge_centrality_delete --components 1 --epochs $epochs

python run_simulation.py --net_name $net_name --budget $budget\
    --strategy global --top_k 100 --components 1 \
    --lp_budget 0.20 \
    --lp_components 7 \
    --lp_epochs 12 --epochs $epochs