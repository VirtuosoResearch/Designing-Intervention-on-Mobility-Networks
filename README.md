# Spectrally Optimal Intervention on Weighted Networks via Iterative Edge Centrality Minimization

### Overview

In this work, we propose a spectral method on networks to reduce the number of infections in the spread of an epidemic. Our algorithm, Frank-Wolfe-Edge Centrality, aims to minimize the sum of the largest r singular values in the network. Specifically, our algorithm iterates through finding the gradient descent direction of the objective. The descent direction is given by a greedy selection of edges with the highest edge centrality values. We provide the code to run on both mobility networks and weighted graphs in our experiments.

### Requirements

To install requirements:

``` pip install -r requirements.txt ```

### Data Preparation

For **mobility networks**, we follow the procedures of (Chang et al., 2021) to derive the networks from [SafeGraph COVID-19 Consortium](https://www.safegraph.com/academics). We provide an example of New York mobility network in this code. For other MSA networks, we will release the networks we use after the review process. For source data to generate the networks, we refer to the SafeGraph website for getting access to the data.
- The data for New York mobility network is placed in `./data/generated_networks/NY/`. Our code handles loading the network.

For **weighted graphs**, We list the link for downloading these datasets and describe how to prepare data to run our code below.
- [Airport](http://opsahl.co.uk/tnet/datasets/openflights.txt): download and place the data in `./data/social_networks/airport/`. 
- [Advogato](https://downloads.skewed.de/mirror/konect.cc/files/download.tsv.advogato.tar.bz2): download and place the data in `./data/social_networks/advogato/`. 
- [Bitcoin](http://snap.stanford.edu/data/soc-sign-bitcoinalpha.html): download and place the data in `./data/social_networks/bitcoin/`. 

If one places the data in a different path, please change the path name in `./social/build_social_networks/`. Our code handles loading the networks.

### Usage

For **mobility networks**, run the experiments inside the `mobility` path. We provide scripts in `./mobility/scripts/run_static_simulation_NY.sh` to replicate the experiments for New York mobility network. For running on other mobility networks, one needs to specify the `--MSA` argument. In the script, specify the `--strategy` argument to run different strategies:
- `none` for no intervention
- `uniform` for uniform scaling
- `edge_weight` for edge weighted reduction
- `capped` for max occupancy capping
- `category` for capping by POI category
- `edge_centrality_delete` for K-EdgeDeletion
- `global` for our Frank-Wolfe-EC algorithm

For **weighted graphs**, run the experiments inside the `social` path. We provide scripts in `./social/scripts/` to replicate the experiments:
- run `run_intervention_AIR.sh` for the Airport network
- run `run_intervention_AD.sh` for the Advogato network
- run `run_intervention_BI.sh` for the Bitcoin network

