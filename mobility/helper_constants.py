import os

DATA_DIR = "../data/"
NETWORK_DIR = os.path.join(DATA_DIR, "network")
CENSUS_DIR = os.path.join(DATA_DIR, "census/safegraph_open_census_data_2018/data/cbg_b01.csv")
PATTERN_DIR = lambda msa_name: os.path.join(DATA_DIR, f"patterns/{msa_name}/core_poi-patterns.csv")
GEOMETRY_DIR = os.path.join(DATA_DIR, "geometry/GA - IL - TX - TX - CA - FL - NY - PA - CA - DC.csv")
WEEKLY_DIR = lambda msa_name: os.path.join(DATA_DIR, f"weekly_patterns/{msa_name}/")

# Real infected cases from New York Times
CASES_DIR = os.path.join(DATA_DIR, "real_infected_cases/us-counties.csv")

# Generated Static Network (Monthly)
MONTHLY_NETWORK_DIR = lambda msa_name: os.path.join(DATA_DIR, f"generated_networks/{msa_name}/")
# Generated Temporal Network (Weekly)
WEEKLY_NETWORK_DIR = lambda msa_name: os.path.join(DATA_DIR, f"generated_weekly_networks/{msa_name}/")

# Network directory constants
MSA_NAMES = ["AT", "CH", "DA", "HO", "LA", "MI", "NY", "PH", "SF", "WA"]
CITY_NAMES = {
 "AT": 'Atlanta',
 "CH": 'Chicago',
 "DA": 'Dallas',
 "HO": 'Houston',
 "LA": 'Los Angeles',
 "MI": 'Miami',
 "NY": 'New York',
 "PH": 'Philadelphia',
 "SF": 'San Francisco',
 "WA": 'Washington'
}
MSA_NETWORKS_NAMES = {
  "NY": "New_York_Newark_Jersey_City_NY_NJ_PA_",
  "SF": "San_Francisco_Oakland_Hayward_CA_"
}

# Suffixes
NETWORK_SUFFIX = "2020-03-01_to_2020-05-02.pkl"
CBG_SUFFIX = "cbg_ids.csv"
POI_SUFFIX = "poi_ids.csv"

''' 
Network Size:
AT: network: (2799, 8433)
CH: network: (5784, 26606)
DA: network: (4069, 15000)
HO: network: (4029, 34866)
LA: network: (69, 25542) TODO
MI: network: (2279, 15559)
NY: network: (10170, 24046)
PH: network: (3547, 15102)
SF: network: (26, 11557) TODO
WA: network: (2564, 8026) whole_population: 5283835.0
'''

Counties = \
{
"AT": ["Fulton", "DeKalb"],
"CH": ["Cook", "DuPage"],
"DA": ["Dallas", "Collin", "Denton", "Rockwall", "Kaufman"],
"HO": ["Harris", "Fort Bend", "Montgomery"],
"LA": ["Los Angeles"],
"MI": ["Miami-Dade"],
"NY": ["New York City"], # ["Bronx", "Kings", "New York", "Queens", "Richmond"],
"PH": ["Philadelphia"],
"SF": ["San Francisco"],
"WA": ["District of Columbia"]
}

States = \
{
"AT": "Georgia",
"CH": "Illinois",
"DA": "Texas",
"HO": "Texas",
"LA": "California",
"MI": "Florida",
"NY": "New York",
"PH": "Pennsylvania",
"SF": "California",
"WA": "District of Columbia"
}

lockdown_poi_list =  set([ 
 'Full-Service Restaurants',
 'Fitness and Recreational Sports Centers',
 'Snack and Nonalcoholic Beverage Bars',
 'Hotels (except Casino Hotels) and Motels',
 'Limited-Service Restaurants',
 'Religious Organizations',
 'Offices of Physicians (except Mental Health Specialists)',
 'Convenience Stores',
 'Supermarkets and Other Grocery (except Convenience) Stores',
 'Used Merchandise Stores',
])

# From 8, March to 8, May
infection_dates = [
 '2020-03-08',
 '2020-03-09',
 '2020-03-10',
 '2020-03-11',
 '2020-03-12',
 '2020-03-13',
 '2020-03-14',
 '2020-03-15',
 '2020-03-16',
 '2020-03-17',
 '2020-03-18',
 '2020-03-19',
 '2020-03-20',
 '2020-03-21',
 '2020-03-22',
 '2020-03-23',
 '2020-03-24',
 '2020-03-25',
 '2020-03-26',
 '2020-03-27',
 '2020-03-28',
 '2020-03-29',
 '2020-03-30',
 '2020-03-31',
 '2020-04-01',
 '2020-04-02',
 '2020-04-03',
 '2020-04-04',
 '2020-04-05',
 '2020-04-06',
 '2020-04-07',
 '2020-04-08',
 '2020-04-09',
 '2020-04-10',
 '2020-04-11',
 '2020-04-12',
 '2020-04-13',
 '2020-04-14',
 '2020-04-15',
 '2020-04-16',
 '2020-04-17',
 '2020-04-18',
 '2020-04-19',
 '2020-04-20',
 '2020-04-21',
 '2020-04-22',
 '2020-04-23',
 '2020-04-24',
 '2020-04-25',
 '2020-04-26',
 '2020-04-27',
 '2020-04-28',
 '2020-04-29',
 '2020-04-30',
 '2020-05-01',
 '2020-05-02',
 '2020-05-03',
 '2020-05-04',
 '2020-05-05',
 '2020-05-06',
 '2020-05-07',
 '2020-05-08',
]