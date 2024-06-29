import pandas as pd

TIDY_TUESDAY_BASE_URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/'

def load_tidy_tuesday_dataset(year, month, day, dataset_name):
    url = f'{TIDY_TUESDAY_BASE_URL}/master/data/{year}/{year}-{month:02d}-{day:02d}/{dataset_name}.csv'
    return pd.read_csv(url)


# # Example datasets
# nhl_player_births = load_tidy_tuesday_dataset(2024, 1, 9, 'nhl_player_births')
# print(nhl_player_births.iloc[:5])
#
#
# bird_counts = load_tidy_tuesday_dataset(2019, 6, 18, 'bird_counts')
# print(bird_counts.iloc[:5])
#
#
# volcano_eruptions = load_tidy_tuesday_dataset(2020, 5, 12, 'eruptions')
# print(volcano_eruptions.iloc[:5])
#
#
# gbb_bakers = load_tidy_tuesday_dataset(2022, 10, 25, 'bakers')
# print(gbb_bakers.iloc[:5])