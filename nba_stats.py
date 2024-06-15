import csv
from basketball_reference_web_scraper import client

# Function to pull data for every year from 1950 to 2024 and write to a CSV
def get_all_season_totals(start_year, end_year, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header_written = False

        for year in range(start_year, end_year + 1):
            try:
                season_data = client.players_season_totals(season_end_year=year)
                
                if season_data and not header_written:
                    # Write header based on the keys of the first dictionary in season_data
                    header = season_data[0].keys()
                    writer.writerow(['Year'] + list(header))
                    header_written = True
                
                for player_stats in season_data:
                    writer.writerow([year] + list(player_stats.values()))
                print(f"Successfully retrieved and wrote data for {year}")
            except Exception as e:
                print(f"Failed to retrieve data for {year}: {e}")

# Example usage: Retrieve data from 2010 to 2024 and write to a CSV file
output_file = "nba_season_totals_2010_2024.csv"
get_all_season_totals(2010, 2024, output_file)
