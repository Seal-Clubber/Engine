import pandas as pd

class Test:
    def __init__(self, datapath: str = None) -> bool:
        self.datapath = datapath

    def check_csv_observations(self, has_header: bool = False):
        """
        Check if a CSV file has fewer than 3 observations.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            bool: True if file has fewer than 3 rows, False otherwise
        """
        return len(pd.read_csv(self.datapath, nrows=3, header=0 if has_header else None)) < 3
    

natgas = Test("aggregatee.csv")
print(natgas.check_csv_observations())