import pandas as pd
import os
from typing import List, Tuple
import glob

class CollegeDataLoader:
    """
    Data loader for college admission dataset
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.data = None
        
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all CSV files from all years and rounds
        """
        all_files = glob.glob(f"{self.base_dir}/*/*.csv")
        dataframes = []
        
        for file in all_files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
                print(f"✅ Loaded: {file} ({len(df)} records)")
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
        
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            print(f"\n📊 Total records loaded: {len(self.data)}")
            return self.data
        else:
            raise ValueError("No data files found!")
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        summary = {
            'total_records': len(self.data),
            'years': sorted(self.data['year'].unique()),
            'rounds': sorted(self.data['round'].unique()),
            'institutes': self.data['institute_name'].nunique(),
            'branches': self.data['branch'].nunique(),
            'categories': list(self.data['category'].unique()),
            'quotas': list(self.data['quota'].unique()),
            'institute_types': list(self.data['institute_type'].unique())
        }
        
        return summary
    
    def filter_data(self, 
                   year: int = None, 
                   round_num: int = None, 
                   category: str = None,
                   institute_type: str = None,
                   branch: str = None) -> pd.DataFrame:
        """
        Filter data based on given criteria
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        filtered_data = self.data.copy()
        
        if year:
            filtered_data = filtered_data[filtered_data['year'] == year]
        if round_num:
            filtered_data = filtered_data[filtered_data['round'] == round_num]
        if category:
            filtered_data = filtered_data[filtered_data['category'] == category]
        if institute_type:
            filtered_data = filtered_data[filtered_data['institute_type'] == institute_type]
        if branch:
            filtered_data = filtered_data[filtered_data['branch'].str.contains(branch, case=False, na=False)]
        
        return filtered_data
    
    def get_rank_trends(self, institute_name: str, branch: str) -> pd.DataFrame:
        """
        Get rank trends for a specific institute and branch over years
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        filtered = self.data[
            (self.data['institute_name'] == institute_name) & 
            (self.data['branch'] == branch)
        ]
        
        return filtered.groupby(['year', 'round', 'category']).agg({
            'opening_rank': 'mean',
            'closing_rank': 'mean'
        }).reset_index()
