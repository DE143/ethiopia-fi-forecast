import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataLoader:
    def __init__(self, data_path='data/raw'):
        self.data_path = data_path
        self.raw_data = None
        self.reference_codes = None
        
    def load_data(self):
        """Load all raw data files"""
        try:
            # Load main dataset
            self.raw_data = pd.read_csv(
                os.path.join(self.data_path, 'ethiopia_fi_unified_data.csv'),
                parse_dates=['observation_date', 'created_at'],
                low_memory=False
            )
            
            # Load reference codes
            self.reference_codes = pd.read_csv(
                os.path.join(self.data_path, 'reference_codes.csv')
            )
            
            print(f"Loaded {len(self.raw_data)} records")
            print(f"Record types: {self.raw_data['record_type'].unique()}")
            
            return self.raw_data, self.reference_codes
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            # Create sample data structure if files don't exist
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data structure for testing"""
        print("Creating sample data structure...")
        
        # Sample observations (Findex data)
        observations = pd.DataFrame({
            'record_type': ['observation'] * 5,
            'pillar': ['access', 'access', 'access', 'usage', 'usage'],
            'indicator': [
                'Account Ownership Rate',
                'Account Ownership Rate',
                'Account Ownership Rate',
                'Digital Payment Adoption',
                'Mobile Money Account Ownership'
            ],
            'indicator_code': [
                'ACC_OWNERSHIP',
                'ACC_OWNERSHIP',
                'ACC_OWNERSHIP',
                'USG_DIGITAL_PAYMENT',
                'ACC_MM_ACCOUNT'
            ],
            'value_numeric': [14, 22, 35, 10, 4.7],
            'observation_date': pd.to_datetime([
                '2011-01-01', '2014-01-01', '2017-01-01', '2021-01-01', '2021-01-01'
            ]),
            'source_name': ['Global Findex'] * 5,
            'confidence': ['high'] * 5
        })
        
        # Sample events
        events = pd.DataFrame({
            'record_type': ['event'] * 3,
            'name': ['Telebirr Launch', 'M-Pesa Entry', 'Interoperability Launch'],
            'category': ['product_launch', 'market_entry', 'infrastructure'],
            'event_date': pd.to_datetime(['2021-05-01', '2023-08-01', '2022-07-01'])
        })
        
        # Combine
        self.raw_data = pd.concat([observations, events], ignore_index=True)
        
        # Sample reference codes
        self.reference_codes = pd.DataFrame({
            'field': ['record_type', 'pillar', 'category', 'confidence'],
            'code': [
                'observation', 'event', 'impact_link', 'target',
                'access', 'usage', 'quality',
                'policy', 'product_launch', 'market_entry', 'infrastructure',
                'high', 'medium', 'low'
            ],
            'description': [
                'Measured value', 'Historical event', 'Modeled relationship', 'Policy target',
                'Access dimension', 'Usage dimension', 'Quality dimension',
                'Policy change', 'Product/service launch', 'Market entry', 'Infrastructure investment',
                'High confidence', 'Medium confidence', 'Low confidence'
            ]
        })
        
        return self.raw_data, self.reference_codes
    
    def validate_data(self):
        """Validate data against reference codes"""
        if self.raw_data is None or self.reference_codes is None:
            print("Data not loaded. Call load_data() first.")
            return False
        
        validation_results = {}
        
        # Check record_type
        valid_record_types = self.reference_codes[
            self.reference_codes['field'] == 'record_type'
        ]['code'].tolist()
        invalid_record_types = set(self.raw_data['record_type']) - set(valid_record_types)
        validation_results['record_type'] = len(invalid_record_types) == 0
        
        # Check pillar for observations
        valid_pillars = self.reference_codes[
            self.reference_codes['field'] == 'pillar'
        ]['code'].tolist()
        observations = self.raw_data[self.raw_data['record_type'] == 'observation']
        invalid_pillars = set(observations['pillar']) - set(valid_pillars)
        validation_results['pillar'] = len(invalid_pillars) == 0
        
        print("Validation Results:")
        for field, valid in validation_results.items():
            status = "✓" if valid else "✗"
            print(f"  {field}: {status}")
        
        return all(validation_results.values())
    
    def get_data_summary(self):
        """Generate summary statistics"""
        if self.raw_data is None:
            print("Data not loaded. Call load_data() first.")
            return None
        
        summary = {
            'total_records': len(self.raw_data),
            'record_type_counts': self.raw_data['record_type'].value_counts().to_dict(),
            'pillar_counts': self.raw_data['pillar'].value_counts().to_dict(),
            'temporal_range': {
                'start': self.raw_data['observation_date'].min(),
                'end': self.raw_data['observation_date'].max()
            } if 'observation_date' in self.raw_data.columns else None,
            'unique_indicators': self.raw_data['indicator'].nunique() 
            if 'indicator' in self.raw_data.columns else 0
        }
        
        return summary