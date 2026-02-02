# Ethiopia Financial Inclusion Dashboard

An interactive dashboard for forecasting and analyzing Ethiopia's financial inclusion progress.

## Features

- **Overview Dashboard**: Key metrics and P2P/ATM crossover analysis
- **Trends Analysis**: Interactive time series visualizations with comparative views
- **Forecasts**: 2025-2027 projections with scenario analysis
- **Policy Simulator**: What-if analysis for different interventions
- **Insights & Recommendations**: Answers to consortium questions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ethiopia-fi-forecast.git
cd ethiopia-fi-forecast
```
2. Install dependencies
```pip install -r requirements.txt```
Running the Dashboard

    1. Ensure you have completed Tasks 1-4 to generate the data files

    2. Run the Streamlit app:
    ```
    streamlit run dashboard/app.py
    ```
3.     Open your browser and navigate to http://localhost:8501

Data Requirements

The dashboard requires the following data files in data/processed/:

    enriched_data.csv - Enriched dataset from Task 1

    impact_matrix.csv - Event-impact matrix from Task 3

    forecast_results.csv - Forecast results from Task 4

If these files are not available, the dashboard will use sample data.
Dashboard Sections
1. Overview

    Current inclusion metrics

    P2P vs ATM crossover visualization

    Key growth drivers and challenges

2. Trends Analysis

    Interactive time series charts

    Gender gap analysis

    Urban-rural divide visualization

    Infrastructure correlations

3. Forecasts

    2025-2027 projections for key indicators

    Scenario analysis (optimistic/base/pessimistic)

    Target achievement assessment

    Uncertainty visualization

4. Policy Simulator

    Interactive policy levers

    Impact simulation for different interventions

    Recommendation generation

5. Insights & Recommendations

    Answers to consortium questions

    Strategic recommendations

    Actionable insights for stakeholders

Technical Architecture

    Frontend: Streamlit

    Visualization: Plotly

    Data Processing: Pandas, NumPy

    Forecasting: Scikit-learn, Statsmodels

    Data Storage: CSV files

Customization

To customize the dashboard:

    Add new indicators: Update the indicator lists in app.py

    Modify forecasts: Update the forecasting model in src/forecast_model.py

    Change visualizations: Modify the Plotly charts in the respective render methods

    Add new data: Ensure new data follows the unified schema format

Troubleshooting

Issue: "Data not loaded" error
Solution: Run Tasks 1-4 first to generate required data files

Issue: Slow performance
Solution: Reduce data sampling frequency or use caching

Issue: Visualizations not updating
Solution: Clear browser cache or restart Streamlit app
Contributing

    Fork the repository

    Create a feature branch

    Make your changes

    Submit a pull request

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Global Findex Database team

    National Bank of Ethiopia

    GSMA for mobile money data

    All data contributors and partners
    ```

### 5.3 Unit Tests

**tests/test_data_loader.py:**
```python
import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        
    def test_load_data(self):
        """Test data loading functionality"""
        data, ref_codes = self.loader.load_data()
        
        self.assertIsNotNone(data)
        self.assertIsNotNone(ref_codes)
        
        # Check required columns exist
        required_cols = ['record_type', 'pillar', 'indicator']
        for col in required_cols:
            self.assertIn(col, data.columns)
            
    def test_validate_data(self):
        """Test data validation"""
        data, ref_codes = self.loader.load_data()
        is_valid = self.loader.validate_data()
        
        self.assertIsInstance(is_valid, bool)
        
    def test_get_data_summary(self):
        """Test summary generation"""
        data, ref_codes = self.loader.load_data()
        summary = self.loader.get_data_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_records', summary)
        self.assertIn('record_type_counts', summary)

if __name__ == '__main__':
    unittest.main()
    ```
    