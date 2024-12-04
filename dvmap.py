import numpy as np
import pandas as pd
#import tensorflow as tf
from faker import Faker
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import json
import csv

class AdvancedDataGenerator:
    def __init__(self, config_path='data_config.json'):
        """
        Initialize Advanced Data Generator with configuration support
        
        Args:
            config_path (str): Path to JSON configuration file
        """
        # Initialize random seeds
        np.random.seed(42)
  
        
        # Faker for generating realistic data
        self.fake = Faker()
        
        # Load or create configuration
        self.config = self._load_or_create_config(config_path)
        
        # Data generation methods mapping
        self.data_type_generators = {
            'medical': self._generate_medical_data,
            'crop': self._generate_crop_data,
            'student': self._generate_student_data,
            'financial': self._generate_financial_data,
            'customer': self._generate_customer_data
        }
    
    def _load_or_create_config(self, config_path):
        """
        Load existing configuration or create default
        
        Args:
            config_path (str): Path to configuration file
        
        Returns:
            dict: Configuration dictionary
        """
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            'medical': {
                'num_samples': 1000,
                'fields': [
                    {'name': 'patient_age', 'type': 'int', 'min': 18, 'max': 90},
                    {'name': 'blood_pressure_systolic', 'type': 'float', 'min': 90, 'max': 200},
                    {'name': 'blood_pressure_diastolic', 'type': 'float', 'min': 60, 'max': 120},
                    {'name': 'cholesterol_level', 'type': 'float', 'min': 100, 'max': 300},
                    {'name': 'diabetes_risk', 'type': 'categorical', 'categories': ['Low', 'Medium', 'High']}
                ]
            },
            'crop': {
                'num_samples': 500,
                'fields': [
                    {'name': 'crop_type', 'type': 'categorical', 'categories': ['Wheat', 'Corn', 'Rice', 'Barley']},
                    {'name': 'soil_ph', 'type': 'float', 'min': 5.5, 'max': 7.5},
                    {'name': 'rainfall', 'type': 'float', 'min': 200, 'max': 1000},
                    {'name': 'yield_per_hectare', 'type': 'float', 'min': 1, 'max': 10},
                    {'name': 'fertilizer_usage', 'type': 'float', 'min': 50, 'max': 300}
                ]
            }
        }
        
        # Save default configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    
    def _generate_field_data(self, field):
        """
        Generate data for a specific field based on its configuration
        
        Args:
            field (dict): Field configuration
        
        Returns:
            np.ndarray: Generated data for the field
        """
        if field['type'] == 'int':
            return np.random.randint(field['min'], field['max'], size=1)[0]
        elif field['type'] == 'float':
            return np.random.uniform(field['min'], field['max'])
        elif field['type'] == 'categorical':
            return np.random.choice(field['categories'])
        elif field['type'] == 'date':
            start = datetime.strptime(field['start'], '%Y-%m-%d')
            end = datetime.strptime(field['end'], '%Y-%m-%d')
            return start + timedelta(days=np.random.randint((end - start).days))
    
    def _generate_medical_data(self, num_samples):
        """
        Generate synthetic medical dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated medical dataset
        """
        data = {}
        for field in self.config['medical']['fields']:
            data[field['name']] = [
                self._generate_field_data(field) for _ in range(num_samples)
            ]
        return pd.DataFrame(data)
    
    def _generate_crop_data(self, num_samples):
        """
        Generate synthetic crop dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated crop dataset
        """
        data = {}
        for field in self.config['crop']['fields']:
            data[field['name']] = [
                self._generate_field_data(field) for _ in range(num_samples)
            ]
        return pd.DataFrame(data)
    
    def _generate_student_data(self, num_samples=1000):
        """
        Generate synthetic student dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated student dataset
        """
        return pd.DataFrame({
            'student_id': range(1, num_samples + 1),
            'age': np.random.randint(18, 25, num_samples),
            'gpa': np.random.uniform(2.0, 4.0, num_samples).round(2),
            'major': np.random.choice(['Computer Science', 'Engineering', 'Biology', 'Economics'], num_samples),
            'scholarship_eligibility': np.random.choice([True, False], num_samples, p=[0.3, 0.7])
        })
    
    def _generate_financial_data(self, num_samples=1000):
        """
        Generate synthetic financial dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated financial dataset
        """
        return pd.DataFrame({
            'annual_income': np.random.uniform(20000, 200000, num_samples).round(2),
            'credit_score': np.random.randint(300, 850, num_samples),
            'loan_amount': np.random.uniform(5000, 500000, num_samples).round(2),
            'loan_purpose': np.random.choice(['Personal', 'Home', 'Education', 'Business'], num_samples),
            'default_risk': np.random.choice(['Low', 'Medium', 'High'], num_samples)
        })
    
    def _generate_customer_data(self, num_samples=1000):
        """
        Generate synthetic customer dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated customer dataset
        """
        return pd.DataFrame({
            'customer_id': range(1, num_samples + 1),
            'age': np.random.randint(18, 70, num_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
            'annual_spend': np.random.uniform(1000, 100000, num_samples).round(2),
            'loyalty_level': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_samples)
        })
    
    def generate_dataset(self, dataset_type='medical', num_samples=None):
        """
        Generate a specific type of dataset
        
        Args:
            dataset_type (str): Type of dataset to generate
            num_samples (int, optional): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated dataset
        """
        if dataset_type not in self.data_type_generators:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Use configured sample size if not specified
        if num_samples is None:
            num_samples = self.config.get(dataset_type, {}).get('num_samples', 1000)
        
        # Generate dataset
        dataset = self.data_type_generators[dataset_type](num_samples)
        return dataset
    
    def export_dataset(self, dataset, output_format='csv', output_dir='generated_datasets'):
        """
        Export generated dataset to specified format
        
        Args:
            dataset (pd.DataFrame): Dataset to export
            output_format (str): Export format (csv, json, excel)
            output_dir (str): Directory to save exported datasets
        
        Returns:
            str: Path to exported dataset
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"dataset_{timestamp}"
        
        # Export based on format
        if output_format == 'csv':
            filepath = os.path.join(output_dir, f"{base_filename}.csv")
            dataset.to_csv(filepath, index=False)
        elif output_format == 'json':
            filepath = os.path.join(output_dir, f"{base_filename}.json")
            dataset.to_json(filepath, orient='records')
        elif output_format == 'excel':
            filepath = os.path.join(output_dir, f"{base_filename}.xlsx")
            dataset.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
        
        print(f"Dataset exported to: {filepath}")
        return filepath

def main():
    try:
        # Initialize data generator
        generator = AdvancedDataGenerator()
        
        # Generate different types of datasets
        dataset_types = ['medical', 'crop', 'student', 'financial', 'customer']
        
        for dataset_type in dataset_types:
            # Generate dataset
            dataset = generator.generate_dataset(dataset_type)
            
            # Export dataset in multiple formats
            formats = ['csv', 'json']
            for export_format in formats:
                generator.export_dataset(
                    dataset, 
                    output_format=export_format, 
                    output_dir=f'generated_datasets/{dataset_type}'
                )
            
            # Print dataset preview
            print(f"\n{dataset_type.capitalize()} Dataset Preview:")
            print(dataset.head())
            print("\n" + "="*50)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
