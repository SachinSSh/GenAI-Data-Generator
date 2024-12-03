import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import json

class AdvancedDataGenerator:
    def __init__(self):
        """
        Initialize Advanced Data Generator
        """
        # Initialize random seeds
        np.random.seed(42)
        
        # Data generation methods mapping
        self.data_type_generators = {
            'medical': self._generate_medical_data,
            'crop': self._generate_crop_data,
            'student': self._generate_student_data,
            'financial': self._generate_financial_data,
            'customer': self._generate_customer_data
        }
        
        # Default configurations
        self.default_config = {
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
            },
            'student': {
                'num_samples': 1000,
                'fields': [
                    {'name': 'student_id', 'type': 'int', 'min': 1, 'max': 10000},
                    {'name': 'age', 'type': 'int', 'min': 18, 'max': 25},
                    {'name': 'gpa', 'type': 'float', 'min': 2.0, 'max': 4.0},
                    {'name': 'major', 'type': 'categorical', 'categories': ['Computer Science', 'Engineering', 'Biology', 'Economics']},
                    {'name': 'scholarship_eligibility', 'type': 'categorical', 'categories': [True, False]}
                ]
            },
            'financial': {
                'num_samples': 1000,
                'fields': [
                    {'name': 'annual_income', 'type': 'float', 'min': 20000, 'max': 200000},
                    {'name': 'credit_score', 'type': 'int', 'min': 300, 'max': 850},
                    {'name': 'loan_amount', 'type': 'float', 'min': 5000, 'max': 500000},
                    {'name': 'loan_purpose', 'type': 'categorical', 'categories': ['Personal', 'Home', 'Education', 'Business']},
                    {'name': 'default_risk', 'type': 'categorical', 'categories': ['Low', 'Medium', 'High']}
                ]
            },
            'customer': {
                'num_samples': 1000,
                'fields': [
                    {'name': 'customer_id', 'type': 'int', 'min': 1, 'max': 10000},
                    {'name': 'age', 'type': 'int', 'min': 18, 'max': 70},
                    {'name': 'gender', 'type': 'categorical', 'categories': ['Male', 'Female', 'Other']},
                    {'name': 'annual_spend', 'type': 'float', 'min': 1000, 'max': 100000},
                    {'name': 'loyalty_level', 'type': 'categorical', 'categories': ['Bronze', 'Silver', 'Gold', 'Platinum']}
                ]
            }
        }
    
    def _generate_field_data(self, field, num_samples):
        """
        Generate data for a specific field based on its configuration
        
        Args:
            field (dict): Field configuration
            num_samples (int): Number of samples to generate
        
        Returns:
            list: Generated data for the field
        """
        if field['type'] == 'int':
            return np.random.randint(field['min'], field['max'], size=num_samples).tolist()
        elif field['type'] == 'float':
            return np.random.uniform(field['min'], field['max'], size=num_samples).tolist()
        elif field['type'] == 'categorical':
            return np.random.choice(field['categories'], size=num_samples).tolist()
    
    def _generate_medical_data(self, num_samples):
        """
        Generate synthetic medical dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated medical dataset
        """
        data = {}
        for field in self.default_config['medical']['fields']:
            data[field['name']] = self._generate_field_data(field, num_samples)
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
        for field in self.default_config['crop']['fields']:
            data[field['name']] = self._generate_field_data(field, num_samples)
        return pd.DataFrame(data)
    
    def _generate_student_data(self, num_samples):
        """
        Generate synthetic student dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated student dataset
        """
        data = {}
        for field in self.default_config['student']['fields']:
            data[field['name']] = self._generate_field_data(field, num_samples)
        return pd.DataFrame(data)
    
    def _generate_financial_data(self, num_samples):
        """
        Generate synthetic financial dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated financial dataset
        """
        data = {}
        for field in self.default_config['financial']['fields']:
            data[field['name']] = self._generate_field_data(field, num_samples)
        return pd.DataFrame(data)
    
    def _generate_customer_data(self, num_samples):
        """
        Generate synthetic customer dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated customer dataset
        """
        data = {}
        for field in self.default_config['customer']['fields']:
            data[field['name']] = self._generate_field_data(field, num_samples)
        return pd.DataFrame(data)
    
    def generate_dataset(self, dataset_type='medical', num_samples=None):
        """
        Generate a specific type of dataset
        
        Args:
            dataset_type (str): Type of dataset to generate
            num_samples (int, optional): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated dataset
        """
        # Check if dataset type is supported
        if dataset_type.lower() not in self.data_type_generators:
            # Try to find the closest match
            possible_matches = [
                match for match in self.data_type_generators.keys() 
                if dataset_type.lower() in match
            ]
            
            if possible_matches:
                dataset_type = possible_matches[0]
                print(f"Using closest match: {dataset_type}")
            else:
                raise ValueError(f"Unsupported dataset type. Available types: {', '.join(self.data_type_generators.keys())}")
        
        # Use configured sample size if not specified
        if num_samples is None:
            num_samples = self.default_config.get(dataset_type, {}).get('num_samples', 1000)
        
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
    # Initialize data generator
    generator = AdvancedDataGenerator()
    
    # Get user input
    print("Synthetic Data Generator")
    print("Available dataset types: medical, crop, student, financial, customer")
    
    # Prompt for dataset type
    while True:
        dataset_type = input("Enter the type of dataset you want to generate (or 'quit' to exit): ").lower()
        
        if dataset_type == 'quit':
            break
        
        try:
            # Prompt for number of samples
            while True:
                try:
                    num_samples = input("Enter number of samples (default is 1000): ")
                    num_samples = int(num_samples) if num_samples else 1000
                    break
                except ValueError:
                    print("Please enter a valid number.")
            
            # Prompt for export format
            export_format = input("Enter export format (csv/json/excel, default is csv): ").lower() or 'csv'
            
            # Generate dataset
            dataset = generator.generate_dataset(dataset_type, num_samples)
            
            # Print dataset preview
            print(f"\n{dataset_type.capitalize()} Dataset Preview:")
            print(dataset.head())
            
            # Export dataset
            output_dir = f'generated_datasets/{dataset_type}'
            filepath = generator.export_dataset(dataset, output_format=export_format, output_dir=output_dir)
            
            print("\nDataset generated successfully!")
        
        except ValueError as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
