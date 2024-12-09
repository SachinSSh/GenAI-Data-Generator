import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import requests
from bs4 import BeautifulSoup
import random
import re

class AdvancedDataGenerator:
    def __init__(self):
        """
        Initialize Advanced Data Generator
        """
        # Initialize random seeds
        np.random.seed(42)
        random.seed(42)
        
        # Data sources for web scraping
        self.data_sources = {
            'universities': 'https://www.4icu.org/top-universities/index1.htm',
            'companies': 'https://www.forbes.com/lists/americas-largest-private-companies/',
            'tech_skills': 'https://www.tiobe.com/tiobe-index/'
        }
        
        # Enhanced data generation methods
        self.data_type_generators = {
            'medical': self._generate_enhanced_medical_data,
            #'crop': self._generate_enhanced_crop_data,
            'student': self._generate_enhanced_student_data,
            #'financial': self._generate_enhanced_financial_data,
            #'customer': self._generate_enhanced_customer_data,
            'tech': self._generate_tech_professional_data
        }
        
        # Cached web scraping results
        self._cached_web_data = {}
    
    def _safe_web_scrape(self, url, parser='html.parser'):
        """
        Safely scrape web data with caching
        
        Args:
            url (str): URL to scrape
            parser (str): BeautifulSoup parser type
        
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        if url in self._cached_web_data:
            return self._cached_web_data[url]
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, parser)
            self._cached_web_data[url] = soup
            return soup
        except Exception as e:
            print(f"Web scraping error for {url}: {e}")
            return None
    
    def _scrape_universities(self):
        """
        Scrape university names
        
        Returns:
            list: University names
        """
        soup = self._safe_web_scrape(self.data_sources['universities'])
        if not soup:
            return ['Harvard University', 'Stanford University', 'MIT', 'Yale University', 
                    'Princeton University', 'Columbia University', 'UC Berkeley', 'CalTech']
        
        universities = []
        for row in soup.find_all('tr'):
            cols = row.find_all('td')
            if cols and len(cols) > 1:
                uni_name = cols[1].get_text(strip=True)
                if uni_name and uni_name not in universities:
                    universities.append(uni_name)
        
        return universities[:50] or ['Harvard University', 'Stanford University', 'MIT', 'Yale University']
    
    def _generate_enhanced_medical_data(self, num_samples):
        """
        Generate enhanced synthetic medical dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated medical dataset
        """
        data = {
            'patient_id': range(1, num_samples + 1),
            'patient_age': np.random.randint(18, 90, num_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
            'blood_pressure_systolic': np.random.normal(130, 20, num_samples).round(1),
            'blood_pressure_diastolic': np.random.normal(80, 15, num_samples).round(1),
            'cholesterol_level': np.random.normal(200, 40, num_samples).round(1),
            'diabetes_risk': np.random.choice(['Low', 'Medium', 'High'], num_samples, 
                                               p=[0.6, 0.3, 0.1]),
            'smoker': np.random.choice([True, False], num_samples, p=[0.2, 0.8]),
            'exercise_hours_per_week': np.random.uniform(0, 10, num_samples).round(1),
            'bmi': np.random.normal(25, 5, num_samples).round(1),
            'medical_condition': np.random.choice([
                'Healthy', 'Hypertension', 'Diabetes', 'High Cholesterol', 
                'Heart Disease', 'Respiratory Issue'
            ], num_samples)
        }
        return pd.DataFrame(data)
    
    def _generate_enhanced_student_data(self, num_samples):
        """
        Generate enhanced synthetic student dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated student dataset
        """
        # Scrape universities for more realistic data
        universities = self._scrape_universities()
        
        data = {
            'student_id': range(1, num_samples + 1),
            'age': np.random.randint(18, 25, num_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
            'gpa': np.random.uniform(2.0, 4.0, num_samples).round(2),
            'university': np.random.choice(universities, num_samples),
            'major': np.random.choice([
                'Computer Science', 'Engineering', 'Biology', 'Economics', 
                'Psychology', 'Mathematics', 'Chemistry', 'Data Science'
            ], num_samples),
            'scholarship_eligibility': np.random.choice([True, False], num_samples, p=[0.3, 0.7]),
            'international_student': np.random.choice([True, False], num_samples, p=[0.2, 0.8]),
            'programming_skills': np.random.choice([
                'Beginner', 'Intermediate', 'Advanced', 'Expert'
            ], num_samples),
            'extracurricular_activities': np.random.choice([
                'Sports', 'Music', 'Debate', 'Research', 'Volunteering', 'None'
            ], num_samples)
        }
        return pd.DataFrame(data)
    
    
    
    def _generate_tech_professional_data(self, num_samples):
        """
        Generate synthetic tech professional dataset
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated tech professional dataset
        """
        programming_languages = [
            'Python', 'JavaScript', 'Java', 'C++', 'Ruby', 'Go', 'Rust', 
            'Swift', 'Kotlin', 'TypeScript'
        ]
        
        data = {
            'professional_id': range(1, num_samples + 1),
            'age': np.random.randint(22, 55, num_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
            'primary_programming_language': np.random.choice(programming_languages, num_samples),
            'years_of_experience': np.random.randint(0, 20, num_samples),
            'current_role': np.random.choice([
                'Software Engineer', 'Data Scientist', 'DevOps Engineer', 
                'Machine Learning Engineer', 'Full Stack Developer', 
                'Cloud Architect', 'Security Specialist'
            ], num_samples),
            'salary': np.random.normal(90000, 30000, num_samples).round(2),
            'remote_work': np.random.choice([True, False], num_samples, p=[0.6, 0.4]),
            'highest_education': np.random.choice([
                'Bachelor', 'Master', 'PhD', 'Bootcamp Graduate'
            ], num_samples),
            'tech_certifications': np.random.choice([
                'AWS Certified', 'CISSP', 'Google Cloud', 'Azure', 'None'
            ], num_samples)
        }
        return pd.DataFrame(data)
    
    def export_dataset(self, dataset, output_format='csv', output_dir='generated_datasets', dataset_type='generic'):
        """
        Export generated dataset to specified format with improved naming
        
        Args:
            dataset (pd.DataFrame): Dataset to export
            output_format (str): Export format (csv, json, excel)
            output_dir (str): Directory to save exported datasets
            dataset_type (str): Type of dataset for filename
        
        Returns:
            str: Path to exported dataset
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename with dataset type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{dataset_type}_dataset_{timestamp}"
        
        # Export based on format
        try:
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
        except Exception as e:
            print(f"Export error: {e}")
            return None

def main():
    # Initialize data generator
    generator = AdvancedDataGenerator()
    
    # Get user input
    print("Enhanced Synthetic Data Generator")
    print("Available dataset types: medical, student, tech, financial")
    
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
            dataset = generator.data_type_generators[dataset_type](num_samples)
            
            # Print dataset preview
            print(f"\n{dataset_type.capitalize()} Dataset Preview:")
            print(dataset.head())
            
            # Export dataset
            output_dir = f'generated_datasets/{dataset_type}'
            filepath = generator.export_dataset(
                dataset, 
                output_format=export_format, 
                output_dir=output_dir, 
                dataset_type=dataset_type
            )
            
            print("\nDataset generated successfully!")
        
        except KeyError:
            print(f"Invalid dataset type. Available types: {', '.join(generator.data_type_generators.keys())}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
