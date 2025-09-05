#!/usr/bin/env python3
"""
College Predictor - Main Application
This script demonstrates how to use the college admission prediction system
"""

from data_loader import CollegeDataLoader
from ml_model import CollegePredictionModel
import pandas as pd

print("Welcome to the college prdictor : ""🎓")
def main():
    print("🎓 College Admission Predictor")
    print("=" * 50)
    
    # 1. Load data
    print("\n📂 Loading admission data...")
    loader = CollegeDataLoader()
    
    try:
        data = loader.load_all_data()
        summary = loader.get_data_summary()
        
        print(f"📊 Data Summary:")
        print(f"   • Total records: {summary['total_records']:,}")
        print(f"   • Years: {summary['years']}")
        print(f"   • Rounds: {summary['rounds']}")
        print(f"   • Institutes: {summary['institutes']}")
        print(f"   • Branches: {summary['branches']}")
        print(f"   • Categories: {summary['categories']}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Train ML model
    print("\n🤖 Training Machine Learning Model...")
    model = CollegePredictionModel()
    
    try:
        performance = model.train_models(data)
        model.save_model('college_predictor_model.pkl')
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # 3. Model trained successfully
    print("\n✅ College Prediction System Ready!")
    print("💡 Use 'python predictor_interface.py' for recommendations based on your rank!")

def interactive_prediction():
    """
    Redirects to the proper interactive interface
    """
    print("\n🎯 Interactive College Prediction")
    print("=" * 40)
    print("💡 For the best experience, please use:")
    print("   python predictor_interface.py")
    print("\n🏫 This will show you college recommendations based on your rank!")
    print("   (No need to guess round numbers - just enter your rank!)")

if __name__ == "__main__":
    # Run the main analysis
    main()
    
    # Optional: Run interactive prediction
    while True:
        user_choice = input("\n🤔 Do you want to see interactive prediction info? (y/n): ").lower()
        if user_choice == 'y':
            interactive_prediction()
        else:
            break
    
    print("\n👋 Thank you for using College Admission Predictor!")
