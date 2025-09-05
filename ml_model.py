import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
from typing import Dict, Tuple, List, Any
warnings.filterwarnings('ignore')



class CollegePredictionModel:
    """
    Machine Learning model for predicting college admission cutoff ranks
    """
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_performance = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning
        """
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Remove rows with invalid ranks
        data = data[(data['opening_rank'] > 0) & (data['closing_rank'] > 0)]
        
        # Create additional features
        data['rank_difference'] = data['closing_rank'] - data['opening_rank']
        data['avg_rank'] = (data['opening_rank'] + data['closing_rank']) / 2
        data['is_competitive'] = (data['closing_rank'] <= 10000).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['category', 'quota', 'gender', 'institute_type', 'institute_name', 'branch']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                # For new data, handle unseen categories
                try:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].astype(str))
                except ValueError:
                    # Handle unseen categories by assigning a default value
                    unique_values = list(self.label_encoders[col].classes_)
                    # Use the most common encoded value (usually 0)
                    data[f'{col}_encoded'] = 0
        
        # Select features for training
        self.feature_columns = [
            'year', 'round', 'is_pwd', 'category_encoded', 'quota_encoded', 
            'gender_encoded', 'institute_type_encoded', 'institute_name_encoded', 
            'branch_encoded', 'opening_rank'
        ]
        
        return data
    
    def train_models(self, df: pd.DataFrame, target: str = 'closing_rank'):
        """
        Train multiple models and select the best one
        """
        print("🚀 Preparing data for training...")
        data = self.prepare_features(df)
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("🤖 Training models...")
        
        best_score = float('inf')
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['linear', 'ridge']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.model_performance[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'R2': r2
            }
            
            print(f"  {name}: MAE={mae:.2f}, RMSE={np.sqrt(mse):.2f}, R2={r2:.4f}")
            
            # Select best model based on MAE
            if mae < best_score:
                best_score = mae
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best model: {self.best_model_name} (MAE: {best_score:.2f})")
        
        return self.model_performance
    
    def predict_cutoff(self, 
                      year: int,
                      round_num: int,
                      category: str,
                      quota: str,
                      gender: str,
                      institute_type: str,
                      institute_name: str,
                      branch: str,
                      is_pwd: int = 0,
                      opening_rank: int = None) -> Dict[str, Any]:
        """
        Predict closing rank cutoff for given parameters
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Create input data
        input_data = pd.DataFrame({
            'year': [year],
            'round': [round_num],
            'category': [category],
            'is_pwd': [is_pwd],
            'quota': [quota],
            'gender': [gender],
            'institute_type': [institute_type],
            'institute_name': [institute_name],
            'branch': [branch],
            'opening_rank': [opening_rank] if opening_rank else [1000],  # Default value
            'closing_rank': [0]  # Dummy value, will be predicted
        })
        
        # Prepare features
        prepared_data = self.prepare_features(input_data)
        
        # Check if all required columns exist
        missing_cols = [col for col in self.feature_columns if col not in prepared_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in prepared data: {missing_cols}")
        
        X = prepared_data[self.feature_columns]
        
        # Make prediction
        if self.best_model_name in ['linear', 'ridge']:
            X_scaled = self.scaler.transform(X)
            prediction = self.best_model.predict(X_scaled)[0]
        else:
            prediction = self.best_model.predict(X)[0]
        
        # Get feature importance (if available)
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance = importance_df.to_dict('records')[:5]  # Top 5 features
        
        return {
            'predicted_closing_rank': max(1, int(prediction)),
            'model_used': self.best_model_name,
            'confidence': 'High' if self.model_performance[self.best_model_name]['R2'] > 0.8 else 'Medium',
            'feature_importance': feature_importance
        }
    
    def get_admission_probability(self, 
                                student_rank: int,
                                year: int,
                                round_num: int,
                                category: str,
                                quota: str,
                                gender: str,
                                institute_type: str,
                                institute_name: str,
                                branch: str,
                                is_pwd: int = 0) -> Dict[str, Any]:
        """
        Calculate probability of admission based on predicted cutoff
        """
        prediction = self.predict_cutoff(
            year, round_num, category, quota, gender,
            institute_type, institute_name, branch, is_pwd
        )
        
        predicted_cutoff = prediction['predicted_closing_rank']
        
        if student_rank <= predicted_cutoff:
            probability = min(95, 80 + (predicted_cutoff - student_rank) / predicted_cutoff * 15)
            status = "High Chance"
        elif student_rank <= predicted_cutoff * 1.1:
            probability = 50 + (predicted_cutoff * 1.1 - student_rank) / (predicted_cutoff * 0.1) * 30
            status = "Moderate Chance"
        else:
            probability = max(5, 50 - (student_rank - predicted_cutoff * 1.1) / predicted_cutoff * 45)
            status = "Low Chance"
        
        return {
            'student_rank': student_rank,
            'predicted_cutoff': predicted_cutoff,
            'admission_probability': round(probability, 1),
            'status': status,
            'recommendation': self._get_recommendation(student_rank, predicted_cutoff, status)
        }
    
    def _get_recommendation(self, student_rank: int, cutoff: int, status: str) -> str:
        """
        Provide recommendation based on admission probability
        """
        if status == "High Chance":
            return "Great choice! You have a strong chance of admission."
        elif status == "Moderate Chance":
            return "Consider this as a target option. Also explore similar alternatives."
        else:
            return "This might be challenging. Consider it as a reach option and have backup plans."
    
    def save_model(self, filepath: str):
        """
        Save the trained model and encoders
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_performance': self.model_performance
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a previously trained model
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_performance = model_data['model_performance']
        print(f"✅ Model loaded from {filepath}")
