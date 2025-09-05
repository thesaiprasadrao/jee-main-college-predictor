#!/usr/bin/env python3
"""
College Predictor - Simple Prediction Interface
Easy-to-use interface for making predictions
"""

from data_loader import CollegeDataLoader
from ml_model import CollegePredictionModel
import pandas as pd

class CollegePredictor:
    """Simple interface for college admission predictions"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.unique_values = {}
        
    def load_model_and_data(self):
        """Load the trained model and data"""
        print("📂 Loading data and model...")
        
        # Load data
        loader = CollegeDataLoader()
        self.data = loader.load_all_data()
        
        # Load model
        self.model = CollegePredictionModel()
        try:
            self.model.load_model('trained_college_predictor.pkl')
            print("✅ Model loaded successfully!")
        except:
            print("⚠️  Pre-trained model not found. Training new model...")
            sample_data = self.data.sample(n=min(20000, len(self.data)), random_state=42)
            self.model.train_models(sample_data)
            self.model.save_model('trained_college_predictor.pkl')
            print("✅ New model trained and saved!")
        
        # Store unique values for validation
        self.unique_values = {
            'categories': list(self.data['category'].unique()),
            'quotas': list(self.data['quota'].unique()),
            'genders': list(self.data['gender'].unique()),
            'institute_types': list(self.data['institute_type'].unique()),
            'institutes': list(self.data['institute_name'].unique()),
            'branches': list(self.data['branch'].unique())
        }
        
    def abbreviate_institute(self, institute_name: str) -> str:
        """Create abbreviated form of institute names"""
        abbreviations = {
            'National Institute of Technology': 'NIT',
            'Indian Institute of Technology': 'IIT',
            'Indian Institute of Information Technology': 'IIIT',
            'Birla Institute of Technology and Science': 'BITS',
            'Vellore Institute of Technology': 'VIT',
            'Malaviya National Institute of Technology': 'MNIT',
            'Maulana Azad National Institute of Technology': 'MANIT',
            'Motilal Nehru National Institute of Technology': 'MNNIT',
            'Sardar Vallabhbhai National Institute of Technology': 'SVNIT',
            'Visvesvaraya National Institute of Technology': 'VNIT',
            'Sant Longowal Institute of Technology': 'SLIET',
            'Atal Bihari Vajpayee Indian Institute of Information Technology': 'ABV-IIIT',
            'University of Hyderabad': 'UoH',
            'Assam University': 'AU',
            'Tezpur University': 'TU',
            'Mizoram University': 'MU',
            'Dr. B R Ambedkar National Institute of Technology': 'NIT Jalandhar',
            'National Institute of Advanced Materials Technology': 'NIAMT'
        }
        
        name = institute_name
        for full_name, abbrev in abbreviations.items():
            if full_name in name:
                name = name.replace(full_name, abbrev)
        
        # Extract city/location for NITs and IIITs
        if 'NIT' in name and ',' in name:
            parts = name.split(',')
            city = parts[-1].strip()
            name = f"NIT {city}"
        elif 'IIIT' in name and ',' in name:
            parts = name.split(',')
            city = parts[-1].strip()
            name = f"IIIT {city}"
        
        return name[:25]  # Ensure it fits in the column
    
    def abbreviate_branch(self, branch_name: str) -> str:
        """Create abbreviated form of branch names"""
        abbreviations = {
            'Computer Science and Engineering': 'CSE',
            'Electronics and Communication Engineering': 'ECE',
            'Electrical and Electronics Engineering': 'EEE',
            'Electrical Engineering': 'EE',
            'Mechanical Engineering': 'ME',
            'Civil Engineering': 'CE',
            'Chemical Engineering': 'ChE',
            'Information Technology': 'IT',
            'Computer Science': 'CS',
            'Artificial Intelligence': 'AI',
            'Machine Learning': 'ML',
            'Data Science': 'DS',
            'Electronics': 'EC',
            'Communication': 'Comm',
            'Biotechnology': 'BT',
            'Biomedical Engineering': 'BME',
            'Metallurgical Engineering': 'MetE',
            'Materials Science': 'MatSci',
            'Aerospace Engineering': 'AE',
            'Automobile Engineering': 'Auto',
            'Production Engineering': 'PE',
            'Industrial Engineering': 'IE',
            'Mining Engineering': 'MinE',
            'Petroleum Engineering': 'PetE',
            'Environmental Engineering': 'EnvE',
            'Instrumentation': 'IC',
            'Mathematics and Computing': 'MnC',
            'Mathematics and Scientific Computing': 'MSC',
            'Computational and Data Science': 'CDS',
            'Bachelor of Technology': 'B.Tech',
            'Bachelor of Architecture': 'B.Arch',
            'Integrated Master of Technology': 'M.Tech',
            '(4 Years, Bachelor of Technology)': '',
            '(5 Years, Integrated Master of Technology)': '(M.Tech)',
            'Under Flexible Academic Program': 'FAP',
            'VLSI Design': 'VLSI',
            'Signal Processing': 'SP',
            'Power Systems': 'PS',
            'Thermal Engineering': 'Thermal',
            'Structural Engineering': 'Structural'
        }
        
        name = branch_name
        
        # Apply abbreviations in order
        for full_name, abbrev in abbreviations.items():
            name = name.replace(full_name, abbrev)
        
        # Clean up extra spaces and empty brackets
        name = ' '.join(name.split())  # Remove extra spaces
        name = name.replace('( ', '(').replace(' )', ')')
        
        # Remove empty parentheses and common patterns
        patterns_to_remove = ['()', '( )', '(,)', '(, )', ', ', '  ']
        for pattern in patterns_to_remove:
            name = name.replace(pattern, ' ')
        
        # Final cleanup
        name = ' '.join(name.split())  # Remove extra spaces again
        name = name.strip(', ')  # Remove trailing commas
        
        return name[:35]  # Ensure it fits in the column

    def find_similar_options(self, search_term: str, option_list: list, max_results: int = 5):
        """Find similar options based on search term"""
        search_lower = search_term.lower()
        matches = [opt for opt in option_list if search_lower in opt.lower()]
        return matches[:max_results]
    
    def get_recommendations_by_rank(self, rank: int, category: str = 'OPEN', top_n: int = 10):
        """Get college recommendations based on rank - aggregated across all rounds"""
        
        # Filter data for the given category
        filtered_data = self.data[self.data['category'] == category].copy()
        
        # Group by institute and branch to get the best and worst cutoffs across all rounds
        grouped = filtered_data.groupby(['institute_name', 'branch']).agg({
            'opening_rank': 'min',  # Best opening rank (lowest)
            'closing_rank': 'max',  # Worst closing rank (highest) - most lenient
            'year': 'max'  # Most recent year
        }).reset_index()
        
        # Find colleges where the student's rank falls within the cutoff range
        suitable = grouped[
            (grouped['opening_rank'] <= rank) & 
            (grouped['closing_rank'] >= rank)
        ].copy()
        
        if len(suitable) == 0:
            # If no exact matches, find colleges with cutoffs close to student rank
            tolerance = rank * 0.15  # 15% tolerance
            suitable = grouped[
                abs(grouped['closing_rank'] - rank) <= tolerance
            ].copy()
        
        if len(suitable) == 0:
            return pd.DataFrame()
        # Abbreviate 
        
        # Calculate admission probability based on how close the rank is to the cutoff
        suitable['probability'] = suitable.apply(
            lambda row: min(95, max(5, 100 - abs(row['closing_rank'] - rank) / rank * 100)), 
            axis=1
        ).round(1)
        
        # Sort by probability and rank
        result = suitable.sort_values(['probability', 'closing_rank'], ascending=[False, True])
        
        return result[['institute_name', 'branch', 'opening_rank', 'closing_rank', 
                      'probability', 'year']].head(top_n)
    
    def interactive_session(self):
        """Run an interactive prediction session"""
        print("\n🎓 Interactive College Prediction Session")
        print("=" * 50)
        
        if self.model is None:
            self.load_model_and_data()
    
        while True:
            try:
                print("\n📝 Choose an option:")
                print("1. Get college recommendations for your rank")
                print("2. View safest ranks by branch category")
                print("3. Exit")
                
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == '3':
                    break
                elif choice == '2':
                    print(f"\nAvailable categories: {', '.join(self.unique_values['categories'])}")
                    category = input("Enter category (default: OPEN): ").upper() or 'OPEN'
                    
                    if category not in self.unique_values['categories']:
                        print(f"⚠️  Category '{category}' not found. Using 'OPEN'.")
                        category = 'OPEN'
                    
                    self.get_safest_ranks_by_branch(category)
                    continue
                elif choice != '1':
                    print("❌ Please enter 1, 2, or 3.")
                    continue
                
                print("\n📝 Enter your details:")
                
                # Get user inputs
                rank = int(input("Your JEE Main Rank: "))
                if rank <= 0:
                    print("❌ Please enter a valid positive rank.")
                    continue
                
                print(f"\nAvailable categories: {', '.join(self.unique_values['categories'])}")
                category = input("Your category (default: OPEN): ").upper() or 'OPEN'
                
                if category not in self.unique_values['categories']:
                    print(f"⚠️  Category '{category}' not found. Using 'OPEN'.")
                    category = 'OPEN'
                
                # Get recommendations
                print(f"\n🔍 Finding recommendations for Rank {rank:,} in {category} category...")
                
                recommendations = self.get_recommendations_by_rank(
                    rank=rank, 
                    category=category, 
                    top_n=15
                )
                
                if len(recommendations) > 0:
                    print(f"\n🏫 Top {len(recommendations)} Recommendations:")
                    print("-" * 120)
                    print(f"{'#':<3} | {'Institute':<25} | {'Branch':<35} | {'Cutoff Range':<15} | {'Chance':<8} | {'Year'}")
                    print("-" * 120)
                    
                    for idx, row in recommendations.iterrows():
                        institute = self.abbreviate_institute(row['institute_name'])
                        branch = self.abbreviate_branch(row['branch'])
                        opening = row['opening_rank']
                        closing = row['closing_rank']
                        prob = row['probability']
                        year = int(row['year'])
                        
                        print(f"{idx + 1:<3} | {institute:<25} | {branch:<35} | {opening:>6,}-{closing:>6,} | {prob:>5.1f}% | {year}")
                
                else:
                    print("❌ No suitable recommendations found for your rank and category.")
                    print("💡 Try a different category or check if your rank is realistic.")
                
                # Additional analysis
                self.show_category_analysis(rank)
                self.show_rank_safety_guide(rank, category)
                self.show_branch_analysis()
                
            except ValueError:
                print("❌ Please enter a valid number for rank.")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                continue
            
            # Ask if user wants to continue
            choice = input("\n🤔 Do you want to try another prediction? (y/n): ").lower()
            if choice != 'y':
                break
        
        print("\n✅ Thanks for using College Predictor!")
    
    def show_category_analysis(self, user_rank: int):
        """Show how the user's rank compares across categories"""
        print(f"\n📊 How rank {user_rank:,} compares across categories:")
        
        for category in self.unique_values['categories']:
            category_data = self.data[self.data['category'] == category]
            
            # Find percentile
            better_ranks = len(category_data[category_data['closing_rank'] <= user_rank])
            total_records = len(category_data)
            percentile = (better_ranks / total_records * 100) if total_records > 0 else 0
            
            avg_cutoff = category_data['closing_rank'].mean()
            median_cutoff = category_data['closing_rank'].median()
            
            status = "✅ Good" if user_rank <= avg_cutoff else "⚠️  Challenging"
            
            print(f"  {category:7s}: Avg={avg_cutoff:6.0f}, Median={median_cutoff:6.0f} | {status} (Top {percentile:.1f}%)")
    
    def show_branch_analysis(self):
        """Show most competitive branches"""
        print(f"\n🏆 Most Competitive Branches (Top 10):")
        
        top_branches = self.data.groupby('branch')['closing_rank'].mean().sort_values().head(10)
        
        for i, (branch, avg_cutoff) in enumerate(top_branches.items(), 1):
            short_branch = branch[:50] + "..." if len(branch) > 50 else branch
            print(f"  {i:2d}. {short_branch:<53} → {avg_cutoff:6.0f}")
    
    def get_safest_ranks_by_branch(self, category: str = 'OPEN'):
        """
        Calculate meaningful safety thresholds for each major branch category
        Shows 90th percentile ranks (ranks that have 90% chance of admission)
        """
        print(f"\n🛡️ Safety Thresholds for {category} Category (JEE Main Paper 1)")
        print("These ranks have ~90% chance of admission (90th percentile cutoffs)")
        print("-" * 70)
        
        # Filter data for the given category
        filtered_data = self.data[self.data['category'] == category].copy()
        
        if len(filtered_data) == 0:
            print(f"❌ No data found for category: {category}")
            return
        
        # Define major branch categories with keywords
        branch_categories = {
            'Computer Science & IT': ['computer science', 'information technology', 'software', 'data science', 'artificial intelligence', 'machine learning', 'cyber security'],
            'Electronics & Communication': ['electronics', 'communication', 'ece', 'vlsi', 'embedded', 'signal processing'],
            'Electrical Engineering': ['electrical', 'power', 'energy', 'instrumentation'],
            'Mechanical Engineering': ['mechanical', 'thermal', 'manufacturing', 'automobile', 'aerospace', 'production'],
            'Civil Engineering': ['civil', 'structural', 'transportation', 'environmental', 'construction'],
            'Chemical Engineering': ['chemical', 'process', 'petroleum', 'polymer'],
            'Biotechnology': ['biotechnology', 'biomedical', 'bioinformatics'],
            'Metallurgy & Materials': ['metallurgical', 'materials', 'ceramic'],
            'Mining Engineering': ['mining', 'geology'],
            'Other Engineering': []  # Will catch remaining branches
        }
        
        safety_thresholds = {}
        
        for category_name, keywords in branch_categories.items():
            if keywords:  # For defined categories
                # Create mask for branches containing any of the keywords
                mask = filtered_data['branch'].str.lower().apply(
                    lambda x: any(keyword in x for keyword in keywords)
                )
                category_data = filtered_data[mask]
            else:  # For 'Other Engineering' - branches not in other categories
                # Find branches not covered by other categories
                covered_branches = set()
                for other_keywords in [v for k, v in branch_categories.items() if k != 'Other Engineering' and v]:
                    mask = filtered_data['branch'].str.lower().apply(
                        lambda x: any(keyword in x for keyword in other_keywords)
                    )
                    covered_branches.update(filtered_data[mask]['branch'].unique())
                
                # Get uncovered branches
                category_data = filtered_data[~filtered_data['branch'].isin(covered_branches)]
            
            if len(category_data) > 0:
                # Calculate meaningful safety thresholds
                safety_rank_90 = category_data['closing_rank'].quantile(0.90)  # 90th percentile
                avg_rank = category_data['closing_rank'].mean()
                median_rank = category_data['closing_rank'].median()
                max_rank = category_data['closing_rank'].max()
                count = len(category_data)
                
                # Get example institutes at safety threshold
                safety_examples = category_data[
                    (category_data['closing_rank'] >= safety_rank_90 * 0.95) & 
                    (category_data['closing_rank'] <= safety_rank_90 * 1.05)
                ]
                if len(safety_examples) > 0:
                    example_institute = safety_examples.iloc[0]['institute_name'][:25] + "..."
                else:
                    example_institute = "Various institutes"
                
                safety_thresholds[category_name] = {
                    'safety_rank_90': int(safety_rank_90),
                    'avg_rank': avg_rank,
                    'median_rank': median_rank,
                    'max_rank': max_rank,
                    'count': count,
                    'example': example_institute
                }
        
        # Sort by safety rank (highest to lowest)
        sorted_categories = sorted(safety_thresholds.items(), key=lambda x: x[1]['safety_rank_90'], reverse=True)
        
        print(f"{'Branch Category':<25} | {'90% Safe':<10} | {'Median':<8} | {'Max Ever':<10} | {'Records':<8} | Example")
        print("-" * 95)
        
        for category_name, data in sorted_categories:
            safety = f"{data['safety_rank_90']:,}"
            median = f"{data['median_rank']:,.0f}"
            max_ever = f"{data['max_rank']:,}"
            count = f"{data['count']:,}"
            example = data['example']
            
            print(f"{category_name:<25} | {safety:<10} | {median:<8} | {max_ever:<10} | {count:<8} | {example}")
        
        print(f"\n💡 Interpretation:")
        print(f"   • 90% Safe: Ranks that have ~90% chance of getting admission")
        print(f"   • Median: Middle rank that got admission")
        print(f"   • Max Ever: Highest rank that ever got admission (can be outlier)")
        print(f"   • For {category} category, focus on '90% Safe' column for realistic planning")
        
        return safety_thresholds
    
    def show_rank_safety_guide(self, user_rank: int, category: str = 'OPEN'):
        """
        Show where the user's rank stands in terms of safety for different branches
        """
        print(f"\n🎯 Safety Analysis for Your Rank {user_rank:,} ({category} category)")
        print("-" * 60)
        
        safety_thresholds = self.get_safest_ranks_by_branch(category)
        
        safe_branches = []
        moderate_branches = []
        risky_branches = []
        
        for branch_name, data in safety_thresholds.items():
            safety_rank_90 = data['safety_rank_90']
            median_rank = data['median_rank']
            
            if user_rank <= median_rank * 0.7:  # Much better than median
                safe_branches.append((branch_name, safety_rank_90, median_rank))
            elif user_rank <= median_rank:  # Better than median
                moderate_branches.append((branch_name, safety_rank_90, median_rank))
            elif user_rank <= safety_rank_90:  # Within 90% safety range
                risky_branches.append((branch_name, safety_rank_90, median_rank))
        
        if safe_branches:
            print(f"\n✅ SAFE BRANCHES (High chance of admission):")
            for branch, safety_90, median in safe_branches:
                print(f"  • {branch:<25} | Median: {median:6,.0f} | 90% Safe: {safety_90:,}")
        
        if moderate_branches:
            print(f"\n⚠️  MODERATE BRANCHES (Good chance, consider multiple options):")
            for branch, safety_90, median in moderate_branches:
                print(f"  • {branch:<25} | Median: {median:6,.0f} | 90% Safe: {safety_90:,}")
        
        if risky_branches:
            print(f"\n🎲 REACH BRANCHES (Possible but risky, have backups):")
            for branch, safety_90, median in risky_branches:
                print(f"  • {branch:<25} | Median: {median:6,.0f} | 90% Safe: {safety_90:,}")
        
        if not (safe_branches or moderate_branches or risky_branches):
            print("😔 Your rank might be challenging for most engineering branches in this category.")
            print("   Consider:")
            if category == 'OPEN':
                print("   • Applying in EWS/OBC-NCL/SC/ST if eligible")
            print("   • Private colleges")
            print("   • State counseling")
            print("   • Different quotas (HS/OS)")
        
        # Category-specific advice
        if category in ['ST', 'SC']:
            print(f"\n💡 {category} Category Advantage:")
            print(f"   • Reserved seats available with lower competition")
            print(f"   • Focus on good institutes rather than just ranks")
        elif category == 'OBC-NCL':
            print(f"\n💡 OBC-NCL Category Advantage:")
            print(f"   • 27% reservation in central institutions")
            print(f"   • Good balance of opportunity and competition")
        elif category == 'EWS':
            print(f"\n💡 EWS Category Advantage:")
            print(f"   • 10% reservation with lower competition than OPEN")
            print(f"   • Good opportunities in premier institutions")

def main():
    predictor = CollegePredictor()
    predictor.interactive_session()

if __name__ == "__main__":
    main()
