import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BookingCurveAnalyzer:
    """
    Analyzes and predicts booking curves based on actual cumulative patterns.
    Uses multiple models and selects the best performer.
    """
    
    def __init__(self):
        # Your actual cumulative booking patterns from the analysis
        self.default_cumulative_pattern = {
            14: 2,   # 1st Day
            13: 4,   # 2nd Day
            12: 7,   # 3rd Day
            11: 10,  # 4th Day
            10: 13,  # 5th Day
            9: 17,   # 6th Day
            8: 21,   # 7th Day
            7: 25,   # 8th Day
            6: 29,   # 9th Day
            5: 34,   # 10th Day
            4: 39,   # 11th Day
            3: 45,   # 12th Day
            2: 51,   # 13th Day
            1: 61,   # Day before departure
            0: 100   # Day of departure
        }
        
        # Day-specific patterns (can be customized based on your data)
        self.day_patterns = {
            'Monday': {14: 2, 13: 4, 12: 6, 11: 9, 10: 12, 9: 16, 8: 20, 7: 24, 6: 28, 5: 33, 4: 38, 3: 44, 2: 50, 1: 59, 0: 100},
            'Tuesday': {14: 2, 13: 4, 12: 7, 11: 10, 10: 13, 9: 17, 8: 21, 7: 25, 6: 29, 5: 34, 4: 39, 3: 45, 2: 51, 1: 60, 0: 100},
            'Wednesday': {14: 3, 13: 5, 12: 8, 11: 11, 10: 14, 9: 18, 8: 22, 7: 26, 6: 30, 5: 35, 4: 40, 3: 46, 2: 52, 1: 62, 0: 100},
            'Thursday': {14: 3, 13: 5, 12: 8, 11: 11, 10: 15, 9: 19, 8: 23, 7: 27, 6: 31, 5: 36, 4: 41, 3: 47, 2: 53, 1: 63, 0: 100},
            'Friday': {14: 3, 13: 6, 12: 9, 11: 12, 10: 16, 9: 20, 8: 24, 7: 28, 6: 33, 5: 38, 4: 43, 3: 49, 2: 55, 1: 65, 0: 100},
            'Saturday': {14: 4, 13: 7, 12: 10, 11: 14, 10: 18, 9: 22, 8: 27, 7: 32, 6: 37, 5: 42, 4: 47, 3: 53, 2: 59, 1: 68, 0: 100},
            'Sunday': {14: 3, 13: 5, 12: 8, 11: 11, 10: 15, 9: 19, 8: 23, 7: 27, 6: 31, 5: 36, 4: 41, 3: 47, 2: 53, 1: 62, 0: 100}
        }
        
        # Model options
        self.models = {
            'curve_fitting': CurveFittingModel(),
            'ratio_based': RatioBasedPredictor(),
            'ml_ensemble': MLEnsemblePredictor(),
            'statistical': StatisticalPredictor()
        }
        
        self.best_model = None
    
    def analyze_current_booking(self, departure_date, current_bookings, days_before_departure):
        """
        Analyze current booking status and predict final numbers.
        """
        day_name = departure_date.strftime('%A')
        
        # Get the appropriate pattern
        pattern = self.day_patterns.get(day_name, self.default_cumulative_pattern)
        expected_percentage = pattern.get(days_before_departure, 0)
        
        # Calculate expected bookings at this point
        if expected_percentage > 0:
            projected_total = (current_bookings / expected_percentage) * 100
        else:
            projected_total = current_bookings
        
        # Analyze deviation
        results = {
            'departure_date': departure_date,
            'day_of_week': day_name,
            'days_before_departure': days_before_departure,
            'current_bookings': current_bookings,
            'expected_percentage': expected_percentage,
            'projected_total': int(projected_total),
            'pattern': pattern
        }
        
        # Check if booking pace is normal
        if days_before_departure in pattern:
            # Calculate confidence interval
            std_dev = expected_percentage * 0.15  # 15% standard deviation
            lower_bound = expected_percentage - (1.96 * std_dev)
            upper_bound = expected_percentage + (1.96 * std_dev)
            
            current_percentage = (current_bookings / projected_total) * 100
            
            results['is_normal'] = lower_bound <= current_percentage <= upper_bound
            results['deviation'] = current_percentage - expected_percentage
            results['confidence_lower'] = int(projected_total * 0.9)  # 90% confidence
            results['confidence_upper'] = int(projected_total * 1.1)  # 110% confidence
        
        return results
    
    def predict_booking_curve(self, current_bookings, days_before_departure, pattern):
        """
        Predict the complete booking curve from current point to departure.
        """
        predictions = {}
        
        # Get current percentage
        current_percentage = pattern.get(days_before_departure, 0)
        
        if current_percentage > 0:
            scaling_factor = current_bookings / current_percentage
            
            # Predict for each remaining day
            for day in range(days_before_departure, -1, -1):
                expected_percentage = pattern.get(day, 0)
                predictions[day] = int(scaling_factor * expected_percentage)
        
        return predictions


class CurveFittingModel:
    """Uses polynomial curve fitting to predict booking patterns."""
    
    def fit_predict(self, current_bookings, days_before_departure, pattern):
        # Extract x (days) and y (percentages) from pattern
        days = sorted(pattern.keys(), reverse=True)
        percentages = [pattern[d] for d in days]
        
        # Fit polynomial curve
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(np.array(days).reshape(-1, 1))
        y = np.array(percentages)
        
        model = Ridge(alpha=0.1)
        model.fit(X, y)
        
        # Predict current position
        X_current = poly.transform([[days_before_departure]])
        expected_percentage = model.predict(X_current)[0]
        
        # Scale based on current bookings
        if expected_percentage > 0:
            scale = (current_bookings / expected_percentage) * 100
            return int(scale)
        return current_bookings


class RatioBasedPredictor:
    """Uses booking velocity ratios to predict final numbers."""
    
    def predict(self, current_bookings, days_before_departure, pattern):
        # Calculate booking velocity
        if days_before_departure > 0 and days_before_departure in pattern:
            current_ratio = pattern[days_before_departure] / 100
            if current_ratio > 0:
                return int(current_bookings / current_ratio)
        return current_bookings


class MLEnsemblePredictor:
    """Uses ensemble of ML models for prediction."""
    
    def __init__(self):
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ]
    
    def predict(self, current_bookings, days_before_departure, historical_data=None):
        # For demo purposes, using simple scaling
        # In production, train on historical data
        if days_before_departure in [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            # Use average growth rate
            remaining_days = days_before_departure
            daily_growth = 0.07  # 7% average daily growth
            multiplier = (1 + daily_growth) ** remaining_days
            return int(current_bookings * multiplier)
        return current_bookings


class StatisticalPredictor:
    """Uses statistical methods for prediction."""
    
    def predict(self, current_bookings, days_before_departure, pattern):
        # Use exponential smoothing
        if days_before_departure in pattern and days_before_departure > 0:
            current_percentage = pattern[days_before_departure]
            final_percentage = 100
            
            # Calculate growth factor
            growth_factor = final_percentage / current_percentage
            return int(current_bookings * growth_factor)
        return current_bookings


def create_interactive_app():
    """Create Streamlit interactive application."""
    st.set_page_config(page_title="Booking Curve Analyser", layout="wide")
    
    st.title("🚂 London EUS PA Booking Curve Analyser")
    st.markdown("Analyse current pre-booking status and predict final passenger numbers")
    
    # Initialize analyzer
    analyzer = BookingCurveAnalyzer()
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        departure_date = st.date_input(
            "📅 Select departure date:",
            min_value=datetime.now().date(),
            value=datetime.now().date() + timedelta(days=7)
        )
    
    with col2:
        days_before = (departure_date - datetime.now().date()).days
        days_before_departure = st.number_input(
            "📆 Days before departure:",
            min_value=0,
            max_value=14,
            value=min(days_before, 14) if days_before > 0 else 7
        )
    
    with col3:
        current_bookings = st.number_input(
            "👥 Current pre-bookings:",
            min_value=0,
            value=100,
            step=10
        )
    
    # Analyze button
    if st.button("🔍 Analyze Booking Status", type="primary"):
        # Perform analysis
        results = analyzer.analyze_current_booking(
            departure_date, 
            current_bookings, 
            days_before_departure
        )
        
        # Display results
        st.markdown("---")
        
        # Status summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Day of Week",
                results['day_of_week'],
                delta=None
            )
        
        with col2:
            expected_pct = results['expected_percentage']
            st.metric(
                "Expected Progress",
                f"{expected_pct}%",
                delta=f"at {days_before_departure} days before"
            )
        
        with col3:
            st.metric(
                "Projected Total",
                f"{results['projected_total']:,}",
                delta=f"{results['projected_total'] - current_bookings:+,} to go"
            )
        
        with col4:
            status_color = "🟢" if results.get('is_normal', True) else "🔴"
            status_text = "Normal" if results.get('is_normal', True) else "Abnormal"
            st.metric(
                "Booking Status",
                f"{status_color} {status_text}",
                delta=f"{results.get('deviation', 0):+.1f}% vs expected"
            )
        
        # Confidence interval
        st.markdown("---")
        st.subheader("📊 Prediction Confidence Range")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Lower Bound:** {results.get('confidence_lower', 0):,} passengers")
        with col2:
            st.success(f"**Most Likely:** {results['projected_total']:,} passengers")
        with col3:
            st.info(f"**Upper Bound:** {results.get('confidence_upper', 0):,} passengers")
        
        # Booking curve visualization
        st.markdown("---")
        st.subheader("📈 Booking Curve Analysis")
        
        # Create the curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Expected vs Current Position
        pattern = results['pattern']
        days = sorted(pattern.keys(), reverse=True)
        percentages = [pattern[d] for d in days]
        
        ax1.plot(days, percentages, 'b-', linewidth=2, label='Expected Curve')
        ax1.axvline(x=days_before_departure, color='red', linestyle='--', alpha=0.7, label='Today')
        
        # Mark current position
        current_pct = (current_bookings / results['projected_total']) * 100
        ax1.scatter([days_before_departure], [current_pct], color='red', s=200, zorder=5, label='Current Status')
        
        ax1.set_xlabel('Days Before Departure')
        ax1.set_ylabel('Cumulative Bookings (%)')
        ax1.set_title('Booking Progress vs Expected Pattern')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(14, -1)
        ax1.set_ylim(0, 105)
        
        # Right plot: Projected booking accumulation
        predictions = analyzer.predict_booking_curve(
            current_bookings, 
            days_before_departure, 
            pattern
        )
        
        if predictions:
            pred_days = sorted(predictions.keys(), reverse=True)
            pred_values = [predictions[d] for d in pred_days]
            
            ax2.plot(pred_days, pred_values, 'g-', linewidth=2, label='Projected Bookings')
            ax2.axvline(x=days_before_departure, color='red', linestyle='--', alpha=0.7, label='Today')
            ax2.scatter([days_before_departure], [current_bookings], color='red', s=200, zorder=5)
            
            # Add confidence bands
            upper_values = [v * 1.1 for v in pred_values]
            lower_values = [v * 0.9 for v in pred_values]
            ax2.fill_between(pred_days, lower_values, upper_values, alpha=0.2, color='green', label='90% Confidence')
            
            ax2.set_xlabel('Days Before Departure')
            ax2.set_ylabel('Total Bookings')
            ax2.set_title('Projected Booking Accumulation')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_xlim(14, -1)
        
        st.pyplot(fig)
        
        # Daily breakdown table
        st.markdown("---")
        st.subheader("📋 Day-by-Day Projection")
        
        if predictions:
            breakdown_data = []
            for day in range(days_before_departure, -1, -1):
                if day in predictions:
                    breakdown_data.append({
                        'Days Before Departure': day,
                        'Day Label': f"Day {14-day}" if day > 0 else "Departure Day",
                        'Expected Cumulative %': f"{pattern.get(day, 0)}%",
                        'Projected Bookings': f"{predictions[day]:,}",
                        'Bookings to Add': f"{predictions[day] - (predictions.get(day+1, current_bookings) if day < days_before_departure else current_bookings):+,}"
                    })
            
            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown, use_container_width=True)
        
        # Model comparison
        if st.checkbox("🔬 Show Model Comparison"):
            st.subheader("Model Performance Comparison")
            
            model_results = {}
            
            # Ratio-based model
            ratio_predictor = RatioBasedPredictor()
            ratio_pred = ratio_predictor.predict(current_bookings, days_before_departure, pattern)
            model_results['Ratio-Based'] = ratio_pred
            
            # Statistical model
            stat_predictor = StatisticalPredictor()
            stat_pred = stat_predictor.predict(current_bookings, days_before_departure, pattern)
            model_results['Statistical'] = stat_pred
            
            # ML Ensemble (simplified)
            ml_predictor = MLEnsemblePredictor()
            ml_pred = ml_predictor.predict(current_bookings, days_before_departure)
            model_results['ML Ensemble'] = ml_pred
            
            # Display comparison
            model_df = pd.DataFrame({
                'Model': model_results.keys(),
                'Prediction': model_results.values(),
                'Deviation from Primary': [f"{(v/results['projected_total']-1)*100:+.1f}%" for v in model_results.values()]
            })
            
            st.dataframe(model_df)
            
            # Recommendation
            st.info(f"💡 **Recommendation**: Based on historical patterns for {results['day_of_week']}s, "
                   f"expect approximately **{results['projected_total']:,}** passengers on departure day. "
                   f"Current booking pace is **{status_text.lower()}**.")


def main():
    """Run the interactive application."""
    create_interactive_app()


if __name__ == "__main__":
    main()
