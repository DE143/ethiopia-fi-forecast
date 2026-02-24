import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


class FinancialInclusionForecaster:
    def __init__(self, data_path="data/processed/enriched_data.csv"):
        self.data = pd.read_csv(
            data_path, parse_dates=["observation_date", "event_date", "created_at"]
        )
        self.observations = self.data[self.data["record_type"] == "observation"]

    def prepare_forecast_data(self, indicator_code="ACC_OWNERSHIP"):
        """Prepare time series data for forecasting"""

        # Extract indicator data
        indicator_data = self.observations[
            self.observations["indicator_code"] == indicator_code
        ].copy()

        if len(indicator_data) == 0:
            raise ValueError(f"No data found for indicator: {indicator_code}")

        # Sort by date and create time series
        indicator_data = indicator_data.sort_values("observation_date")
        indicator_data = indicator_data.drop_duplicates("observation_date", keep="last")

        # Create annual series
        indicator_data["year"] = indicator_data["observation_date"].dt.year
        annual_data = (
            indicator_data.groupby("year")["value_numeric"].mean().reset_index()
        )

        return annual_data

    def forecast_trend_model(self, indicator_code, forecast_years=[2025, 2026, 2027]):
        """Forecast using trend-based models"""

        # Prepare data
        data = self.prepare_forecast_data(indicator_code)

        if len(data) < 3:
            return None

        # Models to try
        models = {}

        # 1. Linear trend
        X = data["year"].values.reshape(-1, 1)
        y = data["value_numeric"].values

        # Linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        linear_predictions = lr.predict(np.array(forecast_years).reshape(-1, 1))

        models["linear"] = {
            "model": lr,
            "predictions": linear_predictions,
            "r_squared": lr.score(X, y),
            "mae": mean_absolute_error(y, lr.predict(X)),
        }

        # 2. Polynomial trend (quadratic)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        lr_poly = LinearRegression()
        lr_poly.fit(X_poly, y)

        poly_predictions = lr_poly.predict(
            poly.transform(np.array(forecast_years).reshape(-1, 1))
        )

        models["polynomial"] = {
            "model": lr_poly,
            "predictions": poly_predictions,
            "r_squared": lr_poly.score(X_poly, y),
            "mae": mean_absolute_error(y, lr_poly.predict(X_poly)),
        }

        # 3. Logistic growth (S-curve) - appropriate for adoption curves
        try:
            # Transform for logistic: y = L / (1 + exp(-k*(x - x0)))
            # Using simplified approach
            max_val = max(y) * 2  # Assume saturation at double current max
            y_scaled = np.log((max_val / y) - 1)
            valid_idx = ~np.isinf(y_scaled) & ~np.isnan(y_scaled)

            if sum(valid_idx) >= 3:
                X_valid = X[valid_idx]
                y_valid = y_scaled[valid_idx]

                lr_log = LinearRegression()
                lr_log.fit(X_valid, y_valid)

                # Predict and transform back
                log_predictions_scaled = lr_log.predict(
                    np.array(forecast_years).reshape(-1, 1)
                )
                log_predictions = max_val / (1 + np.exp(log_predictions_scaled))

                models["logistic"] = {
                    "model": lr_log,
                    "predictions": log_predictions,
                    "r_squared": lr_log.score(X_valid, y_valid),
                    "mae": None,  # Not directly comparable
                }
        except:
            pass

        return models, data

    def forecast_event_augmented(
        self, indicator_code, forecast_years=[2025, 2026, 2027]
    ):
        """Forecast incorporating event impacts"""

        from event_analyzer import EventImpactModeler

        # Get base trend forecast
        trend_models, data = self.forecast_trend_model(indicator_code, forecast_years)
        if trend_models is None:
            return None

        # Get event impacts
        event_modeler = EventImpactModeler()

        # Calculate event impacts for forecast years
        event_impacts = {}
        for year in forecast_years:
            target_date = pd.Timestamp(f"{year}-12-31")
            impact, _ = event_modeler.estimate_event_impacts(
                indicator_code, target_date
            )
            event_impacts[year] = impact

        # Augment trend forecasts with event impacts
        augmented_forecasts = {}

        for model_name, model_info in trend_models.items():
            predictions = model_info["predictions"]
            augmented = {}

            for i, year in enumerate(forecast_years):
                base_pred = predictions[i]
                event_impact = event_impacts[year]
                augmented_pred = base_pred + event_impact

                augmented[year] = {
                    "base": base_pred,
                    "event_impact": event_impact,
                    "augmented": augmented_pred,
                }

            augmented_forecasts[model_name] = augmented

        return augmented_forecasts, event_impacts, trend_models

    def generate_scenarios(self, indicator_code, forecast_years=[2025, 2026, 2027]):
        """Generate optimistic, base, and pessimistic scenarios"""

        # Get event-augmented forecast as base
        augmented_results = self.forecast_event_augmented(
            indicator_code, forecast_years
        )

        if augmented_results is None:
            return None

        augmented_forecasts, event_impacts, trend_models = augmented_results

        # Use polynomial model as base (often best fit for adoption curves)
        base_model = (
            "polynomial"
            if "polynomial" in augmented_forecasts
            else list(augmented_forecasts.keys())[0]
        )

        scenarios = {}

        for year in forecast_years:
            base_value = augmented_forecasts[base_model][year]["augmented"]

            # Scenario definitions
            scenarios[year] = {
                "optimistic": {
                    "value": base_value * 1.15,  # 15% higher than base
                    "description": "Rapid adoption, successful policies, strong economic growth",
                },
                "base": {
                    "value": base_value,
                    "description": "Current trajectory with expected event impacts",
                },
                "pessimistic": {
                    "value": base_value * 0.85,  # 15% lower than base
                    "description": "Economic challenges, slower adoption, regulatory hurdles",
                },
            }

        return scenarios

    def calculate_uncertainty(self, forecasts, confidence_level=0.95):
        """Calculate confidence intervals for forecasts"""

        import scipy.stats as stats

        uncertainty_results = {}

        for model_name, model_forecasts in forecasts.items():
            predictions = []
            years = []

            for year, forecast in model_forecasts.items():
                predictions.append(forecast["augmented"])
                years.append(year)

            if len(predictions) >= 2:
                # Simple uncertainty estimation based on historical error
                historical_data = self.prepare_forecast_data("ACC_OWNERSHIP")
                if len(historical_data) >= 3:
                    # Calculate prediction intervals
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions)

                    # t-distribution for small samples
                    t_value = stats.t.ppf(
                        (1 + confidence_level) / 2, len(predictions) - 1
                    )
                    margin_error = t_value * std_pred / np.sqrt(len(predictions))

                    uncertainty_results[model_name] = {
                        "mean": mean_pred,
                        "std": std_pred,
                        "confidence_interval": (
                            mean_pred - margin_error,
                            mean_pred + margin_error,
                        ),
                        "range": (min(predictions), max(predictions)),
                    }

        return uncertainty_results

    def forecast_all_indicators(self):
        """Forecast all key indicators"""

        key_indicators = {
            "ACC_OWNERSHIP": {
                "name": "Account Ownership Rate",
                "current_value": 49,
                "unit": "%",
                "target_2027": 60,
            },
            "USG_DIGITAL_PAYMENT": {
                "name": "Digital Payment Adoption",
                "current_value": 35,
                "unit": "%",
                "target_2027": 50,
            },
            "ACC_MM_ACCOUNT": {
                "name": "Mobile Money Account Ownership",
                "current_value": 9.45,
                "unit": "%",
                "target_2027": 25,
            },
        }

        forecasts = {}

        for indicator_code, indicator_info in key_indicators.items():
            print(f"\nForecasting: {indicator_info['name']}")

            try:
                # Generate scenarios
                scenarios = self.generate_scenarios(
                    indicator_code, forecast_years=[2025, 2026, 2027]
                )

                if scenarios:
                    forecasts[indicator_code] = {
                        "info": indicator_info,
                        "scenarios": scenarios,
                        "target_gap_2027": {},
                    }

                    # Calculate target achievement
                    for scenario in ["optimistic", "base", "pessimistic"]:
                        pred_2027 = scenarios[2027][scenario]["value"]
                        target = indicator_info["target_2027"]
                        gap = pred_2027 - target

                        forecasts[indicator_code]["target_gap_2027"][scenario] = {
                            "prediction": pred_2027,
                            "target": target,
                            "gap": gap,
                            "achievement": "Achieved" if gap >= 0 else "Not achieved",
                        }

            except Exception as e:
                print(f"Error forecasting {indicator_code}: {e}")
                forecasts[indicator_code] = {"error": str(e)}

        return forecasts

    def create_forecast_visualizations(self, forecasts, output_dir="reports/figures"):
        """Create forecast visualizations"""

        import os

        os.makedirs(output_dir, exist_ok=True)

        viz_files = {}

        # 1. Account Ownership Forecast
        if "ACC_OWNERSHIP" in forecasts:
            acc_forecast = forecasts["ACC_OWNERSHIP"]

            fig, ax = plt.subplots(figsize=(12, 8))

            # Historical data
            historical_data = self.prepare_forecast_data("ACC_OWNERSHIP")
            ax.plot(
                historical_data["year"],
                historical_data["value_numeric"],
                "bo-",
                linewidth=2,
                markersize=8,
                label="Historical",
            )

            # Forecast scenarios
            years = [2024, 2025, 2026, 2027]
            for scenario, color in [
                ("optimistic", "green"),
                ("base", "blue"),
                ("pessimistic", "red"),
            ]:
                values = [acc_forecast["info"]["current_value"]]
                for year in [2025, 2026, 2027]:
                    values.append(acc_forecast["scenarios"][year][scenario]["value"])

                ax.plot(
                    years,
                    values,
                    "--",
                    color=color,
                    linewidth=2,
                    label=f"{scenario.capitalize()}",
                )
                ax.fill_between(years, values, alpha=0.1, color=color)

            # Target line
            ax.axhline(
                y=60,
                color="black",
                linestyle=":",
                linewidth=1,
                label="NFIS-II Target (60%)",
            )

            ax.set_xlabel("Year")
            ax.set_ylabel("Account Ownership (%)")
            ax.set_title("Account Ownership Forecast 2025-2027 with Scenarios")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(40, 70)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/account_ownership_forecast.png", dpi=300)
            viz_files["account_forecast"] = (
                f"{output_dir}/account_ownership_forecast.png"
            )

        # 2. All Indicators Forecast Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        indicators_to_plot = ["ACC_OWNERSHIP", "USG_DIGITAL_PAYMENT", "ACC_MM_ACCOUNT"]

        for idx, indicator_code in enumerate(indicators_to_plot):
            if indicator_code in forecasts and "scenarios" in forecasts[indicator_code]:
                ax = axes[idx]

                # Get forecast data
                forecast_data = forecasts[indicator_code]
                scenarios = forecast_data["scenarios"]

                # Plot scenarios
                years = [2024, 2025, 2026, 2027]
                current_value = forecast_data["info"]["current_value"]

                for scenario, color in [
                    ("optimistic", "green"),
                    ("base", "blue"),
                    ("pessimistic", "red"),
                ]:
                    values = [current_value]
                    for year in [2025, 2026, 2027]:
                        values.append(scenarios[year][scenario]["value"])

                    ax.plot(
                        years,
                        values,
                        "--",
                        color=color,
                        linewidth=2,
                        label=scenario.capitalize(),
                    )

                ax.set_xlabel("Year")
                ax.set_ylabel(f"{forecast_data['info']['name']} (%)")
                ax.set_title(forecast_data["info"]["name"])
                ax.grid(True, alpha=0.3)
                if idx == 2:  # Only show legend on last plot
                    ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_indicators_forecast.png", dpi=300)
        viz_files["all_indicators"] = f"{output_dir}/all_indicators_forecast.png"

        # 3. Target Achievement Visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        indicators = []
        base_2027 = []
        targets = []

        for indicator_code in indicators_to_plot:
            if (
                indicator_code in forecasts
                and "target_gap_2027" in forecasts[indicator_code]
            ):
                indicator_name = forecasts[indicator_code]["info"]["name"]
                base_pred = forecasts[indicator_code]["target_gap_2027"]["base"][
                    "prediction"
                ]
                target = forecasts[indicator_code]["info"]["target_2027"]

                indicators.append(indicator_name[:20])
                base_2027.append(base_pred)
                targets.append(target)

        x = np.arange(len(indicators))
        width = 0.35

        ax.bar(x - width / 2, base_2027, width, label="Base Forecast 2027", alpha=0.8)
        ax.bar(x + width / 2, targets, width, label="Target 2027", alpha=0.8)

        ax.set_xlabel("Indicator")
        ax.set_ylabel("Value (%)")
        ax.set_title("2027 Forecast vs Targets")
        ax.set_xticks(x)
        ax.set_xticklabels(indicators, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_achievement.png", dpi=300)
        viz_files["target_achievement"] = f"{output_dir}/target_achievement.png"

        plt.close("all")

        return viz_files
