import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class EventImpactModeler:
    def __init__(self, data_path="data/processed/enriched_data.csv"):
        self.data = pd.read_csv(
            data_path, parse_dates=["observation_date", "event_date", "created_at"]
        )
        self.events = self.data[self.data["record_type"] == "event"]
        self.impact_links = self.data[self.data["record_type"] == "impact_link"]
        self.observations = self.data[self.data["record_type"] == "observation"]

    def create_impact_matrix(self):
        """Create event-indicator impact matrix"""

        # Merge events with impact links
        event_impacts = pd.merge(
            self.impact_links,
            self.events[["name", "event_date", "category"]],
            left_on="parent_id",
            right_on="name",
            how="left",
        )

        # Create pivot table
        impact_matrix = event_impacts.pivot_table(
            index=["parent_id", "event_date", "category"],
            columns="related_indicator",
            values="impact_magnitude",
            aggfunc="first",
        ).reset_index()

        # Add lag information
        lag_info = event_impacts.groupby("parent_id")["lag_months"].first()
        impact_matrix["lag_months"] = impact_matrix["parent_id"].map(lag_info)

        # Add evidence basis
        evidence_info = event_impacts.groupby("parent_id")["evidence_basis"].first()
        impact_matrix["evidence"] = impact_matrix["parent_id"].map(evidence_info)

        # Add confidence
        confidence_info = event_impacts.groupby("parent_id")["confidence"].first()
        impact_matrix["confidence"] = impact_matrix["parent_id"].map(confidence_info)

        return impact_matrix, event_impacts

    def estimate_event_impacts(self, indicator, event_date):
        """Estimate cumulative impact of events on an indicator"""

        # Get relevant impact links
        relevant_impacts = self.impact_links[
            self.impact_links["related_indicator"] == indicator
        ].copy()

        # Merge with event dates
        relevant_impacts = pd.merge(
            relevant_impacts,
            self.events[["name", "event_date"]],
            left_on="parent_id",
            right_on="name",
            how="left",
        )

        # Calculate impact at given date
        total_impact = 0
        impact_details = []

        for _, impact in relevant_impacts.iterrows():
            if pd.isna(impact["event_date"]):
                continue

            # Calculate effective date (event + lag)
            effective_date = impact["event_date"] + pd.DateOffset(
                months=impact["lag_months"]
            )

            # If event has taken effect by our target date
            if effective_date <= event_date:
                # Calculate time since effectiveness (for gradual effects)
                months_since = (event_date - effective_date).days / 30.44

                # Simple impact model: full impact reached after 6 months
                if months_since >= 6:
                    impact_strength = impact["impact_magnitude"]
                else:
                    # Gradual ramp-up
                    impact_strength = impact["impact_magnitude"] * (months_since / 6)

                total_impact += impact_strength
                impact_details.append(
                    {
                        "event": impact["parent_id"],
                        "impact_magnitude": impact["impact_magnitude"],
                        "lag_months": impact["lag_months"],
                        "effective_date": effective_date,
                        "realized_impact": impact_strength,
                        "evidence": impact["evidence_basis"],
                    }
                )

        return total_impact, impact_details

    def validate_impacts_historically(self):
        """Validate impact estimates against historical data"""

        validation_results = []

        # Key events to validate
        key_events = {
            "Telebirr Launch": {
                "event_date": pd.Timestamp("2021-05-01"),
                "affected_indicators": ["ACC_MM_ACCOUNT", "USG_DIGITAL_PAYMENT"],
                "pre_period": pd.Timestamp("2021-01-01"),
                "post_period": pd.Timestamp("2024-01-01"),
            },
            "M-Pesa Entry": {
                "event_date": pd.Timestamp("2023-08-01"),
                "affected_indicators": ["ACC_MM_ACCOUNT"],
                "pre_period": pd.Timestamp("2023-01-01"),
                "post_period": pd.Timestamp("2024-12-31"),
            },
        }

        for event_name, event_info in key_events.items():
            for indicator in event_info["affected_indicators"]:
                # Get actual pre/post values
                pre_value = self.get_indicator_value(
                    indicator, event_info["pre_period"]
                )
                post_value = self.get_indicator_value(
                    indicator, event_info["post_period"]
                )

                if pre_value is not None and post_value is not None:
                    actual_change = post_value - pre_value

                    # Get modeled impact
                    modeled_impact, details = self.estimate_event_impacts(
                        indicator, event_info["post_period"]
                    )

                    validation_results.append(
                        {
                            "event": event_name,
                            "indicator": indicator,
                            "pre_value": pre_value,
                            "post_value": post_value,
                            "actual_change": actual_change,
                            "modeled_impact": modeled_impact,
                            "difference": modeled_impact - actual_change,
                            "accuracy": 1
                            - abs(modeled_impact - actual_change)
                            / max(abs(actual_change), 0.1),
                            "validation_period": f"{event_info['pre_period'].year}-{event_info['post_period'].year}",
                        }
                    )

        return pd.DataFrame(validation_results)

    def get_indicator_value(self, indicator_code, date):
        """Get value of an indicator at a specific date"""

        indicator_data = self.observations[
            self.observations["indicator_code"] == indicator_code
        ].copy()

        if len(indicator_data) == 0:
            return None

        # Find closest observation date
        indicator_data["date_diff"] = abs(indicator_data["observation_date"] - date)
        closest = indicator_data.loc[indicator_data["date_diff"].idxmin()]

        # Only return if within reasonable time window (2 years)
        if closest["date_diff"].days > 730:
            return None

        return closest["value_numeric"]

    def create_impact_timeline(self, indicator, start_date, end_date):
        """Create timeline of cumulative impacts"""

        dates = pd.date_range(start=start_date, end=end_date, freq="MS")  # Monthly
        impacts = []
        cumulative_impacts = []

        for date in dates:
            impact, _ = self.estimate_event_impacts(indicator, date)
            impacts.append(impact)
            cumulative_impacts.append(sum(impacts))

        timeline_df = pd.DataFrame(
            {
                "date": dates,
                "monthly_impact": impacts,
                "cumulative_impact": cumulative_impacts,
            }
        )

        return timeline_df

    def scenario_analysis(self, future_events):
        """Analyze impact of potential future events"""

        scenarios = []

        for scenario_name, events in future_events.items():
            # Create hypothetical impact links
            scenario_impacts = []

            for event in events:
                # Estimate impact based on comparable events
                base_impact = self.estimate_base_impact(event)
                scenario_impacts.append(
                    {
                        "event": event["name"],
                        "indicator": event["indicator"],
                        "estimated_impact": base_impact["impact"],
                        "confidence": base_impact["confidence"],
                        "lag": event.get("lag_months", 12),
                    }
                )

            scenarios.append(
                {
                    "scenario": scenario_name,
                    "events": events,
                    "total_impact": sum(
                        imp["estimated_impact"] for imp in scenario_impacts
                    ),
                    "details": scenario_impacts,
                }
            )

        return scenarios

    def estimate_base_impact(self, event):
        """Estimate base impact for new events based on comparable ones"""

        comparable_events = {
            "new_mobile_money_operator": {
                "comparable": "M-Pesa Entry",
                "adjustment_factor": 0.8,  # Assuming diminishing returns
                "confidence": "medium",
            },
            "major_policy_reform": {
                "comparable": "Interest Rate Cap Removal",
                "adjustment_factor": 1.2,
                "confidence": "low",
            },
            "infrastructure_expansion": {
                "comparable": "Fayda Digital ID National Rollout",
                "adjustment_factor": 1.0,
                "confidence": "medium",
            },
        }

        event_type = event.get("type", "new_mobile_money_operator")
        comparable = comparable_events.get(
            event_type, comparable_events["new_mobile_money_operator"]
        )

        # Get impact from comparable event
        comparable_impact = self.impact_links[
            self.impact_links["parent_id"] == comparable["comparable"]
        ]

        if len(comparable_impact) > 0:
            base_impact = comparable_impact["impact_magnitude"].iloc[0]
            estimated_impact = base_impact * comparable["adjustment_factor"]
        else:
            estimated_impact = 5.0  # Default impact
            comparable["confidence"] = "low"

        return {
            "impact": estimated_impact,
            "confidence": comparable["confidence"],
            "method": f"Based on {comparable['comparable']} with adjustment factor {comparable['adjustment_factor']}",
        }

    def create_visualizations(self, output_dir="reports/figures"):
        """Create impact modeling visualizations"""

        import os

        os.makedirs(output_dir, exist_ok=True)

        # 1. Impact Matrix Heatmap
        impact_matrix, _ = self.create_impact_matrix()

        # Prepare data for heatmap
        heatmap_data = impact_matrix.set_index("parent_id")
        numeric_cols = heatmap_data.select_dtypes(include=[np.number]).columns
        heatmap_data = heatmap_data[numeric_cols].fillna(0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
        plt.title("Event-Impact Matrix (Magnitude of Impact)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/impact_matrix.png", dpi=300)

        # 2. Historical Validation Plot
        validation_df = self.validate_impacts_historically()

        if not validation_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(validation_df))
            width = 0.35

            ax.bar(
                x - width / 2,
                validation_df["actual_change"],
                width,
                label="Actual Change",
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                validation_df["modeled_impact"],
                width,
                label="Modeled Impact",
                alpha=0.8,
            )
            ax.set_xlabel("Event-Indicator Pair")
            ax.set_ylabel("Change (percentage points)")
            ax.set_title("Model Validation: Actual vs Modeled Impacts")
            ax.set_xticks(x)
            ax.set_xticklabels(
                validation_df.apply(
                    lambda row: f"{row['event']}\n{row['indicator']}", axis=1
                ),
                rotation=45,
                ha="right",
            )
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_validation.png", dpi=300)

        # 3. Impact Timeline
        timeline_df = self.create_impact_timeline(
            "ACC_OWNERSHIP", pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeline_df["date"], timeline_df["cumulative_impact"], linewidth=2)
        ax.fill_between(
            timeline_df["date"], 0, timeline_df["cumulative_impact"], alpha=0.3
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Impact (percentage points)")
        ax.set_title("Cumulative Impact of Events on Account Ownership")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/impact_timeline.png", dpi=300)

        plt.close("all")

        return {
            "impact_matrix": f"{output_dir}/impact_matrix.png",
            "model_validation": (
                f"{output_dir}/model_validation.png"
                if not validation_df.empty
                else None
            ),
            "impact_timeline": f"{output_dir}/impact_timeline.png",
        }
