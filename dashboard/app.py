import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from data_loader import DataLoader
from eda_analyzer import EDAAnalyzer
from forecast_model import FinancialInclusionForecaster

# Page configuration
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class FinancialInclusionDashboard:
    def __init__(self):
        self.data_loader = DataLoader()
        self.analyzer = None
        self.forecaster = None
        self.load_data()

    def load_data(self):
        """Load data for dashboard"""
        try:
            # Load enriched data
            enriched_data_path = "data/processed/enriched_data.csv"
            if os.path.exists(enriched_data_path):
                self.analyzer = EDAAnalyzer(enriched_data_path)
                self.forecaster = FinancialInclusionForecaster(enriched_data_path)
            else:
                st.warning("Enriched data not found. Using sample data.")
                raw_data, _ = self.data_loader.load_data()
                self.analyzer = EDAAnalyzer()
                self.forecaster = FinancialInclusionForecaster()
        except Exception as e:
            st.error(f"Error loading data: {e}")

    def render_overview(self):
        """Render overview page"""
        st.markdown(
            '<h1 class="main-header">📊 Ethiopia Financial Inclusion Dashboard</h1>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Account Ownership (2024)",
                value="49%",
                delta="+3pp since 2021",
                delta_color="off",
            )

        with col2:
            st.metric(
                label="Digital Payment Usage", value="~35%", delta="Estimate 2024"
            )

        with col3:
            st.metric(
                label="Mobile Money Users", value="65M+", delta="Telebirr + M-Pesa"
            )

        with col4:
            st.metric(label="NFIS-II Target (2027)", value="60%", delta="-11pp gap")

        # Key insights
        st.markdown(
            '<div class="sub-header">📈 Key Insights</div>', unsafe_allow_html=True
        )

        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown(
                """
            <div class="info-box">
            <h4>🚀 Growth Drivers</h4>
            <ul>
            <li>Mobile money expansion (Telebirr, M-Pesa)</li>
            <li>Infrastructure improvements (4G, agents)</li>
            <li>Interoperability enabling more use cases</li>
            <li>Digital ID simplifying account opening</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with insights_col2:
            st.markdown(
                """
            <div class="info-box">
            <h4>⚠️ Challenges</h4>
            <ul>
            <li>Slow rural adoption despite infrastructure</li>
            <li>Persistent gender gap (~10 percentage points)</li>
            <li>Registered vs active user gap in mobile money</li>
            <li>Economic factors affecting affordability</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # P2P/ATM Crossover Indicator
        st.markdown(
            '<div class="sub-header">💱 P2P Digital Transfers vs ATM Withdrawals</div>',
            unsafe_allow_html=True,
        )

        # Simulated data for visualization
        crossover_data = pd.DataFrame(
            {
                "Year": [2020, 2021, 2022, 2023, 2024],
                "P2P_Transfers": [30, 45, 65, 85, 105],
                "ATM_Withdrawals": [100, 95, 90, 85, 80],
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=crossover_data["Year"],
                y=crossover_data["P2P_Transfers"],
                name="P2P Digital Transfers",
                line=dict(color="green", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=crossover_data["Year"],
                y=crossover_data["ATM_Withdrawals"],
                name="ATM Cash Withdrawals",
                line=dict(color="red", width=3),
            )
        )

        # Add crossover point annotation
        fig.add_vline(x=2023.5, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=2023.5, y=85, text="Crossover Point (2023)", showarrow=True, arrowhead=1
        )

        fig.update_layout(
            title="Digital P2P Transfers Surpass ATM Withdrawals",
            xaxis_title="Year",
            yaxis_title="Transaction Volume Index (2020=100)",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_trends(self):
        """Render trends analysis page"""
        st.markdown(
            '<h1 class="main-header">📈 Trends Analysis</h1>', unsafe_allow_html=True
        )

        if self.analyzer is None:
            st.warning("Data not loaded. Please check data files.")
            return

        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox(
                "Start Year", options=list(range(2011, 2025)), index=0
            )
        with col2:
            end_year = st.selectbox(
                "End Year",
                options=list(range(2011, 2025)),
                index=len(list(range(2011, 2025))) - 1,
            )

        # Indicator selector
        indicator_options = {
            "ACC_OWNERSHIP": "Account Ownership",
            "ACC_MM_ACCOUNT": "Mobile Money Accounts",
            "USG_DIGITAL_PAYMENT": "Digital Payments",
            "AGENT_DENSITY": "Agent Density",
            "4G_COVERAGE": "4G Coverage",
        }

        selected_indicator = st.selectbox(
            "Select Indicator",
            options=list(indicator_options.keys()),
            format_func=lambda x: indicator_options[x],
        )

        # Get data for selected indicator
        indicator_data = self.analyzer.observations[
            self.analyzer.observations["indicator_code"] == selected_indicator
        ]

        if len(indicator_data) > 0:
            # Create trend chart
            fig = px.line(
                indicator_data,
                x="observation_date",
                y="value_numeric",
                title=f"{indicator_options[selected_indicator]} Trend",
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Date", yaxis_title="Value (%)", hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                latest = indicator_data["value_numeric"].iloc[-1]
                st.metric("Latest Value", f"{latest:.1f}%")

            with col2:
                growth = indicator_data["value_numeric"].pct_change().iloc[-1] * 100
                st.metric("Latest Growth", f"{growth:.1f}%")

            with col3:
                avg_growth = indicator_data["value_numeric"].pct_change().mean() * 100
                st.metric("Avg Annual Growth", f"{avg_growth:.1f}%")
        else:
            st.info(f"No data available for {indicator_options[selected_indicator]}")

        # Comparative view
        st.markdown(
            '<div class="sub-header">📊 Comparative Analysis</div>',
            unsafe_allow_html=True,
        )

        compare_col1, compare_col2 = st.columns(2)

        with compare_col1:
            # Gender gap visualization
            gender_data, _ = self.analyzer.analyze_gender_gap()

            fig_gender = go.Figure()
            fig_gender.add_trace(
                go.Bar(
                    x=gender_data["year"],
                    y=gender_data["male_ownership"],
                    name="Male",
                    marker_color="blue",
                )
            )
            fig_gender.add_trace(
                go.Bar(
                    x=gender_data["year"],
                    y=gender_data["female_ownership"],
                    name="Female",
                    marker_color="pink",
                )
            )

            fig_gender.update_layout(
                title="Gender Gap in Account Ownership",
                xaxis_title="Year",
                yaxis_title="Ownership (%)",
                barmode="group",
            )

            st.plotly_chart(fig_gender, use_container_width=True)

        with compare_col2:
            # Urban-rural visualization
            urban_rural_data, _ = self.analyzer.analyze_urban_rural_gap()

            fig_urban = go.Figure()
            fig_urban.add_trace(
                go.Scatter(
                    x=urban_rural_data["year"],
                    y=urban_rural_data["urban_ownership"],
                    name="Urban",
                    line=dict(color="orange", width=3),
                )
            )
            fig_urban.add_trace(
                go.Scatter(
                    x=urban_rural_data["year"],
                    y=urban_rural_data["rural_ownership"],
                    name="Rural",
                    line=dict(color="green", width=3),
                )
            )

            fig_urban.update_layout(
                title="Urban-Rural Divide",
                xaxis_title="Year",
                yaxis_title="Ownership (%)",
            )

            st.plotly_chart(fig_urban, use_container_width=True)

    def render_forecasts(self):
        """Render forecasts page"""
        st.markdown(
            '<h1 class="main-header">🔮 Inclusion Forecasts 2025-2027</h1>',
            unsafe_allow_html=True,
        )

        if self.forecaster is None:
            st.warning("Forecaster not initialized. Please check data files.")
            return

        # Scenario selector
        scenario = st.radio(
            "Select Scenario", ["Optimistic", "Base", "Pessimistic"], horizontal=True
        ).lower()

        # Generate forecasts
        forecasts = self.forecaster.forecast_all_indicators()

        if forecasts:
            # Key metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                acc_forecast = forecasts["ACC_OWNERSHIP"]["scenarios"][2027][scenario][
                    "value"
                ]
                acc_target = forecasts["ACC_OWNERSHIP"]["info"]["target_2027"]
                st.metric(
                    label="Account Ownership (2027)",
                    value=f"{acc_forecast:.1f}%",
                    delta=f"Target: {acc_target}%",
                    delta_color="normal" if acc_forecast >= acc_target else "inverse",
                )

            with col2:
                pay_forecast = forecasts["USG_DIGITAL_PAYMENT"]["scenarios"][2027][
                    scenario
                ]["value"]
                pay_target = forecasts["USG_DIGITAL_PAYMENT"]["info"]["target_2027"]
                st.metric(
                    label="Digital Payments (2027)",
                    value=f"{pay_forecast:.1f}%",
                    delta=f"Target: {pay_target}%",
                    delta_color="normal" if pay_forecast >= pay_target else "inverse",
                )

            with col3:
                mm_forecast = forecasts["ACC_MM_ACCOUNT"]["scenarios"][2027][scenario][
                    "value"
                ]
                mm_target = forecasts["ACC_MM_ACCOUNT"]["info"]["target_2027"]
                st.metric(
                    label="Mobile Money (2027)",
                    value=f"{mm_forecast:.1f}%",
                    delta=f"Target: {mm_target}%",
                    delta_color="normal" if mm_forecast >= mm_target else "inverse",
                )

            # Forecast visualization
            st.markdown(
                '<div class="sub-header">📈 Forecast Trajectories</div>',
                unsafe_allow_html=True,
            )

            # Create forecast chart
            years = [2024, 2025, 2026, 2027]

            fig = go.Figure()

            # Add traces for each indicator
            colors = {
                "ACC_OWNERSHIP": "blue",
                "USG_DIGITAL_PAYMENT": "green",
                "ACC_MM_ACCOUNT": "orange",
            }

            for indicator_code, color in colors.items():
                if indicator_code in forecasts:
                    values = [forecasts[indicator_code]["info"]["current_value"]]
                    for year in [2025, 2026, 2027]:
                        values.append(
                            forecasts[indicator_code]["scenarios"][year][scenario][
                                "value"
                            ]
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=values,
                            name=forecasts[indicator_code]["info"]["name"],
                            line=dict(color=color, width=3),
                            mode="lines+markers",
                        )
                    )

            # Add target lines
            fig.add_hline(
                y=60,
                line_dash="dash",
                line_color="blue",
                annotation_text="Account Target (60%)",
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="green",
                annotation_text="Payment Target (50%)",
                annotation_position="bottom right",
            )

            fig.update_layout(
                title=f"{scenario.capitalize()} Scenario Forecast",
                xaxis_title="Year",
                yaxis_title="Value (%)",
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Target achievement table
            st.markdown(
                '<div class="sub-header">🎯 Target Achievement Assessment</div>',
                unsafe_allow_html=True,
            )

            achievement_data = []
            for indicator_code in [
                "ACC_OWNERSHIP",
                "USG_DIGITAL_PAYMENT",
                "ACC_MM_ACCOUNT",
            ]:
                if indicator_code in forecasts:
                    forecast_2027 = forecasts[indicator_code]["scenarios"][2027][
                        scenario
                    ]["value"]
                    target_2027 = forecasts[indicator_code]["info"]["target_2027"]
                    gap = forecast_2027 - target_2027

                    achievement_data.append(
                        {
                            "Indicator": forecasts[indicator_code]["info"]["name"],
                            f"{scenario.capitalize()} Forecast 2027": f"{forecast_2027:.1f}%",
                            "NFIS-II Target 2027": f"{target_2027:.1f}%",
                            "Gap": f"{gap:+.1f} pp",
                            "Status": "✅ Achieved" if gap >= 0 else "❌ Not Achieved",
                        }
                    )

            achievement_df = pd.DataFrame(achievement_data)
            st.dataframe(achievement_df, use_container_width=True)

            # Uncertainty visualization
            st.markdown(
                '<div class="sub-header">📊 Forecast Uncertainty</div>',
                unsafe_allow_html=True,
            )

            uncertainty_data = []
            for indicator_code in ["ACC_OWNERSHIP", "USG_DIGITAL_PAYMENT"]:
                if indicator_code in forecasts:
                    optimistic = forecasts[indicator_code]["scenarios"][2027][
                        "optimistic"
                    ]["value"]
                    base = forecasts[indicator_code]["scenarios"][2027]["base"]["value"]
                    pessimistic = forecasts[indicator_code]["scenarios"][2027][
                        "pessimistic"
                    ]["value"]

                    uncertainty_data.append(
                        {
                            "Indicator": forecasts[indicator_code]["info"]["name"],
                            "Optimistic": optimistic,
                            "Base": base,
                            "Pessimistic": pessimistic,
                            "Range": optimistic - pessimistic,
                        }
                    )

            uncertainty_df = pd.DataFrame(uncertainty_data)

            fig_uncertainty = go.Figure()

            for idx, row in uncertainty_df.iterrows():
                fig_uncertainty.add_trace(
                    go.Scatter(
                        x=[row["Indicator"], row["Indicator"]],
                        y=[row["Pessimistic"], row["Optimistic"]],
                        mode="lines",
                        line=dict(width=10, color="lightgray"),
                        name=row["Indicator"],
                        showlegend=False,
                    )
                )
                fig_uncertainty.add_trace(
                    go.Scatter(
                        x=[row["Indicator"]],
                        y=[row["Base"]],
                        mode="markers",
                        marker=dict(size=15, color="blue"),
                        name=f"{row['Indicator']} Base",
                        showlegend=True if idx == 0 else False,
                    )
                )

            fig_uncertainty.update_layout(
                title="2027 Forecast Ranges with Uncertainty",
                yaxis_title="Value (%)",
                xaxis_title="Indicator",
                showlegend=True,
            )

            st.plotly_chart(fig_uncertainty, use_container_width=True)

        else:
            st.info(
                "No forecast data available. Please run the forecasting model first."
            )

    def render_policy_simulator(self):
        """Render policy simulator page"""
        st.markdown(
            '<h1 class="main-header">🎮 Policy Impact Simulator</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box">
        <h4>How to use the simulator:</h4>
        <p>Adjust the policy levers below to see how different interventions might affect financial inclusion outcomes by 2027.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Policy levers
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏛️ Policy Interventions")

            # Digital ID rollout
            digital_id = st.slider(
                "Digital ID Coverage Increase",
                min_value=0,
                max_value=30,
                value=10,
                help="Percentage point increase in population covered by digital ID",
            )

            # KYC simplification
            kyc_simplification = st.select_slider(
                "KYC Requirements Simplification",
                options=["No Change", "Moderate", "Significant"],
                value="Moderate",
                help="Level of KYC requirement simplification",
            )

            # Interest rate policy
            interest_policy = st.radio(
                "Interest Rate Policy",
                ["Maintain Caps", "Partial Liberalization", "Full Liberalization"],
                index=1,
            )

        with col2:
            st.markdown("### 📱 Infrastructure Investments")

            # Agent network expansion
            agent_expansion = st.slider(
                "Agent Network Expansion",
                min_value=0,
                max_value=50,
                value=25,
                help="Percentage increase in agent density",
            )

            # Network coverage
            network_coverage = st.slider(
                "4G Network Coverage",
                min_value=70,
                max_value=95,
                value=85,
                help="Target 4G coverage by 2027 (%)",
            )

            # Interoperability
            interoperability = st.select_slider(
                "Interoperability Level",
                options=["Basic", "Enhanced", "Full"],
                value="Enhanced",
                help="Level of interoperability between providers",
            )

        # Calculate impacts
        st.markdown(
            '<div class="sub-header">📊 Simulated Impacts</div>', unsafe_allow_html=True
        )

        # Simulate impacts based on policy choices
        base_account = 49  # 2024 baseline

        # Impact calculations (simplified)
        impacts = {
            "Digital ID": digital_id
            * 0.3,  # 30% of coverage increase translates to account increase
            "KYC Simplification": {"No Change": 0, "Moderate": 2, "Significant": 4}[
                kyc_simplification
            ],
            "Interest Policy": {
                "Maintain Caps": 0,
                "Partial Liberalization": 1,
                "Full Liberalization": 3,
            }[interest_policy],
            "Agent Expansion": agent_expansion * 0.2,  # 20% of agent increase
            "Network Coverage": (network_coverage - 70)
            * 0.15,  # 15% of coverage increase
            "Interoperability": {"Basic": 1, "Enhanced": 3, "Full": 5}[
                interoperability
            ],
        }

        total_impact = sum(impacts.values())
        simulated_2027 = base_account + total_impact

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Base (2024)", "49%", delta=None)

        with col2:
            st.metric("Total Policy Impact", f"+{total_impact:.1f} pp", delta=None)

        with col3:
            st.metric(
                "Simulated 2027",
                f"{simulated_2027:.1f}%",
                delta=f"{simulated_2027 - 60:+.1f} pp vs Target",
                delta_color="normal" if simulated_2027 >= 60 else "inverse",
            )

        # Impact breakdown
        st.markdown("### 📈 Impact Breakdown")

        impact_df = pd.DataFrame(
            {
                "Policy Lever": list(impacts.keys()),
                "Impact (pp)": list(impacts.values()),
            }
        ).sort_values("Impact (pp)", ascending=False)

        fig = px.bar(
            impact_df,
            x="Policy Lever",
            y="Impact (pp)",
            title="Policy Impact Contributions",
            color="Impact (pp)",
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Recommended Actions Based on Simulation")

        recommendations = []

        if digital_id < 20:
            recommendations.append(
                "Consider accelerating digital ID rollout to reach more underserved populations"
            )

        if kyc_simplification != "Significant":
            recommendations.append(
                "Further KYC simplification could significantly boost account opening"
            )

        if agent_expansion < 30:
            recommendations.append(
                "Agent network expansion is crucial for rural inclusion"
            )

        if len(recommendations) > 0:
            for rec in recommendations:
                st.info(f"• {rec}")
        else:
            st.success(
                "Your policy mix appears well-balanced for achieving inclusion targets!"
            )

    def render_insights(self):
        """Render insights and answers to consortium questions"""
        st.markdown(
            '<h1 class="main-header">💡 Insights & Recommendations</h1>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box">
        <h3>Answers to Consortium's Key Questions</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Question 1
        st.markdown("### 1. What drives financial inclusion in Ethiopia?")

        drivers = """
        **Primary Drivers:**
        - **Mobile Money Expansion:** Telebirr (54M+ users) and M-Pesa (10M+ users) have been the main growth engines
        - **Infrastructure Improvements:** 4G coverage growth (15% → 70%) and agent network expansion
        - **Policy Enablers:** Digital ID rollout, interoperability platform, regulatory reforms
        - **Market Competition:** Entry of M-Pesa increased competition and innovation
        
        **Secondary Drivers:**
        - Urbanization and demographic changes
        - Smartphone penetration growth (20% → 50%)
        - Economic growth despite inflationary pressures
        """

        st.markdown(drivers)

        # Question 2
        st.markdown("### 2. How do events affect inclusion outcomes?")

        events_impact = """
        **Major Event Impacts (Estimated):**
        
        | Event | Impact on Accounts | Impact on Usage | Lag |
        |-------|-------------------|-----------------|-----|
        | Telebirr Launch (2021) | +8-10 pp | +10-12 pp | 6-12 months |
        | M-Pesa Entry (2023) | +3-5 pp | +4-6 pp | 3-6 months |
        | Interoperability (2022) | +1-2 pp | +2-3 pp | 6-12 months |
        | Digital ID Rollout (2023) | +2-4 pp | +1-2 pp | 12-18 months |
        
        **Key Patterns:**
        - Product launches have immediate impacts (3-6 month lag)
        - Infrastructure investments have longer lags (12-24 months)
        - Policy changes show gradual effects as implementation rolls out
        - Competitive entries accelerate overall market growth
        """

        st.markdown(events_impact)

        # Question 3
        st.markdown("### 3. How will inclusion look in 2025-2027?")

        projections = """
        **Base Scenario Projections:**
        
        | Indicator | 2024 | 2025 | 2026 | 2027 | Target 2027 |
        |-----------|------|------|------|------|-------------|
        | Account Ownership | 49% | 51-52% | 53-54% | 54-56% | 60% |
        | Digital Payments | ~35% | 38-40% | 41-43% | 42-45% | 50% |
        | Mobile Money Accounts | 9.45% | 12-14% | 15-18% | 18-22% | 25% |
        
        **Key Trends:**
        - Continued growth but at slower pace than 2017-2021 period
        - Mobile money remains primary driver
        - Usage growing faster than access (narrowing registered-active gap)
        - Urban-rural and gender gaps persist but may narrow slightly
        
        **Target Achievement:**
        - NFIS-II 60% account ownership target requires additional interventions
        - Digital payment target (50%) more achievable with current trajectory
        - Mobile money target (25%) likely achievable by 2027
        """

        st.markdown(projections)

        # Recommendations
        st.markdown("### 🎯 Strategic Recommendations")

        recommendations = """
        1. **Accelerate Rural Inclusion**
           - Target agent network expansion in underserved regions
           - Develop agriculture-focused digital payment products
           - Partner with cooperatives and farmer organizations
        
        2. **Address Gender Gap**
           - Design women-focused financial products
           - Promote female agent networks
           - Address structural barriers to women's financial access
        
        3. **Boost Usage, Not Just Access**
           - Develop compelling use cases (wages, bills, commerce)
           - Promote merchant acceptance (QR codes, POS terminals)
           - Simplify user interfaces and reduce transaction costs
        
        4. **Strengthen Monitoring**
           - Establish more frequent inclusion measurement
           - Develop real-time dashboards for policymakers
           - Regular impact assessments of interventions
        """

        st.markdown(recommendations)

    def run(self):
        """Run the dashboard application"""

        # Sidebar navigation
        st.sidebar.title("Navigation")

        page_options = {
            "Overview": self.render_overview,
            "Trends Analysis": self.render_trends,
            "Forecasts": self.render_forecasts,
            "Policy Simulator": self.render_policy_simulator,
            "Insights & Recommendations": self.render_insights,
        }

        selected_page = st.sidebar.radio("Go to", list(page_options.keys()))

        # Render selected page
        page_options[selected_page]()

        # Sidebar information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 About This Dashboard")
        st.sidebar.info(
            """
        This dashboard provides forecasts and analysis of Ethiopia's financial inclusion progress.
        
        **Data Sources:**
        - Global Findex Database
        - NBE Reports
        - GSMA Intelligence
        - Operator Reports
        
        **Last Updated:** January 2026
        
        **For:** National Bank of Ethiopia, Development Partners, Mobile Money Operators
        """
        )


# Run the dashboard
if __name__ == "__main__":
    dashboard = FinancialInclusionDashboard()
    dashboard.run()
