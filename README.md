# TheDrifters
Code repository for our CoCothon project - Retail Assortment Intelligence Engine

🛍️ Retail Assortment Intelligence Engine (RAIE)"Moving Retail from One-Size-Fits-All to Exactly-What-Fits-Here"

📌 Executive Summary

The Retail Assortment Intelligence Engine (RAIE) is an AI-powered orchestration layer built on the Snowflake AI Data Cloud. It solves a multi-million dollar inefficiency in fashion and footwear retail: the mismatch between regional body profiles and inventory distribution. By leveraging Snowflake Cortex AI, RAIE transforms demographic and preference data into optimized, store-level SKU distributions.

📉 The Problem: 
The Flaw of "One-Size-Fits-All"In traditional retail, a common practice is to use a generic size assortment, which often allocates equal quantities of all sizes across all stores. This approach fails to account for regional variations in customer profiles.Missing Sizes & Lost Sales: Customers frequently encounter stockouts in high-demand sizes, leading to low conversion rates.Inventory Waste: Retailers face significant markdown pressure from an overstock of irrelevant SKUs in the wrong locations.Revenue Leakage in Footwear: Failing to account for regional variations in foot length and width results in missed opportunities.High Return Rates: Poor fit leads to increased returns and high reverse logistics costs.

🏗️ Technical Architecture
RAIE is architected to move beyond "static planning" into Agentic Retail Orchestration.1. The Foundation Tables (Snowflake)The system is built on a specialized Star Schema designed for biometric intelligence:DIM_REGION: Geographical clusters for demographic analysis.DIM_STORES: Mapping physical locations to regional IDs and state-level metadata.DIM_BODY_PROFILE: Biometric attributes (Height, Weight, Body Type, Age).REGION_BODY_PROFILE_DISTRIBUTION: The "Intelligence Layer" containing population percentages per profile.FACT_SALES: Transactional data used to anchor AI predictions to real-world velocity.2. Synthetic Data Generation (CoCo Powered)To demonstrate the engine's precision, we utilized Snowflake Cortex Code (CoCo) to synthesize high-fidelity demographic data. We purposefully introduced regional outliers—such as higher obesity indices in certain clusters—to prove the engine's ability to detect and react to localized demand signals.

🚀 Streamlit App: 
The Merchant War RoomThe application provides an end-to-end interface for merchandise planners, divided into four intelligent pillars:

Tab 1: 
📊 Current Data (Intelligence Foundation)Performance Audit: 
Visualizes historical sales against regional biometric baselines.Biometric Distribution: Interactive charts for Height, Weight, Body Type, and Age profiles.Intelligent Attribution: Uses Snowpark to join raw sales with dominant body profiles to identify fit mismatches.

Tab 2: 
🔮 Future Predictions (Demand Outlook)Demand Forecasting: 
Calculates projected units for planning horizons ranging from 1 month to 2 years.Stockout Risk: Proactively identifies sizes where current stock covers less than 30% of projected demand.

Tab 3: 
🧠 Assortment Insights (Optimization Engine)Largest Remainder Allocation: 
A mathematical algorithm ensuring 100% of the "Open-to-Buy" budget is distributed as whole units.Gap Analysis: A real-time audit between "Recommended" vs. "Actual" stock to highlight overstocked and understocked variants.

Tab 4:
🤖 AI Advisor (Cortex Orchestration)Cortex COMPLETE:
Generates a professional buying brief using the snowflake-arctic model.Cortex SENTIMENT: Analyzes unstructured customer return reasons to identify "Fit Flags" (e.g., items being returned for being "too narrow").


🛠️ Setup & Installation

Clone the Repository:
Bashgit clone https://github.com/[your-username]/raie-engine.git

Snowflake Setup:
Execute the DDL scripts provided in the /sql folder to create the RAIE_ANALYTICS database.
Populate tables using the provided mock data scripts.
Cortex Access:Ensure your Snowflake user role has the necessary privileges to call SNOWFLAKE.CORTEX.COMPLETE and SENTIMENT.
Run Streamlit:Upload streamlit_app.py to a Streamlit in Snowflake stage or run locally by configuring secrets.toml.
