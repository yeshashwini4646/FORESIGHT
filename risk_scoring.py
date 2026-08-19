import pandas as pd
import numpy as np


def calculate_risk(df, inventory_multiplier=1.5, overstock_multiplier=2.0):
    """
    Calculate transparent demonstration inventory risk.

    Online Retail II does not contain stock snapshots. Estimated stock is
    therefore based on recent demand and must not be presented as actual stock.
    """
    if inventory_multiplier <= 0 or overstock_multiplier <= inventory_multiplier:
        raise ValueError(
            "overstock_multiplier must be greater than a positive inventory_multiplier"
        )

    risk_df = df.copy()

    demand_source = "Predicted_Quantity" if "Predicted_Quantity" in risk_df else "Quantity"
    risk_df["Forecast_Demand"] = risk_df[demand_source].clip(lower=0)
    reference_demand = risk_df.get("Rolling_7", risk_df["Forecast_Demand"])
    reference_demand = reference_demand.fillna(risk_df["Forecast_Demand"])
    risk_df["Current_Stock"] = (reference_demand * inventory_multiplier).round()
    risk_df["Inventory_Demand_Gap"] = (
        risk_df["Current_Stock"] - risk_df["Forecast_Demand"]
    )
    risk_df["Stock_Coverage_Days"] = (
        risk_df["Current_Stock"] / risk_df["Forecast_Demand"].replace(0, pd.NA)
    ).round(2)

    # Risk Rules
    conditions = [
        risk_df["Forecast_Demand"] > risk_df["Current_Stock"],
        risk_df["Current_Stock"] >= (
            risk_df["Forecast_Demand"] * overstock_multiplier
        ),
    ]

    choices = [
        "Stockout Risk",
        "Overstock Risk"
    ]

    risk_df["Risk"] = np.select(
        conditions,
        choices,
        default="Healthy"
    )

    risk_df["Recommendation"] = risk_df["Risk"].map(
        {
            "Stockout Risk": "Increase replenishment",
            "Healthy": "Maintain current inventory",
            "Overstock Risk": "Reduce replenishment / promotional action",
        }
    )
    risk_df["Inventory_Assumption"] = (
        "Estimated from recent demand; Online Retail II has no stock snapshots"
    )

    return risk_df