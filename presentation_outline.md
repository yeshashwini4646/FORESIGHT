# Project FORESIGHT Presentation Outline

## 1. Business Problem
Retail teams need a repeatable way to understand demand, anticipate replenishment pressure, and identify potential overstock from sales history.

## 2. Client Background
Project FORESIGHT is presented as a retail intelligence prototype for online retail operations. The current dataset is historical transaction data rather than a live client inventory feed.

## 3. Dataset
- Source: Online Retail II
- 525,461 raw transaction rows loaded
- 504,730 rows retained after cleaning
- Transaction date, product, quantity, price, customer, and country fields
- No inventory snapshots or actual stock levels are included

## 4. Data Cleaning
- Removed exact duplicates
- Excluded cancelled invoices
- Removed missing descriptions
- Coerced and validated dates, quantities, and prices
- Removed non-positive quantities and prices
- Calculated revenue as quantity multiplied by price

## 5. EDA Insights
The dashboard uses daily revenue and quantity trends, country/product breakdowns, and top-product views. The current evidence supports transaction and demand analysis; inventory conclusions are estimates because stock data is unavailable.

## 6. Forecast Model
Features include lag-1 quantity, lag-7 quantity, a prior 7-day rolling mean, and calendar variables. The workflow uses an ordered 80/20 time split and prevents the target day from entering rolling features.

## 7. Model Performance
Results from the regenerated cleaned, leakage-safe pipeline:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Lag-7 baseline | 11,322.73 | 17,304.36 |
| Random Forest | 10,646.54 | 15,045.50 |
| XGBoost | 8,629.22 | 11,743.91 |
| LightGBM | 7,396.95 | 9,918.68 |

LightGBM is the best model in this regenerated run. The previously reported XGBoost metrics remain historical results from the prior artifact and should not be presented as the current run.

## 8. Risk Scoring
The risk engine estimates inventory from recent demand because actual stock snapshots are unavailable. It reports forecast demand, estimated stock, inventory-demand gap, coverage days, risk category, recommendation, and the inventory assumption.

## 9. Dashboard
The Streamlit project contains Home, Sales Analytics, Forecast, Inventory, Risk Dashboard, Product Details, and Executive Summary pages. Existing pages are retained and are being expanded incrementally.

## 10. Deployment and Recommendations
- Run the data pipeline before serving predictions.
- Serve the FastAPI model with Uvicorn.
- Deploy Streamlit and FastAPI separately.
- Replace estimated inventory with inventory snapshots before making operational stock decisions.
- Monitor model performance on a future time window after deployment.
