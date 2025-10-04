###demand_forecasting_supply – Case Study  

---

##1 Structure  
From a local `.xlsx/.csv` file (imported via *choose file location*), build a robust Python workflow on Google Colab.  

#Global objectives 
Reduce stockouts and optimize stocks 

#How?
Improve demand forecasting from the demand forecast baseline

#Output 
Forecasting future demand (Horizon of 7 and 14 days)
Determine impact on KPI supply chain (stockout rate, fill rate)
Calculate impact on overall cost (Cost avoided from improved inventory management with new demand forecasting)

forecast demand → calculate projected stock = inventory - demand forecast  
                → compare to reorder_point → stockout risk (yes/no) 

# Columns of the dataset 

Data columns (total 15 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   Date                     91250 non-null  object 
 1   SKU_ID                   91250 non-null  object 
 2   Warehouse_ID             91250 non-null  object 
 3   Supplier_ID              91250 non-null  object 
 4   Region                   91250 non-null  object 
 5   Units_Sold               91250 non-null  int64  
 6   Inventory_Level          91250 non-null  int64  
 7   Supplier_Lead_Time_Days  91250 non-null  int64  
 8   Reorder_Point            91250 non-null  int64  
 9   Order_Quantity           91250 non-null  int64  
 10  Unit_Cost                91250 non-null  float64
 11  Unit_Price               91250 non-null  float64
 12  Promotion_Flag           91250 non-null  int64  
 13  Stockout_Flag            91250 non-null  int64  
 14  Demand_Forecast          91250 non-null  float64

## 2 Data processing 

a)Create a continue calendar, 'Date' for 'SKU_ID'

Ensure proper date parsing:
- Parse 'Date' column with infer_datetime_format=True.
- Assume dataset in UTC unless otherwise stated. 
- If format not ISO, user must provide format string.

b)For null cells of 'Units_Sold' --> 0 
c)For null cells of 'Inventory_Level':

- If explicit 0 in dataset → keep as 0 (means no stock).
- If missing but previous value exists → forward-fill.
- If true NA (not filled anywhere) → leave NA and flag for review.
d)For null cells of 'Supplier_Lead_Time_Days' → impute with median per supplier.

e)Check logic between 'Units sold' and 'Inventory_Level'

Fill missing values systematically with 0 (for Units_Sold, Inventory_Level) or forward-fill (for lags/moving averages).
Verify that the dataset contains only numeric values before splitting.

f)Data split, 80% training, 20% test respecting chronology 

## 3 Baseline 

To assess the added value of the ML forecasting model, we define several baselines:  
1. **Naïve forecast**: Units_Sold (j) = Units_Sold (j-1).  
2. **Moving average**: Average demand over the last 7 days.  
3. **Dataset baseline**: Compare predictions to the provided "Demand_Forecast" column.  

Performance metrics (RMSE, MAPE, WAPE) will be computed for each baseline and for the ML model.  
The ML model will be considered successful if it reduces forecast error compared to these baselines
and If this improvement translates into better business KPIs (fill rate, stockout rate, avoided costs).  

Create baseline forecasts to compare with ML models:

- Baseline_Naive = Units_Sold lagged by 1 day (lag_sell_j-1).  
- Baseline_MA7 = 7-day moving average of Units_Sold.  

Add these columns into the test dataset (test_baseline) for later evaluation.

## 4 Analysis explanation, supply chain oriented 

- Calculate : mean, standard_deviation for all columns, coeffficient_of_variation for 'SKU_ID'
- Identifying saisonality : mean per week, per month for --> 'Units_Sold'
- Identifying impact of 'Promotion_Flag' : mean of 'Units_Sold' with 'Promotion_Flag'=0(No promotion) and 'Promotion_Flag'=1(Promotion)
- Identifying impact of 'Supplier_Lead_Time_Days': correlation with 'Stockout_Flag'
- Segmentation of 'SKU_ID' : Regular vs erratic (ABC/XYZ classification)
- Classification with ABC (cumulated value) and XYZ ( demand variability)
ABC = CA_{SKU} = {Units_Sold}\times\{Unit_Price}
     A = 20 % of SKU represent 80% of CA_{SKU}
     B = 30 % of SKU represent 15% of CA_{SKU}
     C = 50 % of SKU represent 5% of CA_{SKU}
Descending sorting + accumulation of A b and c

XYZ = coeffficient_of_variation <= 0.5 --> X , low variability
      coeffficient_of_variation 0.5 < CV ≤ 1.0 --> Y , medium variability
      coeffficient_of_variation > 1.0 --> Z , high variability                                     
"z values are parameters depending on SKU class and can be adjusted for scenario testing (sensitivity analysis)."

##Feature engineering, creating variables for forecasting from dataset data

time of 'Date'  --> 'week_day', 'month', 'week_number', 'day_trend' (number of days elapsed)
sell_lag of 'Units_Sold'  --> 'lag_sell_j-1', 'lag_sell_j7', 'lag_sell_j14'
moving_average of 'lag_sell_j7''and'lag_sell_j14'  --> 'ma_7j', 'ma28j' 
volatility, standard deviation of 'lag_sell_j7''and'lag_sell_j14' --> 'volatility_j7', 'volatility_j14'
Inventory from 'lag_sell_j-1' --> days_of_stock = 'Inventory_Level'/'ma_7j'
Promotion impact from 'Promotion_Flag' --> 'promotion_of_the_day'(if 0= no promotion, if =1 promotion), 'j-1_promotion', 'promotion_density'(=density of promotion over the last 30 days)
lead time and supplier from 'Supplier_Lead_Time_Days'  = average_lead_time( average of 'Supplier_Lead_Time_Days'), leadtime_variability (variability of 'Supplier_Lead_Time_Days'
Categorization features:
- For SKU_ID: use target encoding or frequency encoding (not one-hot, to avoid dimensional explosion).
- For Warehouse_ID and Region: one-hot encoding if low cardinality, otherwise target encoding.

Encode categorical identifiers ('SKU_ID', 'Warehouse_ID', 'Region', 'Supplier_ID') into numeric format before training (LabelEncoding or similar), so the model can process them.
Ensure all input features are numeric.

##5 Forecasting model 

#Baseline : (mandatory benchmark)

Naïve forecast: Units_Sold (j) = Units_Sold (j-1).  
Moving average: Average demand over the last 7 days.  
Naive saisonality : Units_Sold (j) = Units_Sold (j-7)
Dataset baseline: Compare predictions to the provided "Demand_Forecast" column.

#Machine learning / Statistic model: 
#Input : All variable created in feature engineering 

- Time-Related Features (from 'Date') 🗓️
These features capture temporal patterns:

'week_day': The day of the week (e.g., Monday, Tuesday, or 1 to 7).
'month': The calendar month (e.g., January, February, or 1 to 12).
'week_number': The week number within the year.
'day_trend': The number of days elapsed since a fixed starting point (e.g., the beginning of the dataset).

Sales Lag Features (from 'Units_Sold') 🚀
These features look at past sales performance:

'lag_sell_j-1': Units Sold from the previous day (j−1).
'lag_sell_j7': Units Sold from 7 days ago (j−7).
'lag_sell_j14': Units Sold from 14 days ago (j−14).

- Moving Average Features (from 'lag_sell_j7' and 'lag_sell_j14') 📈
These features smooth out sales data to show underlying trends:

'ma_7j': Moving average of sales over the last 7 days (likely using 'lag_sell_j7' as a base, or perhaps the actual 'Units_Sold' data).
'ma28j': Moving average of sales over the last 28 days (likely based on a wider window of 'Units_Sold').

- Volatility and Standard Deviation Features (from 'lag_sell_j7' and 'lag_sell_j14') 📉
These features measure the fluctuation in sales:

'volatility_j7': Standard deviation of sales over the last 7 days.
'volatility_j14': Standard deviation of sales over the last 14 days.

- Inventory Feature (from 'lag_sell_j-1' and 'Inventory_Level') 📦
This feature estimates the current stock coverage:

'days_of_stock': Calculated as ’Inventory_Level’/’ma_7j’. This estimates how many days the current inventory will last based on the 7-day moving average sales rate.

- Promotion Impact Features (from 'Promotion_Flag') 🎉
These features quantify the presence and density of promotions:

'promotion_of_the_day': A binary flag (0 or 1) indicating if there is a promotion on the current day.
'j-1_promotion': A binary flag (0 or 1) indicating if there was a promotion on the previous day.
'promotion_density': The density (proportion or count) of promotions over the last 30 days.

- Lead Time and Supplier Features (from 'Supplier_Lead_Time_Days') 🚚
These features characterize the supply chain time:

'average_lead_time': The average of the 'Supplier_Lead_Time_Days' over a specified historical period.
'leadtime_variability': The variability (e.g., standard deviation) of the 'Supplier_Lead_Time_Days'.

- Categorization Features (from 'SKU_ID', 'Warehouse_ID', and 'Region') 🏷️
These features are numerical representations of categorical identifiers, typically created via encoding (e.g., One-Hot Encoding, Label Encoding, or Target Encoding):

Features derived from 'SKU_ID': (Encoded representation)
Features derived from 'Warehouse_ID': (Encoded representation)
Features derived from 'Region': (Encoded representation)

"Forecasting models must use time-series aware validation (walk-forward or expanding window), not random split, to avoid data leakage."

# Ouput 
prediction_units_sold (j+1, j+7, j+14)

# Machine learning model 
XGBOOST

## 6 Metrics and validation 

# Forecast metrics (Technical)
Compare naive with forecast (Units_sold vs prediction_units_sold)

MAE = mean(|y - ŷ|)
RMSE = sqrt(mean((y - ŷ)²))
SMAPE = 2 × mean(|y - ŷ| / (|y| + |ŷ|))
WAPE = sum(|y - ŷ|) / sum(y)

Models must generate forecasts for every day up to horizon 14 (multi-step daily forecast),
but we will evaluate performance specifically at j+1, j+7, and j+14.

Use time-series validation: walk-forward (expanding window). 
Example: 5 folds; at each fold expand training window and test on next period. Evaluate at horizons j+1, j+7, j+14.

Use direct multi-step forecasting: train separate model (or separate head) per horizon (j+1, j+7, j+14). 
Evaluate each independently.

Perform a check on dataset structure before fitting: verify X_train / y_train shapes and ensure there are no null values. 
Print summary of columns and datatypes to avoid errors.

#Forecast business (supply chain)

a) fill_rate --> (Higher fill_rate is, the better the customer satisfaction)
fill_rate =  served_demand / total_demand
served_demand = min('Units_Sold', 'Inventory_Level')
total_demand = 'Units_sold'

b) stockout_rate --> % of days with stockout 
stockout rate = (# days with unmet demand) / total number of days
days_with_stockout = 'Inventory_Level' < 'Units_sold' or 'Stockout_Flag'=1

c) holding_cost (€) = sum(Inventory_Level_day * unit_storage_cost_sku) over all days
annual_storage_rate = 0.2 ( We assume a cost of 20% of storage)
unit_storage_cost_sku = Unit_Cost_sku * (annual_storage_rate / 365)
holding_cost_per_day = Inventory_Level_day × unit_storage_cost
total_holding_cost = sum(holding_cost_per_day over all days of simulation)

d) stockout_cost = sum(unmet_demand_units_day * unit_stockout_cost_sku) over simulation horizon
unmet_demand_units_day = max(0, demand_day - onhand_after_fulfilment_day)
penalty_factor = 1.5 # multiplier applied to margin (e.g. 1.5 = 150% of margin)
unit_stockout_cost_sku = (Unit_Price - Unit_Cost)_sku * penalty_factor

e) total_cost = holding_cost + stockout_cost
Compare baseline (naive) and ML forecast 

Each KPI are calculated by SKU_Id/day, then aggregated
Tous les KPI sont calculés par SKU/jour, aggregated by weighting by demand (Unit_Sold)

## 7 Inventory Simulation with forecast

#Starting point logic 

a)z = target service level (Value depend on ABC classification) 
for SKU_ID in group A --> 1.28 (90 %)
for SKU_ID in group B --> 1.06 (85 %)
for SKU_ID in group C --> 0.84 (80 %)
safety_stock = z * σ_daily_demand * √Supplier_Lead_Time_Days

b) reorder_point(ROP) : 

If use_dataset_ROP = True → use dataset 'Reorder_Point' as authoritative.
Else → ROP = average_forecast × Supplier_Lead_Time_Days + safety_stock.
Parameter: use_dataset_ROP (default = False).

c)Loop for every day 

if onhand <= reorder_point --> initiate_order of reorder_quantity 
if 'Order_Quantity' from dataset = True --> use dataset 'Order_Quantity' as authoritative for reorder_quantity
Else --> reorder_quantity = 0.5 * mean('Order_Quantity) per 'SKU_ID'  
reception time = Supplier_Lead_Time_Days

Orders and receipts management:
- Allow multiple outstanding orders per SKU (FIFO queue).
- Track order queue with expected arrival dates and update inventory on arrival.
- Order quantity = dataset 'Order_Quantity' unless otherwise specified.

selling = min(onhand, daily_demand)
stockout = if daily_demand > onhand

# Simulating variability with monte carlo simulation 

Simulate 1000 random scenarios 

Random parameter : 
- daily demand = forecast ± gauss noise (N(μ,σ)
- supplier lead time = empiric distribution of 'Supplier_Lead_Time_Days'

Baseline simulation: lead_time = mean(Supplier_Lead_Time_Days) per supplier.
Monte Carlo simulation: sample lead_time from the empirical distribution of that supplier (round up to integer days)

Output --> Calculate distribution per # Forecast business KPI's ( fill_rate, stockout_rate, total_cost)

Scaling rule:
- If SKU_count × n_runs × horizon is too large → run full Monte Carlo on top-N SKUs by revenue (default N=100).
- Others aggregated via analytical approximation.
- Default n_mc = 200 for all SKUs; 1000 for top-N SKUs.

## 8 Business impact estimation (Forecast horizon j+7)

Objective --> Compare business value of ML modelling from Baseline data(naive)
Anwering the following questions : "Does the new forecast reduce costs and improve customer service?

a) Calculate technicals forecast metrics 

# Forecast metrics (Technical)
Compare naive with forecast (Units_sold from vs prediction_units_sold)

MAE  
RMSE 
SMAPE 

b) Add every forecast into inventory simulation ( 7 Inventory Simulation with forecast)

- Baseline (Naïve j-1, Naïve Season j-7, Moving Avg 7d, Dataset Forecast)
- Machine learning modelling 

c) compare results with # Forecast business KPI's

fill_rate 
stockout_rate 
holding_cost 
stockout_cost 
total_cost 

d) measure improvements from the ML modelling 

fill_rate (% improvements)
stockout_rate (% avoided)
total_cost (€ savings)

e) Aggregation method
Aggregate global KPIs by demand-weighting:
weight_SKU = total_units_sold_SKU / total_units_sold_all_SKUs
Global KPI = Σ (KPI_SKU × weight_SKU)

f) Ouput:

- Comparative table

Modèle	RMSE	SMAPE	Fill Rate	Stockout Rate	Holding Cost (€)	Stockout Cost (€)	Total Cost (€)       WAPE
Naïve	 …	 …	…	…	…	…	                                                  …
Moving Avg	…	…	…	…	…	                        …	                                          …
Dataset	…	…	…	…	…	…	                                                  …
ML Forecast	…	…	…	…	        …	                        …	                                          …

## 8 bis Business impact estimation (Forecast horizon j+7)

Objective --> Compare business value of ML modelling from Baseline data(naive) for horizon = 7 days ahead.
Answering: "Does the new forecast reduce costs and improve customer service at medium-term horizon?"

a) Forecast metrics (Technical, horizon j+7)
Compare naive vs ML → Units_sold[t] vs prediction_units_sold[t+7]

b) Inject j+7 forecasts into inventory simulation
- Baseline models (Naïve j-1, Naïve j-7, Moving Avg 7d, Dataset Forecast)
- Machine learning (7-day horizon)

c) Compute forecast business KPIs (fill_rate, stockout_rate, holding_cost, stockout_cost, total_cost)

d) Compare ML vs baselines
- fill_rate (% improvements)
- stockout_rate (% avoided)
- total_cost (€ savings)

e) Aggregation method (same weighting by SKU demand)

f) Output → Comparative table (horizon j+7)


## 9 Data export for visalisation 
After computing forecasts, simulations, and business KPIs, export the results in structured CSV/Excel files for external visualization (dashboard, BI tool).
"Export all outputs in a single Excel file, with one sheet per section (forecast, simulation, KPI comparison, segmentation)." via pd.ExcelWriter

Additionally:
- Export 'quick_check.csv' with first 100 rows of key outputs for quick manual validation.
- Generate 'validation_report.txt' listing imputations, missing values, and rows dropped.

### Modifications for SKU segmentation export
# Calculate coefficient of variation (CV) per SKU
sku_params['CV'] = sku_params['std_demand'] / sku_params['avg_demand']

# Combine ABC and XYZ class into a single column
sku_params['ABC_XYZ_Class'] = sku_params['ABC_Class'] + "_" + sku_params['XYZ_Class']

# Prepare DataFrame for export
sku_segmentation = sku_params[[
    'SKU_ID', 'ABC_Class', 'XYZ_Class', 'ABC_XYZ_Class',
    'avg_demand', 'std_demand', 'CV', 'avg_lead_time',
    'avg_inventory', 'total_units_sold'

a) Export forecast results (per SKU_ID, Date, with baselines, ML forecasts, and actuals).

b) Export simulation results (Monte Carlo KPIs per SKU_ID and per model).

c) Export aggregated KPI comparison (baseline vs ML).

d) Export SKU segmentation (ABC/XYZ classification).

Files to be saved: demand_forecasting_supply.xlsx
Ensure data is clean, standardized, and properly formatted for dashboard integration.

Export metadata sheet with key parameters:
- annual_storage_rate
- penalty_factor
- n_mc
- random_seed (default = 42)
- use_dataset_ROP
- validation config (walk-forward folds, horizons)
