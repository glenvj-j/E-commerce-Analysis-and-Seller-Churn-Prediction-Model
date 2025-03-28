# E-commerce Analysis and Seller Churn Prediction Model

![github cover](https://github.com/user-attachments/assets/e8184c52-bb85-4288-baea-1b02d612e9d8)


**Data Period:** September 2016 - October 2018  

**Contributors:** [Glen Valencius Joydi](https://github.com/glenvj-j) | [Khrysna Taruna Putra](https://github.com/krisnatp-gh) | [Daffa Dzaky Naufal](https://github.com/daffa-dzn)

  
> [!NOTE]
> Presentation Deck : [View the Google Slides Presentation](https://docs.google.com/presentation/d/1RXtLS-ZDupxqhABBnfjqadugFRhJX3UAFSvMGAnbRYs/edit?usp=sharing)<br>
> Dashboard : [View Tableau](https://public.tableau.com/app/profile/glen.joy2546/viz/OlistDashboard_17407870344560/Homepage?publish=yes)<br>
> Streamlit Deploy : [View here](https://olist-seller-churn-dashboard-team-alpha-jcds0508.streamlit.app/)
---

## 1. Business Problem

Olist generates revenue from two primary sources:  
- **Seller subscriptions**  
- **Commissions from sold items**  

To maximize commissions, we need to identify the best-performing products. Since Olist's products come from sellers, finding the top sellers is key to increasing commission-based revenue.

### **Our Approach**
We will focus on three main areas:
1. **Analyzing Olist’s Business Performance**  
2. **Understanding Buyer Preferences in Shopping on Olist**  
3. **Analyzing and modeling which sellers to retain and prioritize for churn prevention**  

## 2. Analysis

### **2.1. Objective: Maximizing Profit & Preventing Losses**  

Our primary goal is to **maximize profit** by identifying the best in each category:  
- 🛍️ **Best Products** – High-demand and high-profit items  
- 👤 **Best Buyers** – Loyal customers with high spending  
- 🏪 **Best Sellers** – Reliable sellers with strong performance  

Additionally, we aim to **prevent profit loss** by detecting and addressing seller churn. Future growth opportunities include identifying **promising cities for expansion** and **high-potential product categories** for investment.  

### **2.2. How Does Olist Generate Revenue?**

- **Commission on Sales:** 18% platform fee and 5% operating cost per item sold.
- **Subscription Fees:** $39 BRL monthly subscription.

### **2.3. Stakeholders**

#### 1. Olist (Company Perspective)  
- **Problem:** Identifying high-performing sellers who are at risk of churning.  
- **Impact:** Loss of revenue **commissions** if top sellers leave or stop selling.  

#### 2. Seller Perspective  
- **Problem:** High-performing sellers may churn if they don't get sales for a long time.  
- **Impact:** Losing **top sellers**, reducing product availability and sales volume.  

#### 3. Buyer Perspective  
- **Problem:** A decline in product variety on Olist Store makes the platform less attractive.  
- **Impact:** **Lower sales revenue** due to reduced customer engagement and purchases.

### **2.4. Analytical Approach**
We will separate our analysis into two major categories:
- **Descriptive Analytics**: Focuses on understanding business, seller, and buyer performance.
- **Predictive Analytics**: Focuses on modeling to predict churn for the next quarter based on seller performance from the last three months.

## 3. Data Overview

We have combined 8 CSV datasets into 3 focused datasets:

| **Original Dataset** | **New Dataset** |
| --- | --- |
| olist_customers_dataset.csv | business_side.csv |
| olist_geolocation_dataset.csv | buyer_side.csv |
| olist_order_items_dataset.csv | seller_side.csv |
| olist_order_payments_dataset.csv |  |
| olist_order_reviews_dataset.csv |  |
| olist_orders_dataset.csv |  |
| olist_products_dataset.csv |  |
| olist_sellers_dataset.csv |  |
| product_category_name_translation |  |

- The dataset represents **sales data from Olist**.  
- Each row in the dataset corresponds to **a transaction of a product** in Olist’s partner marketplace.

### **3.1. Attributes Information**

| **Attribute** | **Data Type** | **Description** |
| --- | --- | --- |
| customer_id | Object | Each order has a unique customer_id. (NOTE: Hash) |
| customer_unique_id | Object | Unique identifier of a customer. (NOTE: Hash) |
| customer_zip_code_prefix | Object | First five digits of customer ZIP code |
| customer_state | Object | Customer state |
| geolocation_zip_code_prefix | Object | First five digits of ZIP code |
| geolocation_lat | float64 | Latitude of ZIP area |
| geolocation_lng | float64 | Longitude of ZIP area |
| geolocation_city | Object | City of ZIP area |
| geolocation_state | Object | Province of ZIP area |
| product_id | Object | Represents a unique product |
| product_category_name_english | Object | Category of Product |
| price | float64 | Item price |
| order_id | Object | Represents a unique order, each order_id contains multiple product_id |
| order_item_id | int64 | The quantity of items and their order sequence in a single order |
| order_purchase_timestamp | datetime64[ns] | Purchase timestamp by buyer |
| avg_items_per_order | int64 | Number of products per transaction |
| net_profit | int64 | Net Profit (Price * 18%) |
| payment_type | Object | Customer’s chosen payment method |
| order_status | Object | Order status (delivered, shipped, etc.) |
| review_score | int64 | Customer satisfaction rating (1 to 5) |
| seller_id | Object | Represents a unique seller |
| seller_zip_code_prefix | Object | ZIP code of Seller |
| seller_city | Object | City of Seller |
| seller_state | Object | State of Seller |
| order_approved_at | datetime64[ns] | Payment approval timestamp |
| shipping_limit_date | datetime64[ns] | Deadline for seller to ship the item |
| order_delivered_carrier_date | datetime64[ns] | Timestamp when the order was handed to the logistics partner |
| order_delivered_customer_date | datetime64[ns] | Actual order delivery date |
| order_estimated_delivery_date | datetime64[ns] | Estimated delivery date provided at the time of purchase |
| time_deliver_to_carrier | int64 | Time taken for the seller to deliver the package to the logistics partner |

## 4. Data Cleaning Process

Here’s how we will clean the data:

| **Column** | **Cleaning Method** |
| --- | --- |
| Product Category | Fill missing values as "Unknown" |
| Review | Set missing values to 0 (indicating no reviews or not delivered yet) |
| Payment | Fill missing values with the most common method (Credit Card) |
| Order Approved At | Fill missing values using the median approval time per seller |
| Order Delivered Carrier Date | Fill missing values using the median delivery time per seller |
| Order Delivered Customer Date | Align missing values with the estimated delivery date |

## 5. Analysis

Here is the outline of the analysis :

### **5.1. Business Performance Analysis**
- Total Profit & Revenue Trends
- Seasonal Trends (Year-on-Year Analysis)
- Seller & Buyer Growth Trends
- Correlation Between Product Variety and Profit
- Pareto Analysis
- Black Friday Anticipation
- Delivery Status

### **5.2. Buyer Analysis**
- Best Selling Products & High-Profit Items
- Customer Loyalty & Spending Analysis
- Geographic Trends & Buyer Behavior
- Payment Type Preferences
- Customer Feedback & Sentiment Analysis

### **5.3. Seller Analysis**
- Geographic Analysis & Best-Selling Products by Region
- Service Quality & Delivery Performance
- Seller Issues & Risk Analysis
- Financial & Profitability Analysis
- Identifying Top Sellers to Retain

# 7. Modeling Introduction

## 7.1. Understanding Churn

For Olist's marketplace business model where revenue comes from sales commissions, churn is defined as a seller ceasing sales activity on the platform for an extended period. Specifically:

- Churn occurs when a seller has no sales activity in the quarter following their last active quarter
- The primary indicator of sales activity is a seller approving an order
- Churn is considered recurring, as sellers may become inactive due to temporary factors and later return

We selected quarterly measurement intervals rather than monthly to:
1. Reduce the impact of seasonal e-commerce fluctuations
2. Align with standard fiscal reporting periods
3. Provide more stable data for prediction models

## 7.2. Label and Churn Data Preparation

The churn dataset was structured with each row representing an active seller in a given quarter. To determine whether a seller churned, we checked if they remained active in the subsequent quarter. This approach created a clear binary classification target variable for our model.

![image](https://github.com/user-attachments/assets/fa529807-e333-4e97-92dd-fc82eb1e0b5c)

Following the churn labeling process, we engineered additional features to create a comprehensive dataset for modeling seller churn:

- **`year`** and **`quarter`**: Temporal identifiers indicating the specific period of seller activity
- **`seller_id`**: Unique identifier assigned to each seller on the platform
- **`is_churn`**: Binary target variable (1 = churned, 0 = retained) indicating whether the seller remained active in the following quarter

**Order fulfillment metrics:**
- **`n_orders`**: Total count of unique orders received during the quarter
- **`n_approved_orders`**: Count of orders that received seller approval for processing
- **`n_delivered_carrier`**: Count of orders successfully transferred to logistics carriers
- **`n_orders_late_to_carrier`**: Count of orders delivered to carriers after the scheduled timeframe
- **`n_delivered_customers`**: Count of orders successfully delivered to end customers
- **`n_orders_late_to_customer`**: Count of orders that reached customers after the promised delivery date

**Performance indicators:**
- **`sales`**: Gross quarterly revenue generated before platform commission (denominated in R$)
- **`median_approve_time`**: Median time interval between an order and seller approval (minutes)
- **`median_review_score`**: Median customer satisfaction rating on a scale of 0-5

**Seller characteristics:**
- **`tenure`**: Cumulative months the seller has been active on the platform
- **`n_months_active_quarter`**: Number of months with activity within the analyzed quarter (range: 1-3)
- **`last_month_active_quarter`**: Final month of activity within the quarter (values: 1, 2, or 3)
- **`seller_city`** and **`seller_state`**: Geographical location identifiers for the seller



## 7.3. Outlier Handling

Our analysis revealed significant right skew in several metrics (`n_orders`, `n_approved_orders`, `n_delivered_carrier`, etc.) which is common in e-commerce where high-performing sellers represent a small percentage of the overall population. These patterns represent meaningful business segments rather than errors, so we retained these outliers for modeling.

For `median_review_score`, we observed left skew, likely correlating with unprofitable sellers. Only for `median_approve_time` did we remove one extreme outlier that appeared to be an anomalous data point rather than a meaningful business pattern.

## 7.4. Seller Segmentation

We segmented sellers into **profitable** and **unprofitable** categories based on their sales performance. This segmentation is critical as our analysis determined that unprofitable sellers should not be prioritized for churn intervention efforts. Within the profitable segment, we further classified sellers as **Top** or **Regular** based on sales volume.

# 8. Cost Evaluation

We developed a comprehensive cost structure to evaluate our churn prediction model, considering different costs for Top and Regular sellers:

| Cost Type | Description | Top Sellers | Regular Sellers |
|-----------|-------------|------------|----------------|
| False Negative | Lost commission when churning seller is misclassified | R$625.0 | R$86.0 |
| False Positive | Intervention costs for non-churning sellers | R$150.0 | R$10.0 |
| True Positive | Net benefit of successful intervention | -R$37.0 | -R$3.0 |

For Top Sellers, interventions include a 2% commission reduction, CRM support (R$36), marketing assistance (R$40), and vouchers (R$5), with an estimated 30% retention success rate. Regular Sellers receive a 1% commission reduction and vouchers, with an estimated 15% retention success rate.

# 9. Modeling and Results

We implemented an XGBoost classification model to predict seller churn, focusing on **Recall** as our primary metric due to the higher cost of false negatives compared to false positives. Key results include:

- Model recall improved from a baseline of 0.5269 to 0.9132 through hyperparameter tuning
- Robust performance on out-of-time data with an average recall of 0.92 across all prediction windows
- Solid recall for Regular sellers (0.8668) with some false positive errors
- Excellent performance for Top sellers with both high recall (0.9259) and accuracy (0.9091)

The optimized model delivers significant cost efficiencies:
- 3.6× cheaper than the base model (R$6,204 vs R$22,214)
- 4.9× savings compared to no intervention (R$6,204 vs R$30,365)
- 5.9× more cost-effective than blanket intervention for all top sellers (R$6,204 vs R$36,761)

This demonstrates that our targeted approach to seller churn prediction and intervention can substantially reduce costs while effectively retaining valuable sellers.

## 10. Conclusion & Recommendation

### 10.1. Analysis
#### 10.1.1. Conclusion

##### **1. Olist’s Business Performance Analysis**  

- **Revenue Growth:** From 2017 to early 2018, Olist’s revenue increased by **21%**, despite the year not being complete.  
- **Black Friday Effect:** Sales spiked significantly in **November 2017**, likely due to **Black Friday promotions**.  
- **Seller & Buyer Growth:**  
  - **Sellers** tend to join more at the end and start of the year.  
  - **Buyers** also peak around **November** and at the **beginning of the year**.  
- **Product Variety & Profitability:**  
  - Increasing the variety of products leads to higher profits.  
  - **Pareto Analysis:** By focusing on just **16 key categories**, we can cover **80% of net profit**.
  
   These categories include:  

    | health_beauty | 'watches_gifts' | 'bed_bath_table' | 'sports_leisure' | 'computers_accessories' | 
    | --- | --- | --- | --- | --- |
    |'furniture_decor' | 'housewares' | 'cool_stuff' | 'auto' | 'toys' | 
    | 'garden_tools' | 'baby' | 'perfumery' | 'telephony' | 'office_furniture' | 
    | 'stationery' | 'computers'`.  

**Black Friday Strategy:**  

We can prioritize different product categories based on our business goals. Whether we aim to maximize overall revenue, focus on high net profit per order, or explore hidden opportunities, the table below provides clear guidance:  

| **Focus on High-Profit Categories** | **Highest Net Profit per Order** | **Hidden Opportunities (Low Orders, High Profit)** |
|---|---|---|
| bed_bath_table                       | computers                        | computers                                      |
| furniture_decor                      | agro_industry_and_commerce       | agro_industry_and_commerce                     |
| sports_leisure                        | musical_instruments              | musical_instruments                            |
| health_beauty                         | home_appliances_2                | home_appliances_2                              |
| garden_tools                          | industry_commerce_and_business   | industry_commerce_and_business                 |
| computers_accessories                 | furniture_bedroom                | small_appliances                               |
| toys                                  | party_supplies                   | construction_tools_safety                      |
| watches_gifts                         | small_appliances                 | construction_tools_lights                      |
| housewares                            | cine_photo                       | air_conditioning                               |
| telephony                             | construction_tools_safety        | watches_gifts                                  |


- **Order Cancellations:** Olist’s **cancellation rate is below 1%**, which is considered acceptable.  


---

##### **2. Buyer Preferences in Shopping on Olist**  

- **Basket Size & Average Order Size:**  
  - Most buyers purchase **only 1 item per order**.  
  - **Solution:** Implement **bundling & cross-selling** strategies to encourage multiple purchases.
  - Average order **value 96 BR.**
  - **Solution:** Create a coupon with a minimum order requirement
   **Loyal & High-Spending Customers:**  
  - By combining **loyalty (repeat purchases)** and **profitability (high spenders)**, we can target high-value customers for **personalized promotions and loyalty rewards**.  
- **Average Buyer Spending:**  
  - The **median spending per buyer** is **around 90**.  
- **Best Cities for Expansion:**  
  - Cities with **high median profit** should be prioritized for expansion.  
- **Payment Preferences:**  
  - **Credit cards** are the most preferred payment method.  
  - **Opportunity:** Partnering with credit card companies for promotional discounts.  
- **Customer Sentiment & Ratings:**  
  - **13% of reviews** gave a rating of **1 star**, and **3% gave a 2-star rating**.  
  - The main complaints were **product quality issues** and **delivery problems**.  

---

##### **3. Seller Retention & Churn Prevention Strategy**  

- **Best Cities for Expansion:**  
  - **Filtering cities** where **total orders are above median** but **total sellers are below median** helps identify areas with **high demand but few sellers**.  
  - **Benefit:** Faster deliveries & improved customer satisfaction.  
- **Product Category Optimization for Expansion:**  
  - By identifying the **most ordered product categories** in potential expansion areas, we can **attract new sellers** to fulfill demand.  
- **Seller Performance Evaluation:**  
  - By analyzing **buyer ratings**, we can identify **underperforming sellers** for evaluation and guidance.  
  - Sellers with **high late deliveries & cancellations** should be prioritized for **further investigation**.  
- **Seller Profitability Analysis:**  
  - **Strategy 1:** Focus on sellers with **high net profit per order** and at least **2+ orders**, ensuring long-term profitability.  
  - **Strategy 2:** Focus on sellers with **one-time high-profit orders**, analyzing if their product category is **niche or repeatable**.  
  - **Next Step:** Decide on the **long-term seller retention strategy** based on these insights.  

#### 10.1.2. Recommendation
1. **Prepare for Black Friday (November 2018)**  
   - Focus on high-performing product categories listed in the previous analysis to maximize sales potential.  

2. **Increase Bucket Size**  
   - Encourage larger purchases by offering **bundled deals** or **discounted add-ons**.  

3. **Expand Seller Network in High-Demand Areas**  
   - Identify regions with **high order volume but few sellers** and recruit new sellers to improve delivery speed and customer satisfaction.  

4. **Collaborate with Credit Card Companies**  
   - Partner with banks to offer **exclusive promotions** for credit card users, incentivizing more purchases.  

5. **Support Low-Rated Sellers**  
   - Provide **training or guidance** to sellers receiving consistently low ratings to improve product quality and service. 

### 10.2. Modeling

#### 10.2.1. Conclusion

- We segmented sellers into **profitable** and **unprofitable** categories, determining that **unprofitable sellers should not be prioritized for churn intervention efforts**.

- Among profitable sellers, we further classified them as **Top** or **Regular** sellers. Top sellers are defined as those in the highest 20% of total sales (measured over the previous year) among all active sellers in a given quarter.

- We developed a classification model using XGBoost to predict seller churn, using recall as our primary optimization metric due to the high cost of false negatives. **Through tuning, we improved the model's recall from a baseline of `0.5269` to `0.9132`.**

- The tuned model demonstrates **robust performance on out-of-time data, achieving an average recall of 0.92 across all prediction windows**.

- The optimized model achieves **high recall for both Regular (`0.8688`) and Top sellers (`0.9259`)**, though Regular sellers exhibits more false positive errors.

- The model **performs exceptionally well for Top sellers, with both high recall (`0.9259`) and high accuracy (`0.9091`)**, making it particularly effective at predicting churn within this valuable segment.

- **Our tuned model delivers significant cost efficiencies across multiple comparison scenarios**: it reduces total costs from R$22,214 to R$6,204 compared to the base model (**3.6x cheaper**), outperforms using no intervention (R$30,635 vs R$6,204, representing **4.9x savings**), and proves far more cost-effective than blanket intervention for all top sellers (R$36,761 vs R$6,204, or **5.9x cheaper**).


#### 10.2.1. Recommendations

- Deploy a tiered intervention strategy, prioritizing top sellers with our model, as its performance suggests targeting will be applied effectively.

- Consider reducing incentives for regular sellers below initial cost assumptions, given the model's higher false positive rate in this segment.

- Further segment regular sellers to address the notable false positive errors within this group. Develop tailored targeting interventions for each new segment.

- Direct account managers using the model to focus on the three highest-impact features: the last active month in a quarter, delivered orders per active month in that quarter, and total active months in that quarter.

- Initiate cross-departmental discussions on retention strategies for priority sellers.

- Validate the model quarterly and re-train with new data to maintain model quality.

- Conduct economic evaluations of the model's performance after retention campaign to measure ROI and business impact.


---


## 11. Deployment

We deployed this model using Streamlit to assist Account Managers (B2B Seller Relations) in identifying and prioritizing sellers at risk of churn.
By uploading quarterly data, the model predicts whether a seller is likely to churn or not. <br> The results can be downloaded based on priority levels, highlighting **Top Priority Sellers** (high-value sellers at risk of churn) and **Standard Priority Sellers** (moderate-value sellers at risk of churn).

Streamlit Deploy : [View here](https://olist-seller-churn-dashboard-team-alpha-jcds0508.streamlit.app/)

---

Resource :
[Brazil Shapefiles](https://www.kaggle.com/datasets/rodsaldanha/brazilianstatesshapefiles)
