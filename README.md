# Predicting Restaurant Success in Hawaii
**Name**: Justin James

## Introduction

This project explores the "Hawaii Google Maps Reviews" dataset, which contains rich information on business attributes and customer sentiment across the Hawaiian islands. The dataset consists of two main tables: a metadata table containing 21,507 unique businesses, and a reviews table containing 1,504,347 individual customer reviews. 

**Research Question:** Positioning myself as a prospective entrepreneur, what factors make a business successful in Hawaii, and where is the optimal location to open a new restaurant? Initially, my project was centered around finding the "Goldilocks Zone": a location where nearby competitor restaurants are rated "medium" on average, allowing a new business to stand out without competing against heavily saturated areas or settling in naturally poor locations. However, as the data science lifecycle often dictates, my data exploration forced me to pivot and look at a broader combination of operational decisions and geographic clustering.

**Why It Matters:** Hawaii has a unique, heavily tourist-driven economy intertwined with local communities. By predicting the viability of a business based on operational features and geographic market saturation, entrepreneurs can maximize their chances of success before investing capital.

**Relevant Columns:**
To answer this question, my analysis relies primarily on the metadata dataset, focusing on the following pre-opening columns:
* **`name`**: The name of the business to identify specific establishments.
* **`category`**: The category tags used to isolate the restaurant market.
* **`latitude` & `longitude`**: The geographic coordinates used for plotting spatial distributions and clustering.
* **`avg_rating`**: The average rating of the business, serving as my primary metric for determining a business's success.
* **`price`**: The affordability tier (1.0 to 4.0), which dictates the target demographic.
* **`hours` & `MISC`**: Messy columns from which I engineered specific operational booleans (like weekend availability and delivery options).
* **`relative_results`**: A list of related businesses used to calculate local market saturation (my core "Goldilocks Zone" metric).

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
To ensure my predictive model strictly adhered to a "Time of Prediction" constraint (only using data an entrepreneur would know *before* opening), I had to clean and extract intentional business decisions from the messy raw data. 

1. **Filtering the Market**: I filtered the 21,507 businesses down to only those containing "restaurant" in their category, dropping entries with missing or invalid (0.0) coordinates. 
2. **Type Casting**: I converted object columns to strictly typed Pandas `string` objects and mapped the string `price` tiers ('$') to floats (1.0 - 4.0).
3. **Feature Engineering from Messy Data**: 
   * I exploded the nested `MISC` and `Service options` dictionaries to extract pre-opening operational decisions, engineering booleans like `has_delivery`.
   * I parsed the complex `hours` dictionary to create an `is_open_weekends` boolean (capturing crucial tourist traffic).
   * I counted the items in the `relative_results` list to create a `num_related_competitors` integer, capturing the immediate market saturation of the chosen location to test my Goldilocks theory.
4. **Final Cleanup**: I dropped the original, un-parsable object columns to save memory. My final cleaned `restaurants_df` contains 4,301 valid Hawaiian restaurants.

**Cleaned DataFrame Head:**

<div style="overflow-x: auto;">

| name                     | category                         |   avg_rating |   price | has_delivery   | is_open_weekends   |   num_related_competitors |
|:-------------------------|:---------------------------------|-------------:|--------:|:---------------|:-------------------|--------------------------:|
| Hale Pops                | Restaurant                       |          4.4 |     nan | True           | True               |                         5 |
| Akasatana Ramen Kyoto    | Ramen restaurant                 |          5   |     nan | True           | True               |                         2 |
| Grill City               | Restaurant                       |          3.5 |     nan | True           | True               |                         5 |
| Buona Sera               | Italian restaurant, Restaurant   |          3.7 |       2 | True           | True               |                         2 |
| Tucker & Bevvy Breakfast | Fast food restaurant, Restaurant |          4.2 |     nan | True           | True               |                         4 |

</div>

### Univariate Analysis

To understand the baseline landscape of Hawaii's restaurant industry, I first looked at the distributions of my two most important individual variables: average ratings and market saturation (competitors).

**Distribution of Average Ratings**

<iframe src="assets/rating_dist.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

The distribution of average ratings is heavily left-skewed, with a massive spike between 4.5 and 5.0. This indicates that the vast majority of restaurants in Hawaii are highly rated by customers. For a prospective entrepreneur, this reveals that the baseline for an "average" restaurant is exceptionally high; simply having a rating of 4.0 might actually place a business in the bottom tier compared to its peers.

**Distribution of Market Saturation**

<iframe src="assets/competitor_dist.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

The distribution of nearby related competitors shows that a typical restaurant in Hawaii operates in close proximity to about 4 to 5 direct competitors. While there are a few isolated businesses with very few competitors, and a long right tail of highly saturated hotspots with 10 or more competitors, the bulk of the market clusters around this 4-5 competitor mark. This visual validates my search for a "Goldilocks Zone"—aiming for areas that demonstrate proven foot traffic and market viability (around the 4-5 competitor average) without crossing into the brutal, high-competition tail of the distribution.

### Bivariate Analysis

Now we will explore how different variables interact with a restaurant's success, specifically looking at geography and market saturation.

**1. Geographical Distribution vs. Rating**

<iframe src="assets/map_distribution.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

This map visualization illustrates the geographical distribution of restaurants across the Hawaiian islands, color-coded by their average rating. By zooming into specific islands, we can observe distinct, dense clusters in major commercial and tourist hubs (like Honolulu), allowing us to visually identify where high-performing and low-performing businesses group together.

**2. Competitor Saturation vs. Rating**

<iframe src="assets/competitor_scatter.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

This scatter plot directly tests my "Goldilocks Zone" hypothesis by plotting the number of nearby competitors against a restaurant's average rating. The OLS trendline reveals a slight negative correlation; as competitor saturation increases, the average rating tends to dip slightly, suggesting that hyper-saturated areas make it marginally harder to maintain top-tier customer satisfaction. Additionally, the variance in ratings is highest in low-competition areas, meaning being isolated is a gamble that could result in either a perfect 5.0 or a devastating 1.0 rating.

### Interesting Aggregates

To further understand how operational decisions impact a restaurant's success, I aggregated the data to look at the relationships between price tiers, weekend availability, foot traffic, and market saturation. 

**Mean Average Rating by Price Tier and Weekend Availability**

|   price |   False |   True |
|--------:|--------:|-------:|
|       1 |    3.68 |   4.09 |
|       2 |    4.08 |   4.25 |
|       3 |    4.39 |   4.27 |
|       4 |    4.4  |   4.49 |


**Significance:** This pivot table reveals a fascinating dynamic regarding operational hours. While average ratings generally scale upward with higher price tiers (likely due to higher quality of food and service), simply choosing to stay open on weekends has an incredibly marginal impact on a restaurant's average rating. Across almost all price tiers, being open on weekends only bumps the average rating by 0.01 to 0.03 points. Interestingly, at the highest price tier (4.0), opening on weekends actually results in a slight *decrease* in average ratings, perhaps suggesting that the sheer volume of weekend tourist traffic puts a strain on the highly curated experience expected at luxury establishments.

**Average Foot Traffic & Competition by Price Tier**

|   price |   num_of_reviews |   num_related_competitors |
|--------:|-----------------:|--------------------------:|
|       1 |           249.36 |                      4.45 |
|       2 |           469.74 |                      4.2  |
|       3 |           485.75 |                      3.98 |
|       4 |           746.92 |                      4.3  |

**Significance:** This groupby table is crucial for a prospective entrepreneur looking for the "Goldilocks Zone". It shows that foot traffic (measured by the number of reviews) and competitor saturation peak at the 3.0 ($$$) price tier. The cheapest restaurants (1.0) have the lowest review volume and lowest competition, while the most expensive (4.0) see a slight drop-off in both metrics, reflecting their exclusivity. For a new business owner, this indicates that opening a tier 3.0 restaurant means entering the most hyper-active, saturated market segment, whereas aiming for a 2.0 or 4.0 tier might offer a slightly more balanced environment.

---

## Assessment of Missingness

### NMAR Analysis

When evaluating the missingness in my dataset, I believe the `price` column is **NMAR** (Not Missing At Random).

**Reasoning:**
Data is NMAR when the likelihood of it being missing depends on the actual, unobserved value itself. In the context of Google Maps restaurant listings, the data-generating process relies on business owners voluntarily providing their price tier (e.g., $, $$, $$$). It is highly likely that extremely expensive, exclusive fine-dining restaurants intentionally omit their pricing information online (subscribing to the adage, "if you have to ask, you can't afford it"). Conversely, very cheap "hole-in-the-wall" local spots might lack the organized online presence required to update their Google business profiles. In both cases, the missingness of the price is directly caused by the actual price tier of the restaurant itself.

**Making it MAR:**
To explain this missingness and make the `price` column MAR (Missing At Random), I would need to obtain additional data about the restaurant's characteristics that correlate with these extremes. For example, if I could obtain a dataset containing the **"Michelin Star Status"** or a **"Fine-Dining Classification"** for these restaurants, I could likely explain the missingness. If the data showed that the probability of the `price` being missing heavily depended on whether the restaurant was classified as a luxury fine-dining establishment, the missingness would then be dependent on that observed category, shifting it from NMAR to MAR.

### Missingness Dependency

To better understand the data-generating process, I investigated the missingness in the `price` column using two separate permutation tests. My goal was to determine if the missingness is Dependent (MAR) on other specific columns, or completely random.

**Test 1: Price Missingness vs. Average Rating (MAR)**

I first tested if a restaurant's average rating dictates whether or not it lists a price. 
* **Null Hypothesis:** The missingness of `price` does not depend on `avg_rating` (MCAR).
* **Alternative Hypothesis:** The missingness of `price` does depend on `avg_rating` (MAR).
* **Test Statistic:** Absolute difference in mean `avg_rating` between restaurants with missing prices and non-missing prices.
* **Significance Level:** 0.05

<iframe src="assets/missing_price_dist.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

**Interpretation:** The permutation test yielded a p-value of 0.0. As seen in the plot above, our observed difference in means (the red dashed line) falls drastically far outside the empirical distribution of our shuffled statistics. Therefore, I reject the null hypothesis. The missingness of the `price` column strongly depends on the restaurant's average rating.

---

**Test 2: Price Missingness vs. Random ID Hash (MCAR)**

To verify my methodology and prove that the missingness doesn't just arbitrarily depend on *everything*, I ran a second permutation test against a completely random feature: whether the last character of the restaurant's internal Google Maps ID (`gmap_id`) is a digit.
* **Null Hypothesis:** The missingness of `price` does not depend on whether the last character of the `gmap_id` is a digit.
* **Alternative Hypothesis:** The missingness of `price` does depend on whether the last character of the `gmap_id` is a digit.
* **Test Statistic:** Total Variation Distance (TVD) between the distributions of the `gmap_id` ending type for missing vs. non-missing prices.
* **Significance Level:** 0.05

**Interpretation:** This permutation test yielded a high p-value of 0.604 (well above the 0.05 threshold). We fail to reject the null hypothesis. As expected, the missingness of a restaurant's price has absolutely no dependency on the random, internal hash generated by Google Maps' servers.

---

## Hypothesis Testing

**Test 1: The Goldilocks Zone (Geography)**

My core research question initially asked if there is a "Goldilocks Zone" for opening a new restaurant—a market that isn't completely barren, but also isn't oversaturated with competitors. Using the `num_related_competitors` feature, I defined this zone as having moderate competition (2 to 4 nearby related competitors). I wanted to test if restaurants in this zone perform significantly better.

* **Null Hypothesis:** The average rating of restaurants in the Goldilocks Zone is the same as the average rating of restaurants outside of it. Any observed difference is due to random chance.
* **Alternative Hypothesis:** Restaurants in the Goldilocks Zone have a strictly higher average rating than restaurants outside of it.
* **Test Statistic:** Signed Difference in Means (Goldilocks Mean Rating - Other Mean Rating). I chose the *signed* difference rather than the absolute difference because my alternative hypothesis is directional; I specifically want to know if the Goldilocks mean is *greater*.
* **Significance Level:** 0.05

<iframe src="assets/hypo1_goldilocks.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

**Conclusion:** * **Observed Difference:** -0.0069
* **P-Value:** 0.696

With a p-value of 0.696, which is vastly larger than my significance level of 0.05, I **fail to reject the null hypothesis**. The data suggests that placing a restaurant in a moderately saturated area does not reliably yield a higher average rating. The observed difference was actually slightly negative, meaning we do not have sufficient evidence to support the "Goldilocks Zone" theory geographically.

---

**Test 2: Operational Success (Weekend Availability)**

Since geographic saturation alone did not appear to be a strong indicator of success, I pivoted to test an operational feature: weekend availability. I wanted to see if being open on the weekends (Saturday or Sunday) correlates with a significantly different average rating compared to restaurants that only operate on weekdays.

* **Null Hypothesis:** There is no difference in the mean average rating between restaurants that are open on weekends and those that are closed on weekends. Any observed difference is due to random chance.
* **Alternative Hypothesis:** There is a significant difference in the mean average rating between restaurants open on weekends and those that are closed on weekends.
* **Test Statistic:** Absolute Difference in Means. I chose the absolute difference because my alternative hypothesis simply looks for *any* significant difference (whether higher or lower) between the two groups. 
* **Significance Level:** 0.05

<iframe src="assets/hypo2_weekends.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

**Conclusion:** * **Observed Absolute Difference:** 0.1083
* **P-Value:** 0.0

With a p-value of 0.0, which is strictly less than my significance level of 0.05, I **reject the null hypothesis**. The data strongly suggests there is a significant difference in the average ratings between restaurants that open on weekends and those that do not. While this does not absolutely prove causation, it provides highly compelling evidence that operational decisions (like weekend hours) play a crucial role in a restaurant's success, making this an excellent feature to utilize in my predictive modeling.

---

## Framing a Prediction Problem

**The Prediction Problem**
Building on my exploratory analysis and hypothesis testing, my goal is to predict the overall success of a new restaurant in Hawaii based on its operational decisions and geographical location. 

**Type of Problem & Response Variable**
This is a **regression** problem. My response variable is a restaurant's **`avg_rating`** (a continuous quantitative value ranging from 1.0 to 5.0 stars). I chose this variable because a restaurant's average Google Maps rating is the most direct, quantifiable proxy for customer satisfaction, overall success, and long-term viability in a tourist-heavy economy like Hawaii. Prospective entrepreneurs rely heavily on simulating this exact metric when deciding whether a new business venture will be viable.

**Evaluation Metric**
To evaluate my model, I will use **Root Mean Squared Error (RMSE)**. I chose RMSE over Mean Absolute Error (MAE) or R-squared because RMSE is highly interpretable—it expresses the error margin in the exact same units as the target variable (stars). Furthermore, RMSE penalizes large prediction errors much more heavily than MAE. In the restaurant industry, a prediction being off by a full star is catastrophically worse for a business plan than being off by a fraction of a star, so I need a metric that aggressively penalizes being far off the mark.

**Time of Prediction Constraints**
To ensure this model is genuinely useful for a prospective entrepreneur, the "time of prediction" is strictly defined as the moment *before* the restaurant opens. Therefore, the model will only be trained using features that are known during the planning phase:
* **Geographic Decisions:** Exact planned location (`latitude`, `longitude`).
* **Market Saturation:** The number of competitors already operating in that immediate area (`num_related_competitors`).
* **Operational Decisions:** Planned hours (`is_open_weekends`), menu pricing tier (`price`), and planned amenities (such as offering delivery via `has_delivery`, or accepting credit cards via `accepts_credit_cards`).

To prevent data leakage, I am strictly excluding any post-opening customer interaction data. Features like the text of customer reviews or the total accumulated foot traffic (`num_of_reviews`) mathematically do not exist before a restaurant opens, and using them would invalidate the realistic predictive power of the model.

---

## Baseline Model

**Model Description & Features**
For my baseline model, I built a **Linear Regression** to predict a restaurant's overall success, quantified by its `avg_rating`. To strictly adhere to my "time of prediction" constraint (only using data known before a restaurant opens), my baseline model currently uses exactly two pre-opening operational features:
1. **`has_delivery` (Nominal):** A boolean feature indicating whether the restaurant plans to offer delivery services.
2. **`is_open_weekends` (Nominal):** A boolean feature indicating whether the restaurant operates on Saturdays or Sundays.

In total, my baseline model utilizes **0 quantitative features, 0 ordinal features, and 2 nominal features**. 

**Encodings**
Because both of my features are categorical (nominal), I transformed them using an `sklearn` `OneHotEncoder`. I specifically set `drop='first'` to avoid multicollinearity issues between the generated columns. This encoding step was grouped inside a `ColumnTransformer` and passed directly into a single `sklearn` `Pipeline` alongside the `LinearRegression` estimator.

**Model Performance**
To evaluate my model's ability to generalize to unseen data, I performed a 75/25 train-test split. After training the pipeline on the training set, I evaluated it on the unseen testing set:
* **RMSE:** 0.4675 stars
* **R-squared:** 0.0042

**Is this a "good" model?**
Currently, I **do not** believe this is a "good" model. While an RMSE of 0.4675 might not sound massive on a 1-to-5 star scale, the R-squared value of 0.0042 is incredibly poor. It reveals that this model is explaining less than 0.5% of the variance in restaurant ratings. A model relying solely on basic operational decisions—like delivery and weekend hours—is far too simplistic to capture the complex nuances of Hawaii's dining economy. It completely ignores a restaurant's geographic location, target demographic (price), and competitor saturation. I will introduce these complex, engineered features in my Final Model to capture these missing relationships and improve the predictive accuracy.

---

## Final Model

**New Engineered Features**
To improve upon my baseline model and capture the complex realities of the Hawaiian restaurant economy, I engineered several new features. Crucially, I maintained my strict "time of prediction" constraint by only using data a business owner would know prior to opening day. 

Alongside my operational booleans (`is_open_weekends`, `has_delivery`, and `accepts_credit_cards`), I added:
1. **Market Saturation (`num_related_competitors`)**: Standardized using a `StandardScaler`. Based on the data generating process, a restaurant's success heavily depends on local foot traffic. High competition indicates a proven market but splits the customer base, while zero competition might mean the location is naturally unviable.
2. **Target Demographic (`price`)**: I used a `SimpleImputer` (median strategy) to handle missing values, followed by a `StandardScaler`. In the real world, customer expectations scale dramatically with price; a $ restaurant is judged on entirely different criteria than a $$$$ fine-dining establishment. 
3. **Geographic Clustering (`KMeans`)**: Instead of passing raw, abstract `latitude` and `longitude` coordinates into the model, I built a `KMeans` clusterer into my pipeline. This mathematically transforms grid coordinates into distance metrics from major Hawaiian "hotspots". This aligns perfectly with the geographic reality of Hawaii, where a restaurant located in the dense tourist hub of Waikiki operates under completely different economic forces than a rural spot in Maui.

**Algorithm & Hyperparameter Selection**
Because geographic location and non-linear feature interactions are highly influential in this dataset, I tested two algorithms: a `RandomForestRegressor` and a `KNeighborsRegressor`. To ensure the best performance and avoid overfitting, I utilized `GridSearchCV` with 5-fold cross-validation, optimizing for the lowest Root Mean Squared Error (RMSE) on unseen test data. 

The **`RandomForestRegressor`** was the official winner. The Grid Search identified the following optimal hyperparameters:
* **`n_clusters`: 5** (The KMeans algorithm found 5 optimal distinct geographic sub-markets/tourist hubs across the islands).
* **`max_depth`: 10** (Deep enough to learn complex interactions between price and location, but shallow enough to prevent the tree from memorizing specific restaurants).
* **`min_samples_split`: 10** (Requires at least 10 restaurants in a node before splitting, ensuring the model makes generalizable rules rather than hyper-specific assumptions).

**Model Performance & Baseline Comparison**

<iframe src="assets/feature_importance.html" width="800" height="450" frameborder="0" class="plotly-graph"></iframe>

My Final Model represents a massive improvement over the Baseline Model:
* **Baseline RMSE:** 0.4675 stars  ->  **Final RMSE:** 0.4369 stars
* **Baseline R-squared:** 0.0042 -> **Final R-squared:** 0.1302

By incorporating geography, market saturation, and target demographics, **the model's error dropped**, and its **variance explained (R-squared) skyrocketed from 0.4% to over 13%**. As seen in the Feature Importances plot above, the geographic cluster distance and the `price` tier are overwhelmingly the most important predictors of a restaurant's success. While predicting highly subjective human opinions (star ratings) will always carry inherent noise and variance, this 30x improvement in R-squared proves that an entrepreneur's pre-opening geographic and operational decisions directly and significantly impact their future rating.

---

## Fairness Analysis

**The Fairness Question:**
Earlier in my project, my hypothesis testing revealed that a restaurant's weekend availability (`is_open_weekends`) was a statistically significant operational feature. But does my Final Model treat these two groups fairly? I want to evaluate if my model is unfairly biased (less accurate) when predicting ratings for weekday-only restaurants compared to those open on the weekends.

**The Groups:**
* **Group X (Weekend Operators):** Restaurants that are open on Saturdays or Sundays (`is_open_weekends = True`).
* **Group Y (Weekday-Only Operators):** Restaurants that are closed all weekend (`is_open_weekends = False`).

**Evaluation Metric:**
Because my Final Model is a regression model, I cannot use classification parity metrics like precision or accuracy. Instead, I am using **RMSE parity** to evaluate fairness.

**Hypotheses & Test Setup:**
* **Null Hypothesis ($H_0$):** My model is fair. The RMSE for predicting ratings of weekend restaurants is roughly the same as the RMSE for weekday-only restaurants. Any observed difference is due to random chance.
* **Alternative Hypothesis ($H_1$):** My model is unfair. The RMSE for weekday-only restaurants is significantly *higher* (worse) than the RMSE for weekend restaurants.
* **Test Statistic:** Difference in RMSE (Weekday-Only RMSE - Weekend RMSE).
* **Significance Level:** $\alpha = 0.05$

To test this, I performed a permutation test with 500 iterations, shuffling the `is_open_weekends` labels on my unseen test data to create an empirical distribution of the RMSE differences under the null hypothesis.

<iframe src="assets/fairness_test.html" width="800" height="600" frameborder="0" class="plotly-graph"></iframe>

**Conclusion: Interpreting the Fairness Analysis**

* **Observed Difference:** 0.1967 stars
* **P-Value:** 0.0360

**The Statistical Verdict:**
With a p-value of 0.0360 (which strictly falls below our $\alpha$ threshold of 0.05), we **reject the null hypothesis**. The permutation test strongly suggests that this model does *not* achieve RMSE parity. Mathematically, the model is significantly worse at predicting the ratings of weekday-only restaurants (RMSE: 0.6216) compared to those open on weekends (RMSE: 0.4249).

**The Real-World Business Story:**
Why is our model "unfairly" inaccurate for weekday-only spots? It comes down to human behavior. In a tourist-heavy economy like Hawaii, restaurants open on the weekends are likely catering to vacationers. Tourist behavior is highly geographic and heavily influenced by proximity to hubs—a pattern our K-Means clustering captured beautifully. 

On the flip side, a restaurant that can survive while being *closed* every weekend is likely a highly specialized, hyper-local business (such as a lunch spot catering exclusively to downtown Honolulu office workers). The success of these weekday-only spots relies on complex local dynamics, community word-of-mouth, and specific niche markets that simple pre-opening features just cannot capture.

**Final Project Takeaway:**
This "unfairness" is actually the perfect conclusion to this predictive project. It illustrates that while machine learning can help entrepreneurs find a geographic "Goldilocks Zone" and evaluate baseline market saturation, it cannot fully simulate the human element of the restaurant industry, especially for niche, local businesses. An algorithm can help an entrepreneur find the perfect location, but ultimately, the food, service, and community connection will dictate their true success.