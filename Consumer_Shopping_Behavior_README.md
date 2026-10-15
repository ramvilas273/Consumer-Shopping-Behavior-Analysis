# How Do People Shop? — Consumer Shopping Behavior Analysis

A data analysis and machine learning practice project exploring how people shop, what influences their purchases, and whether shoppers can be grouped into behavioral segments.

The project uses a consumer shopping behavior survey dataset and a Jupyter Notebook (`sc.ipynb`) to perform data cleaning, exploratory data analysis (EDA), clustering, and impulse-buyer prediction.

> **Important data limitation:** The notebook states that the 500 rows were generated from only 7 real answers. Treat the findings as a learning exercise, not as representative evidence about shoppers in general.

## Project Questions

1. What does the data look like, and what needs cleaning?
2. Who are the shoppers, and what are their shopping preferences?
3. Can shoppers be grouped into behavioral types?
4. Can we predict who is an impulse buyer?

## Dataset

**File:** `consumer_shopping_behavior_survey.csv`

The dataset includes fields covering:

- Respondent details: age, gender, occupation, country/region, and income range
- Shopping behavior: shopping channel, frequency, platform, and device
- Purchase preferences: spending category, payment method, and monthly non-essential spending
- Marketing influence: reviews, social ads, and likelihood of buying after an ad
- Shopping habits: impulse purchase frequency, return frequency, and shopping-list habits
- Attitudes: online shopping satisfaction, trust in reviews, and regret after purchase
- Other responses: reasons for shopping online or in-store and preferred shopping mode

## Project Workflow

### 1. Data Loading and Inspection

- Load the CSV with Pandas.
- Inspect rows, columns, data types, missing values, and duplicates.
- Check invalid age values, including non-numeric entries such as `test` or `xyz`.

### 2. Data Cleaning

- Convert age to numeric, coercing invalid values to missing.
- Parse timestamps using `%d/%m/%Y %H:%M:%S`.
- Standardize text labels, such as spending-category capitalization.
- Fill missing values in selected categorical columns with `Unknown`.
- Inspect missing responses in later survey questions to understand possible survey drop-off.

### 3. Exploratory Data Analysis (EDA)

The notebook explores:

- Age distribution and age groups
- Where respondents shop most often
- Preferred payment methods
- Top spending categories
- Online shopping satisfaction by age group
- Discount importance by shopping mode
- Shopping behavior by occupation and country/region
- Relationships between marketing influence, impulse buying, satisfaction, trust, and regret

Visualizations include histograms, count plots, bar charts, box plots, heatmaps, and grouped summaries.

### 4. Shopper Clustering

- Scale selected numeric features with `StandardScaler`.
- Apply K-Means clustering.
- Compare clustering results using silhouette scores.
- Use three clusters for a more useful exploratory description, while noting that the notebook reports low silhouette scores (approximately 0.15–0.20), indicating substantial overlap between groups.

The clusters should not be interpreted as sharply separated or definitive shopper types.

### 5. Impulse-Buyer Prediction

The notebook imports and uses machine learning tools including:

- `LogisticRegression`
- `RandomForestClassifier`
- `Pipeline`
- `SimpleImputer`
- `train_test_split`
- `accuracy_score`
- `roc_auc_score`

The objective is to explore whether survey features can help predict impulse-purchase behavior. Evaluate results carefully and avoid treating performance on this small, synthetic-style dataset as evidence of real-world predictive reliability.

## Key Observations in the Notebook

The notebook describes these exploratory findings:

- The median respondent age is about 22, with many respondents between 20 and 30.
- Around half of respondents shop through a mix of online and in-store channels; roughly one-third shop online only.
- Digital wallets are the most common payment method in the notebook's analysis.
- Clothing is reported as the largest top-spending category.
- Satisfaction differences across age groups appear small, with no clear age effect identified.
- Review influence, social-ad influence, and buying after an ad show positive correlations in the approximate range of 0.40–0.48.
- Impulse buying has a positive relationship with social-ad influence and purchase regret in the notebook's correlation analysis.
- The clustering silhouette scores are low, so the identified groups overlap considerably.

These are descriptive observations from the provided dataset and should not be generalized beyond it. Correlation does not establish causation.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
consumer-shopping-behavior/
├── sc.ipynb
├── consumer_shopping_behavior_survey.csv
└── README.md
```

## Installation

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run the Notebook

1. Place `sc.ipynb` and `consumer_shopping_behavior_survey.csv` in the same folder.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `sc.ipynb`.
4. Run the cells in order.

## Limitations

- The notebook reports 500 rows generated from only 7 real answers; the dataset is not suitable for reliable population-level conclusions.
- Missing responses and invalid entries may affect summaries and model results.
- Low silhouette scores indicate that clusters are not clearly separated.
- Predictive metrics may be unstable and should not be interpreted as production-ready performance.
- Associations between variables do not prove cause and effect.

## Objective

This project demonstrates a learning workflow for survey data: inspection, cleaning, visualization, feature scaling, clustering, and introductory classification.

## Author

**Ram Vilas**
