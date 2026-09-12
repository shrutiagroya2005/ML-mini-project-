ML Mini Projects
Two small, self-contained machine learning projects exploring regression regularization and market basket analysis — built and run in Google Colab.

Project 1: Regression Shrinkage Methods (Ridge vs Lasso)
Compares three regression approaches — plain Linear Regression, Ridge (L2), and Lasso (L1) — on a synthetic grocery spending dataset, to see how each handles noisy, irrelevant features.

What It Does
Generates a synthetic dataset of grocery transactions with realistic item-association patterns (e.g., people who buy Bread often also buy Butter)
Adds several "noise" features that have no real relationship to the target, to test how each model handles irrelevant inputs
Trains Linear Regression, Ridge, and Lasso to predict total spend from basket contents
Compares model performance and, more importantly, compares the learned coefficients side by side
Results
Model	R² Score
Linear Regression	0.9509
Ridge (L2)	0.9507
Lasso (L1)	0.8899
Key finding: Lasso pushes the coefficients of the noise features completely to zero, effectively removing them from the model entirely. Ridge instead just shrinks their coefficients to be very small, but doesn't eliminate them. This is the core practical difference between L1 and L2 regularization — Lasso can perform automatic feature selection, Ridge cannot.

A coefficient comparison chart (project1_shrinkage_coefficients.png) visualizes this: real features (Bread, Milk, Chicken, etc.) keep meaningful coefficient values across all three models, while the injected noise features collapse to (or near) zero specifically under Lasso.

Tech Used
scikit-learn — LinearRegression, Ridge, Lasso, StandardScaler, train_test_split
NumPy / Pandas — synthetic data generation and manipulation
Matplotlib — coefficient comparison visualization
Project 2: Market Basket Analysis (Apriori / Association Rule Mining)
Finds which grocery items are frequently bought together, using the classic Apriori algorithm — the same technique behind "customers who bought this also bought..." recommendations.

What It Does
Generates a synthetic dataset of 2,000 grocery transactions with built-in purchase patterns (e.g., Bread → Butter, Milk → Eggs)
One-hot encodes the transactions using TransactionEncoder
Runs the Apriori algorithm to find frequent itemsets (combinations of items that appear together often enough to be meaningful)
Extracts association rules from those itemsets — statements like "if a customer buys X, they are also likely to buy Y" — ranked by lift
Results
Top frequent itemsets included single items like Milk, Bread, and Eggs, as well as 2-item combinations like (Eggs, Butter).

Top association rules (sorted by lift) included patterns such as:

Antecedent	Consequent	Support	Confidence	Lift
Butter, Milk	Bread	0.1955	0.5891	2.0823
Butter, Eggs	Bread	0.1955	0.7995	2.0823
Bread, Milk	Butter	0.1955	0.6157	2.0186
Bread, Eggs	Butter	0.1955	0.6400	2.0186
Butter	Bread, Milk	0.2645	0.5939	1.5439
Interpretation: A lift greater than 1 means the items are bought together more often than would be expected by chance — for example, customers who buy Butter and Milk together are over 2x more likely than average to also buy Bread.

Tech Used
mlxtend — apriori, association_rules, TransactionEncoder
NumPy / Pandas — synthetic data generation and manipulation
How to Run
Both projects were built and run in Google Colab. To run locally instead:

bash
pip install pandas numpy scikit-learn mlxtend matplotlib

# Then open the notebook and run all cells:
jupyter notebook ML_Mini_Projects.ipynb
Project Structure
ML-Mini-Projects/
├── ML_Mini_Projects.ipynb          # Both projects, in one notebook
├── project1_shrinkage_coefficients.png   # Ridge vs Lasso coefficient chart
└── README.md
What This Is / Isn't
Both projects use synthetically generated data (not a real-world dataset), designed specifically to have known, injected patterns — this makes it easy to verify the models are actually learning what they're supposed to learn (e.g., confirming Lasso really does zero out the noise features we added on purpose). For a portfolio piece using real-world retail data, a public dataset like the Instacart Market Basket dataset would be a natural next step.


