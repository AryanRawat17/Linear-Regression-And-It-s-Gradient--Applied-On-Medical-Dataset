# Downloadin A Dataset #

medical_charge_url="https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
from urllib.request import urlretrieve
urlretrieve(medical_charge_url,"medical.csv")




# Now we can import pandas to analyze our dataset #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




medical_df= pd.read_csv("medical.csv")
medical_df


#Inspect Your Data: Check for non-numeric values in your numeric columns.#
# Now let's check their datatype#

print(medical_df.head())
print(medical_df.info())




#Now convert Columns to Numeric:Use pd.to_numeric with the errors='coerce'parameter to convert non-numeric values to NaN.#
medical_df = medical_df.apply(pd.to_numeric, errors='coerce')






#Now we get it's statistical value#
# Calculate and print mean for each column
means = medical_df.mean()
print("Means:\n", means)

# Calculate and print standard deviation for each column
stds = medical_df.std()
print("Standard Deviations:\n", stds)

# Calculate and print median for each column
medians = medical_df.median()
print("Medians:\n", medians)

# Calculate and print minimum for each column
mins = medical_df.min()
print("Minimums:\n", mins)

# Calculate and print maximum for each column
maxs = medical_df.max()
print("Maximums:\n", maxs)

# Calculate and print 25th percentile for each column
percentile_25 = medical_df.quantile(0.25)
print("25th Percentiles:\n", percentile_25)

# Calculate and print 50th percentile for each column (this is the median)
percentile_50 = medical_df.quantile(0.50)
print("50th Percentiles (Medians):\n", percentile_50)

# Calculate and print 75th percentile for each column
percentile_75 = medical_df.quantile(0.75)
print("75th Percentiles:\n", percentile_75)

# Get a summary of statistics for each column using describe
summary = medical_df.describe()
print("Summary Statistics:\n", summary)






# Here we create a histogram for age #



# Convert the 'age' column to numeric, coercing errors to NaN
medical_df['age'] = pd.to_numeric(medical_df['age'], errors='coerce')

# Option 1: Drop rows with NaN values in the 'age' column
medical_df = medical_df.dropna(subset=['age'])

# Option 2: Fill NaN values with the mean of the 'age' column
# medical_df['age'].fillna(medical_df['age'].mean(), inplace=True)

# Calculate statistics for the 'age' column
age_count = medical_df['age'].count()
age_mean = medical_df['age'].mean()
age_std = medical_df['age'].std()
age_median = medical_df['age'].median()
age_min = medical_df['age'].min()
age_max = medical_df['age'].max()
age_25th = medical_df['age'].quantile(0.25)
age_50th = medical_df['age'].quantile(0.50)
age_75th = medical_df['age'].quantile(0.75)

# Print the statistics
print(f"Age Count: {age_count}")
print(f"Age Mean: {age_mean}")
print(f"Age Standard Deviation: {age_std}")
print(f"Age Median: {age_median}")
print(f"Age Minimum: {age_min}")
print(f"Age Maximum: {age_max}")
print(f"Age 25th Percentile: {age_25th}")
print(f"Age 50th Percentile: {age_50th}")
print(f"Age 75th Percentile: {age_75th}")

# Create a histogram for the 'age' column
plt.hist(medical_df['age'], bins=47, edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()







# Here we create a histogram for bmi #


# Convert the 'bmi' column to numeric, coercing errors to NaN
medical_df['bmi'] = pd.to_numeric(medical_df['bmi'], errors='coerce')

# Option 1: Drop rows with NaN values in the 'bmi' column
medical_df = medical_df.dropna(subset=['bmi'])

# Option 2: Fill NaN values with the mean of the 'bmi' column
# medical_df['bmi'].fillna(medical_df['bmi'].mean(), inplace=True)

# Calculate statistics for the 'bmi' column
bmi_count = medical_df['bmi'].count()
bmi_mean = medical_df['bmi'].mean()
bmi_std = medical_df['bmi'].std()
bmi_median = medical_df['bmi'].median()
bmi_min = medical_df['bmi'].min()
bmi_max = medical_df['bmi'].max()
bmi_25th = medical_df['bmi'].quantile(0.25)
bmi_50th = medical_df['bmi'].quantile(0.50)
bmi_75th = medical_df['bmi'].quantile(0.75)

# Print the statistics
print(f"BMI Count: {bmi_count}")
print(f"BMI Mean: {bmi_mean}")
print(f"BMI Standard Deviation: {bmi_std}")
print(f"BMI Median: {bmi_median}")
print(f"BMI Minimum: {bmi_min}")
print(f"BMI Maximum: {bmi_max}")
print(f"BMI 25th Percentile: {bmi_25th}")
print(f"BMI 50th Percentile: {bmi_50th}")
print(f"BMI 75th Percentile: {bmi_75th}")

# Create a histogram for the 'bmi' column with a specified color
plt.hist(medical_df['bmi'], bins=47, edgecolor='black', color='darkred')
plt.title('Histogram of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




# Here we create a histogram for Smoker and Non-Smoker #





# Important to note #
medical_df= pd.read_csv("medical.csv")
medical_df


#Inspect Your Data: Check for non-numeric values in your numeric columns.#
# Now let's check their datatype#

print(medical_df.head())
print(medical_df.info())


# Ensure that the 'smoker' and 'charges' columns exist
if 'smoker' not in medical_df.columns or 'charges' not in medical_df.columns:
    print("Error: The dataset does not contain 'smoker' and 'charges' columns.")
else:
    # Print the unique values in the 'smoker' column for debugging
    print("Unique values in 'smoker' column:", medical_df['smoker'].unique())

    # Print basic statistics of the 'charges' column for debugging
    print(medical_df['charges'].describe())

    # Convert the 'charges' column to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')

    # Drop rows with NaN values in the 'charges' column
    medical_df = medical_df.dropna(subset=['charges'])

    # Separate data for smokers and non-smokers
    smokers = medical_df[medical_df['smoker'] == 'yes']
    non_smokers = medical_df[medical_df['smoker'] == 'no']

    # Print the number of rows in each group for debugging
    print("Number of smokers:", len(smokers))
    print("Number of non-smokers:", len(non_smokers))

    # Plot histograms
    plt.figure(figsize=(10, 6))

    plotted_any = False

    if not non_smokers.empty:
        plt.hist(non_smokers['charges'], bins=47, edgecolor='black', color='grey', alpha=0.7, label='Non-Smokers')
        plotted_any = True

    if not smokers.empty:
        plt.hist(smokers['charges'], bins=47, edgecolor='black', color='green', alpha=0.7, label='Smokers')
        plotted_any = True

    if plotted_any:
        # Add titles and labels
        plt.title('Histogram of Charges by Smoker Status')
        plt.xlabel('Charges')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data available to plot.")
        
        
        
        
        
        
        
        
        
        
        
# Scatter Ploting Starts From Here#





#Inspect Your Data: Check for non-numeric values in your numeric columns.#
# Now let's check their datatype#

print(medical_df.head())
print(medical_df.info())





# Ensure that the 'age', 'charges', and 'smoker' columns exist
if 'age' not in medical_df.columns or 'charges' not in medical_df.columns or 'smoker' not in medical_df.columns:
    print("Error: The dataset does not contain 'age', 'charges', and 'smoker' columns.")
else:
    # Convert the 'charges' column to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')

    # Handle missing values in 'charges' column
    median_charges = medical_df['charges'].median()
    medical_df['charges'] = medical_df['charges'].fillna(median_charges)

    # Handle missing values in 'smoker' column
    medical_df['smoker'] = medical_df['smoker'].fillna('no')  # assuming 'no' as the default value

    # Separate data for smokers and non-smokers
    smokers = medical_df[medical_df['smoker'] == 'yes']
    non_smokers = medical_df[medical_df['smoker'] == 'no']

    # Plot scatter plot
    plt.figure(figsize=(10, 6))

    # Plot data for non-smokers
    plt.scatter(non_smokers['age'], non_smokers['charges'], alpha=0.5, color='blue', edgecolors='w', s=40, label='Non-Smokers')

    # Plot data for smokers
    plt.scatter(smokers['age'], smokers['charges'], alpha=0.5, color='red', edgecolors='w', s=40, label='Smokers')

    # Add titles and labels
    plt.title('Scatter Plot of Charges vs Age by Smoker Status')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
#Inspect Your Data: Check for non-numeric values in your numeric columns.#
# Now let's check their datatype#

print(medical_df.head())
print(medical_df.info())


# Ensure that the 'bmi', 'charges', and 'smoker' columns exist
if 'bmi' not in medical_df.columns or 'charges' not in medical_df.columns or 'smoker' not in medical_df.columns:
    print("Error: The dataset does not contain 'bmi', 'charges', and 'smoker' columns.")
else:
    # Convert the 'charges' column to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')

    # Handle missing values in 'charges' column
    median_charges = medical_df['charges'].median()
    medical_df['charges'] = medical_df['charges'].fillna(median_charges)

    # Handle missing values in 'bmi' column
    median_bmi = medical_df['bmi'].median()
    medical_df['bmi'] = medical_df['bmi'].fillna(median_bmi)

    # Handle missing values in 'smoker' column
    medical_df['smoker'] = medical_df['smoker'].fillna('no')  # assuming 'no' as the default value

    # Separate data for smokers and non-smokers
    smokers = medical_df[medical_df['smoker'] == 'yes']
    non_smokers = medical_df[medical_df['smoker'] == 'no']

    # Plot scatter plot
    plt.figure(figsize=(10, 6))

    # Plot data for non-smokers
    plt.scatter(non_smokers['bmi'], non_smokers['charges'], alpha=0.9, color='blue', edgecolors='w', s=40, label='Non-Smokers')

    # Plot data for smokers
    plt.scatter(smokers['bmi'], smokers['charges'], alpha=0.9, color='red', edgecolors='w', s=40, label='Smokers')

    # Add titles and labels
    plt.title('Scatter Plot of Charges vs BMI by Smoker Status')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


    

# Here We Learn Corelation #





# Downloadin A Dataset #

medical_charge_url="https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
from urllib.request import urlretrieve
urlretrieve(medical_charge_url,"medical.csv")




# Now we can import pandas to analyze our dataset #
import pandas as pd
medical_df= pd.read_csv("medical.csv")
medical_df



# Ensure that the 'age' and 'charges' columns exist
if 'age' not in medical_df.columns or 'charges' not in medical_df.columns:
    print("Error: The dataset does not contain 'age' and 'charges' columns.")
else:
    # Convert the 'charges' column to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')

    # Handle missing values in 'charges' column
    median_charges = medical_df['charges'].median()
    medical_df['charges'] = medical_df['charges'].fillna(median_charges)

    # Calculate the correlation between 'age' and 'charges'
    correlation = medical_df['age'].corr(medical_df['charges'])
    print(f"The correlation between age and charges is: {correlation}")

   


# Ensure that the 'bmi' and 'charges' columns exist
if 'bmi' not in medical_df.columns or 'charges' not in medical_df.columns:
    print("Error: The dataset does not contain 'bmi' and 'charges' columns.")
else:
    # Convert the 'charges' and 'bmi' columns to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')
    medical_df['bmi'] = pd.to_numeric(medical_df['bmi'], errors='coerce')

    # Handle missing values in 'charges' and 'bmi' columns
    median_charges = medical_df['charges'].median()
    medical_df['charges'] = medical_df['charges'].fillna(median_charges)

    median_bmi = medical_df['bmi'].median()
    medical_df['bmi'] = medical_df['bmi'].fillna(median_bmi)

    # Calculate the correlation between 'bmi' and 'charges'
    correlation = medical_df['bmi'].corr(medical_df['charges'])
    print(f"The correlation between BMI and charges is: {correlation}")
    
    
    
    
# Ensure that the 'smoker' and 'charges' columns exist
if 'smoker' not in medical_df.columns or 'charges' not in medical_df.columns:
    print("Error: The dataset does not contain 'smoker' and 'charges' columns.")
else:
    # Convert the 'charges' column to numeric, coercing errors to NaN
    medical_df['charges'] = pd.to_numeric(medical_df['charges'], errors='coerce')

    # Handle missing values in 'charges' column
    median_charges = medical_df['charges'].median()
    medical_df['charges'] = medical_df['charges'].fillna(median_charges)

    # Encode 'smoker' column: 'yes' -> 1, 'no' -> 0
    medical_df['smoker'] = medical_df['smoker'].map({'yes': 1, 'no': 0})

    # Calculate the correlation between 'smoker' and 'charges'
    correlation = medical_df['smoker'].corr(medical_df['charges'])
    print(f"The correlation between smoking status and charges is: {correlation}")








# Here we described correlation between all pairs of numeric column #




# Ensure the dataset has been loaded correctly
if medical_df.empty:
    print("Error: The dataset is empty or could not be loaded.")
else:
    # Convert relevant columns to numeric, coercing errors to NaN
    numeric_columns = ['age', 'bmi', 'children', 'charges']
    for column in numeric_columns:
        medical_df[column] = pd.to_numeric(medical_df[column], errors='coerce')


    # Select only numeric columns for correlation analysis
    numeric_df = medical_df[numeric_columns]

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    print("Correlation matrix:")
    print(correlation_matrix)







# Here we present the Heatmap of our correlation #




# Display the first few rows to understand the structure of the dataset
print(medical_df.head())

# Ensure the dataset has been loaded correctly
if medical_df.empty:
    print("Error: The dataset is empty or could not be loaded.")
else:
    # Convert relevant columns to numeric, coercing errors to NaN
    numeric_columns = ['age', 'bmi', 'children', 'charges']
    for column in numeric_columns:
        medical_df[column] = pd.to_numeric(medical_df[column], errors='coerce')

    # Handle missing values by filling with the median
    for column in numeric_columns:
        median_value = medical_df[column].median()
        medical_df[column] = medical_df[column].fillna(median_value)

    # Select only numeric columns for correlation analysis
    numeric_df = medical_df[numeric_columns]

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    print("Correlation matrix:")
    print(correlation_matrix)

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.show()





"""Linear regression using a single feature - try to find a way of estimating the 
value of charges using the value of age for non smokers create a dataframe containing 
just data for non smokers-


1-Filter the DataFrame to Include Only Non-Smokers.
2-Create a Linear Regression Model Using age as the Predictor.
3-Evaluate the Model.
4-Plot the Results."""





# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
medical_df = pd.read_csv('medical.csv')

# Display the first few rows to understand the structure of the dataset
print(medical_df.head())


# Filter the dataset to include only non-smokers
non_smokers_df = medical_df[medical_df['smoker'] == 'no']

# Select the relevant columns for the analysis
non_smokers_df = non_smokers_df[['age', 'charges']]

# Display the first few rows of the non-smokers dataframe
print("Non-smokers data:")
print(non_smokers_df.head())

# Define the independent variable (age) and dependent variable (charges)
X = non_smokers_df[['age']]
y = non_smokers_df['charges']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the results
plt.figure(figsize=(15,7))
plt.scatter(X, y, color='blue', label='Actual Charges')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Charges')
plt.title('Age vs. Charges for Non-Smokers')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()




# Predict charges for specific ages
ages_to_predict = [25, 30, 35, 40, 45, 50, 55, 60]
predicted_charges = model.predict(pd.DataFrame(ages_to_predict, columns=['age']))

# Print the predicted charges for the given ages
for age, charge in zip(ages_to_predict, predicted_charges):
    print(f'Estimated charge for age {age}: {charge}')








""" find the target charges for non-smokers and compare them with the predicted estimated charges, we will:

Filter the DataFrame to Include Only Non-Smokers.
Create and Train a Linear Regression Model.
Predict Charges for Non-Smokers.
Compare the Actual and Predicted Charges"""






# Display the first few rows to understand the structure of the dataset
print(medical_df.head())

# Filter the dataset to include only non-smokers
non_smokers_df = medical_df[medical_df['smoker'] == 'no']

# Select the relevant columns for the analysis
non_smokers_df = non_smokers_df[['age', 'charges']]

# Display the first few rows of the non-smokers dataframe
print("Non-smokers data:")
print(non_smokers_df.head())

# Define the independent variable (age) and dependent variable (charges)
X = non_smokers_df[['age']]
y = non_smokers_df['charges']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Print performance metrics
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Create a DataFrame to compare actual and predicted charges
comparison_df = non_smokers_df.copy()
comparison_df['predicted_charges'] = y_pred

# Display the comparison DataFrame
print(comparison_df.head())

# Plot the results
plt.figure(figsize=(15, 7))
plt.scatter(X, y, color='blue', label='Actual Charges')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Charges')
plt.title('Age vs. Charges for Non-Smokers')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()

