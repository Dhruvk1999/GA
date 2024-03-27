# run_ga_optimization(stock_data,generate_individual,alpha8,evaluate_strategy)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from strategies.alphas import *

def compute_labels(data, x_pct, n_days):
    labels = []
    for index, row in data.iterrows():
        row_index = data.index.get_loc(index)  # Get the integer index
        future_prices = data['Close'].iloc[row_index + 1:row_index + n_days + 1]
        price_change = (future_prices / row['Close']) - 1
        if any(price_change >= x_pct):
            labels.append(1)  # Price went up by x_pct or more within next n days
        else:
            labels.append(0)  # Price didn't go up by x_pct within next n days
    return labels


def train_random_forest(data,data_test,alpha,*args):
    # Replace X and y with your actual dataset

    data['alpha'] = alpha(data,*args)
    y = compute_labels(data, 0.02, 5)
    data['alpha102'] = alpha102(data,2,10)

    data=data.drop(["Close","Open","Volume",'High', 'Low', 'Adj Close'],axis =1 )
    print("this is being trained upon",data)

    data_test['alpha'] =alpha(data_test,6)
    y_test = compute_labels(data_test,0.02,5)
    data_test['alpha102'] = alpha102(data_test,2,10)

    data_test=data_test.drop(["Close","Open","Volume",'High', 'Low', 'Adj Close'],axis =1 )



    # Split the dataset into training and testing sets

    # Initialize the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=43)

    # Train the random forest classifier
    rf_classifier.fit(data, y)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(data_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return rf_classifier
