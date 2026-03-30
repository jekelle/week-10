import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv"

def roast_category(roast):
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast, float("nan"))

def main():
    df = pd.read_csv(DATA_URL)

    # Exercise 1
    X1 = df[["100g_USD"]]
    y = df["rating"]

    lr = LinearRegression()
    lr.fit(X1, y)

    with open("model_1.pickle", "wb") as f:
        pickle.dump(lr, f)

    # Exercise 2
    df["roast_cat"] = df["roast"].apply(roast_category)
    X2 = df[["100g_USD", "roast_cat"]]

    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(X2, y)

    with open("model_2.pickle", "wb") as f:
        pickle.dump(dtr, f)

if __name__ == "__main__":
    main()
