from sklearn.model_selection import train_test_split
from src.preprocess import standard_scale

def split_data(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_test = standard_scale(x_train, x_test)
    return x_train, x_test, y_train, y_test
