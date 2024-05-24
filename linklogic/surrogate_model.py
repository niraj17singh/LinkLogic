from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class SurrogateModel():
    def __init__(self, model, data, alpha):

        self.data = data
        if model == 'Lasso':
            self.model = Lasso(alpha=alpha, precompute=True, max_iter=1000,
                        positive=True, random_state=9999, selection='random')
        else:
            print("Currently only Linear Regression supported")

    def split_data(self):
        train_x, test_x, train_y, test_y = train_test_split(self.data.iloc[:,:-1],
                                                            self.data.iloc[:,-1],
                                                            test_size=0.33,
                                                            random_state=42)

        return train_x, test_x, train_y, test_y

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        pred_train = self.model.predict(train_x)
        train_acc = r2_score(train_y, pred_train)
#         print('Range of coefficients:', min(self.model.coef_), max(self.model.coef_))
        print('\nTrain_acc: %.4f' % train_acc)
        return train_acc, pred_train

    def test(self, test_x, test_y):
        pred_test = self.model.predict(test_x)
        test_acc = r2_score(test_y, pred_test)
        print('Test_acc: %.4f' % test_acc + '\n')
        return test_acc, pred_test

    def run(self):
        train_x, test_x, train_y, test_y = self.split_data()
        train_acc, pred_train = self.train(train_x, train_y)
        test_acc, pred_test = self.test(test_x, test_y)
        coef = self.model.coef_
        return train_acc, test_acc, coef, train_x, test_x, train_y, test_y, pred_train, pred_test

    def append_stats_to_features(self, feature_stats, coef_):

        feature_stats['Coefficient'] = coef_


        # print features with positive coefficients in descending order
        feature_stats_sort =  feature_stats.sort_values(['Coefficient'], ascending=False)
        
        print("Explanations: \n")
        for row in feature_stats_sort.iterrows():
            if row[1][1] > 0:
                print("coef:", round(row[1][1], 4), row[1][0])

        return feature_stats
