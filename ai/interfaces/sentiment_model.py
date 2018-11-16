import pickle
from ai.algorithms.text_classification import *
from ai.ai_ultis.data_utils import *


def predict_test_datas(model, testing_data):
    test_data = []
    test_label = []
    sum = {'average': 0}
    for tag in testing_data:
        test_data.extend(testing_data[tag])
        test_label.extend([tag] * len(testing_data[tag]))
        sum[tag] = 0
    for i in range(len(test_data)):
        predicted_tag = model.predict(test_data[i])
        if predicted_tag == test_label[i]:
            sum['average'] += 1
            sum[predicted_tag] += 1
        print("sentence: ", test_data[i], " -- pred : ", predicted_tag)
    print("average acc: ", sum['average'] / len(test_data))
    tag='search'
    print(tag, " acc: ", sum[tags[tag]] / len(testing_data[tags[tag]]))
    print("correct: ", sum[tags[tag]], " - - Total: ", len(testing_data[tags[tag]]))
    print("- - - - - - - - - - - -")
    return sum['average'] / len(test_data)


class Classifier(object):
    def __init__(self):
        pass

    def train_clf(self, excel_file, excel_sheet="Sheet1"):
        self.tag_model = {}
        training_data = load_excel_training_data(excel_file, excel_sheet)
        training_data, testing_data = split_data(training_data, 0.8)
        vocabulary, corpus = build_vocabulary(training_data)
        self.vectorizer = train_vectorizer(vocabulary, corpus)
        tag='search'
        print("train model for tag: ", tag)
        tag_excluded_data_dict = copy.deepcopy(training_data)
        texts = copy.deepcopy(training_data[tag])
        print(len(texts))
        del tag_excluded_data_dict[tag]
        other_texts = []
        for other_tag in tag_excluded_data_dict:
            other_texts.extend(tag_excluded_data_dict[other_tag])
        print(len(other_texts))
        curr_clf = train_MLP_title_prediction(tag, self.vectorizer, texts, other_texts, MLP_structure=[4, 256])
            # print(curr_clf.predict(texts))
        self.tag_model[tag] = curr_clf
        print("training completed. testing . . . . . .")
        self.predict_test_data(testing_data)
        pass

    def dump(self, storage_path):
        with open(storage_path, "wb") as f:
            pickle.dump(self.tag_model, f)
            f.close()

    def load(self, storage_path):
        with open(storage_path, "rb") as f:
            self.tag_model= pickle.load(f)
            f.close()

    def predict_tag(self, tag, text):
        vector= self.tag_model[tag]['feature_gen'].transform([text])
        pred = self.tag_model[tag]['clf'].predict(vector)[0]
        prob = max(self.tag_model[tag]['clf'].predict_proba(vector)[0])
        if pred == 'other':
            return 0
        return prob

    def predict(self, text):
        max_prob=0
        pred="search"
        tag=1
        prob= self.predict_tag(tags[tag], text)
        if prob > max_prob:
            max_prob= prob
            pred= tags[tag]
        if max_prob==0:pred="neutral"
        return pred

    def predict_test_data(self,testing_data):
        test_data = []
        test_label = []
        sum={'average':0}
        for tag in testing_data:
            test_data.extend( testing_data[tag])
            test_label.extend([tag]*len(testing_data[tag]))
            sum[tag]=0
        for i in range(len(test_data)):
            predicted_tag= self.predict(test_data[i])
            if predicted_tag == test_label[i]:
                sum['average']+=1
                sum[predicted_tag]+=1
            print("sentence: ", test_data[i], " -- pred : ", predicted_tag)
        print("average acc: ", sum['average']/len(test_data))
        for tag in tags:
            print(tag," acc: ", sum[tags[tag]]/len(testing_data[tags[tag]]))
            print("correct: ", sum[tags[tag]]," - - Total: ", len(testing_data[tags[tag]]))
            print("- - - - - - - - - - - -")
        return sum['average']/len(test_data)

    def predict_test_data(self,testing_data):
        test_data = []
        test_label = []
        sum={'average':0}
        tag='search'
        test_data.extend( testing_data[tag])
        test_label.extend([tag]*len(testing_data[tag]))
        sum[tag]=0
        for i in range(len(test_data)):
            predicted_tag= self.predict(test_data[i])
            if predicted_tag == test_label[i]:
                sum['average']+=1
                sum[predicted_tag]+=1
            print("sentence: ", test_data[i], " -- pred : ", predicted_tag)
        print("average acc: ", sum['average']/len(test_data))
        tag=1
        print(tag," acc: ", sum[tags[tag]]/len(testing_data[tags[tag]]))
        print("correct: ", sum[tags[tag]]," - - Total: ", len(testing_data[tags[tag]]))
        print("- - - - - - - - - - - -")
        return sum['average']/len(test_data)

mlp=Classifier()
mlp.train_clf("C:/Users/Luong Thanh Long/PycharmProjects/Feature - Copy/data.csv")
mlp.dump("C:\\Users\\Luong Thanh Long\\PycharmProjects\\Feature\\storage\\model\\model.sav")
# if mlp.predict_test_data('D:\\ITSOL\\detect_sentiment_\\storage\\data\\test.xlsx','Sheet1')>predict_test_datas(mlp.load("C:\\Users\\Luong Thanh Long\\PycharmProjects\\Feature\\storage\\model\\model.sav"),load_excel_training_data('D:\\ITSOL\\detect_sentiment_\\storage\\data\\test.xlsx','Sheet1')):
#    mlp.dump("C:\\Users\\Luong Thanh Long\\PycharmProjects\\Feature\\storage\\model\\model.sav")
print(get_normal_word('Theo đó, Xiaomi bất ngờ lỗ ròng 6,9 tỷ USD trong cả năm 2017'))