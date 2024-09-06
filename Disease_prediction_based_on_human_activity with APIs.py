from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, GlobalMaxPooling1D, Reshape
from warnings import simplefilter,filterwarnings
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from experta import *
import pandas as pd
import numpy as np
import scipy.io
import pickle
import glob
import sys
import gc
import os
import re
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Dejavu Sans'
app = Flask(__name__)

data_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\DataSet"
results_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\Results"
code_path = "C:\\Users\\user\\Python_Anaconda\\task\\06\\Code"
model_info_path = os.path.join(results_path,'model info')


def save_object(obj, filename,path):
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
    
def load_object(filename,path):
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object

class Healthy(Fact):
    pass
class FootPain(Fact):
    pass
class Insomnia(Fact):
    pass
class Obesity(Fact):
    pass
class Arthritis(Fact):
    pass
class HeartAndBloodVesselProblems(Fact):
    pass
class Bedsores(Fact):
    pass
class BackPains(Fact):
    pass
    
class DiseasesDetectionBasedActivities(KnowledgeEngine):
    
    def __init__(self, activites):
        super().__init__()
        self.activites = activites
        self.important_activites = []
        self.important_number_activites = []
        self.other_activites = []
        self.other_number_activites = []        
        self.Disease = 'Healthy'

    @Rule(NOT(Fact(ID = W())))
    def FillFacts(self):
        activites_count, base_number = self.calculate_counter()
        for key in activites_count:
            if key[1]>=base_number:
                self.important_activites.append(key[0])
                self.important_number_activites.append(key[1])
            else:
                self.other_activites.append(key[0])
                self.other_number_activites.append(key[1])
    def calculate_counter(self):
        uniqe_activites = set(self.activites)
        activity_dic = {key:0 for key in uniqe_activites}
        for i, x in enumerate(self.activites):
            activity_dic[x]+=1
        return sorted(activity_dic.items(), key=lambda x: x[1]), np.ceil(len(self.activites)/len(uniqe_activites))
    
    @Rule(NOT(FootPain(foot_pain = W())))
    def FootPain_(self):
        foot_pain = False
        if np.array_equal(np.sort(self.important_activites), np.sort(['WALKING', 'STANDING'])) and self.other_activites[self.other_number_activites.index(max(self.other_number_activites))]=='SITTING':
            foot_pain = True
        self.declare(FootPain(foot_pain = foot_pain))

    @Rule(NOT(Insomnia(insomnia = W())))
    def Insomnia_(self):
        insomnia = False
        if np.array_equal(np.sort(self.important_activites), np.sort(['LAYING', 'SITTING', 'WALKING'])) and np.array_equal(np.sort(self.other_activites), np.sort(['WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS', 'STANDING'])):
            insomnia  = True
        self.declare(Insomnia(insomnia = insomnia))
    
    @Rule(NOT(Obesity(obesity = W())))
    def Obesity_(self):
        obesity = False
        if np.array_equal(np.sort(self.important_activites), np.sort(['SITTING', 'LAYING'])) and self.other_activites[self.other_number_activites.index(max(self.other_number_activites))]=='STANDING':
            obesity = True
        self.declare(Obesity(obesity = obesity))
    
    @Rule(NOT(Bedsores(bedsores = W())))
    def Bedsores_(self):
        bedsores = False
        if np.array_equal(self.important_activites, ['LAYING']):
            bedsores = True
        self.declare(Bedsores(bedsores = bedsores))
    
    @Rule(NOT(Arthritis(arthritis = W())))
    def Arthritis_(self):
        arthritis = False
        if np.array_equal(np.sort(self.important_activites), np.sort(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])):
            arthritis = True
        self.declare(Arthritis(arthritis = arthritis))
        
    @Rule(NOT(HeartAndBloodVesselProblems(heart_and_blood_vessel_problems = W())))
    def Heart_and_blood_vessel_problems_(self):
        heart_and_blood_vessel_problems = False
        if self.important_activites[self.important_number_activites.index(max(self.important_number_activites))]=='SITTING' and self.other_activites[self.other_number_activites.index(max(self.other_number_activites))]=='LAYING':
            heart_and_blood_vessel_problems=True
        self.declare(HeartAndBloodVesselProblems(heart_and_blood_vessel_problems = heart_and_blood_vessel_problems))
    
    @Rule(NOT(BackPains(back_pains = W())))
    def BackPains_(self):
        back_pains = False
        if self.important_activites[self.important_number_activites.index(max(self.important_number_activites))]=='STANDING' and np.array_equal(np.sort(self.important_activites), np.sort(['STANDING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])): 
            back_pains = True
        self.declare(BackPains(back_pains = back_pains))

    @Rule(FootPain(foot_pain = L(True)))
    def Foot_Pain_test(self):
        self.Disease = 'Foot Pain'
        
    @Rule(Insomnia(insomnia = L(True)))
    def Insomnia_test(self):
        self.Disease = 'Healthy But sometimes Have Insomnia'
        
    @Rule(Obesity(obesity = L(True)))
    def Obesity_test(self):
        self.Disease = 'Obesity'
        
    @Rule(Bedsores(bedsores = L(True)))
    def Bedsores_test(self):
        self.Disease = 'Bedsores'
        
    @Rule(Arthritis(arthritis = L(True)))
    def Arthritis_test(self):
        self.Disease = 'Arthritis'
 
    @Rule(HeartAndBloodVesselProblems(heart_and_blood_vessel_problems = L(True)))
    def HeartAndBloodVesselProblems_test(self):
        self.Disease = 'Heart And Blood Vessel Problems'
   
    @Rule(BackPains(back_pains = L(True)))
    def BackPains_test(self):
        self.Disease = 'Back Pains'
        
def ExpertSystem(activites):
    engine = DiseasesDetectionBasedActivities(activites)
    engine.reset()
    engine.run()
    return engine.Disease
    
    
def HumanActivityPredection(X, y, label_encoder):
    model_name = "CNN_Network_"
    model = load_model(filepath=os.path.join(model_info_path ,model_name+".h5"))
    y_pred = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    activites = label_encoder.inverse_transform(y_pred)
    return activites
    
def Pipeline(subject_id, CSV_file=None):
    CSV_file = pd.read_csv(os.path.join(data_path,"new_test_dayaset.csv"),) if CSV_file is None else CSV_file
    subject_recordes = CSV_file[CSV_file['subject']==int(subject_id)]
    label_encoder = load_object("label_encoder", results_path)
    X = subject_recordes.drop('Activity', axis=1)
    y = np.array(subject_recordes['Activity'].tolist())
    y = label_encoder.transform(y)
    y = to_categorical(y, 6)
    X = X.values.reshape(-1, 1, X.shape[1])
    activites = HumanActivityPredection(X, y, label_encoder)
    Disease = ExpertSystem(activites)
    DiseaseFromReallActivites = ExpertSystem(subject_recordes['Activity'].tolist())
    return Disease, DiseaseFromReallActivites

@app.route('/predict', methods=['POST'])
def predict_pipeline():
    try:
        data = request.get_json()

        subject_id = int(data.get('subject_id'))
        csv_file = data.get('csv_file')

        disease_prediction, real_activities_disease = Pipeline(subject_id, csv_file)

        return jsonify({
            "subject_id": subject_id,
            "predicted_disease": disease_prediction,
            "disease_from_real_activities": real_activities_disease
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
