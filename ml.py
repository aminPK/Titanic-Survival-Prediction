import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# Daten laden
daten= sns.load_dataset("titanic")
print(daten.head())

# daten_auswahl
df= daten[["survived","sex", "age","pclass","embarked"]].copy()

# Fehlende Daten in Age und Emabarked beseitigen
df["age"]= df["age"].fillna(df["age"].mean())
df["embarked"]= df["embarked"].fillna(df["embarked"].mode() [0])   

# daten in sex, embarked binäre Form umwandeln
le_sex= LabelEncoder()
le_embarked= LabelEncoder()
df["sex"]= le_sex.fit_transform(df["sex"])
df["embarked"]= le_embarked.fit_transform(df["embarked"])

# future und lable anführen in x, y
x= df[["age", "sex", "pclass", "embarked"]]
y= df["survived"]

# model mit train_test_split trainieren
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)

# Tree_model trainnieren und Bewrtungen durchführen
tree_model= DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
y_pre_tree_model= tree_model.predict(x_test)
print(y_pre_tree_model)
print("___Decition_Tree___")
print("classification_report", classification_report(y_test, y_pre_tree_model))
print("Confusion_matrix",confusion_matrix(y_test, y_pre_tree_model))
# Wichtigste Future_Merkmale in Tree_model visualiesiren
importances= tree_model.feature_importances_
feature= x.columns

for name, importances in zip(x.columns, tree_model.feature_importances_):
    print(f"{name}: {importances:.2f}")
    
# Visualiesiren
plt.figure(figsize=(8, 5))
plt.barh(feature, importances)
plt.xlabel("Wichtigkeit")
plt.title("Future_Wichtigkeit_in_Decision_Tree")
plt.show()

# Cross_validation untersuchung, wie gut funktioniert das Model auf mehreren Datenaufteilungen.
scores= cross_val_score(tree_model, x,y, cv= 5)
print("Cross_validation_score:", scores)
print("Die durchschnittliche Genauigkeit:", scores.mean())


# mit Grid_searchCV die Besten Parametern für das Model bestimmen(mit einem frischen tree_model)
# model
dt= DecisionTreeClassifier()

# welche Parametern sollen untersucht werden
param_grid= {"max_depth":[2,3,4,5,6],
             "criterion": ["gini","entropy"],
             "min_samples_split": [2,4,6]} 

# Gridsearch mit 5 facher CV
grid_search= GridSearchCV(dt,param_grid, cv=5)

# training auf gesamten trainingsdaten.
grid_search.fit(x_train, y_train)

# bestes Model + bester Parameter zeigen
print("Das beste Modell:", grid_search.best_estimator_)
print("Beste Parameter:", grid_search.best_params_)
print("Beste Genauigkeit:", grid_search.best_score_)

# Vorhersage mit GridsearchCV (das beste Modell) treffen und das Modell bewerten

# das beste modell
best_modell= grid_search.best_estimator_

# vorhersagen
y_pred_best= grid_search.predict(x_test)

# modell bewerten
print("Confusion_matrix_best:", confusion_matrix(y_test,y_pred_best ))
print("Classification_reporter_best:", classification_report( y_test,y_pred_best))





# Logestic_Regression trainieren vorhersagen und die Vorhesage Bewerten
log_model= LogisticRegression()
log_model.fit(x_train, y_train)
y_pre_log_model= log_model.predict(x_test)
print("Logestic_Regression")
print("Confusion_matrix", confusion_matrix(y_test, y_pre_log_model))
print("Classification_reporter", classification_report(y_test, y_pre_log_model))

# Wichtige Feature bei Coeffizient in Log_model herausfinden und visualiesiren
coeffizients= log_model.coef_[0]
feautre= x.columns

for name, coef in zip(feature, coeffizients):
    print(f"{name}: {coef:.3f}")
    
plt.figure(figsize=(8, 5))
plt.bar(feature, coeffizients)
plt.xlabel("Coeffizients")
plt.title("feautre_Wichtigkeit in (Logestic_Regression)")
plt.axvline(0, color= "gray", linestyle="--")
plt.show()


# Die Daten mit Roc_curve, und AUC bewerten und optimieren. nur für Logestic_Regression und Random_forest, weil die Predict.proba(Wahrscheinlichkeiten für Kategorisierte Daten anwenden)

models_to_plot= {"Logestic Regression": LogisticRegression(),
                 "Random Forest": RandomForestClassifier()}

plt.figure(figsize=(8, 6))
for name, model in models_to_plot.items():
    model.fit(x_train, y_train)
    y_proba= model.predict_proba(x_test)[:, 1]
    fpr, tpr, _= roc_curve(y_test, y_proba)
    roc_auc= auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name}, (AUC= {roc_auc:.2f})") 
    
# Diagonale Linie (Zufallsdaten) erstellen
plt.plot([0, 1], [0, 1], "k--", label= "Zufall")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positve Rate")
plt.title("Roc_Kurve")
plt.legend()
plt.grid
plt.show()

# mit neuen Model Cross_validation mit obigem Code kombinieren

models={"LogesticRegression:": LogisticRegression(max_iter= 1000),
        "RandomForest:": RandomForestClassifier()}

for name, model in models.items():
    scores_cross= cross_val_score(model, x,y, cv= 5, scoring= "roc_auc")
    print(f"{name}")
    print(f"AUC Score pro Fold {scores_cross}")
    print(f"Mittelwert AUC: {scores_cross.mean():.3f}")
    print(f"Standardabweichungen: {scores_cross.std():.3f}")
     



#[]{} \n,