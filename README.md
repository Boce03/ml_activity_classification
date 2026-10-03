# ml_activity_classification
Projekat iz predmeta Istraživanje podataka 2 na Matematičkom fakultetu

## Opis projekta
Projekat sadrži eksploraciju skupa podataka koji sadrži različita fiziološka merenja ispitanika merena tokom četiri aktivnost: neutralna, emocionalna, mentalna i fizička. Signali koji su mereni su ECG, TEB, EDA ruke i EDA šake. Cilj projekta je eksploracija skupa podataka i klasifikacija podataka na osnovu fizioloških merenja po aktivnostima koje predstavljaju klase.

## Struktura projekta
1. data direktorijum sadrži ulazne podatke, kao i kreirani data.csv fajl u fazi preprocesiranja podataka
2. docs direktorijum sadrži dokumentaciju, koja se sastoji od inicijalnog opisa skupa podataka i izveštaja u PDF formatu (IP2_isvestaj.pdf)
3. models direktorijum koji sadrži sačuvane konstruisane modele sa najboljim hiperparametrima u okviru GridSearchCV za svaku kombinaciju modela i skupa atributa (u models_notebook.ipynb se nalazi export i import deo)
4. feature_reduction je python paket koji sadrži klasu InterpretableFeatureReducer koja može kao transformator da se koristi u okviru pipeline-a

### Opis jupyter notebook sveski, u redosledu u kojem treba da se čitaju:
1. interpretable_feature_reduction.ipynb sadrži eksploraciju podataka i interpretabilnu redukciju skupa atributa zasnovanu na uzajamnoj informaciji
2. feature_reduction.ipynb sadrži PCA, t-SNE i LDA transformacije skupa atributa
3. models_notebook.ipynb sadrži treniranje modela KNN, Decision Tree, SVM, Naive Bayes (GaussianNB) i Neural Networks (MLP) nad različitim skupovima atributa i uporedjivanje rezultata

## Resursi
Sve potrebne biblioteke zajedno za verzija se nalaze u requirements.txt. Preporuka za pokretanje projekta lokalno je pokretanje virtuelnog okruženja (venv) i instaliranje potrebnih resursa preko pip-a i potom pokretanje jupyter notebook servera iz python virtuelnog okruženja.

