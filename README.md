# Analyse des Prix de Voitures - Projet ML

## Description
Ce projet utilise le machine learning pour analyser et prédire les prix des voitures en se basant sur diverses caractéristiques comme la marque, le modèle, l'année, la taille du moteur, etc. Il comprend à la fois une analyse exploratoire des données et un modèle de prédiction.

## Dataset
Le jeu de données "Car Price Dataset" provient de Kaggle et contient 10 000 entrées avec les caractéristiques suivantes :
- Brand : Marque du véhicule
- Model : Modèle du véhicule
- Year : Année de fabrication
- Engine_Size : Taille du moteur
- Fuel_Type : Type de carburant
- Transmission : Type de transmission
- Mileage : Kilométrage
- Doors : Nombre de portes
- Owner_Count : Nombre de propriétaires précédents
- Price : Prix du véhicule (variable cible)

## Structure du Projet
1. **Prétraitement des données**
   - Séparation features/target
   - Split train/test
   - One-hot encoding pour les variables catégorielles
   - Standardisation des variables numériques

2. **Modélisation**
   - Modèle : Réseau de neurones (Deep Learning)
   - Architecture : 5 couches denses (52-30-15-5-1 neurones)
   - Optimiseur : Adam avec learning rate adaptatif
   - Callbacks : ReduceLROnPlateau et EarlyStopping

3. **Analyse d'Impact des Features**
   - Analyse des corrélations pour les variables numériques
   - Évaluation de l'impact des variables catégorielles
   - Visualisations des influences positives et négatives

4. **Interface Web**
   - Une interface web simple et épurée a été ajoutée pour permettre aux utilisateurs de saisir les caractéristiques de leur voiture et obtenir une estimation du prix en temps réel.
   - L'interface web est construite avec Flask et permet aux utilisateurs de sélectionner les valeurs pour les caractéristiques catégorielles (marque, modèle, transmission, type de carburant) et de saisir les valeurs pour les caractéristiques numériques (année, taille du moteur, etc.).

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
1. Charger et prétraiter les données :
```python
df = pd.read_csv('car_price_dataset.csv')
```

2. Préparer les données :
```python
X = df.drop(columns=["Price"])
y = df["Price"]
```

3. Entraîner le modèle :
```python
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), 
          callbacks=[reduce_lr, early_stop])
```

4. Analyser l'impact des features :
```python
# Voir le code d'analyse d'impact pour les détails
```

## Résultats
Le projet permet de :
- Comprendre les facteurs qui influencent le plus le prix des voitures
- Identifier les caractéristiques ayant un impact positif ou négatif sur le prix
- Prédire le prix d'une voiture en fonction de ses caractéristiques

## Améliorations Possibles
1. Implémenter d'autres algorithmes (Random Forest, XGBoost, etc.)

## Auteur
Berthod Guillaume

