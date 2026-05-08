"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""
import pandas as pd

# ===== CHARGER LES DONNEES =====

# recuperer les donnees sous forme json dans un fichier csv
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APER US =====
print("=" * 50)

print("SENSANTE - Exploration du dataset")

print("=" * 50)

# Dimensions du dataset

# Pour compter le nombre d'elements (enregistrements) recupérer depuis le fichier csv
print(f"\nNombre de patients : {len(df)}")

# Récuperer le nombre de colonne contenu dans la fichier csv grace a .shape ("age","sexe","temperature","tension_sys","toux","fatigue","maux_tete","frissons","nausee","region","diagnostic")
print(f"Nombre de colonnes : {df.shape [1]}")

# Lister les colonnes du fichier csv avec .columns et list() qui va transformer l'affiche en liste
print(f"Colonnes : {list(df.columns)}")

# Apercu des 5 premieres lignes
print(f"\n--- 5 premiers patients ---")

# Afficher l'entete des données (c'est à dire les noms des colonnes)
print(df.head())

# ===== STATISTIQUES DE BASE =====
print(f"\n--- Statistiques descriptives ---")

# .describe() sert a generer un résumé statistique d'un dataframe (patients_dakar.csv) (donc ne prend que les colonnes avec des nombres)
# .round(2) permet d'arrodir les nombres à 2 chiffres après la virgule
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics ---")

# permet de compter combien de fois chaque valeur apparait dans la colonne "diagnostic"
# df["diagnostic"] recuperer les elements de la colonne "diagnostic"
# .value_counts() compte le nombre d'occurence d'un element
diag_counts = df["diagnostic"].value_counts()

# .items() permet de recuperer les elements sous forme de paire (clé, valeur)
for diag, count in diag_counts.items():
    pct = count / len ( df ) * 100
    print(f"{diag:12s} : {count:3d} patients({pct:.1f}%)")

# ===== REPARTITION PAR REGION =====
print(f"\n--- Repartition par region (top 5) ---")

# Fait la meme chose qu'en haut mais pour "region" et ici avec .head(5) il recupere seulement 5 elements
region_counts = df["region"].value_counts().head(5)

for region, count in region_counts.items():
    print(f"{region:15s} : {count:3d} patients")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic ---")

# Avec .groupy()[] permet de regrouper des colonnes (mais ne peut pas etre utiliser seule il faut toujours une autre fonction (.mean(),.sum()))
# .mean permet de calculer la moyenne d'un element en fonction d'un autre (Exemple ici: on calcul la moyenne de la temperature en fonction du diagnostic)
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()

for diag, temp in temp_by_diag.items() :
    print(f"{diag:12s} : {temp:.1f} C")

print(f"\n{'= ' * 50}")

# ===== NOMBRE DE PATIENTS PAR SEXE ET DIAGNOSTIC =====
print(f"\n--- Nombre de patients par sexe et diagnostic ---")

# fait la meme chose que precedemment mais ici on compte le nombre de diagnostic par sexe (exemple : nombre de femme ayant la grippe,....)
sexe_diag_counts = df.groupby(["sexe", "diagnostic"]).size()

for (sexe, diag), count in sexe_diag_counts.items():
    print(f"Sexe: {sexe:6s} | Diagnostic: {diag:12s} : {count:3d} patients")

print("Exploration terminee !")

print("Prochain lab : entrainer un modele ML")

print(f"{'= ' * 50}")









""""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Verifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Encoder les variables categoriques en nombres

# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Definir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys','toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]

y = df['diagnostic']

print(f"Features : {X.shape}") 

print(f"Cible : {y.shape}")


# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,

    test_size=0.2, # 20% pour le test

    random_state=42, # Pour avoir les memes resultats a chaque fois

    stratify=y # Garder les memes proportions de diagnostics
)

print(f"Entrainement : {X_train.shape[0]} patients")

print(f"Test : {X_test.shape[0]} patients")


# Creer le modele
model = RandomForestClassifier(
    n_estimators=100, # 100 arbres de decision

    random_state=42 # Reproductibilite
)

# Entrainer sur les donnees d'entrainement
model.fit(X_train, y_train)

print("Modele entraine !")

print(f"Nombre d'arbres : {model.n_estimators}")

print(f"Nombre de features : {model.n_features_in_}")

print(f"Classes : {list(model.classes_)}")


# Predire sur les donnees de test
y_pred = model.predict(X_test)

# Comparer les 10 premieres predictions avec la realite
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],

    'Prediction': y_pred[:10]
})

print(comparison)

# Pourcentage de prediction
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.2%}")


# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Matrice de confusion :")

print(cm)

# Rapport de classification
print("\nRapport de classification :")

print(classification_report(y_test, y_pred))


# Visualiser avec seaborn
plt.figure(figsize=(8, 6))

sns.heatmap( cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_,
            yticklabels=model.classes_)

plt.xlabel('Prediction du modele')

plt.ylabel('Vrai diagnostic')

plt.title('Matrice de confusion- SenSante')

plt.tight_layout()

plt.savefig('figures/confusion_matrix.png', dpi=150)

plt.show()

print("Figure sauvegardee dans figures/confusion_matrix.png")


# Creer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Serialiser le modele
joblib.dump(model, "models/model.pkl")

# Verifier la taille du fichier
size = os.path.getsize("models/model.pkl")

print(f"Modele sauvegarde : models/model.pkl")

print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs (indispensables pour les nouvelles donnees)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")

joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features (pour reference)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")


# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded = joblib.load("models/model.pkl")

le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")

le_region_loaded = joblib.load("models/encoder_region.pkl")


print(f"Modele recharge : {type(model_loaded).__name__}")

print(f"Classes : {list(model_loaded.classes_)}")


# === 3 nouveaux patients ===

patients_test = [
    # 1. Jeune sans symptômes
    {
        'nom': 'Jeune sain',
        'age': 20,
        'sexe': 'M',
        'temperature': 36.7,
        'tension_sys': 120,
        'toux': False,
        'fatigue': False,
        'maux_tete': False,
        'region': 'Dakar'
    },

    # 2. Adulte avec forte fièvre
    {
        'nom': 'Forte fievre',
        'age': 35,
        'sexe': 'F',
        'temperature': 40.0,
        'tension_sys': 115,
        'toux': True,
        'fatigue': True,
        'maux_tete': True,
        'region': 'Dakar'
    },

    # 3. Patient âgé avec toux
    {
        'nom': 'Age avec toux',
        'age': 70,
        'sexe': 'M',
        'temperature': 37.8,
        'tension_sys': 130,
        'toux': True,
        'fatigue': True,
        'maux_tete': False,
        'region': 'Dakar'
    }
]


# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
'age': 28,
'sexe': 'F',
'temperature': 39.5,
'tension_sys': 110,
'toux': True,
'fatigue': True,
'maux_tete': True,
'region': 'Dakar'
}

# Encoder les valeurs categoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]

region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Preparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Predire
diagnostic = model_loaded.predict([features])[0]

probas = model_loaded.predict_proba([features])[0]

proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic---")

print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")

print(f"Diagnostic : {diagnostic}")

print(f"Probabilite : {proba_max:.1%}")

print(f"\nProbabilites par classe :")

for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f" {classe:8s} : {proba:.1%} {bar}")






importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f" {name:20s} : {imp:.3f}")
"""