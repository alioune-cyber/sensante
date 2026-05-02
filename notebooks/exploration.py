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