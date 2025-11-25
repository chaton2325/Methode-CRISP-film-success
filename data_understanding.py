import pandas

#Importation du dataset
df = pandas.read_csv('original dataset/movie_dataset.csv')

#Affichage des attributs du dataset
print('attributes in the dataset: ', df.columns.tolist())

#Affichage du nombre de lignes dans le dataset
print('Nombre de lignes dans le dataset: ', len(df))