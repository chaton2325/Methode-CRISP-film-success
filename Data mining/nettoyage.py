#1) Suppression des colonnes inutiles (attributs qui n'aideront pas)
import pandas
df = pandas.read_csv('original dataset/movie_dataset.csv')
df = df.drop(columns=['index','homepage', 'id', 'keywords', 'overview', 'release_date', 'status', 'tagline', 'title', 'original_title', 'crew'])

output_file = 'phases de traitement dataset/movie_dataset_cleaned1.csv'
df.to_csv(output_file, index=False)


df1 = pandas.read_csv('phases de traitement dataset/movie_dataset_cleaned1.csv')
print('Nouvelles colonnes : ', df1.columns.tolist())