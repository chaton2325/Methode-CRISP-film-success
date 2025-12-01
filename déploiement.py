import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
from pathlib import Path


class FilmSuccessPredictorApp:
    """
    Application de prédiction de succès film basée sur K-Means clustering
    """
    
    def __init__(self, dataset_path='movie_dataset_cleaned_final.csv'):
        """
        Initialise l'application avec le dataset
        
        Parameters:
        -----------
        dataset_path : str
            Chemin vers le fichier CSV du dataset
        """
        self.dataset_path = dataset_path
        self.df = None
        self.X_normalized = None
        self.scaler = None
        self.kmeans_model = None
        self.numeric_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
        self.df_km = None
        self.k_optimal = None
        
        print("\n" + "="*80)
        print("🎬 FILM SUCCESS PREDICTOR - Initialisation")
        print("="*80)
        
    def load_data(self):
        """Charge et prépare les données"""
        try:
            print(f"\n📂 Chargement du dataset: {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            print(f"✓ Dataset chargé: {len(self.df)} films")
            print(f"✓ Colonnes disponibles: {self.df.columns.tolist()}")
            
            # Sélectionner les données sans NaN
            self.df_km = self.df[self.numeric_cols + ['revenue']].dropna()
            print(f"✓ Films avec données complètes: {len(self.df_km)}")
            
            return True
        except Exception as e:
            print(f"✗ Erreur lors du chargement: {e}")
            return False
    
    def preprocess_data(self):
        """Normalise les données"""
        try:
            print(f"\n🔧 Normalisation des données (0-1)")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X = self.df_km[self.numeric_cols].values
            self.X_normalized = self.scaler.fit_transform(X)
            print(f"✓ Données normalisées: shape={self.X_normalized.shape}")
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la normalisation: {e}")
            return False
    
    def find_optimal_k(self):
        """Trouve le k optimal avec score de silhouette"""
        try:
            print(f"\n🔍 Recherche du k optimal (silhouette score)")
            silhouette_scores = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.X_normalized)
                score = silhouette_score(self.X_normalized, labels)
                silhouette_scores.append(score)
                print(f"  k={k}: silhouette_score={score:.3f}")
            
            self.k_optimal = K_range[np.argmax(silhouette_scores)]
            print(f"\n✓ K optimal trouvé: {self.k_optimal} (score={max(silhouette_scores):.3f})")
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la recherche du k optimal: {e}")
            return False
    
    def train_kmeans(self):
        """Entraîne le modèle K-means avec k optimal"""
        try:
            print(f"\n🤖 Entraînement du modèle K-Means (k={self.k_optimal})")
            self.kmeans_model = KMeans(n_clusters=self.k_optimal, random_state=42, n_init=10)
            self.df_km['Cluster'] = self.kmeans_model.fit_predict(self.X_normalized)
            print(f"✓ Modèle entraîné et clusters assignés")
            
            # Statistiques par cluster
            print(f"\n📊 Statistiques par cluster:")
            for cluster in range(self.k_optimal):
                cluster_data = self.df_km[self.df_km['Cluster'] == cluster]
                print(f"\n  Cluster {cluster}: {len(cluster_data)} films")
                print(f"    - Revenue moyen: ${cluster_data['revenue'].mean():,.0f}")
                print(f"    - Revenue médian: ${cluster_data['revenue'].median():,.0f}")
            
            return True
        except Exception as e:
            print(f"✗ Erreur lors de l'entraînement: {e}")
            return False
    
    def evaluate_film(self, budget, popularity, runtime, vote_average, vote_count):
        """
        Évalue un film et prédiction de succès
        
        Parameters:
        -----------
        budget, popularity, runtime, vote_average, vote_count : float
            Caractéristiques du film
        
        Returns:
        --------
        dict : Résultats de la prédiction
        """
        try:
            # Créer l'instance utilisateur
            user_film = np.array([[budget, popularity, runtime, vote_average, vote_count]])
            user_film_normalized = self.scaler.transform(user_film)
            
            # Prédire le cluster
            cluster_pred = self.kmeans_model.predict(user_film_normalized)[0]
            
            # Distance au centre du cluster pour l'utilisateur
            user_distance = np.linalg.norm(
                user_film_normalized[0] - self.kmeans_model.cluster_centers_[cluster_pred]
            )
            
            # Distance au centre du cluster pour tous les points du dataset
            distances = np.linalg.norm(
                self.X_normalized - self.kmeans_model.cluster_centers_[cluster_pred],
                axis=1
            )
            
            # Calcul de la probabilité appartenance au cluster
            # (basé sur les distances dans le cluster)
            cluster_distances = distances[self.df_km['Cluster'] == cluster_pred]
            confidence = max(0, 1 - (user_distance / (cluster_distances.max() + 1)))
            
            # Évaluer le succès basé sur les statistiques du cluster
            cluster_data = self.df_km[self.df_km['Cluster'] == cluster_pred]
            avg_revenue = cluster_data['revenue'].mean()
            median_revenue = cluster_data['revenue'].median()
            
            # Déterminer le succès
            success_threshold = self.df_km['revenue'].median()
            predicted_success = avg_revenue >= success_threshold
            
            return {
                'cluster': cluster_pred,
                'confidence': confidence,
                'avg_revenue_cluster': avg_revenue,
                'median_revenue_cluster': median_revenue,
                'success': predicted_success,
                'user_film_normalized': user_film_normalized,
                'cluster_center': self.kmeans_model.cluster_centers_[cluster_pred],
                'cluster_films': cluster_data
            }
        except Exception as e:
            print(f"✗ Erreur lors de l'évaluation: {e}")
            return None
    
    def visualize_prediction(self, results, budget, popularity, runtime, vote_average, vote_count):
        """
        Visualise la prédiction avec le film utilisateur et les clusters
        
        Parameters:
        -----------
        results : dict
            Résultats de la prédiction
        budget, popularity, runtime, vote_average, vote_count : float
            Caractéristiques du film utilisateur
        """
        try:
            # Créer score combiné pour visualisation 2D (moyenne des variables normalisées)
            df_display = self.df_km.copy()
            df_display['Score_Combine'] = self.X_normalized.mean(axis=1)
            
            # Normaliser revenue pour visualisation (avec arrays numpy directement)
            revenue_values = df_display['revenue'].values.reshape(-1, 1)
            scaler_revenue = MinMaxScaler(feature_range=(0, 1))
            revenue_normalized = scaler_revenue.fit_transform(revenue_values).flatten()
            df_display['Revenue_Normalized'] = revenue_normalized
            
            # Score utilisateur - utiliser directement des arrays numpy
            user_array = np.array([[budget, popularity, runtime, vote_average, vote_count]])
            user_normalized = self.scaler.transform(user_array)
            user_score_combine = float(user_normalized.mean())
            
            # Calculer revenue approximatif utilisateur avec array numpy
            user_revenue_value = (budget + popularity + runtime + vote_average + vote_count) / 5
            user_revenue = np.array([[user_revenue_value]])
            user_revenue_norm = float(scaler_revenue.transform(user_revenue)[0][0])
            
            # Créer la visualisation
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Graphique 1: Tous les clusters avec le film utilisateur
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for cluster in range(self.k_optimal):
                cluster_data = df_display[df_display['Cluster'] == cluster]
                axes[0].scatter(cluster_data['Score_Combine'], 
                               cluster_data['Revenue_Normalized'],
                               label=f'Cluster {cluster}', 
                               s=50, alpha=0.6, c=colors[cluster % len(colors)])
            
            # Ajouter le film utilisateur
            axes[0].scatter(user_score_combine, user_revenue_norm, 
                           s=400, marker='*', c='gold', edgecolors='black', 
                           linewidth=2, label='Film utilisateur', zorder=10)
            
            axes[0].set_xlabel('Score Combiné (Moyenne variables 0-1)', fontsize=11)
            axes[0].set_ylabel('Revenue Normalisé (0-1)', fontsize=11)
            axes[0].set_title('Visualisation: Film Utilisateur vs Clusters K-Means', 
                             fontsize=12, fontweight='bold')
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Graphique 2: Zoom sur le cluster prédit
            cluster_pred = results['cluster']
            cluster_data = df_display[df_display['Cluster'] == cluster_pred]
            
            axes[1].scatter(cluster_data['Score_Combine'], 
                           cluster_data['Revenue_Normalized'],
                           s=50, alpha=0.6, c=colors[cluster_pred % len(colors)],
                           label=f'Cluster {cluster_pred}')
            
            # Centre du cluster (calculé à partir des points du cluster)
            center_score = cluster_data['Score_Combine'].mean()
            center_revenue = cluster_data['Revenue_Normalized'].mean()
            
            axes[1].scatter(center_score, center_revenue,
                           s=400, marker='X', c='red', edgecolors='black',
                           linewidth=2, label='Centre du cluster', zorder=9)
            
            # Film utilisateur
            axes[1].scatter(user_score_combine, user_revenue_norm, 
                           s=400, marker='*', c='gold', edgecolors='black', 
                           linewidth=2, label='Film utilisateur', zorder=10)
            
            axes[1].set_xlabel('Score Combiné (Moyenne variables 0-1)', fontsize=11)
            axes[1].set_ylabel('Revenue Normalisé (0-1)', fontsize=11)
            axes[1].set_title(f'Zoom: Cluster {cluster_pred} + Film Utilisateur', 
                             fontsize=12, fontweight='bold')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()
            print("\n✓ Visualisation affichée avec succès!")
            
        except Exception as e:
            print(f"✗ Erreur lors de la visualisation: {e}")
    
    def display_results(self, results, user_input):
        """Affiche les résultats de la prédiction"""
        print("\n" + "="*80)
        print("📋 RÉSULTATS DE LA PRÉDICTION")
        print("="*80)
        
        print(f"\n📥 Film entré par l'utilisateur:")
        print(f"  • Budget: ${user_input['budget']:,.0f}")
        print(f"  • Popularité: {user_input['popularity']:.2f}")
        print(f"  • Runtime: {user_input['runtime']:.0f} minutes")
        print(f"  • Vote Average: {user_input['vote_average']:.2f}/10")
        print(f"  • Vote Count: {user_input['vote_count']:,.0f}")
        
        print(f"\n🎯 Prédiction K-Means:")
        print(f"  • Cluster assigné: {results['cluster']}")
        print(f"  • Confiance: {results['confidence']*100:.1f}%")
        print(f"  • Revenue moyen du cluster: ${results['avg_revenue_cluster']:,.0f}")
        print(f"  • Revenue médian du cluster: ${results['median_revenue_cluster']:,.0f}")
        
        if results['success']:
            print(f"\n✅ PRÉDICTION: SUCCÈS POTENTIEL")
            print(f"   Le film appartient à un cluster à succès commercial")
        else:
            print(f"\n⚠️  PRÉDICTION: SUCCÈS MODÉRÉ")
            print(f"   Le film appartient à un cluster avec succès modéré")
        
        print(f"\n📊 Statistiques du cluster {results['cluster']}:")
        cluster_data = results['cluster_films']
        print(f"  • Nombre de films dans le cluster: {len(cluster_data)}")
        print(f"  • Revenue min: ${cluster_data['revenue'].min():,.0f}")
        print(f"  • Revenue max: ${cluster_data['revenue'].max():,.0f}")
        print(f"  • Revenue std: ${cluster_data['revenue'].std():,.0f}")
    
    def run_interactive_mode(self):
        """Mode interactif de l'application"""
        print("\n" + "="*80)
        print("🎬 ENTREZ LES CARACTÉRISTIQUES DE VOTRE FILM")
        print("="*80)
        
        try:
            budget = float(input("\n💰 Budget (en dollars): "))
            popularity = float(input("📈 Popularité (0-100): "))
            runtime = float(input("⏱️  Runtime (en minutes): "))
            vote_average = float(input("⭐ Vote Average (0-10): "))
            vote_count = float(input("🗳️  Vote Count (nombre de votes): "))
            
            user_input = {
                'budget': budget,
                'popularity': popularity,
                'runtime': runtime,
                'vote_average': vote_average,
                'vote_count': vote_count
            }
            
            # Prédire
            results = self.evaluate_film(budget, popularity, runtime, vote_average, vote_count)
            
            if results:
                # Afficher résultats
                self.display_results(results, user_input)
                
                # Visualiser
                print("\n📊 Génération de la visualisation...")
                self.visualize_prediction(results, budget, popularity, runtime, vote_average, vote_count)
            
        except ValueError as e:
            print(f"\n✗ Erreur: Veuillez entrer des nombres valides. {e}")
        except Exception as e:
            print(f"\n✗ Erreur lors du traitement: {e}")
    
    def run(self):
        """Lance l'application complète"""
        if not self.load_data():
            return False
        
        if not self.preprocess_data():
            return False
        
        if not self.find_optimal_k():
            return False
        
        if not self.train_kmeans():
            return False
        
        print("\n" + "="*80)
        print("✅ APPLICATION PRÊTE - Mode interactif activé")
        print("="*80)
        
        # Mode interactif
        while True:
            self.run_interactive_mode()
            
            another = input("\n\n🔄 Tester un autre film? (oui/non): ").lower()
            if another not in ['oui', 'o', 'yes', 'y']:
                print("\n👋 Merci d'avoir utilisé Film Success Predictor!")
                break


def main():
    """Point d'entrée principal"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "🎬 FILM SUCCESS PREDICTOR 🎬" + " "*31 + "║")
    print("║" + " "*15 + "Prédiction de succès commercial basée sur K-Means" + " "*15 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Créer l'application
    app = FilmSuccessPredictorApp(dataset_path='movie_dataset_cleaned_final.csv')
    
    # Lancer
    app.run()


if __name__ == "__main__":
    main()
