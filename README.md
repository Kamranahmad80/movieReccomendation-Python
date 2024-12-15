 
# **Movie Recommendation System**

## **Overview**
This project implements a movie recommendation system using the **MovieLens 20M Dataset**. It includes basic and advanced recommendation algorithms, ranging from cosine similarity and SVD to Neural Collaborative Filtering (NCF) and Recurrent Neural Networks (RNN). The system evaluates the models using various metrics, including MAE, RMSE, Precision, Recall, and F1-Score.

## **Features**
- **Content-Based Filtering**: Using Cosine Similarity to recommend movies based on attributes.
- **Collaborative Filtering**: Leveraging SVD to predict user preferences.
- **Advanced Models**:
  - Neural Collaborative Filtering (NCF) for learning user-movie interactions.
  - Recurrent Neural Network (RNN) for sequence-based recommendations.
- **Evaluation and Fine-Tuning**: Using metrics and hyperparameter optimization to improve performance.

---

## **Folder Structure**
```
movie-recommendation-system/
├── data/                # Contains dataset files (movies.csv, ratings.csv, etc.)
├── notebooks/           # Jupyter notebooks for different stages of development
├── outputs/             # Results, presentation slides, and reports
├── src/                 # Python scripts for utilities and models
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
```

---

## **Datasets**
The project uses the **MovieLens 20M Dataset**:
- `ratings.csv`: User ratings for movies.
- `movies.csv`: Movie metadata such as titles and genres.
- `tags.csv` and `links.csv`: Additional information for enrichment.

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

---

## **Setup Instructions**

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset files (`movies.csv`, `ratings.csv`, etc.) in the `data/` folder.

4. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   - Start with `notebooks/01_data_exploration.ipynb` for data analysis and preprocessing.
   - Proceed with `notebooks/02_basic_models.ipynb` for basic recommendation algorithms.
   - Use `notebooks/03_advanced_models.ipynb` for advanced models.

5. View outputs in the `outputs/` folder after running the notebooks.

---

## **Project Workflow**

### 1. Data Exploration & Preprocessing
- Load and clean the dataset.
- Perform exploratory data analysis (EDA) and visualize distributions.
- Feature engineering to extract relevant attributes.

### 2. Basic Models
- **Content-Based Filtering**: Cosine Similarity.
- **Collaborative Filtering**: Singular Value Decomposition (SVD).

### 3. Advanced Models
- **Neural Collaborative Filtering (NCF)**: Deep learning-based recommendation system.
- **Recurrent Neural Network (RNN)**: Capturing temporal patterns in user preferences.

### 4. Evaluation & Optimization
- Metrics: MAE, RMSE, Precision, Recall, F1-Score.
- Train-test splits: 70-30, 80-20, and 90-10.
- Hyperparameter tuning for optimal performance.

### 5. Deliverables
- Python code and Jupyter Notebooks.
- Final report in `outputs/`.
- Presentation slides summarizing the project.

---

## **Evaluation Metrics**
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Precision & Recall**
- **F1-Score**
- **Accuracy**

---

## **Challenges**
- Large dataset handling and preprocessing.
- Hyperparameter tuning for advanced models.
- Balancing precision and recall in recommendations.

---

## **Contributors**
- **Kamran Ahmad Khan**
  - *Role*: Developer and Project Lead
  - *Email*: kamranahmadkhan110@gmail.com
  - *GitHub*: [github.com/Kamranahmad80](https://github.com/Kamranahmad80)
