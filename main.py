import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import os

# Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

# Loaders
from sklearn.datasets import fetch_covtype, fetch_openml

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay

# Suprimir avisos de convergência
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- CONFIGURAÇÃO ---
CREDITCARD_CSV_PATH = "creditcard.csv"
PLOT_DIR = "plots" # Pasta para salvar as imagens

# Modelos a serem testados
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (Depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

# --- FUNÇÕES HELPER ---
def evaluate_model(clf, X_train, y_train, X_test, y_test, model_name, dataset_name):
    """Treina, avalia, imprime métricas e salva a matriz de confusão."""
    print(f"\n--- Avaliando {model_name} no {dataset_name} ---")
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_test)
    
    # Métricas
    print(f"Tempo de Treino: {train_time:.4f}s")
    print("Relatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    try:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except AttributeError:
        print("ROC-AUC Score: N/A (modelo não suporta predict_proba)")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão: {model_name}")
    
    filename = os.path.join(PLOT_DIR, f"{dataset_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_CM.png")
    plt.savefig(filename)
    print(f"Matriz de Confusão salva em: {filename}")
    plt.close() # Fecha a figura para liberar memória

def plot_decision_boundaries(X_train_2f, y_train_2f, X_test_2f, y_test_2f, models, dataset_name, feature_names):
    """Treina modelos em 2 features e salva suas fronteiras de decisão."""
    
    print(f"\n--- Plotando Fronteiras de Decisão (2 Features) para {dataset_name} ---")
    print(f"Features usadas: {feature_names[0]} vs {feature_names[1]}")

    plot_idx = np.random.choice(X_train_2f.shape[0], size=min(1000, X_train_2f.shape[0]), replace=False)
    X_plot, y_plot = X_train_2f[plot_idx], y_train_2f.iloc[plot_idx]

    fig_size = (len(models) * 5, 5)
    fig, axes = plt.subplots(1, len(models), figsize=fig_size, squeeze=False)
    
    for ax, (model_name, clf) in zip(axes.flat, models.items()):
        
        pipeline_2f = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])
        
        start_time = time.time()
        pipeline_2f.fit(X_train_2f, y_train_2f)
        print(f"Treino (2f) {model_name} levou {time.time() - start_time:.3f}s")
        
        score = pipeline_2f.score(X_test_2f, y_test_2f)

        DecisionBoundaryDisplay.from_estimator(
            pipeline_2f,
            X_plot,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            alpha=0.6,
            xlabel=feature_names[0],
            ylabel=feature_names[1],
        )

        ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k', s=20, alpha=0.7)
        ax.set_title(f"{model_name}\nAcc (2f): {score:.3f}")

    plt.suptitle(f"Fronteiras de Decisão - {dataset_name}", fontsize=16, y=1.05)
    plt.tight_layout()
    
    filename = os.path.join(PLOT_DIR, f"{dataset_name.replace(' ', '_')}_Fronteiras.png")
    plt.savefig(filename)
    print(f"Gráfico de Fronteiras salvo em: {filename}")
    plt.close() # Fecha a figura para liberar memória

# --- ANÁLISE 1: COVERTYPE ---
def run_covertype_analysis(models):
    print("\n" + "="*80)
    print("Iniciando Análise: 1) Covertype (Florestas)")
    print("="*80)

    cov = fetch_covtype(as_frame=True)
    X = cov.data
    y_mult = cov.target
    
    y = (y_mult == 1).astype(int)
    
    print(f"Covertype Carregado: {X.shape} | Positivos (Classe 1): {y.sum()} | Negativos: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, clf in models.items():
        evaluate_model(clf, X_train_scaled, y_train, X_test_scaled, y_test, model_name, "Covertype")
    
    features_to_plot = [X.columns[0], X.columns[1]]
    X_train_2f = X_train[features_to_plot].values
    X_test_2f = X_test[features_to_plot].values
    
    plot_decision_boundaries(X_train_2f, y_train, X_test_2f, y_test, models, "Covertype", features_to_plot)


# --- ANÁLISE 2: ADULT / CENSUS INCOME ---
def run_adult_analysis(models):
    print("\n" + "="*80)
    print("Iniciando Análise: 2) Adult / Census Income")
    print("="*80)

    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame
    
    X = df.drop(columns=["class"])
    y = (df["class"] == ">50K").astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder='passthrough'
    )
    
    X_train_proc = preprocess.fit_transform(X_train)
    X_test_proc = preprocess.transform(X_test)
    
    print(f"Adult Carregado: {X.shape} -> Após OHE/Scale (Treino): {X_train_proc.shape}")

    for model_name, clf in models.items():
        evaluate_model(clf, X_train_proc, y_train, X_test_proc, y_test, model_name, "Adult")

    features_to_plot = ['age', 'hours-per-week']
    
    X_train_2f = X_train[features_to_plot].values
    X_test_2f = X_test[features_to_plot].values

    imputer_2f = SimpleImputer(strategy='median')
    X_train_2f = imputer_2f.fit_transform(X_train_2f)
    X_test_2f = imputer_2f.transform(X_test_2f)
    
    plot_decision_boundaries(X_train_2f, y_train, X_test_2f, y_test, models, "Adult", features_to_plot)


# --- ANÁLISE 3: CREDIT CARD FRAUD ---
def run_creditcard_analysis(models):
    print("\n" + "="*80)
    print("Iniciando Análise: 3) Credit Card Fraud")
    print("="*80)
    
    try:
        df = pd.read_csv(CREDITCARD_CSV_PATH)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{CREDITCARD_CSV_PATH}'")
        return

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    
    print(f"CreditCard Carregado: {X.shape} | Fraudes (1): {y.sum()} | Não-Fraudes (0): {(y==0).sum()}")
    print("ATENÇÃO: Dataset extremamente desbalanceado. Focar em Recall e ROC-AUC.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_balanced = {}
    for name, model in models.items():
        models_balanced[name] = model

    models_balanced["Logistic Regression"] = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
    models_balanced["Random Forest (100 Trees)"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    print("\n(Usando 'class_weight=balanced' para LR e RF devido ao desbalanceamento)")
    
    for model_name, clf in models_balanced.items():
        evaluate_model(clf, X_train_scaled, y_train, X_test_scaled, y_test, model_name, "Credit Card")

    features_to_plot = ['V1', 'V2']
    X_train_2f = X_train[features_to_plot].values
    X_test_2f = X_test[features_to_plot].values
    
    plot_decision_boundaries(X_train_2f, y_train, X_test_2f, y_test, models_balanced, "Credit Card", features_to_plot)

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Cria o diretório de plots se não existir
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    run_covertype_analysis(models)
    run_adult_analysis(models)
    run_creditcard_analysis(models)
    print("\n" + "="*80)
    print(f"Análises Concluídas. Gráficos salvos em '{PLOT_DIR}'.")
    print("="*80)