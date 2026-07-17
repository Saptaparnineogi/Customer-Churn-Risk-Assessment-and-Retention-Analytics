from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.preprocessing import get_feature_types, build_preprocessor
from src.model_training import get_models, train_models
from src.evaluation import evaluate_models


def main():

    df = load_data()

    df = engineer_features(df)

    X = df.drop(columns="Churn")
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    num_features, cat_features = get_feature_types(X_train)

    preprocessor = build_preprocessor(
        num_features,
        cat_features,
    )

    models = get_models()

    trained_models = train_models(
        preprocessor,
        models,
        X_train,
        y_train,
    )

    evaluate_models(
        trained_models,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()