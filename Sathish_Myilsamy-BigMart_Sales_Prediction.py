import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42

# 1. Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = np.log1p(train["Item_Outlet_Sales"])
train.drop("Item_Outlet_Sales", axis=1, inplace=True)

# 2. Create Item_Type_Combined early
for df in [train, test]:
    df["Item_Type_Combined"] = df["Item_Identifier"].str[:2].map({
        "FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"
    })

# 3. Combine for preprocessing
data = pd.concat([train, test], axis=0, ignore_index=True)

# 4. Cleaning
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({
    "LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"
})
data["Item_Weight"] = data.groupby("Item_Identifier")["Item_Weight"] \
    .transform(lambda x: x.fillna(x.mean()))
data["Item_Weight"].fillna(data["Item_Weight"].mean(), inplace=True)

outlet_size_mode = data.groupby("Outlet_Type")["Outlet_Size"].agg(lambda x: x.mode().iloc[0])
data["Outlet_Size"] = data.apply(lambda row: outlet_size_mode[row["Outlet_Type"]] 
                                 if pd.isna(row["Outlet_Size"]) else row["Outlet_Size"], axis=1)

data["Item_Visibility"] = data["Item_Visibility"].replace(0, np.nan)
data["Item_Visibility"] = data.groupby("Item_Identifier")["Item_Visibility"] \
    .transform(lambda x: x.fillna(x.mean()))
data["Item_Visibility"].fillna(data["Item_Visibility"].mean(), inplace=True)

# 5. Feature Engineering
data["Outlet_Age"] = 2025 - data["Outlet_Establishment_Year"]
data["Item_MRP_Tier"] = pd.cut(data["Item_MRP"], bins=[0,69,136,203,270],
                               labels=["Low","Medium","High","Very High"])
data["Price_per_Weight"] = data["Item_MRP"] / (data["Item_Weight"] + 1)
data["Visibility_log"] = np.log1p(data["Item_Visibility"])
data["MRP_log"] = np.log1p(data["Item_MRP"])
data["MRP_by_Age"] = data["Item_MRP"] / (data["Outlet_Age"] + 1)

# Target encoding features
train_target = pd.DataFrame({
    "Item_Identifier": train["Item_Identifier"],
    "Outlet_Identifier": train["Outlet_Identifier"],
    "Item_Type_Combined": train["Item_Type_Combined"],
    "y": np.expm1(y)
})
item_mean = train_target.groupby("Item_Identifier")["y"].mean().rename("Mean_Item_Sales")
outlet_mean = train_target.groupby("Outlet_Identifier")["y"].mean().rename("Mean_Outlet_Sales")
type_mean = train_target.groupby("Item_Type_Combined")["y"].mean().rename("Mean_Type_Sales")

data = data.merge(item_mean, on="Item_Identifier", how="left")
data = data.merge(outlet_mean, on="Outlet_Identifier", how="left")
data = data.merge(type_mean, on="Item_Type_Combined", how="left")

for col in ["Mean_Item_Sales", "Mean_Outlet_Sales", "Mean_Type_Sales"]:
    data[col].fillna(data[col].mean(), inplace=True)

# 6. Split Back
train_clean = data.iloc[:len(y), :]
test_clean = data.iloc[len(y):, :]

# Ensure correct dtypes for preprocessing
categorical_cols = train_clean.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = train_clean.select_dtypes(exclude=["object"]).columns.tolist()

for col in categorical_cols:
    train_clean[col] = train_clean[col].fillna("missing").astype(str)
    test_clean[col] = test_clean[col].fillna("missing").astype(str)

for col in numeric_cols:
    train_clean[col] = pd.to_numeric(train_clean[col], errors="coerce")
    test_clean[col] = pd.to_numeric(test_clean[col], errors="coerce")

# 7. Preprocessor
preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
])

X_all_enc = preprocessor.fit_transform(train_clean)
X_test_enc = preprocessor.transform(test_clean)

# 8. Model parameters
xgb_params = dict(
    n_estimators=800, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    eval_metric="rmse"
)
lgb_params = dict(
    n_estimators=800, learning_rate=0.05, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
)
cat_params = dict(
    iterations=800, learning_rate=0.05, depth=8,
    silent=True, random_state=RANDOM_STATE
)

# 9. 5-Fold OOF Stacking
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ridge_oof = np.zeros((len(train_clean), 3))
ridge_test_folds = np.zeros((len(test_clean), 3, 5))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all_enc)):
    X_tr, X_va = X_all_enc[tr_idx], X_all_enc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr)
    
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr.toarray(), y_tr)  # LGBM needs dense
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr.toarray(), y_tr, verbose=False)  # CatBoost needs dense
    
    ridge_oof[va_idx, 0] = xgb_model.predict(X_va)
    ridge_oof[va_idx, 1] = lgb_model.predict(X_va.toarray())
    ridge_oof[va_idx, 2] = cat_model.predict(X_va.toarray())
    
    ridge_test_folds[:, 0, fold] = xgb_model.predict(X_test_enc)
    ridge_test_folds[:, 1, fold] = lgb_model.predict(X_test_enc.toarray())
    ridge_test_folds[:, 2, fold] = cat_model.predict(X_test_enc.toarray())

ridge_test_mean = ridge_test_folds.mean(axis=2)

# 10. Ridge Meta learner
meta_model = Ridge(alpha=1.0)
meta_model.fit(ridge_oof, y)
final_preds_log = meta_model.predict(ridge_test_mean)
final_preds = np.expm1(final_preds_log)

# 11. Save Submission
submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"],
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")
