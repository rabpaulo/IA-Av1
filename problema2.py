import numpy as np

data = np.loadtxt('dose_radiacao_expandido.csv', delimiter=',', skiprows=1)

print(data)

y = data[:, 1]
X = data[:, 2:]


class LinearRegression:
    def __init__(self, X, y, intercept=True): # terceiro parametro para a resolucao do item f
        self.X = np.array(X)
        self.y = np.array(y)
        self.intercept = intercept
        self.beta = None

    def fit(self):
        if self.intercept:
            X_mat = np.column_stack((np.ones(self.X.shape[0]), self.X))
        else:
            X_mat = self.X

        # Equação Normal
        self.beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ self.y
        return self

    def predict(self, X_new):
        X_new = np.array(X_new)
        
        if self.intercept:
            if X_new.ndim == 1:
                X_new = np.insert(X_new, 0, 1)
            else:
                X_new = np.column_stack((np.ones(X_new.shape[0]), X_new))
        
        return X_new @ self.beta


def calculate_metrics(y_real, y_pred, p):
    n = len(y_real)
    mse = np.mean((y_real - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_real - y_pred))
    
    ss_res = np.sum((y_real - y_pred)**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {"RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2, "R2_adj": r2_adj}


model_full = LinearRegression(X, y).fit()
beta_full = model_full.beta

print(f"(a) Coeficientes (Modelo Completo): Intercepto={beta_full[0]:.4f}, mAmp={beta_full[1]:.4f}, Tempo={beta_full[2]:.4f}")


y_hat_b = model_full.predict([15, 5])
print(f"(b) Previsão para 15mA e 5min: {y_hat_b:.2f} rad")


y_pred_full = model_full.predict(X)
metrics_full = calculate_metrics(y, y_pred_full, p=2)

print(f"(c) R2 Score: {metrics_full['R2']:.4f}")


print(f"(d) R2 Ajustado: {metrics_full['R2_adj']:.4f}")


X_mAmp = data[:, 1:2]

model_mAmp = LinearRegression(X_mAmp, y).fit()
y_pred_mAmp = model_mAmp.predict(X_mAmp)

metrics_mAmp = calculate_metrics(y, y_pred_mAmp, p=1)


model_zero = LinearRegression(X, y, intercept=False).fit()
y_pred_zero = model_zero.predict(X)

metrics_zero = calculate_metrics(y, y_pred_zero, p=2)


print(f"Modelo Completo - RMSE: {metrics_full['RMSE']:.4f}, MAE: {metrics_full['MAE']:.4f}")
print(f"Modelo Apenas Corrente - RMSE: {metrics_mAmp['RMSE']:.4f}, R2: {metrics_mAmp['R2']:.4f}")
print(f"Modelo Intercepto Zero - RMSE: {metrics_zero['RMSE']:.4f}, R2: {metrics_zero['R2']:.4f}")
