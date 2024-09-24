from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# 加载训练好的模型
model = joblib.load('D:\我的文档\我的文件\pythonProject\一键操作集合\cholestasis_risk_model.pkl')

# 创建Flask应用
app = Flask(__name__)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 计算胆汁淤积风险
@app.route('/predict', methods=['POST'])
def predict():
    # 获取表单提交的特征值
    features = [float(request.form[feature]) for feature in [
        'week_2_amino_acid_dose', 'fasting_time', 'week_1_fat_emulsion_dose',
        'gestational_age', 'ast', 'total_bilirubin', 'alkaline_phosphatase',
        'platelet_count', 'ggt', 'white_blood_cell_count', 'mother_age'
    ]]

    # 将特征转为numpy数组，进行预测
    input_data = np.array([features])
    prediction = model.predict_proba(input_data)[0, 1]  # 获取风险概率

    # 将结果返回到网页上
    return render_template('result.html', probability=prediction)

if __name__ == '__main__':
    app.run(debug=True)
