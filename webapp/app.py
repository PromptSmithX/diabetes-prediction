"""
Diabetes Risk Assessment Web App
Flask backend — loads calibrated XGBoost model, handles imputation, returns predictions + advice.
"""

import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ─── ManualCalibratedModel ────────────────────────────────────────────────────
# Must be defined here so pickle can deserialize the saved .pkl file.
class ManualCalibratedModel:
    def __init__(self, base_model, method='sigmoid'):
        self.base_model  = base_model
        self.method      = method
        self._calibrator = None

    def fit(self, X_cal, y_cal):
        raw = self.base_model.predict_proba(X_cal)[:, 1]
        if self.method == 'sigmoid':
            self._calibrator = LogisticRegression(solver='lbfgs', C=1e10, max_iter=1000)
            self._calibrator.fit(raw.reshape(-1, 1), y_cal)
        else:
            self._calibrator = IsotonicRegression(out_of_bounds='clip')
            self._calibrator.fit(raw, y_cal)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.method == 'sigmoid':
            cal = self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            cal = np.clip(self._calibrator.predict(raw), 0, 1)
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'processed_data')

app = Flask(__name__)

# ─── Load model & metadata ────────────────────────────────────────────────────
print("Loading model...")
with open(os.path.join(MODELS_DIR, 'xgboost_calibrated.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'model_metadata.json')) as f:
    META = json.load(f)

FEATURES = META['features']

# ─── Imputation values (median for continuous/ordinal, mode for binary) ───────
# Computed from training set — used when user leaves a field blank
IMPUTE_VALUES = {
    # Binary  → mode
    'HighBP':               1.0,
    'HighChol':             1.0,
    'CholCheck':            1.0,
    'Smoker':               0.0,
    'Stroke':               0.0,
    'HeartDiseaseorAttack': 0.0,
    'PhysActivity':         1.0,
    'Fruits':               1.0,
    'Veggies':              1.0,
    'HvyAlcoholConsump':    0.0,
    'AnyHealthcare':        1.0,
    'NoDocbcCost':          0.0,
    'DiffWalk':             0.0,
    'Sex':                  0.0,
    # Continuous → median
    'BMI':                  29.0,
    'MentHlth':             0.0,
    'PhysHlth':             0.0,
    # Ordinal → median
    'GenHlth':              3.0,
    'Age':                  9.0,
    'Education':            5.0,
    'Income':               6.0,
}

BINARY_FEATURES = [
    'HighBP','HighChol','CholCheck','Smoker','Stroke',
    'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
    'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex'
]

AGE_LABELS = {
    1:'18–24', 2:'25–29', 3:'30–34', 4:'35–39', 5:'40–44',
    6:'45–49', 7:'50–54', 8:'55–59', 9:'60–64', 10:'65–69',
    11:'70–74', 12:'75–79', 13:'80+'
}
GEN_HEALTH = {1:'Xuất sắc',2:'Rất tốt',3:'Tốt',4:'Bình thường',5:'Kém'}


# ─── Feature Engineering (phải khớp Notebook 1) ───────────────────────────────
def add_engineered_features(p: dict) -> dict:
    d = p.copy()
    bmi = d['BMI']
    d['BMI_Cat'] = (0 if bmi < 18.5 else 1 if bmi < 25 else
                    2 if bmi < 30 else 3 if bmi < 35 else
                    4 if bmi < 40 else 5)
    d['CardioRisk']         = (d['HighBP'] + d['HighChol'] +
                                d['HeartDiseaseorAttack'] + d['Stroke'] + d['DiffWalk'])
    d['HealthyScore']       = (d['PhysActivity'] + d['Fruits'] + d['Veggies'] +
                                (1 - d['Smoker']) + (1 - d['HvyAlcoholConsump']))
    d['TotalBadHealthDays'] = d['MentHlth'] + d['PhysHlth']
    d['BMI_Age']            = d['BMI'] * d['Age']
    age = d['Age']
    d['AgeGroup']           = (0 if age <= 3 else 1 if age <= 6 else
                                2 if age <= 9 else 3)
    d['SocioIndex']         = d['Income'] + d['Education']
    return d


# ─── Rule-based advice engine ─────────────────────────────────────────────────
def generate_advice(p: dict, prob: float) -> list:
    """Generate personalized rule-based recommendations."""
    advices = []

    bmi = float(p.get('BMI', 25))
    if bmi >= 40:
        advices.append({'icon':'⚖️','cat':'Cân nặng (BMI)',
            'detail': f'BMI = {bmi:.1f} — Béo phì độ III. Nguy cơ cực cao.',
            'action': 'Cần can thiệp y tế ngay về kế hoạch giảm cân toàn diện.',
            'priority': 1})
    elif bmi >= 35:
        advices.append({'icon':'⚖️','cat':'Cân nặng (BMI)',
            'detail': f'BMI = {bmi:.1f} — Béo phì độ II.',
            'action': 'Tư vấn chuyên gia dinh dưỡng, giảm cân có kiểm soát.',
            'priority': 1})
    elif bmi >= 30:
        advices.append({'icon':'⚖️','cat':'Cân nặng (BMI)',
            'detail': f'BMI = {bmi:.1f} — Béo phì độ I.',
            'action': 'Tập thể dục ≥150 phút/tuần, chế độ ăn ít tinh bột đường.',
            'priority': 2})
    elif bmi >= 25:
        advices.append({'icon':'⚖️','cat':'Cân nặng (BMI)',
            'detail': f'BMI = {bmi:.1f} — Thừa cân.',
            'action': 'Tăng vận động, kiểm soát khẩu phần ăn.',
            'priority': 3})

    if float(p.get('HighBP', 0)) == 1:
        advices.append({'icon':'🩸','cat':'Huyết áp cao',
            'detail': 'Huyết áp cao làm tăng đề kháng insulin.',
            'action': 'Theo dõi huyết áp thường xuyên, giảm muối < 5g/ngày, tránh stress.',
            'priority': 1})

    if float(p.get('HighChol', 0)) == 1:
        advices.append({'icon':'🧪','cat':'Cholesterol cao',
            'detail': 'Rối loạn lipid máu liên quan chặt với kháng insulin.',
            'action': 'Xét nghiệm lipid máu, giảm chất béo bão hòa, tăng chất xơ.',
            'priority': 2})

    if float(p.get('PhysActivity', 1)) == 0:
        advices.append({'icon':'🏃','cat':'Thiếu vận động',
            'detail': 'Ít vận động thể chất làm tăng nguy cơ kháng insulin.',
            'action': 'Mục tiêu 30 phút/ngày: đi bộ nhanh, bơi lội, đạp xe.',
            'priority': 2})

    if float(p.get('Smoker', 0)) == 1:
        advices.append({'icon':'🚬','cat':'Hút thuốc lá',
            'detail': 'Hút thuốc tăng viêm mạn tính và đề kháng insulin.',
            'action': 'Bỏ thuốc lá ngay — tham khảo chương trình hỗ trợ cai thuốc.',
            'priority': 2})

    if float(p.get('HeartDiseaseorAttack', 0)) == 1:
        advices.append({'icon':'❤️','cat':'Bệnh tim mạch',
            'detail': 'Tiền sử bệnh tim/nhồi máu — hội chứng chuyển hóa nguy cơ rất cao.',
            'action': 'Theo dõi tim mạch và đường huyết đồng thời, khám định kỳ.',
            'priority': 1})

    if float(p.get('Stroke', 0)) == 1:
        advices.append({'icon':'🧠','cat':'Tiền sử đột quỵ',
            'detail': 'Đột quỵ liên quan chặt với rối loạn chuyển hóa glucose.',
            'action': 'Kiểm soát đường huyết, huyết áp, cholesterol đồng thời.',
            'priority': 1})

    gen = int(float(p.get('GenHlth', 3)))
    if gen >= 4:
        advices.append({'icon':'💊','cat':'Sức khỏe tổng quát',
            'detail': f'Sức khỏe tự đánh giá: {GEN_HEALTH.get(gen)} — cần cải thiện.',
            'action': 'Khám sức khỏe toàn diện, xét nghiệm máu đầy đủ.',
            'priority': 1})

    if float(p.get('HvyAlcoholConsump', 0)) == 1:
        advices.append({'icon':'🍺','cat':'Rượu bia nhiều',
            'detail': 'Uống rượu nặng ảnh hưởng chức năng gan và điều tiết đường huyết.',
            'action': 'Nam ≤ 2 đơn vị/ngày, Nữ ≤ 1 đơn vị/ngày.',
            'priority': 2})

    phys = float(p.get('PhysHlth', 0))
    if phys > 14:
        advices.append({'icon':'🤒','cat':'Sức khỏe thể chất',
            'detail': f'{int(phys)} ngày sức khỏe thể chất kém trong tháng.',
            'action': 'Tìm nguyên nhân triệu chứng mạn tính, tư vấn bác sĩ.',
            'priority': 2})

    ment = float(p.get('MentHlth', 0))
    if ment > 14:
        advices.append({'icon':'🧠','cat':'Sức khỏe tinh thần',
            'detail': f'{int(ment)} ngày sức khỏe tinh thần kém trong tháng — cortisol cao kéo dài.',
            'action': 'Quản lý stress: thiền định, yoga, tư vấn tâm lý.',
            'priority': 3})

    age = int(float(p.get('Age', 1)))
    if age >= 8:
        advices.append({'icon':'👴','cat':'Nhóm tuổi cao',
            'detail': f'Nhóm tuổi {AGE_LABELS.get(age, "")} — nguy cơ TĐ tăng tự nhiên.',
            'action': 'Xét nghiệm đường huyết lúc đói định kỳ ít nhất 1 năm/lần.',
            'priority': 3})

    if float(p.get('DiffWalk', 0)) == 1:
        advices.append({'icon':'🦿','cat':'Hạn chế vận động',
            'detail': 'Khó đi bộ/leo cầu thang giới hạn khả năng tập thể dục.',
            'action': 'Bài tập phù hợp: bơi lội, ghế tập, vật lý trị liệu.',
            'priority': 2})

    if float(p.get('Fruits', 1)) == 0 and float(p.get('Veggies', 1)) == 0:
        advices.append({'icon':'🥗','cat':'Chế độ ăn uống',
            'detail': 'Không ăn đủ trái cây và rau xanh — thiếu chất xơ và vi chất.',
            'action': 'Bổ sung ≥ 5 phần rau/trái cây mỗi ngày, hạn chế đường tinh luyện.',
            'priority': 3})

    if not advices:
        advices.append({'icon':'✅','cat':'Lối sống lành mạnh',
            'detail': 'Các chỉ số sức khỏe của bạn nhìn chung tốt.',
            'action': 'Duy trì lối sống hiện tại và khám định kỳ để theo dõi.',
            'priority': 5})

    advices.sort(key=lambda x: x['priority'])
    return advices[:6]  # top 6


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        raw_inputs  = data.get('inputs', {})
        imputed_log = []   # track which fields were imputed

        # ── 1. Parse & impute raw features (original 21) ────────────────────
        ORIG_FEATURES = [
            'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
            'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
            'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost',
            'GenHlth','MentHlth','PhysHlth','DiffWalk',
            'Sex','Age','Education','Income'
        ]

        patient = {}
        for feat in ORIG_FEATURES:
            val = raw_inputs.get(feat)
            if val is None or val == '' or val == 'unknown':
                patient[feat] = IMPUTE_VALUES[feat]
                strat = 'mode' if feat in BINARY_FEATURES else 'median'
                imputed_log.append({
                    'field': feat,
                    'imputed_value': IMPUTE_VALUES[feat],
                    'strategy': strat
                })
            else:
                patient[feat] = float(val)

        # ── 2. Feature engineering ───────────────────────────────────────────
        patient_full = add_engineered_features(patient)

        # ── 3. Build DataFrame in correct feature order ──────────────────────
        df_input = pd.DataFrame([patient_full])[FEATURES]

        # ── 4. Predict ───────────────────────────────────────────────────────
        prob = float(MODEL.predict_proba(df_input)[0, 1])

        # ── 5. Risk level ────────────────────────────────────────────────────
        if prob < 0.20:
            risk_level = 'LOW'
            risk_label = 'Thấp'
            risk_color = '#27ae60'
            urgency    = 'Không cần đi khám ngay'
            summary    = 'Nguy cơ tiểu đường thấp. Duy trì lối sống lành mạnh và kiểm tra định kỳ.'
        elif prob < 0.50:
            risk_level = 'MEDIUM'
            risk_label = 'Trung bình'
            risk_color = '#f39c12'
            urgency    = 'Nên khám trong 3–6 tháng tới'
            summary    = 'Có một số yếu tố nguy cơ. Đặt lịch xét nghiệm đường huyết sớm.'
        else:
            risk_level = 'HIGH'
            risk_label = 'Cao'
            risk_color = '#e74c3c'
            urgency    = 'Nên đi khám càng sớm càng tốt'
            summary    = 'Nguy cơ tiểu đường cao. Cần xét nghiệm đường huyết và tư vấn bác sĩ ngay.'

        # ── 6. Advice ────────────────────────────────────────────────────────
        advices = generate_advice(patient, prob)

        return jsonify({
            'success':     True,
            'probability': round(prob * 100, 1),
            'risk_level':  risk_level,
            'risk_label':  risk_label,
            'risk_color':  risk_color,
            'urgency':     urgency,
            'summary':     summary,
            'advices':     advices,
            'imputed':     imputed_log,
            'n_imputed':   len(imputed_log),
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
