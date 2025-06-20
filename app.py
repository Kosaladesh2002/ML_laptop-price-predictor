from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

def prediction(lst):
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([lst])
    return pred_value

@app.route('/', methods=['POST', 'GET'])
def index():
    pred_value = 0
    if request.method == 'POST':
        try:
            # Step 1: Get form inputs
            ram = int(request.form['ram'])
            weight = float(request.form['weight'])
            company = request.form['company'].lower()
            typename = request.form['typename'].lower()
            opsys = request.form['opsys'].lower()
            cpu = request.form['cpuname'].lower()
            gpu = request.form['gpuname'].lower()
            touchscreen = request.form.get('touchscreen')
            ips = request.form.get('ips')
            screen_size = float(request.form['screen_size'])
            resolution = request.form['resolution']
            hdd = int(request.form['hdd'])
            ssd = int(request.form['ssd'])

            # Step 2: Calculate PPI
            try:
                x_res, y_res = map(int, resolution.lower().split('x'))
                ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
            except:
                ppi = 0  # fallback if resolution is invalid

            # Step 3: Numeric features (7)
            feature_list = [
                ram,
                weight,
                1 if touchscreen else 0,
                1 if ips else 0,
                ppi,
                hdd,
                ssd
            ]

            # Step 4: One-hot categories (total 37 features)
            company_list = ['acer', 'apple', 'asus', 'chuwi', 'dell', 'fujitsu', 'google', 'hp',
                            'lenovo', 'lg', 'msi', 'razer', 'samsung', 'toshiba']  # 14

            typename_list = ['2in1convertible', 'gaming', 'netbook', 'notebook', 'ultrabook', 'workstation']  # 6

            opsys_list = ['android', 'chromeos', 'linux', 'mac', 'windows', 'noos']  # 6

            cpu_list = ['amd', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'intelcorei9', 'intelpentium', 'intelceleron']  # 7

            gpu_list = ['amd', 'intel', 'nvidia', 'apple']  # ✅ 4 ← (removed 'arm')

            def encode_feature(options, selected):
                return [1 if item == selected else 0 for item in options]

            # Add encoded categorical features
            feature_list += encode_feature(company_list, company)
            feature_list += encode_feature(typename_list, typename)
            feature_list += encode_feature(opsys_list, opsys)
            feature_list += encode_feature(cpu_list, cpu)
            feature_list += encode_feature(gpu_list, gpu)

            # Step 5: Final check
            if len(feature_list) == 44:
                pred_value = prediction(feature_list)
        
            else:
                pred_value = f"❌ Feature mismatch: Got {len(feature_list)} features, expected 44."

        except Exception as e:
            pred_value = f"⚠️ Error: {str(e)}"

    return render_template('index.html', pred_value=pred_value)

if __name__ == '__main__':
    app.run(debug=True)
