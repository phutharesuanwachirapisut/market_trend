from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import os

app = Flask(__name__)


def forecast_market_trend(df, location, property_type, Timeframe):
    from prophet import Prophet
    import pandas as pd

    df = df.copy()
    df['Date_Listed'] = pd.to_datetime(df['Date_Listed'], dayfirst=True, errors='coerce')

    tf_parts = Timeframe.split(" ")
    number = int(tf_parts[1])
    unit = tf_parts[2].lower()

    if "year" in unit:
      periods_t = number * 12
    else:
      periods_t = number

    #กรองข้อมูลตาม location และ property_type
    filtered_df = df[(df['Location'] == location) & (df['Property_Type'] == property_type)].copy()
    if filtered_df.empty:
        return {"error": "ไม่มีข้อมูลตรงกับเงื่อนไขที่เลือก"}

    filtered_df['Date_Listed'] = pd.to_datetime(filtered_df['Date_Listed'], dayfirst=True, errors='coerce')
    filtered_df['YearMonth'] = filtered_df['Date_Listed'].dt.to_period('M')

    #สร้างราคาตลาดเฉลี่ยรายเดือน
    monthly_df = (
        filtered_df
        .groupby('YearMonth')['Price']
        .mean()
        .reset_index()
        .rename(columns={'YearMonth': 'ds', 'Price': 'y'})
    )
    monthly_df['ds'] = monthly_df['ds'].dt.to_timestamp()

    #Train Prophet model
    model = Prophet()
    model.fit(monthly_df)

    #ทำนายล่วงหน้า
    future = model.make_future_dataframe(periods=periods_t, freq='M')
    forecast = model.predict(future)
    forecast_next = forecast[['ds', 'yhat']].tail(periods_t)

    #หาค่าต่ำสุด
    best_row = forecast_next.loc[forecast_next['yhat'].idxmin()]
    best_month = best_row['ds'].strftime('%b %Y')
    best_price = round(best_row['yhat'], 2)

    #สร้างผลลัพธ์
    result = {
        "forecast": [
            {"month": row['ds'].strftime('%b %Y'), "price": round(row['yhat'], 2)}
            for _, row in forecast_next.iterrows()
        ],
        "lowest_month": best_month,
        "lowest_price": best_price
    }

    return result

@app.route('/')
def index():
    return "Property Price Forecast API is running!"

@app.route('/forecast', methods=['POST'])
def forecast_endpoint():
    try:
        data = request.get_json()
        csv_data = data.get('csv_data')
        location = data.get('location')
        property_type = data.get('property_type')
        timeframe = data.get('timeframe')

        if not all([csv_data, location, property_type, timeframe]):
            return jsonify({
                "error": "กรุณาส่งข้อมูล csv_data, location, property_type และ timeframe"
            }), 400
        df = pd.read_json(csv_data)
        result = forecast_market_trend(df, location, property_type, timeframe)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)