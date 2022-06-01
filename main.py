import pandas as pd
import streamlit as st
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from fbprophet import Prophet
import numpy as np
from PIL import Image
image = Image.open('C:\\Users\\sonmyogyeong\\PycharmProjects\\prediction\\logo.png')
st.sidebar.image(image)
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("쌀 값 예측")
st.subheader("쌀 값을 예측합니다.")
st.text("안녕하세요! 쌀값을 알고 싶은 지역을 선택하시면, 예측된 쌀값 그래프가 나타납니다. ")
st.sidebar.header("Menu")
select_city = st.sidebar.selectbox('쌀 값을 알고싶은 도시를 고르세요.', [
    '서울','대전','대구','부산','광주'])
st.sidebar.write("선택하신 도시는 "+select_city+"입니다.")
month = st.sidebar.slider("Year", 1,5)
st.sidebar.write(str(month)+"년 뒤를 예측합니다.")
if select_city == '서울':
    data = pd.read_csv("C:\\Users\\sonmyogyeong\\PycharmProjects\\lstm\\data\\seoul.csv")
elif select_city =='광주':
    data = pd.read_csv("C:\\Users\\sonmyogyeong\\PycharmProjects\\lstm\\data\\gwangju.csv")
elif select_city=='대구':
    data = pd.read_csv("C:\\Users\\sonmyogyeong\\PycharmProjects\\lstm\\data\\daegu.csv")
elif select_city=='대전':
    data = pd.read_csv("C:\\Users\\sonmyogyeong\\PycharmProjects\\lstm\\data\\daejeon.csv")
else :
    data = pd.read_csv("C:\\Users\\sonmyogyeong\\PycharmProjects\\lstm\\data\\busan.csv")


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['price'], name="price"))
    fig.layout.update(title_text = select_city, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

dataframe = data
st.write(data.tail(10))
data['date'] = pd.to_datetime(data['date'])
series = np.asarray(dataframe.columns)

st.subheader(select_city+"의 "+str(month)+"년 뒤 쌀 값을 예측합니다.")

df_train_daejeon = data[['date','price']]
df_train_daejeon = df_train_daejeon.rename(columns={
    "date":"ds",
    "price":"y"
})

m = Prophet()
m.fit(df_train_daejeon)
future = m.make_future_dataframe(periods = month, freq='Y')
forecast = m.predict(future)
forecast1 = forecast[['ds','yhat']]
st.write(forecast1.tail())
result1 = forecast1['yhat']
result = result1.tail(1)
st.write(result)
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)