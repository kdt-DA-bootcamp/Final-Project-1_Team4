import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 TIGER 화장품 ETF", "💬 감성 점수", "📊 LinearRegression", "📊 GradientBoosting", "📊 XGBoost", "📊 LSTM + XGBoost"])

# ------------------ TAB 1: ETF vs KOSPI ------------------ #
with tab1:
    # ░░ 1. 트리맵 데이터 로딩 및 시각화 ░░
    df_tree = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab1_Constituents.csv")
    df_tree.columns = df_tree.columns.str.strip()

    df_tree['비중(%)'] = pd.to_numeric(df_tree['비중(%)'], errors='coerce')
    df_tree['1년 수익률'] = pd.to_numeric(df_tree['1년 수익률'], errors='coerce')
    df_tree = df_tree.dropna(subset=['비중(%)', '1년 수익률'])

    def get_return_category(x):
        if x >= 150: return '≥ +150%'
        elif x >= 100: return '+100% ~ +150%'
        elif x >= 50: return '+50% ~ +100%'
        elif x >= 0: return '0% ~ +50%'
        elif x >= -50: return '0% ~ -50%'
        elif x >= -100: return '-50% ~ -100%'
        else: return '≤ -100%'

    df_tree['수익률 구간'] = df_tree['1년 수익률'].apply(get_return_category)

    color_discrete_map = {
        '≥ +150%': '#ff1a1a', '+100% ~ +150%': '#ff6600',
        '+50% ~ +100%': '#ffcc00', '0% ~ +50%': '#99ccff',
        '0% ~ -50%': '#3399ff', '-50% ~ -100%': '#0066cc', '≤ -100%': '#003366'
    }

    fig_tree = px.treemap(
        df_tree, path=['종목명'], values='비중(%)',
        color='수익률 구간', color_discrete_map=color_discrete_map
    )

    fig_tree.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:.2f}%",
        textfont=dict(size=16),
        selector=dict(type='treemap')
    )

    # 📦 구성종목 트리맵 섹션 제목
    st.markdown("""
    <style>
    .section-title {
        font-size: 28px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .subsection {
        font-size: 16px;
        font-weight: 500;
        margin-top: 30px;
        margin-bottom: 5px;
    }
    .legend-box span {
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 5px;
    }
    </style>

    <div class='section-title'>📦 구성종목 트리맵 및 1년 수익률</div>
    <div class='subsection'>📊 1년 수익률</div>
    <div class='legend-box'>
        <span style='background-color:#ff1a1a;color:white;padding:3px 8px;'>≥ +150%</span>
        <span style='background-color:#ff6600;color:white;padding:3px 8px;'>+100~150%</span>
        <span style='background-color:#ffcc00;padding:3px 8px;'>+50~100%</span>
        <span style='background-color:#99ccff;padding:3px 8px;'>0~+50%</span>
        <span style='background-color:#3399ff;color:white;padding:3px 8px;'>0~-50%</span>
        <span style='background-color:#0066cc;color:white;padding:3px 8px;'>-50~-100%</span>
        <span style='background-color:#003366;color:white;padding:3px 8px;'>≤ -100%</span>
    </div>

    <div class='subsection'>🗂️ 구성종목 트리맵</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("<hr style='margin:40px 0; border:1px solid #555;'>", unsafe_allow_html=True)
    st.markdown("### 📈 TIGER ETF vs KOSPI 지수 (정규화 + 이동평균)")
    # 데이터 로딩
    df_compare = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab1_ETFvsKOSPI.csv")
    df_compare['date'] = pd.to_datetime(df_compare['date'])
    df_compare = df_compare.dropna(subset=['TIGER ETF', 'KOSPI'])

    # 정규화
    df_compare['TIGER ETF'] = pd.to_numeric(df_compare['TIGER ETF'], errors='coerce')
    df_compare['KOSPI'] = pd.to_numeric(df_compare['KOSPI'], errors='coerce')

    df_compare['TIGER ETF_norm'] = df_compare['TIGER ETF'] / df_compare['TIGER ETF'].iloc[0]
    df_compare['KOSPI_norm'] = df_compare['KOSPI'] / df_compare['KOSPI'].iloc[0]

    # 20일 이동 평균선
    df_compare['TIGER_MA'] = df_compare['TIGER ETF_norm'].rolling(window=20).mean()
    df_compare['KOSPI_MA'] = df_compare['KOSPI_norm'].rolling(window=20).mean()

    # 날짜 슬라이더 (최근 2년 기본)
    min_date = df_compare['date'].min().date()
    max_date = df_compare['date'].max().date()
    default_start = max_date - pd.DateOffset(years=2)
    selected_range = st.slider("📅 기간 선택", min_value=min_date, max_value=max_date,
                               value=(default_start.date(), max_date))

    df_filtered = df_compare[(df_compare['date'] >= pd.to_datetime(selected_range[0])) &
                             (df_compare['date'] <= pd.to_datetime(selected_range[1]))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['TIGER ETF_norm'],
                             mode='lines', name='TIGER ETF',
                             line=dict(width=1, color='orange'), opacity=0.5))
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['KOSPI_norm'],
                             mode='lines', name='KOSPI',
                             line=dict(width=1, color='lightgreen'), opacity=0.5))
    # MA 선
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['TIGER_MA'],
                             mode='lines', name='TIGER ETF MA (20D)',
                             line=dict(width=2, dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['KOSPI_MA'],
                             mode='lines', name='KOSPI MA (20D)',
                             line=dict(width=2, dash='dash', color='lightblue')))

    fig.update_layout(
        xaxis_title="날짜", yaxis_title="정규화 지수",
        template="plotly_dark", height=600,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    

# ------------------ TAB 2: 감성 점수 ------------------ #
with tab2:
    df_sentiment = pd.read_csv(r'C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab2_Sentiment_Score_Daily_Combined.csv')
    df_sentiment['날짜'] = pd.to_datetime(df_sentiment['문서발표일'], errors='coerce')
    df_sentiment = df_sentiment.dropna(subset=['날짜'])
    df_sentiment = df_sentiment.loc[:, ~df_sentiment.columns.duplicated()]

    df_avg = df_sentiment[['날짜', '감성점수_평균']].copy().rename(columns={'감성점수_평균': '감성점수'})
    df_avg['기업명'] = '전체 평균'

    df_long = df_sentiment.drop(columns=['감성점수_평균', '문서발표일'], errors='ignore')
    df_long = df_long.melt(id_vars='날짜', var_name='기업명', value_name='감성점수')
    df_long = pd.concat([df_long, df_avg], ignore_index=True)
    df_long = df_long.sort_values(['기업명', '날짜'])
    df_long['감성점수'] = df_long.groupby('기업명')['감성점수'].ffill()

    st.markdown("### 💬 기업별 감성 점수 추이")
    min_date = df_long['날짜'].min().date()
    max_date = df_long['날짜'].max().date()
    date_range = st.slider("감성점수 기간 선택", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    df_filtered = df_long[(df_long['날짜'] >= pd.to_datetime(date_range[0])) & (df_long['날짜'] <= pd.to_datetime(date_range[1]))]

    fig_sent = go.Figure()

    avg_df = df_filtered[df_filtered['기업명'] == '전체 평균']
    fig_sent.add_trace(go.Scatter(
        x=avg_df['날짜'], y=avg_df['감성점수'],
        name='전체 평균', line=dict(color='crimson', width=2), mode='lines'
    ))

    for 기업 in df_filtered['기업명'].unique():
        if 기업 == '전체 평균': continue
        개별 = df_filtered[df_filtered['기업명'] == 기업]
        fig_sent.add_trace(go.Scatter(
            x=개별['날짜'], y=개별['감성점수'],
            name=기업, mode='markers', marker=dict(size=4, opacity=0.3)
        ))

    fig_sent.update_layout(
        height=700, template='plotly_dark',
        hovermode='x unified', title="감성 점수 그래프"
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    

# ------------------ TAB 3: 김은재 : Linear ------------------ #
with tab3:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    st.subheader("📈 2024년 ETF 종가 예측 결과")

    # CSV 로딩
    result_df = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab3_ETF_Predictions_2024.csv")  # 또는 절대 경로 사용

    # 예측 성능 계산
    y_true = result_df['실제 종가']
    y_pred = result_df['예측 종가']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 메트릭 카드 표시
    st.markdown("#### ✅ 예측 성능 지표")
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE", f"{mae:.2f}")
    col2.metric("📈 RMSE", f"{rmse:.2f}")
    col3.metric("📊 R² Score", f"{r2:.4f}")

    # 데이터프레임 출력
    st.markdown("#### 📋 예측 결과 데이터")
    st.dataframe(result_df)

    import plotly.graph_objects as go

    st.markdown("#### 📊 2024년 분기별 실제 종가 vs 예측 종가")

    x = result_df['분기']
    y_true = result_df['실제 종가']
    y_pred = result_df['예측 종가']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y_true,
        mode='lines+markers',
        name='실제 종가',
        line=dict(color='lightskyblue', width=3),
        marker=dict(symbol='circle', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y_pred,
        mode='lines+markers',
        name='예측 종가',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(symbol='square', size=8)
    ))

    fig.update_layout(
        title='2024년 ETF 종가 예측 결과',
        xaxis_title='분기',
        yaxis_title='ETF 종가',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_dark',  # ✅ 여기가 포인트
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)




# ------------------ TAB 4: 김주혜 : GradientBoosting ------------------ #
with tab4:
    # tab4 안의 가장 위에 추가
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    st.markdown("### 📊 GradientBoosting 예측 결과 시각화")

    df_pred = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab4_model_GradientBoosting.csv")
    # 예시 컬럼: 날짜, 실제값, 예측값 등 가정
    df_pred['날짜'] = pd.to_datetime(df_pred['날짜'])
    df_pred = df_pred.sort_values('날짜')

    y_true = df_pred['실제 수익률']
    y_pred = df_pred['예측 수익률']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    
    # MAPE (0 나누기 방지)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    
    st.markdown("#### 📌 예측 성능 지표 (GradientBoosting)")

    col1, col2, col3 = st.columns(3)
    col4, col5, _ = st.columns(3)

    col1.metric("📉 MAE", f"{mae:.4f}")
    col2.metric("📏 MSE", f"{mse:.6f}")
    col3.metric("📐 RMSE", f"{rmse:.4f}")
    col4.metric("📊 MAPE", f"{mape:.2f}%")
    col5.metric("🎯 R² Score", f"{r2:.4f}")

    # 선그래프 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['날짜'], y=y_true,
                             name='실제 수익률', line=dict(color='lightgreen')))
    fig.add_trace(go.Scatter(x=df_pred['날짜'], y=y_pred,
                             name='예측 수익률', line=dict(color='orange', dash='dot')))
    
    st.markdown("#### 📈 2025년 실제 수익률 vs 예측 수익률")

    fig.update_layout(
        template='plotly_dark',
        xaxis_title="날짜", yaxis_title="수익률",
        height=500, hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)




# ------------------ TAB 5: 고민정 : XGboost ------------------ #

with tab5:
    st.markdown("### 📊 모델별 예측력 + 변수 영향력 통합 분석")

    # CSV 로드
    df_eval = pd.read_csv(
        r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab5_XGboost.csv"
    )

    # target + model 조합 컬럼 생성
    df_eval['target'] = df_eval['target'].str.strip()
    df_eval['target+model'] = df_eval['target'] + ' - ' + df_eval['model']

    # 사용할 성능 지표들
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    # 각 성능 지표별 차트를 반복 생성
    for metric in metrics:
        st.markdown(f"### 🔹 {metric}")

        fig = px.bar(
            df_eval,
            x='composition',
            y=metric,
            color='target+model',
            barmode='group',
            text=metric,
            title=f"{metric} by Composition / Target / Model"
        )

        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            template='plotly_dark',
            height=550,
            xaxis_title="Composition", 
            yaxis_title=metric,
            uniformtext_minsize=15,
            uniformtext_mode='hide'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Optional: 데이터프레임 확인
    st.markdown("#### 📋 Raw Evaluation Data")
    st.dataframe(df_eval.reset_index(drop=True))

    st.markdown("<hr style='margin-top:40px; margin-bottom:30px; border:1px solid #555;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:0px; margin-bottom:20px; font-size:28px; font-weight:700'>
    🧠 상위 10개 주요변수 중요도
    </div>
    """, unsafe_allow_html=True)

    # 파일 로드
    df_20 = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab5_Variables_20Days_Later.csv")
    df_q = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab5_Variables_Quarterly.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📅 20일 후 수익률 예측 변수")
        fig_20 = px.bar(
            df_20.sort_values(by='중요도'),
            x='중요도', y='변수명',
            orientation='h',
            text='중요도',
            template='plotly_dark',
            height=500
        )
        fig_20.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside',
            marker=dict(
                color='rgba(51, 153, 255, 0.7)' # 파랑 + 투명도
            )
        )
        fig_20.update_layout(
            xaxis_title='중요도',
            yaxis_title='',
            margin=dict(t=30, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_20, use_container_width=True)

    with col2:
        st.markdown("##### 📅 분기 수익률 예측 변수")
        fig_q = px.bar(
            df_q.sort_values(by='중요도'),
            x='중요도', y='변수명',
            orientation='h',
            text='중요도',
            template='plotly_dark',
            height=500
        )
        fig_q.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside',
            marker=dict(
                color='rgba(255, 102, 102, 0.7)'  # 빨강 + 투명도
            )
        )
        fig_q.update_layout(
            xaxis_title='중요도',
            yaxis_title='',
            margin=dict(t=30, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_q, use_container_width=True)

# ------------------ TAB 6: 권가희 : XGboost + LSTM ------------------ #

with tab6:
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown("### 📈 LSTM + XGBoost 예측 결과 (2025년)")

    # CSV 파일 불러오기
    pred_df = pd.read_csv(r"C:\Users\Vivobook Pro M7600QE\BootCamp\TIL\팀플_화장품\streamlit\data\tab6_XGboost_LSTM.csv")
    pred_df["날짜"] = pd.to_datetime(pred_df["날짜"])

    # Plotly 시각화
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pred_df["날짜"], y=pred_df["실제 ETF"],
        mode='lines+markers',
        name='실제',
        line=dict(color='royalblue', width=2.5),
        marker=dict(symbol='circle', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df["날짜"], y=pred_df["예측 ETF"],
        mode='lines+markers',
        name='예측',
        line=dict(color='darkorange', width=2.5, dash='dash'),
        marker=dict(symbol='square', size=6)
    ))

    fig.update_layout(
        xaxis_title='날짜',
        yaxis_title='ETF 종가',
        legend=dict(orientation='h', y=1.1, x=1, xanchor='right'),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    # 지표 계산
    y_true = pred_df["실제 ETF"]
    y_pred = pred_df["예측 ETF"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 지표 카드 형태 출력
    st.markdown("#### 📅 성능 지표")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 RMSE", f"{rmse:.2f}")
    col2.metric("📏 MAE", f"{mae:.2f}")
    col3.metric("📊 MAPE", f"{mape:.2f}%")
    col4.metric("🎯 R² Score", f"{r2:.4f}")

    st.markdown("#### 📉 2025년 ETF 종가 예측 vs 실제")
    st.plotly_chart(fig, use_container_width=True)
