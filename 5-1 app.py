import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4 = st.tabs(["📈 ETF vs KOSPI", "💬 감성 점수", "📊 모델 예측 결과1", "📊 모델 예측 결과2"])

# ------------------ TAB 1: ETF vs KOSPI ------------------ #
with tab1:
    # ░░ 1. 트리맵 데이터 로딩 및 시각화 ░░
    df_tree = pd.read_csv("/home/ubuntu/team4-db-project/구성종목_1년.csv")
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
    df_compare = pd.read_csv("/home/ubuntu/team4-db-project/ETFvsKOSPI.csv")
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
    df_sentiment = pd.read_csv('/home/ubuntu/team4-db-project/통합_감성점수_일별_정리.csv')
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

# ------------------ TAB 3: 모델 예측 1 ------------------ #

with tab3:
    st.markdown("### 📊 모델별 성능 비교 대시보드")

    # 엑셀 시트 로드 및 정리
    df_eval = pd.read_csv("/home/ubuntu/team4-db-project/모델결과1.csv")
    df_eval.columns = ['모델구성', '모델', '타겟(Y)', 'Accuracy', 'Precision', 'Recall', 'F1-score']

    # 타겟 설정: '분기수익률'과 '20일후 수익률'
    타겟_목록 = ['분기수익률', '20일후 수익률']
    
    # 성능 지표를 Accuracy, Precision, Recall, F1-score 중 하나로 선택
    성능지표 = 'Accuracy'  # 기본 지표 설정

    for 타겟선택 in 타겟_목록:
        st.markdown(f"### {타겟선택}에 대한 모델 성능")

        # 데이터 필터링 & 정렬
        df_filtered = df_eval[df_eval['타겟(Y)'] == 타겟선택].dropna(subset=[성능지표])
        # 데이터가 존재하는지 확인
        st.write(f"필터링된 {타겟선택} 데이터:", df_filtered)
        df_sorted = df_filtered.sort_values(by=성능지표, ascending=False)

        # 바 차트 시각화
        fig = px.bar(
            df_sorted,
            x='모델',
            y=성능지표,
            color='모델구성',
            text=성능지표,
            title=f"📊 모델 성능 비교 - {성능지표} 기준 ({타겟선택})"
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title="모델", 
            yaxis_title=성능지표,
            uniformtext_minsize=8, 
            uniformtext_mode='hide'
        )

        st.plotly_chart(fig, use_container_width=True)

        # 성능 지표 테이블
        st.markdown(f"#### 📋 {타겟선택}에 대한 모델 성능 지표 테이블")
        st.dataframe(df_sorted.reset_index(drop=True))
        
        # 구분선 추가
        st.markdown("<hr style='margin:40px 0; border:1px solid #555;'>", unsafe_allow_html=True)


# ------------------ TAB 4: 모델 예측 2 ------------------ #
with tab4:
    # tab4 안의 가장 위에 추가
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    st.markdown("### 📊 GradientBoosting 예측 결과 시각화")

    df_pred = pd.read_csv("/home/ubuntu/team4-db-project/모델결과2.csv")
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

    # 선그래프 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['날짜'], y=y_true,
                             name='실제 수익률', line=dict(color='lightgreen')))
    fig.add_trace(go.Scatter(x=df_pred['날짜'], y=y_pred,
                             name='예측 수익률', line=dict(color='orange', dash='dot')))

    fig.update_layout(
        template='plotly_dark',
        title="📈 실제 수익률 vs 예측 수익률 (GradientBoosting)",
        xaxis_title="날짜", yaxis_title="수익률",
        height=500, hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 성능 지표 출력
    st.markdown(f"""
    ### 📌 예측 성능 지표 (GradientBoosting)

    - **MAE** (Mean Absolute Error): `{mae:.4f}`
    - **MSE** (Mean Squared Error): `{mse:.6f}`
    - **RMSE** (Root Mean Squared Error): `{rmse:.4f}`
    - **MAPE** (Mean Absolute Percentage Error): `{mape:.2f}%`
    - **R² Score**: `{r2:.4f}`
    """)
