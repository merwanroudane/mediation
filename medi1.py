import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import pingouin as pg
import graphviz
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Google AI
genai.configure(api_key="AIzaSyD9Z1QGHY9BGjEQ_wmFx0xSHHcQLDvu4XM")


def mediations():
    st.set_page_config(layout="wide")
    df = pd.DataFrame()
    st.subheader("نموذج المتغيرات الوسيطية والمعدلة")

    # File upload and example data options
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم مثال تجريبي", False, help="استخدم ملف مثال مدمج لتجربة التطبيق"
    )

    # Create a new example file when requested
    if use_example_file:
        # Creating example data
        np.random.seed(42)
        n = 200

        # Independent variable (X)
        leadership_style = np.random.normal(0, 1, n)

        # Mediator variables (M)
        employee_motivation = 0.5 * leadership_style + np.random.normal(0, 0.7, n)
        work_environment = 0.4 * leadership_style + np.random.normal(0, 0.8, n)

        # Dependent variable (Y)
        productivity = (0.3 * leadership_style + 0.5 * employee_motivation +
                        0.4 * work_environment + np.random.normal(0, 0.6, n))

        # Control variable
        employee_experience = np.random.normal(3, 1, n)

        # Create DataFrame
        example_df = pd.DataFrame({
            'القيادة_الإدارية': leadership_style,
            'دافعية_الموظفين': employee_motivation,
            'بيئة_العمل': work_environment,
            'الإنتاجية': productivity,
            'خبرة_الموظفين': employee_experience
        })

        # Create a buffer to hold the Excel file
        buffer = io.BytesIO()
        example_df.to_excel(buffer, index=False)
        buffer.seek(0)

        # Use the buffer as the uploaded file
        df = example_df.copy()

        # Display info about the example
        st.info("""
        ### مثال تجريبي: تأثير أسلوب القيادة على الإنتاجية

        في هذا المثال، نفترض أن:
        - **المتغير المستقل**: أسلوب القيادة الإدارية
        - **المتغيرات الوسيطة**: دافعية الموظفين وبيئة العمل
        - **المتغير التابع**: الإنتاجية
        - **المتغير الضابط**: خبرة الموظفين

        الفرضية: يؤثر أسلوب القيادة على الإنتاجية من خلال دافعية الموظفين وبيئة العمل.
        """)
    elif uploaded_file:
        df = pd.read_excel(uploaded_file).copy()

    # RTL styling for Arabic
    st.markdown(
        """
        <style>
        /* جعل الاتجاه من اليمين إلى اليسار */
        body {
            direction: rtl;
            text-align: right;
        }

        /* تخصيص جدول البيانات */
        .stDataFrame {
            direction: rtl;
        }

        /* تخصيص عناوين النصوص */
        h1, h2, h3, h4, h5, h6 {
            text-align: right;
        }

        /* تخصيص عناصر الإدخال */
        .stTextInput, .stSelectbox, .stButton {
            text-align: right;
        }

        /* تحسين تصميم الجداول */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: right;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced explanatory section with more details on regression and mediation
    st.markdown("""
        <div style="
    background-color: #f0f8ff; 
    padding: 15px; 
    border-radius: 10px; 
    border: 2px solid #4682b4;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    text-align: right;
    direction: rtl;
    font-size: 18px;
    line-height: 1.8;">
    <h3>🔍 مقدمة في تحليل النماذج الإحصائية المتقدمة:</h3>

    <h4>📊 تحليل الانحدار الأساسي:</h4>
    <p>يدرس العلاقة المباشرة بين متغيرين، ويقيس تغيّر المتغير التابع (Y) مع تغيّر المتغير المستقل (X). لكن هذا النموذج البسيط قد لا يكون كافياً لفهم العلاقات الأكثر تعقيداً.</p>

    <h4>📈 دراسة المتغيرات الوسيطة والمعدلة:</h4>
    <p>تساعد في فهم أفضل للعلاقات بين المتغيرات، وتكشف في بعض الأحيان عن علاقات زائفة أو غير مباشرة.</p>

    <h4>📌 المتغيرات الوسيطة (Mediators):</h4>
    <ul>
        <li><strong>المفهوم:</strong> تنقل تأثير المتغير المستقل (X) إلى المتغير التابع (Y).</li>
        <li><strong>السؤال:</strong> كيف يؤثر المتغير المستقل على المتغير التابع؟</li>
        <li><strong>المسار:</strong> X → M → Y</li>
        <li><strong>الشروط:</strong> علاقة بين X وM، وبين M وY، وتغير العلاقة بين X وY عند إدخال M.</li>
        <li><strong>مثال:</strong> تؤثر القيادة الإدارية على الإنتاجية من خلال رفع دافعية الموظفين.</li>
    </ul>

    <h4>📌 المتغيرات المعدلة (Moderators):</h4>
    <ul>
        <li><strong>المفهوم:</strong> تؤثر على قوة أو اتجاه العلاقة بين المتغير المستقل (X) والمتغير التابع (Y).</li>
        <li><strong>السؤال:</strong> متى تكون العلاقة بين المتغيرات أقوى أو أضعف؟</li>
        <li><strong>المسار:</strong> X * W → Y</li>
        <li><strong>الشروط:</strong> وجود تفاعل إحصائي دال بين المتغير المستقل والمتغير المعدل.</li>
        <li><strong>مثال:</strong> يختلف تأثير الحوافز المادية على الأداء حسب مستوى الرضا الوظيفي.</li>
    </ul>

    <h4>🧪 الأهمية في البحث العلمي:</h4>
    <p>تساعد هذه النماذج في الكشف عن الآليات الداخلية للظواهر، تفسير التناقضات بين نتائج الدراسات، تحسين القدرة التنبؤية للنماذج، وتطوير تدخلات أكثر فاعلية.</p>
</div>

    """, unsafe_allow_html=True)

    # Choose analysis type
    medtype = st.radio(
        "اختر احدى الطرق ثم أكمل",
        ('التحليل الوسيطي', 'التحليل المعدل او التفاعلي'))

    # If there's data uploaded
    if not df.empty:
        st.write("### البيانات المدخلة:")
        st.dataframe(df.head())

        # Data summary
        with st.expander("إحصاءات وصفية للبيانات"):
            st.write(df.describe().round(3))

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.warning("⚠️ يوجد قيم مفقودة في البيانات:")
                st.write(missing_values[missing_values > 0])

    # Mediation Analysis
    if medtype == 'التحليل الوسيطي' and not df.empty:
        with st.form(key="mediation_form"):
            col1, col2 = st.columns(2)

            with col1:
                y = st.selectbox(
                    "المتغير التابع",
                    options=df.columns,
                    help="إختر المتغير التابع")
                x = st.selectbox(
                    "المتغير المستقل",
                    options=df.columns,
                    help="إختر المتغير المستقل")

            with col2:
                m = st.multiselect(
                    "المتغيرات الوسيطة",
                    options=df.columns,
                    help="إختر متغير وسيط او اكثر")
                cov = st.multiselect(
                    "اختر المتغيرات المشتركة (الضابطة)",
                    options=df.columns,
                    help="إختر متغير مشتركا او اكثر")

            Nsim = st.slider(
                'اختر عدد مرات البوتستراب',
                100, 2000, 500, 10)

            submitted = st.form_submit_button("تنفيذ التحليل")

        if submitted and x and y and m:
            # Educational note about mediation analysis steps
            st.markdown("""
            <div style="
                background-color: #e6f7ff; 
                padding: 15px; 
                border-radius: 10px; 
                border: 1px solid #1890ff;
                margin-bottom: 20px;
                text-align: right;
                direction: rtl;">
                <h3>خطوات تحليل الوساطة:</h3>
                <ol>
                    <li><strong>الخطوة الأولى:</strong> التحقق من وجود علاقة بين المتغير المستقل (X) والمتغير التابع (Y)</li>
                    <li><strong>الخطوة الثانية:</strong> التحقق من وجود علاقة بين المتغير المستقل (X) والمتغير الوسيط (M)</li>
                    <li><strong>الخطوة الثالثة:</strong> التحقق من وجود علاقة بين المتغير الوسيط (M) والمتغير التابع (Y) عند التحكم في (X)</li>
                    <li><strong>الخطوة الرابعة:</strong> التحقق من انخفاض العلاقة بين (X) و(Y) عند إدخال (M)</li>
                </ol>
                <p><strong>أنواع الوساطة:</strong></p>
                <ul>
                    <li><strong>وساطة كاملة:</strong> تختفي العلاقة بين X و Y تماماً عند إدخال M</li>
                    <li><strong>وساطة جزئية:</strong> تنخفض العلاقة بين X و Y لكنها تظل دالة إحصائياً عند إدخال M</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Pre-test section
            st.write("## فحص البيانات قبل التحليل")

            # Create tabs for organizing pre-test results
            pre_test_tabs = st.tabs(["الارتباطات", "التوزيع الطبيعي", "الانحدار الخطي البسيط"])

            with pre_test_tabs[0]:
                # Correlation analysis
                st.write("### مصفوفة الارتباط")
                variables = [x, y] + m + cov
                corr_matrix = df[variables].corr().round(3)

                # Heatmap for correlations using Plotly
                fig = px.imshow(corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)
                fig.update_layout(
                    title='مصفوفة الارتباط بين المتغيرات',
                    width=800,
                    height=600
                )
                st.plotly_chart(fig)

                # Table of correlations
                st.dataframe(corr_matrix)

                # Check for multicollinearity
                if len(variables) > 2:
                    st.write("### فحص الارتباط المتعدد (Multicollinearity)")
                    X_multi = df[variables].drop(columns=[y])
                    corr_X = X_multi.corr()

                    # Flag high correlations
                    high_corr = []
                    for i in range(len(corr_X.columns)):
                        for j in range(i + 1, len(corr_X.columns)):
                            if abs(corr_X.iloc[i, j]) > 0.7:
                                high_corr.append(f"{corr_X.columns[i]} & {corr_X.columns[j]}: {corr_X.iloc[i, j]:.3f}")

                    if high_corr:
                        st.warning("⚠️ تم العثور على ارتباطات عالية قد تشير إلى مشكلة الارتباط المتعدد:")
                        for corr in high_corr:
                            st.write(f"- {corr}")
                    else:
                        st.success("✅ لا توجد مؤشرات قوية على وجود مشكلة الارتباط المتعدد.")

            with pre_test_tabs[1]:
                # Normality tests
                st.write("### اختبار التوزيع الطبيعي للمتغيرات")

                normality_results = []
                for var in variables:
                    shapiro_test = stats.shapiro(df[var].dropna())
                    normality_results.append({
                        'المتغير': var,
                        'إحصاء شابيرو-ويلك': shapiro_test[0],
                        'القيمة الاحتمالية p': shapiro_test[1],
                        'التوزيع طبيعي؟': 'نعم' if shapiro_test[1] > 0.05 else 'لا'
                    })

                st.dataframe(pd.DataFrame(normality_results).set_index('المتغير'))

                # Create Q-Q plots
                st.write("### رسوم Q-Q للتحقق من التوزيع الطبيعي")

                # Use Plotly for Q-Q plots
                n_vars = len(variables)
                n_cols = 2
                n_rows = (n_vars + n_cols - 1) // n_cols

                fig = make_subplots(rows=n_rows, cols=n_cols,
                                    subplot_titles=[f'Q-Q Plot: {var}' for var in variables])

                row, col = 1, 1
                for i, var in enumerate(variables):
                    # Get sorted data and theoretical quantiles
                    sorted_data = np.sort(df[var].dropna())
                    n = len(sorted_data)
                    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

                    # Create scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=theoretical_quantiles,
                            y=sorted_data,
                            mode='markers',
                            name=var
                        ),
                        row=row, col=col
                    )

                    # Add diagonal line
                    min_val = min(theoretical_quantiles.min(), sorted_data.min())
                    max_val = max(theoretical_quantiles.max(), sorted_data.max())

                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                    # Update indices
                    col += 1
                    if col > n_cols:
                        col = 1
                        row += 1

                fig.update_layout(
                    height=300 * n_rows,
                    width=900,
                    title_text='رسوم Q-Q للتحقق من التوزيع الطبيعي',
                    showlegend=False
                )

                st.plotly_chart(fig)

                # Histograms
                st.write("### توزيع المتغيرات")

                # Use Plotly for histograms
                fig = make_subplots(rows=n_rows, cols=n_cols,
                                    subplot_titles=[f'توزيع {var}' for var in variables])

                row, col = 1, 1
                for i, var in enumerate(variables):
                    fig.add_trace(
                        go.Histogram(
                            x=df[var].dropna(),
                            name=var,
                            opacity=0.7,
                            nbinsx=20
                        ),
                        row=row, col=col
                    )

                    # Add kernel density estimate
                    kde_x, kde_y = get_kde_values(df[var].dropna())
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y * len(df[var].dropna()) * (kde_x[1] - kde_x[0]),  # Scale KDE to match histogram
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                    # Update indices
                    col += 1
                    if col > n_cols:
                        col = 1
                        row += 1

                fig.update_layout(
                    height=300 * n_rows,
                    width=900,
                    title_text='توزيع المتغيرات',
                    showlegend=False,
                    bargap=0.1
                )

                st.plotly_chart(fig)

            with pre_test_tabs[2]:
                # Simple linear regression
                st.write("### الانحدار الخطي البسيط (X → Y)")

                X_simple = sm.add_constant(df[x])
                model_simple = sm.OLS(df[y], X_simple).fit()

                # Summary of regression results
                st.write("#### ملخص نموذج الانحدار")
                st.write(f"R² = {model_simple.rsquared:.3f}")
                st.write(f"R² المعدل = {model_simple.rsquared_adj:.3f}")
                st.write(f"قيمة F = {model_simple.fvalue:.3f}")
                st.write(f"القيمة الاحتمالية p = {model_simple.f_pvalue:.5f}")

                # Educational note about interpreting R-squared
                st.info("""
                **تفسير معامل التحديد R²**:
                - يمثل نسبة التباين في المتغير التابع الذي يمكن تفسيره بواسطة المتغير المستقل
                - القيمة 0.1-0.3: تأثير ضعيف
                - القيمة 0.3-0.5: تأثير متوسط
                - القيمة > 0.5: تأثير قوي
                """)

                # Coefficients table
                coef_df = pd.DataFrame({
                    'المعاملات': model_simple.params,
                    'الخطأ المعياري': model_simple.bse,
                    'قيمة t': model_simple.tvalues,
                    'القيمة الاحتمالية': model_simple.pvalues
                }).round(3)
                st.dataframe(coef_df)

                # Educational note about interpreting coefficients
                st.info("""
                **تفسير المعاملات**:
                - **الثابت (const)**: قيمة المتغير التابع عندما يكون المتغير المستقل صفراً
                - **معامل الانحدار**: مقدار التغير في المتغير التابع عند تغير المتغير المستقل بوحدة واحدة
                - **القيمة الاحتمالية p**: إذا كانت أقل من 0.05، فالعلاقة دالة إحصائياً
                """)

                # Plot regression using Plotly
                fig = px.scatter(df, x=x, y=y, opacity=0.6, trendline="ols")
                fig.update_layout(
                    title=f'الانحدار الخطي البسيط: {x} → {y}',
                    xaxis_title=x,
                    yaxis_title=y,
                    width=800,
                    height=500
                )
                st.plotly_chart(fig)

                # Residual analysis
                st.write("#### تحليل البواقي")

                # Calculate residuals
                residuals = model_simple.resid
                fitted_values = model_simple.fittedvalues

                # Plot residuals using Plotly
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=['البواقي مقابل القيم المتوقعة', 'Q-Q Plot للبواقي'])

                # Residuals vs Fitted
                fig.add_trace(
                    go.Scatter(
                        x=fitted_values,
                        y=residuals,
                        mode='markers',
                        marker=dict(opacity=0.6),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Add horizontal line at y=0
                fig.add_trace(
                    go.Scatter(
                        x=[fitted_values.min(), fitted_values.max()],
                        y=[0, 0],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Q-Q plot for residuals
                sorted_residuals = np.sort(residuals)
                n = len(sorted_residuals)
                theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_residuals,
                        mode='markers',
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # Add diagonal line
                min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
                max_val = max(theoretical_quantiles.max(), sorted_residuals.max())

                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=2
                )

                fig.update_layout(
                    height=400,
                    width=900,
                    title_text='تحليل البواقي'
                )

                st.plotly_chart(fig)

                # Test for heteroscedasticity
                bp_test = het_breuschpagan(residuals, X_simple)
                dw_value = durbin_watson(residuals)

                st.write("#### اختبارات للبواقي")
                st.write(f"- اختبار Breusch-Pagan للتجانس: p = {bp_test[1]:.4f} " +
                         ("✅" if bp_test[1] > 0.05 else "⚠️"))
                st.write(f"- إحصاء Durbin-Watson للارتباط الذاتي: {dw_value:.4f} " +
                         ("✅" if 1.5 < dw_value < 2.5 else "⚠️"))

                # Educational note about residual tests
                st.info("""
                **تفسير اختبارات البواقي**:

                - **اختبار Breusch-Pagan**:
                  - يفحص تجانس التباين (Homoscedasticity)
                  - إذا كانت القيمة الاحتمالية p > 0.05: البواقي متجانسة (جيد)
                  - إذا كانت p < 0.05: مشكلة عدم تجانس التباين (Heteroscedasticity)

                - **إحصاء Durbin-Watson**:
                  - يفحص الارتباط الذاتي في البواقي
                  - القيمة المثالية حوالي 2
                  - إذا كانت القيمة < 1.5 أو > 2.5: مشكلة الارتباط الذاتي
                """)

            # Main Mediation Analysis
            st.write("## نتائج تحليل الوساطة")

            # Run mediation analysis
            try:
                mod = pg.mediation_analysis(data=df, x=x, y=y, m=m, covar=cov, seed=1235, n_boot=Nsim)
                nm = len(m)

                st.write("### جدول التقديرات")
                st.dataframe(mod.round(3))

                # Educational note about mediation table
                st.info("""
                **شرح جدول التقديرات**:

                - **Total effect (الأثر الكلي)**: التأثير الإجمالي للمتغير المستقل (X) على المتغير التابع (Y)
                - **Direct effect (الأثر المباشر)**: تأثير X على Y بعد التحكم في المتغير الوسيط (M)
                - **Indirect effect (الأثر غير المباشر)**: تأثير X على Y من خلال M (الوساطة)

                **تفسير النتائج**:
                - إذا كان الأثر غير المباشر دالاً إحصائياً (p < 0.05): يوجد وساطة
                - إذا كان الأثر المباشر غير دال: وساطة كاملة
                - إذا كان الأثر المباشر دالاً: وساطة جزئية
                """)

                # Prepare data for Gemini analysis
                table_text = mod.round(3).to_string()

                # Call Gemini for analysis
                prompt = f"""
                        لديك نتائج تحليل نموذج وسيطي حيث المتغير التابع هو {y}، المتغير المستقل هو {x}، والمتغيرات الوسيطة هي {', '.join(m)}.
                        إليك جدول التقديرات:
                        {table_text}
                        قم بتحليل النتائج، واشرح هل هناك تأثير مباشر أو غير مباشر؟ وهل المتغير الوسيط يلعب دورًا مهمًا؟
                        اذكر نسبة التأثير غير المباشر من التأثير الكلي إن وجد، وما هي دلالته العملية؟
                        استخدم شرح سهل وعملي يسهل على الباحث فهمه، مع تفسير النتائج الإحصائية.
                        """

                response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
                analysis_text = response.text if response else "لم يتمكن Gemini من تحليل البيانات."

                # Show Gemini analysis
                st.subheader("📌 **تحليل Gemini**")
                st.markdown(
                    f"<div style='background-color:#f0f8ff;padding:15px;border-radius:10px;border:1px solid #4682b4'>{analysis_text}</div>",
                    unsafe_allow_html=True)

                # Create graphs
                graph = graphviz.Digraph()
                graph1 = graphviz.Digraph()

                graph.attr(rankdir='LR', splines='curved')
                graph.node_attr = {'color': '#6495ED', 'style': 'filled', 'shape': 'box', 'fontname': 'Arial'}
                graph.edge_attr = {'fontname': 'Arial', 'fontsize': '12'}

                graph1.attr(rankdir='LR', splines='curved')
                graph1.node_attr = {'style': 'filled', 'shape': 'box', 'fontname': 'Arial'}
                graph1.edge_attr = {'fontname': 'Arial', 'fontsize': '12'}

                # Add nodes and edges for mediation path
                i = 0
                for mm in m:
                    # Format p-values with stars for significance
                    p_indirect = mod.loc[2 * nm + 2 + i]["pval"].round(3)
                    stars_indirect = get_significance_stars(p_indirect)

                    p_m_y = mod.loc[i + nm]["pval"].round(3)
                    stars_m_y = get_significance_stars(p_m_y)

                    p_x_m = mod.loc[i]["pval"].round(3)
                    stars_x_m = get_significance_stars(p_x_m)

                    # Node for mediator
                    graph.node(mm, label=mm + "\n" + "الاثر غير المباشر = " +
                                         str(mod.loc[2 * nm + 2 + i]["coef"].round(3)) + stars_indirect)

                    # Edges for mediation path
                    graph.edge(mm, y, label=str(mod.loc[i + nm]["coef"].round(3)) + stars_m_y)
                    graph.edge(x, mm, label=str(mod.loc[i]["coef"].round(3)) + stars_x_m)

                    i = i + 1

                # Format p-values for direct effect
                p_direct = mod.loc[2 * nm + 1]["pval"].round(3)
                stars_direct = get_significance_stars(p_direct)

                # Format p-values for total effect
                p_total = mod.loc[2 * nm]["pval"].round(3)
                stars_total = get_significance_stars(p_total)

                # Add direct effect
                graph.edge(x, y,
                           label="اﻷثر المباشر = " + str(mod.loc[2 * nm + 1]["coef"].round(3)) + stars_direct,
                           _attributes={'color': '#FF6347', 'penwidth': '2.0'})

                # Add total effect
                graph1.edge(x, y,
                            label="اﻷثر الكلي = " + str(mod.loc[2 * nm]["coef"].round(3)) + stars_total,
                            _attributes={'color': '#FF6347', 'penwidth': '2.0'})

                # Style the nodes
                graph.node(y, _attributes={'color': '#87CEEB'})
                graph.node(x, _attributes={'color': '#90EE90'})
                graph1.node(y, _attributes={'color': '#87CEEB'})
                graph1.node(x, _attributes={'color': '#90EE90'})

                # Display the graphs
                st.write("### الرسوم البيانية للتحليل الوسيطي")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### اﻷثر الكلي")
                    st.graphviz_chart(graph1)

                with col2:
                    st.write("#### اﻷثر المباشر وغير المباشر")
                    st.graphviz_chart(graph)

                # Add legend for significance
                st.markdown("""
                <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; direction: rtl; text-align: right;">
                <p><strong>دلالة النجوم:</strong></p>
                <ul>
                <li>* = p < 0.05</li>
                <li>** = p < 0.01</li>
                <li>*** = p < 0.001</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                # Post-test analysis section
                st.write("## تحليل ما بعد الوساطة")

                # Calculate proportion of mediation
                post_tabs = st.tabs(["نسب التأثير", "المقارنة بين النماذج", "الرسوم البيانية"])

                with post_tabs[0]:
                    # Calculate the proportion of mediation
                    if nm == 1:  # Single mediator
                        total_effect = mod.loc[2 * nm]["coef"]
                        direct_effect = mod.loc[2 * nm + 1]["coef"]
                        indirect_effect = mod.loc[2 * nm + 2]["coef"]

                        if total_effect != 0:
                            prop_mediated = (indirect_effect / total_effect) * 100
                            st.write(f"### نسبة التأثير المتوسط")
                            st.write(f"- التأثير غير المباشر يمثل {prop_mediated:.2f}% من التأثير الكلي")

                            # Educational note about proportion mediated
                            st.info("""
                            **تفسير نسبة التأثير المتوسط**:
                            - نسبة أقل من 20%: وساطة ضعيفة
                            - نسبة 20% - 80%: وساطة متوسطة
                            - نسبة أكثر من 80%: وساطة قوية (تقترب من الوساطة الكاملة)
                            """)

                            # Create pie chart using Plotly
                            if direct_effect * indirect_effect >= 0:  # Same sign
                                labels = ['التأثير المباشر', 'التأثير غير المباشر']
                                sizes = [abs(direct_effect), abs(indirect_effect)]
                                colors = ['#87CEEB', '#6495ED']

                                fig = go.Figure(data=[go.Pie(
                                    labels=labels,
                                    values=sizes,
                                    hole=.3,
                                    marker_colors=colors,
                                    textinfo='percent+label',
                                    textfont_size=14
                                )])

                                fig.update_layout(
                                    title_text='نسبة التأثير المباشر وغير المباشر',
                                    title_font_size=16,
                                    height=500,
                                    width=700
                                )

                                st.plotly_chart(fig)
                            else:
                                st.write(
                                    "⚠️ التأثير المباشر وغير المباشر لهما إشارات متعاكسة، مما يجعل نسبة الوساطة صعبة التفسير.")
                    else:
                        # Multiple mediators
                        st.write("### تأثير كل متغير وسيط")

                        # Create bar chart of indirect effects using Plotly
                        indirect_effects = [mod.loc[2 * nm + 2 + i]["coef"] for i in range(nm)]
                        p_values = [mod.loc[2 * nm + 2 + i]["pval"] for i in range(nm)]

                        # Add significance markers
                        hovertext = []
                        for i in range(nm):
                            stars = get_significance_stars(p_values[i])
                            hovertext.append(f"p = {p_values[i]:.3f} {stars}")

                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=m,
                            y=indirect_effects,
                            text=[f"{val:.3f}{get_significance_stars(p)}" for val, p in
                                  zip(indirect_effects, p_values)],
                            textposition='auto',
                            hovertext=hovertext,
                            marker_color='#6495ED'
                        ))

                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            x1=len(m) - 0.5,
                            y0=0,
                            y1=0,
                            line=dict(color="black", width=2, dash="solid")
                        )

                        fig.update_layout(
                            title='التأثيرات غير المباشرة لكل متغير وسيط',
                            xaxis_title="المتغيرات الوسيطة",
                            yaxis_title="حجم التأثير",
                            height=500,
                            width=800
                        )

                        st.plotly_chart(fig)

                with post_tabs[1]:
                    # Compare models with and without mediators
                    st.write("### مقارنة نماذج الانحدار")

                    # Add educational note about model comparison
                    st.info("""
                    **أهمية مقارنة النماذج**:
                    - تساعد المقارنة في تحديد مدى إسهام المتغيرات الوسيطة في تفسير التباين
                    - زيادة R² تعني أن المتغيرات الوسيطة تضيف قدرة تفسيرية للنموذج
                    - انخفاض AIC/BIC يشير إلى تحسن جودة النموذج
                    """)

                    # Model without mediators (X -> Y)
                    X_no_med = sm.add_constant(df[[x] + cov]) if cov else sm.add_constant(df[x])
                    model_no_med = sm.OLS(df[y], X_no_med).fit()

                    # Model with mediators (X + M -> Y)
                    X_with_med = sm.add_constant(df[[x] + m + cov]) if cov else sm.add_constant(df[[x] + m])
                    model_with_med = sm.OLS(df[y], X_with_med).fit()

                    # Compare R-squared
                    r2_no_med = model_no_med.rsquared
                    r2_with_med = model_with_med.rsquared
                    r2_increase = r2_with_med - r2_no_med

                    st.write(f"- معامل التحديد R² بدون المتغيرات الوسيطة: {r2_no_med:.3f}")
                    st.write(f"- معامل التحديد R² مع المتغيرات الوسيطة: {r2_with_med:.3f}")
                    st.write(f"- الزيادة في معامل التحديد R²: {r2_increase:.3f}")

                    # Compare F statistics
                    st.write(f"- قيمة F للنموذج بدون المتغيرات الوسيطة: {model_no_med.fvalue:.3f}")
                    st.write(f"- قيمة F للنموذج مع المتغيرات الوسيطة: {model_with_med.fvalue:.3f}")

                    # Compare AIC and BIC
                    st.write(f"- معيار AIC للنموذج بدون المتغيرات الوسيطة: {model_no_med.aic:.3f}")
                    st.write(f"- معيار AIC للنموذج مع المتغيرات الوسيطة: {model_with_med.aic:.3f}")
                    st.write(f"- معيار BIC للنموذج بدون المتغيرات الوسيطة: {model_no_med.bic:.3f}")
                    st.write(f"- معيار BIC للنموذج مع المتغيرات الوسيطة: {model_with_med.bic:.3f}")

                    # Create comparison table
                    comparison_data = {
                        'النموذج': ['بدون متغيرات وسيطة', 'مع متغيرات وسيطة'],
                        'R²': [r2_no_med, r2_with_med],
                        'R² المعدل': [model_no_med.rsquared_adj, model_with_med.rsquared_adj],
                        'قيمة F': [model_no_med.fvalue, model_with_med.fvalue],
                        'القيمة الاحتمالية': [model_no_med.f_pvalue, model_with_med.f_pvalue],
                        'AIC': [model_no_med.aic, model_with_med.aic],
                        'BIC': [model_no_med.bic, model_with_med.bic]
                    }

                    comparison_df = pd.DataFrame(comparison_data).set_index('النموذج').round(3)
                    st.dataframe(comparison_df)

                    # Create bar chart comparing R-squared using Plotly
                    models = ['بدون متغيرات وسيطة', 'مع متغيرات وسيطة']
                    r2_values = [r2_no_med, r2_with_med]
                    colors = ['#87CEEB', '#6495ED']

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=models,
                        y=r2_values,
                        text=[f"{val:.3f}" for val in r2_values],
                        textposition='auto',
                        marker_color=colors
                    ))

                    fig.update_layout(
                        title='مقارنة معامل التحديد R² بين النماذج',
                        xaxis_title="النموذج",
                        yaxis_title="معامل التحديد R²",
                        height=500,
                        width=800,
                        yaxis=dict(range=[0, max(r2_values) * 1.2])
                    )

                    st.plotly_chart(fig)

                with post_tabs[2]:
                    # Additional visualizations
                    st.write("### رسوم بيانية إضافية")

                    # Scatter plot matrix
                    st.write("#### مصفوفة الانتشار للمتغيرات")
                    vars_to_plot = [x, y] + m[:2] if len(m) > 2 else [x, y] + m

                    fig = px.scatter_matrix(
                        df[vars_to_plot],
                        dimensions=vars_to_plot,
                        opacity=0.7,
                        title="العلاقات بين المتغيرات"
                    )

                    fig.update_layout(
                        height=700,
                        width=900
                    )

                    st.plotly_chart(fig)

                    # Standardized coefficients for path comparison
                    st.write("#### المعاملات المعيارية للمسارات")

                    # Educational note about standardized coefficients
                    st.info("""
                    **أهمية المعاملات المعيارية**:
                    - تتيح مقارنة قوة التأثير بين المتغيرات المختلفة بغض النظر عن وحدات القياس
                    - تعبر عن التغير في المتغير التابع بالانحراف المعياري عند تغير المتغير المستقل بانحراف معياري واحد
                    - تساعد في تحديد المسارات الأكثر أهمية في النموذج
                    """)

                    # Standardize variables
                    scaler = StandardScaler()
                    df_std = pd.DataFrame()

                    for var in [x, y] + m + cov:
                        df_std[var] = scaler.fit_transform(df[[var]])

                    # Run mediation on standardized data
                    mod_std = pg.mediation_analysis(data=df_std, x=x, y=y, m=m, covar=cov, seed=1235, n_boot=Nsim)

                    # Create bar chart of standardized effects using Plotly
                    if nm == 1:  # Single mediator
                        effects = [
                            mod_std.loc[0]["coef"],  # X -> M
                            mod_std.loc[1]["coef"],  # M -> Y
                            mod_std.loc[3]["coef"],  # X -> Y (direct)
                            mod_std.loc[2]["coef"],  # X -> Y (total)
                            mod_std.loc[4]["coef"]  # X -> M -> Y (indirect)
                        ]

                        effect_names = [
                            f'{x} → {m[0]}',
                            f'{m[0]} → {y}',
                            f'{x} → {y} (مباشر)',
                            f'{x} → {y} (كلي)',
                            f'{x} → {m[0]} → {y}'
                        ]

                        colors = ['#6495ED', '#6495ED', '#FF6347', '#90EE90', '#6A5ACD']

                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=effect_names,
                            y=effects,
                            text=[f"{val:.3f}" for val in effects],
                            textposition='auto',
                            marker_color=colors
                        ))

                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            x1=len(effect_names) - 0.5,
                            y0=0,
                            y1=0,
                            line=dict(color="black", width=2, dash="solid")
                        )

                        fig.update_layout(
                            title='المعاملات المعيارية للمسارات',
                            xaxis_title="المسارات",
                            yaxis_title="حجم التأثير المعياري",
                            height=500,
                            width=800,
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(fig)
                    else:
                        # Multiple mediators - focus on indirect effects
                        indirect_effects = []
                        effect_names = []

                        for i in range(nm):
                            indirect_effects.append(mod_std.loc[2 * nm + 2 + i]["coef"])
                            effect_names.append(f'{x} → {m[i]} → {y}')

                        # Add direct and total effects
                        indirect_effects.extend([
                            mod_std.loc[2 * nm + 1]["coef"],  # Direct effect
                            mod_std.loc[2 * nm]["coef"]  # Total effect
                        ])
                        effect_names.extend([
                            f'{x} → {y} (مباشر)',
                            f'{x} → {y} (كلي)'
                        ])

                        colors = ['#6495ED'] * nm + ['#FF6347', '#90EE90']

                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=effect_names,
                            y=indirect_effects,
                            text=[f"{val:.3f}" for val in indirect_effects],
                            textposition='auto',
                            marker_color=colors
                        ))

                        fig.add_shape(
                            type="line",
                            x0=-0.5,
                            x1=len(effect_names) - 0.5,
                            y0=0,
                            y1=0,
                            line=dict(color="black", width=2, dash="solid")
                        )

                        fig.update_layout(
                            title='مقارنة المعاملات المعيارية للتأثيرات',
                            xaxis_title="المسارات",
                            yaxis_title="حجم التأثير المعياري",
                            height=500,
                            width=800,
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(fig)

            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
                st.write("يرجى التأكد من عدم وجود قيم مفقودة في البيانات وأن المتغيرات المحددة صحيحة.")
                st.write("ملاحظة: تأكد من أن المتغيرات المختارة مختلفة وذات معنى للتحليل الإحصائي.")

    # Moderation Analysis
    elif medtype == 'التحليل المعدل او التفاعلي' and not df.empty:
        st.write("## التحليل المعدل (Moderation Analysis)")

        # Educational note about moderation
        st.markdown("""
        <div style="
            background-color: #e6f7ff; 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid #1890ff;
            margin-bottom: 20px;
            text-align: right;
            direction: rtl;">
            <h3>مفاهيم أساسية في التحليل المعدل:</h3>
            <p><strong>المتغير المعدل</strong> هو متغير يؤثر على قوة أو اتجاه العلاقة بين المتغير المستقل والمتغير التابع.</p>

            <h4>خطوات التحليل المعدل:</h4>
            <ol>
                <li><strong>الخطوة الأولى:</strong> تمركز المتغيرات المستقلة والمعدلة حول متوسطاتها (لتسهيل التفسير)</li>
                <li><strong>الخطوة الثانية:</strong> إنشاء متغير التفاعل (المتغير المستقل × المتغير المعدل)</li>
                <li><strong>الخطوة الثالثة:</strong> تضمين المتغيرات الأصلية ومتغير التفاعل في نموذج الانحدار</li>
                <li><strong>الخطوة الرابعة:</strong> اختبار دلالة معامل التفاعل</li>
            </ol>

            <h4>أنماط التعديل:</h4>
            <ul>
                <li><strong>تعديل تعزيزي:</strong> المتغير المعدل يقوي العلاقة بين المتغير المستقل والتابع</li>
                <li><strong>تعديل تثبيطي:</strong> المتغير المعدل يضعف العلاقة بين المتغير المستقل والتابع</li>
                <li><strong>تعديل متعاكس:</strong> المتغير المعدل يغير اتجاه العلاقة بين المتغير المستقل والتابع</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.form(key="moderation_form"):
            col1, col2 = st.columns(2)

            with col1:
                y = st.selectbox(
                    "المتغير التابع",
                    options=df.columns,
                    help="إختر المتغير التابع",
                    key="mod_y")

                x = st.selectbox(
                    "المتغير المستقل",
                    options=df.columns,
                    help="إختر المتغير المستقل",
                    key="mod_x")

            with col2:
                w = st.selectbox(
                    "المتغير المعدل",
                    options=df.columns,
                    help="إختر المتغير المعدل",
                    key="mod_w")

                cov = st.multiselect(
                    "اختر المتغيرات المشتركة (الضابطة)",
                    options=df.columns,
                    help="إختر متغير مشتركا او اكثر",
                    key="mod_cov")

            center_vars = st.checkbox("تمركز المتغيرات", value=True,
                                      help="تمركز المتغيرات يساعد في تفسير التفاعلات")

            submitted = st.form_submit_button("تنفيذ التحليل")

        if submitted and x and y and w:
            # Pre-test for moderation
            st.write("## فحص البيانات قبل التحليل")

            # Create tabs for organizing pre-test results
            pre_test_tabs = st.tabs(["الارتباطات", "المتغيرات", "الانحدار الأساسي"])

            with pre_test_tabs[0]:
                # Correlation analysis
                st.write("### مصفوفة الارتباط")
                variables = [x, y, w] + cov
                corr_matrix = df[variables].corr().round(3)

                # Heatmap for correlations using Plotly
                fig = px.imshow(corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)
                fig.update_layout(
                    title='مصفوفة الارتباط بين المتغيرات',
                    width=800,
                    height=600
                )
                st.plotly_chart(fig)

                # Table of correlations
                st.dataframe(corr_matrix)

            with pre_test_tabs[1]:
                # Variable distributions
                st.write("### توزيع المتغيرات")

                # Create subplot for distributions using Plotly
                fig = make_subplots(rows=1, cols=3,
                                    subplot_titles=[f'توزيع {x}', f'توزيع {y}', f'توزيع {w}'])

                # X distribution
                fig.add_trace(
                    go.Histogram(
                        x=df[x],
                        name=x,
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=1
                )

                # Add KDE for X
                kde_x, kde_y = get_kde_values(df[x])
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y * len(df[x]) * (kde_x[1] - kde_x[0]),  # Scale KDE to match histogram
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Y distribution
                fig.add_trace(
                    go.Histogram(
                        x=df[y],
                        name=y,
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=2
                )

                # Add KDE for Y
                kde_x, kde_y = get_kde_values(df[y])
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y * len(df[y]) * (kde_x[1] - kde_x[0]),
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # W distribution
                fig.add_trace(
                    go.Histogram(
                        x=df[w],
                        name=w,
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=3
                )

                # Add KDE for W
                kde_x, kde_y = get_kde_values(df[w])
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y * len(df[w]) * (kde_x[1] - kde_x[0]),
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=1, col=3
                )

                fig.update_layout(
                    height=400,
                    width=900,
                    title_text='توزيع المتغيرات',
                    showlegend=False,
                    bargap=0.1
                )

                st.plotly_chart(fig)

                # Scatter plot using Plotly
                st.write("### العلاقة بين المتغيرات")

                # Create scatter plot with moderator as color
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=w,
                    opacity=0.7,
                    color_continuous_scale='viridis',
                    title=f'العلاقة بين {x} و {y} بناءً على مستويات {w}'
                )

                fig.update_layout(
                    height=500,
                    width=800
                )

                st.plotly_chart(fig)

            with pre_test_tabs[2]:
                # Basic regression without interaction
                st.write("### الانحدار الأساسي بدون تفاعل")

                formula = f"{y} ~ {x} + {w}"
                if cov:
                    formula += " + " + " + ".join(cov)

                X_no_interact = sm.add_constant(df[[x, w] + cov]) if cov else sm.add_constant(df[[x, w]])
                model_no_interact = sm.OLS(df[y], X_no_interact).fit()

                st.write("#### ملخص نموذج الانحدار الأساسي")
                st.write(f"R² = {model_no_interact.rsquared:.3f}")
                st.write(f"R² المعدل = {model_no_interact.rsquared_adj:.3f}")
                st.write(f"قيمة F = {model_no_interact.fvalue:.3f}")
                st.write(f"القيمة الاحتمالية p = {model_no_interact.f_pvalue:.5f}")

                # Educational note about basic regression model
                st.info("""
                **تفسير نموذج الانحدار الأساسي**:
                - هذا النموذج يتضمن التأثيرات المباشرة للمتغير المستقل والمتغير المعدل فقط
                - لا يتضمن تأثير التفاعل بينهما
                - يُستخدم كنموذج أساسي للمقارنة مع نموذج التفاعل
                """)

                # Coefficients table
                coef_df = pd.DataFrame({
                    'المعاملات': model_no_interact.params,
                    'الخطأ المعياري': model_no_interact.bse,
                    'قيمة t': model_no_interact.tvalues,
                    'القيمة الاحتمالية': model_no_interact.pvalues
                }).round(3)
                st.dataframe(coef_df)

            # Main Moderation Analysis
            st.write("## نتائج تحليل التعديل")

            # Create centered variables if requested
            df_mod = df.copy()

            if center_vars:
                df_mod[f"{x}_c"] = df_mod[x] - df_mod[x].mean()
                df_mod[f"{w}_c"] = df_mod[w] - df_mod[w].mean()
                df_mod[f"{x}_c_{w}_c"] = df_mod[f"{x}_c"] * df_mod[f"{w}_c"]

                x_var = f"{x}_c"
                w_var = f"{w}_c"
                interact_var = f"{x}_c_{w}_c"

                st.write("### تم تمركز المتغيرات")
                st.write(f"- متوسط {x}: {df[x].mean():.3f}")
                st.write(f"- متوسط {w}: {df[w].mean():.3f}")

                # Educational note about centering
                st.info("""
                **فائدة تمركز المتغيرات**:
                - يسهل تفسير معاملات الانحدار (تمثل التأثير عند المتوسط)
                - يقلل من مشكلة الارتباط المتعدد بين المتغيرات ومتغير التفاعل
                - يجعل اختبار التأثيرات البسيطة أكثر دقة
                """)
            else:
                df_mod[f"{x}_{w}"] = df_mod[x] * df_mod[w]

                x_var = x
                w_var = w
                interact_var = f"{x}_{w}"

            # Create moderation model
            X_interact = sm.add_constant(df_mod[[x_var, w_var, interact_var] + cov]) if cov else sm.add_constant(
                df_mod[[x_var, w_var, interact_var]])
            model_interact = sm.OLS(df_mod[y], X_interact).fit()

            st.write("### نتائج نموذج التعديل")
            st.write(f"R² = {model_interact.rsquared:.3f}")
            st.write(f"R² المعدل = {model_interact.rsquared_adj:.3f}")
            st.write(f"قيمة F = {model_interact.fvalue:.3f}")
            st.write(f"القيمة الاحتمالية p = {model_interact.f_pvalue:.5f}")

            # Coefficients table
            coef_df = pd.DataFrame({
                'المعاملات': model_interact.params,
                'الخطأ المعياري': model_interact.bse,
                'قيمة t': model_interact.tvalues,
                'القيمة الاحتمالية': model_interact.pvalues
            }).round(3)
            st.dataframe(coef_df)

            # Educational note about interaction coefficient
            st.info("""
            **تفسير معامل التفاعل**:
            - **معامل موجب**: يشير إلى أن العلاقة بين المتغير المستقل والتابع تزداد قوة مع زيادة المتغير المعدل
            - **معامل سالب**: يشير إلى أن العلاقة بين المتغير المستقل والتابع تقل قوة مع زيادة المتغير المعدل
            - **القيمة p < 0.05**: تشير إلى أن تأثير التفاعل دال إحصائياً
            """)

            # Test of interaction
            st.write("### اختبار التفاعل")

            interaction_coef = model_interact.params[interact_var]
            interaction_p = model_interact.pvalues[interact_var]

            if interaction_p < 0.05:
                st.success(f"✅ التفاعل بين {x} و {w} دال إحصائياً (p = {interaction_p:.3f})")
                if interaction_coef > 0:
                    st.write(f"العلاقة بين {x} و {y} تزداد قوةً عندما يزداد {w}")
                else:
                    st.write(f"العلاقة بين {x} و {y} تقل قوةً عندما يزداد {w}")
            else:
                st.warning(f"⚠️ التفاعل بين {x} و {w} غير دال إحصائياً (p = {interaction_p:.3f})")

            # Compare models
            delta_r2 = model_interact.rsquared - model_no_interact.rsquared
            f_change = ((model_interact.rsquared - model_no_interact.rsquared) / (
                        len(model_interact.params) - len(model_no_interact.params))) / \
                       ((1 - model_interact.rsquared) / (len(df) - len(model_interact.params) - 1))

            p_change = 1 - stats.f.cdf(f_change, len(model_interact.params) - len(model_no_interact.params),
                                       len(df) - len(model_interact.params) - 1)

            st.write(f"الزيادة في R² بسبب التفاعل: {delta_r2:.3f} (F-change = {f_change:.3f}, p = {p_change:.3f})")

            # Visualize moderation
            st.write("### الرسوم البيانية للتعديل")

            # Create plots at different levels of moderator
            w_mean = df[w].mean()
            w_sd = df[w].std()

            w_low = w_mean - w_sd
            w_high = w_mean + w_sd

            x_min, x_max = df[x].min(), df[x].max()
            x_range = np.linspace(x_min, x_max, 100)

            # Calculate predicted values at different moderator levels
            if center_vars:
                # For centered variables
                y_low = model_interact.params['const'] + \
                        model_interact.params[x_var] * (x_range - df[x].mean()) + \
                        model_interact.params[w_var] * (w_low - df[w].mean()) + \
                        model_interact.params[interact_var] * (x_range - df[x].mean()) * (w_low - df[w].mean())

                y_mean = model_interact.params['const'] + \
                         model_interact.params[x_var] * (x_range - df[x].mean()) + \
                         model_interact.params[w_var] * (w_mean - df[w].mean()) + \
                         model_interact.params[interact_var] * (x_range - df[x].mean()) * (w_mean - df[w].mean())

                y_high = model_interact.params['const'] + \
                         model_interact.params[x_var] * (x_range - df[x].mean()) + \
                         model_interact.params[w_var] * (w_high - df[w].mean()) + \
                         model_interact.params[interact_var] * (x_range - df[x].mean()) * (w_high - df[w].mean())
            else:
                # For non-centered variables
                y_low = model_interact.params['const'] + \
                        model_interact.params[x_var] * x_range + \
                        model_interact.params[w_var] * w_low + \
                        model_interact.params[interact_var] * x_range * w_low

                y_mean = model_interact.params['const'] + \
                         model_interact.params[x_var] * x_range + \
                         model_interact.params[w_var] * w_mean + \
                         model_interact.params[interact_var] * x_range * w_mean

                y_high = model_interact.params['const'] + \
                         model_interact.params[x_var] * x_range + \
                         model_interact.params[w_var] * w_high + \
                         model_interact.params[interact_var] * x_range * w_high

            # Plot moderation effect using Plotly
            fig = go.Figure()

            # Add scatter points for actual data
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode='markers',
                    marker=dict(
                        color='gray',
                        opacity=0.3,
                        size=8
                    ),
                    name='البيانات الفعلية'
                )
            )

            # Add prediction lines
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_low,
                    mode='lines',
                    line=dict(color="#FF6347", width=2),
                    name=f"{w} منخفض (-1 SD: {w_low:.2f})"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_mean,
                    mode='lines',
                    line=dict(color="#4682B4", width=2),
                    name=f"{w} متوسط (المتوسط: {w_mean:.2f})"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_high,
                    mode='lines',
                    line=dict(color="#90EE90", width=2),
                    name=f"{w} مرتفع (+1 SD: {w_high:.2f})"
                )
            )

            fig.update_layout(
                title=f"تأثير {x} على {y} عند مستويات مختلفة من {w}",
                xaxis_title=x,
                yaxis_title=y,
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bordercolor="Black",
                    borderwidth=1
                ),
                height=500,
                width=800
            )

            st.plotly_chart(fig)

            # Additional perspective plot using Plotly
            st.write("### منظور ثلاثي الأبعاد للتفاعل")

            # Create a meshgrid
            x_grid = np.linspace(x_min, x_max, 30)
            w_grid = np.linspace(df[w].min(), df[w].max(), 30)
            X_grid, W_grid = np.meshgrid(x_grid, w_grid)

            # Calculate Z values
            if center_vars:
                Z_grid = model_interact.params['const'] + \
                         model_interact.params[x_var] * (X_grid - df[x].mean()) + \
                         model_interact.params[w_var] * (W_grid - df[w].mean()) + \
                         model_interact.params[interact_var] * (X_grid - df[x].mean()) * (W_grid - df[w].mean())
            else:
                Z_grid = model_interact.params['const'] + \
                         model_interact.params[x_var] * X_grid + \
                         model_interact.params[w_var] * W_grid + \
                         model_interact.params[interact_var] * X_grid * W_grid

            # Create 3D surface plot
            fig = go.Figure(data=[
                go.Surface(
                    x=X_grid,
                    y=W_grid,
                    z=Z_grid,
                    colorscale='viridis',
                    opacity=0.8
                )
            ])

            # Add actual data points
            fig.add_trace(
                go.Scatter3d(
                    x=df[x],
                    y=df[w],
                    z=df[y],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='black',
                        opacity=0.5
                    ),
                    name='البيانات الفعلية'
                )
            )

            fig.update_layout(
                title=f'العلاقة ثلاثية الأبعاد بين {x}، {w}، و{y}',
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=w,
                    zaxis_title=y
                ),
                height=700,
                width=800
            )

            st.plotly_chart(fig)

            # Gemini analysis
            prompt = f"""
            لديك نتائج تحليل التعديل (Moderation Analysis) حيث المتغير التابع هو {y}، المتغير المستقل هو {x}، والمتغير المعدل هو {w}.

            معامل التفاعل = {interaction_coef:.3f}
            قيمة p للتفاعل = {interaction_p:.3f}

            تغير في R² بسبب إضافة التفاعل = {delta_r2:.3f}

            قم بتحليل النتائج، واشرح هل التفاعل دال إحصائياً؟ وكيف يؤثر المتغير المعدل {w} على العلاقة بين المتغير المستقل {x} والمتغير التابع {y}؟
            اشرح معنى التفاعل بأسلوب سهل وعملي. اعط امثلة تطبيقية لهذا التفاعل.
            """

            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            analysis_text = response.text if response else "لم يتمكن Gemini من تحليل البيانات."

            # Show Gemini analysis
            st.subheader("📌 **تحليل Gemini**")
            st.markdown(
                f"<div style='background-color:#f0f8ff;padding:15px;border-radius:10px;border:1px solid #4682b4'>{analysis_text}</div>",
                unsafe_allow_html=True)

            # Additional analysis section
            st.write("## تحليلات إضافية")

            # Calculate simple slopes
            st.write("### تحليل الانحدارات البسيطة")

            # Educational note about simple slopes
            st.info("""
            **الانحدارات البسيطة (Simple Slopes)**:
            - تمثل تأثير المتغير المستقل على المتغير التابع عند مستويات محددة من المتغير المعدل
            - تساعد في فهم التفاعل بشكل أكثر تفصيلاً
            - عادة ما تُحسب عند: المتوسط، المتوسط + انحراف معياري واحد، المتوسط - انحراف معياري واحد
            """)

            # Calculate simple slopes
            if center_vars:
                # For centered variables, the slope is the coefficient of X plus the interaction term times W
                slope_low = model_interact.params[x_var] + model_interact.params[interact_var] * (w_low - df[w].mean())
                slope_mean = model_interact.params[x_var] + model_interact.params[interact_var] * (
                            w_mean - df[w].mean())
                slope_high = model_interact.params[x_var] + model_interact.params[interact_var] * (
                            w_high - df[w].mean())
            else:
                # For non-centered variables
                slope_low = model_interact.params[x_var] + model_interact.params[interact_var] * w_low
                slope_mean = model_interact.params[x_var] + model_interact.params[interact_var] * w_mean
                slope_high = model_interact.params[x_var] + model_interact.params[interact_var] * w_high

            # Create table for simple slopes
            slopes_data = {
                'مستوى المتغير المعدل': [f'{w} منخفض (-1 SD: {w_low:.2f})',
                                         f'{w} متوسط (المتوسط: {w_mean:.2f})',
                                         f'{w} مرتفع (+1 SD: {w_high:.2f})'],
                'معامل الانحدار': [slope_low, slope_mean, slope_high]
            }

            slopes_df = pd.DataFrame(slopes_data)
            st.dataframe(slopes_df.round(3))

            # Plot simple slopes
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=slopes_df['مستوى المتغير المعدل'],
                    y=slopes_df['معامل الانحدار'],
                    text=[f"{val:.3f}" for val in slopes_df['معامل الانحدار']],
                    textposition='auto',
                    marker_color=['#FF6347', '#4682B4', '#90EE90']
                )
            )

            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=2.5,
                y0=0,
                y1=0,
                line=dict(color="black", width=2, dash="solid")
            )

            fig.update_layout(
                title='معاملات الانحدار (Simple Slopes) عند مستويات مختلفة من المتغير المعدل',
                xaxis_title="مستوى المتغير المعدل",
                yaxis_title=f"تأثير {x} على {y}",
                height=500,
                width=800
            )

            st.plotly_chart(fig)

            # Johnson-Neyman technique
            st.write("### نقطة التحول (Johnson-Neyman Technique)")

            # Educational note about Johnson-Neyman
            st.info("""
            **تقنية Johnson-Neyman**:
            - تحدد القيمة (أو القيم) الدقيقة للمتغير المعدل التي يتحول عندها تأثير المتغير المستقل من دال إلى غير دال
            - تعطي نطاق قيم المتغير المعدل حيث يكون التأثير دالاً إحصائياً
            - تساعد في فهم حدود التفاعل بشكل أدق من مجرد اختبار الانحدارات البسيطة
            """)

            # Only calculate if interaction is significant
            if interaction_p < 0.05:
                # Calculate Johnson-Neyman point
                if center_vars:
                    # For centered variables
                    jn_point = -model_interact.params[x_var] / model_interact.params[interact_var]
                    jn_point_original = jn_point + df[w].mean()
                else:
                    # For non-centered variables
                    jn_point = -model_interact.params[x_var] / model_interact.params[interact_var]
                    jn_point_original = jn_point

                # Check if J-N point is within the range of the moderator
                if df[w].min() <= jn_point_original <= df[w].max():
                    st.write(f"نقطة التحول (Johnson-Neyman): {jn_point_original:.3f}")
                    st.write(f"عند هذه القيمة من {w}، يتحول تأثير {x} على {y} من الدلالة إلى عدم الدلالة الإحصائية.")

                    # Create regions of significance plot
                    w_range = np.linspace(df[w].min(), df[w].max(), 100)

                    if center_vars:
                        slopes = model_interact.params[x_var] + model_interact.params[interact_var] * (
                                    w_range - df[w].mean())
                    else:
                        slopes = model_interact.params[x_var] + model_interact.params[interact_var] * w_range

                    # Calculate standard errors and t-values
                    se_slopes = np.sqrt(model_interact.bse[x_var] ** 2 +
                                        (w_range - df[w].mean()) ** 2 * model_interact.bse[interact_var] ** 2 +
                                        2 * (w_range - df[w].mean()) * model_interact.cov_params().loc[
                                            x_var, interact_var])

                    t_values = slopes / se_slopes
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df.shape[0] - len(model_interact.params)))

                    # Create significance regions plot
                    fig = go.Figure()

                    # Add slopes line
                    fig.add_trace(
                        go.Scatter(
                            x=w_range,
                            y=slopes,
                            mode='lines',
                            line=dict(color='blue', width=2),
                            name='معامل الانحدار'
                        )
                    )

                    # Add significance threshold
                    fig.add_trace(
                        go.Scatter(
                            x=w_range,
                            y=np.zeros_like(w_range),
                            mode='lines',
                            line=dict(color='black', dash='dash'),
                            name='الصفر'
                        )
                    )

                    # Add Johnson-Neyman point
                    fig.add_trace(
                        go.Scatter(
                            x=[jn_point_original, jn_point_original],
                            y=[min(slopes), max(slopes)],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='نقطة التحول'
                        )
                    )

                    # Color regions of significance
                    significant_region = p_values < 0.05

                    # Find the indices where significance changes
                    change_indices = np.where(np.diff(significant_region))[0]

                    # Add shaded regions
                    for i in range(len(change_indices) + 1):
                        if i == 0:
                            start_idx = 0
                        else:
                            start_idx = change_indices[i - 1] + 1

                        if i == len(change_indices):
                            end_idx = len(w_range) - 1
                        else:
                            end_idx = change_indices[i]

                        if significant_region[start_idx]:
                            fig.add_trace(
                                go.Scatter(
                                    x=w_range[start_idx:end_idx + 1],
                                    y=slopes[start_idx:end_idx + 1],
                                    mode='none',
                                    fill='tozeroy',
                                    fillcolor='rgba(0, 255, 0, 0.2)',
                                    name='منطقة دالة إحصائياً',
                                    showlegend=i == 0
                                )
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=w_range[start_idx:end_idx + 1],
                                    y=slopes[start_idx:end_idx + 1],
                                    mode='none',
                                    fill='tozeroy',
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    name='منطقة غير دالة إحصائياً',
                                    showlegend=i == 0
                                )
                            )

                    fig.update_layout(
                        title=f'مناطق الدلالة الإحصائية لتأثير {x} على {y} عبر قيم {w}',
                        xaxis_title=w,
                        yaxis_title=f'تأثير {x} على {y}',
                        height=500,
                        width=800
                    )

                    st.plotly_chart(fig)
                else:
                    st.write("لا توجد نقطة تحول (Johnson-Neyman) ضمن نطاق قيم المتغير المعدل في البيانات.")


# Function to get significance stars for p-values
def get_significance_stars(p_value):
    if p_value < 0.001:
        return " ***"
    elif p_value < 0.01:
        return " **"
    elif p_value < 0.05:
        return " *"
    else:
        return ""


# Function to calculate KDE values
def get_kde_values(data, bw_method='scott'):
    kde = stats.gaussian_kde(data, bw_method=bw_method)
    x_min, x_max = min(data), max(data)
    x_range = np.linspace(x_min - 0.1 * (x_max - x_min),
                          x_max + 0.1 * (x_max - x_min),
                          1000)
    y_kde = kde(x_range)
    return x_range, y_kde


if __name__ == "__main__":
    mediations()