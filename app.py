from datetime import datetime
import pandas as pd
import streamlit as st

from src.model_service import evaluate_application, model_status

st.set_page_config(page_title="CreditRisk AI | Nhóm 14", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

# Visual refresh only: inference, preprocessing, grade thresholds, SHAP and clustering
# remain delegated to src.model_service without changing their logic.
st.markdown("""
<style>
:root {
  --navy:#16345a; --ink:#213653; --muted:#657790; --blue:#3267e9;
  --blue-deep:#244fc7; --cyan:#0891b2; --violet:#7454e8; --mint:#11a99a;
  --line:#dce7f5; --surface:#ffffff; --wash:#f4f8ff;
  --shadow:0 8px 24px rgba(31,70,130,.07);
}
.stApp {
  background: radial-gradient(ellipse at 8% 0%, rgba(101,151,255,.12), transparent 34%),
              radial-gradient(ellipse at 96% 12%, rgba(60,207,220,.10), transparent 28%),
              linear-gradient(135deg,#f7faff 0%,#f1f6ff 52%,#f7fbff 100%);
  color:var(--ink);
}
[data-testid="stHeader"] { background:rgba(247,250,255,.84); }
[data-testid="stToolbar"] { right:1.2rem; }
.block-container { max-width:1440px; padding-top:1.45rem; padding-bottom:3.2rem; }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#fff 0%,#f7faff 58%,#eff6ff 100%);
  border-right:1px solid #dbe7f7; box-shadow:5px 0 24px rgba(24,59,110,.035);
}
[data-testid="stSidebar"] .block-container { padding-top:1.45rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
.brand-mark {
  width:46px;height:46px;border-radius:15px;
  background:linear-gradient(145deg,#3679f6 0%,#5d58e8 62%,#8a4fe5 100%);
  color:#fff;display:inline-flex;align-items:center;justify-content:center;
  font-size:25px;font-weight:800;margin-right:11px;vertical-align:middle;
  box-shadow:0 7px 16px rgba(68,99,226,.24);border:1px solid rgba(255,255,255,.7);
}
.brand-title { display:inline-block;vertical-align:middle;font-size:1.25rem;font-weight:800;color:#17365d;line-height:1.2;letter-spacing:-.025em; }
.brand-sub { font-size:.83rem;color:#657790;margin-top:9px;line-height:1.65; }
.brand-sub b { color:#315b99; }
.hero {
  position:relative;isolation:isolate;overflow:hidden;
  background:linear-gradient(112deg,rgba(255,255,255,.98) 0%,#f0f6ff 56%,#eafaff 100%);
  border:1px solid #d7e6fa;border-radius:21px;padding:1.65rem 1.8rem;
  margin:.15rem 0 1.35rem;box-shadow:0 9px 26px rgba(35,78,145,.055);
}
.hero::before {
  content:"";position:absolute;z-index:-1;width:250px;height:250px;right:3%;top:-145px;
  border-radius:50%;background:radial-gradient(circle,rgba(76,126,245,.15),rgba(76,126,245,0) 70%);
}
.hero::after {
  content:"";position:absolute;z-index:-1;width:170px;height:170px;right:8%;bottom:-120px;
  border-radius:50%;border:25px solid rgba(14,165,233,.055);
}
.eyebrow {
  display:inline-flex;align-items:center;gap:.42rem;text-transform:uppercase;letter-spacing:.13em;
  color:#315fe0;font-weight:800;font-size:.71rem;margin-bottom:.65rem;
  padding:.35rem .68rem;border-radius:999px;background:linear-gradient(90deg,#e8efff,#e7faff);
  border:1px solid #d9e7ff;
}
.eyebrow::before { content:"✦";font-size:.8rem;color:#7454e8; }
.hero h1 { color:#17365d;font-size:2.08rem;line-height:1.2;margin:0;font-weight:820;letter-spacing:-.035em; }
.hero p { color:#536985;font-size:1rem;margin:.68rem 0 0;max-width:880px;line-height:1.7; }
.section-head {
  display:flex;align-items:center;gap:.6rem;color:#1d416d;font-size:1.08rem;
  font-weight:780;margin:.45rem 0 .35rem;letter-spacing:-.012em;
}
.section-head::before {
  content:"✦";flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;
  width:27px;height:27px;border-radius:9px;color:#fff;font-size:.75rem;
  background:linear-gradient(145deg,#3679f6,#7952e8);box-shadow:0 4px 9px rgba(76,100,220,.17);
}
.section-note { color:#718199;font-size:.86rem;margin:0 0 .95rem .1rem;line-height:1.55; }
.panel {
  background:linear-gradient(145deg,#fff 0%,#fbfdff 100%);border:1px solid var(--line);
  border-radius:17px;padding:1.1rem 1.2rem;box-shadow:0 5px 18px rgba(31,70,130,.045);
}
.panel b { color:#254c80; }
.kpi-label { color:#657790;font-size:.82rem;font-weight:650; }
.kpi-value { color:#173b68;font-size:1.8rem;font-weight:800;margin-top:.25rem; }
.kpi-note { color:#718199;font-size:.78rem;margin-top:.15rem; }
.pill { display:inline-block;padding:.3rem .7rem;border-radius:999px;background:#e9efff;color:#315fe0;font-size:.78rem;font-weight:750;border:1px solid #d7e3ff; }
/* Metric cards: a clean accent edge plus a gentle hover lift. */
div[data-testid="stMetric"] {
  position:relative;overflow:hidden;background:linear-gradient(150deg,#fff 0%,#f9fbff 100%);
  border:1px solid #dce7f5;border-top:3px solid #4f7cf0;padding:1.02rem 1.12rem;
  border-radius:16px;box-shadow:0 5px 16px rgba(25,66,128,.045);transition:transform .18s ease,box-shadow .18s ease;
}
div[data-testid="stMetric"]:hover { transform:translateY(-2px);box-shadow:0 10px 22px rgba(25,66,128,.09); }
div[data-testid="stMetricLabel"] { color:#647894;font-weight:650; }
div[data-testid="stMetricValue"] { color:#193f72;font-weight:800;letter-spacing:-.025em; }
div[data-testid="stMetricDelta"] { font-weight:650; }
[data-testid="stForm"] {
  background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(249,252,255,.98));
  border:1px solid #dbe7f7;border-radius:19px;padding:1.25rem 1.35rem;
  box-shadow:0 7px 23px rgba(27,65,125,.045);
}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  border-radius:10px!important;border-color:#d5e1f1!important;background:#fff!important;
  transition:border-color .15s ease,box-shadow .15s ease;
}
[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus {
  border-color:#6794f6!important;box-shadow:0 0 0 3px rgba(50,103,233,.10)!important;
}
[data-testid="stRadio"] label { border-radius:10px; }
[data-testid="stRadio"] label:hover { color:#315fe0; }
.stButton > button,.stDownloadButton > button,button[kind="primary"] {
  border-radius:11px;font-weight:720;min-height:2.7rem;transition:all .18s ease;
}
.stButton > button[kind="primary"],button[kind="primary"] {
  background:linear-gradient(100deg,#3267e9 0%,#5d58e8 100%);border:1px solid #4269e7;
  color:white;box-shadow:0 5px 13px rgba(55,94,218,.18);
}
.stButton > button[kind="primary"]:hover,button[kind="primary"]:hover {
  background:linear-gradient(100deg,#2858d4 0%,#5144d6 100%);border-color:#345bdc;
  box-shadow:0 8px 17px rgba(55,94,218,.23);transform:translateY(-1px);
}
.stDownloadButton > button,.stButton > button:not([kind="primary"]) {
  border-color:#cfddf2;color:#315d9a;background:linear-gradient(180deg,#fff,#f7faff);
}
.stDownloadButton > button:hover,.stButton > button:not([kind="primary"]):hover { border-color:#8eaff5;color:#244fc7;background:#f2f7ff; }
[data-testid="stAlert"] { border-radius:13px;border:1px solid rgba(115,145,190,.18); }
[data-testid="stDataFrame"] { border:1px solid #dce7f5;border-radius:13px;overflow:hidden;box-shadow:0 3px 13px rgba(31,70,130,.035); }
[data-testid="stTable"] { border-radius:12px; }
hr { border-color:#e0e9f6;margin:1.2rem 0; }
[data-testid="stCaptionContainer"] { color:#78879c; }
.footer-note { color:#718199;font-size:.78rem;border-top:1px solid #dfe8f5;margin-top:2.2rem;padding:1.1rem .15rem 0;line-height:1.7; }
.footer-note::before { content:"✦";color:#5e70e8;margin-right:.45rem; }
@media (max-width: 700px) {
  .hero { padding:1.15rem 1.2rem;border-radius:16px; }
  .hero h1 { font-size:1.55rem; }
  .block-container { padding-left:1rem;padding-right:1rem; }
  [data-testid="stForm"] { padding:1rem; }
  div[data-testid="stMetric"] { padding:.85rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none!important; } }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

status = model_status()
with st.sidebar:
    st.markdown('<span class="brand-mark">✦</span><span class="brand-title">CreditRisk AI</span>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Phân tích rủi ro tín dụng<br><b>Machine Learning · Nhóm 14</b></div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("KHÔNG GIAN LÀM VIỆC", ["Tổng quan & kiểm tra", "Lịch sử hồ sơ", "Thông tin hệ thống"], label_visibility="visible")
    st.divider()
    if status["ready"]:
        st.success("Mô hình đã được tìm thấy")
    else:
        st.warning("Chưa tìm thấy mô hình")
    st.markdown('<div class="brand-sub">Bản demo phục vụ học tập. Không dùng để phê duyệt tín dụng thực tế.</div>', unsafe_allow_html=True)

if page == "Tổng quan & kiểm tra":
    st.markdown('''<div class="hero"><div class="eyebrow">CREDIT RISK ANALYTICS · GROUP 14</div><h1>Đánh giá rủi ro tín dụng</h1><p>Phân tích hồ sơ vay bằng Random Forest, xem hạng tín dụng A–D, phân khúc khách hàng và các yếu tố đóng góp vào dự báo.</p></div>''', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("🧠 Mô hình chính", "Random Forest", help="Mô hình được lựa chọn trong báo cáo Nhóm 14")
    with h2:
        st.metric("📈 ROC-AUC báo cáo", "92,10%")
    with h3:
        st.metric("🎯 Recall báo cáo", "73,91%")
    st.markdown('<div class="section-head">🧾 Kiểm tra hồ sơ mới</div><div class="section-note">Nhập thông tin theo đúng đơn vị của bộ dữ liệu dùng để huấn luyện.</div>', unsafe_allow_html=True)
    if not status["ready"]:
        st.info("Giao diện đã sẵn sàng. Để dự báo thật, hãy tạo artifact bằng `python scripts/train_export.py` sau khi đặt `credit_risk_cleaned.csv` ở thư mục gốc. Khi thiếu mô hình, ứng dụng không tự tạo xác suất giả.")

    with st.form("application_form"):
        st.markdown('<div class="section-head">👤 01 · Thông tin khách hàng</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=28, step=1)
            income = st.number_input("Thu nhập hằng năm", min_value=1, value=60000, step=1000, help="Dùng cùng đơn vị với dữ liệu huấn luyện.")
        with c2:
            emp_length = st.number_input("Số năm làm việc", min_value=0, max_value=70, value=4, step=1)
            home = st.selectbox("Tình trạng nhà ở", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        with c3:
            default_history = st.selectbox("Từng vỡ nợ (lịch sử tín dụng)", ["N", "Y"])
            purpose = st.selectbox("Mục đích vay", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        st.divider()
        st.markdown('<div class="section-head">💳 02 · Thông tin khoản vay</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            loan_amount = st.number_input("Số tiền vay", min_value=500, max_value=35000, value=8000, step=500)
        with c5:
            interest_rate = st.number_input("Lãi suất (%)", min_value=0.0, max_value=30.0, value=10.5, step=0.25)
        with c6:
            loan_percent_income = st.number_input("Tỷ lệ khoản vay / thu nhập", min_value=0.0, max_value=1.0, value=min(8000 / 60000, 1.0), step=0.01, format="%.2f", help="Biến loan_percent_income theo định nghĩa trong dữ liệu.")
        submitted = st.form_submit_button("✨ Phân tích hồ sơ", type="primary", use_container_width=True)

    if submitted:
        record = {"person_age": int(age), "person_income": float(income), "person_home_ownership": home,
                  "person_emp_length": int(emp_length), "loan_intent": purpose, "loan_amnt": float(loan_amount),
                  "loan_int_rate": float(interest_rate), "loan_percent_income": float(loan_percent_income),
                  "cb_person_default_on_file": default_history}
        result = evaluate_application(record)
        result["application"] = record
        result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_result = result
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:100]

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.markdown('<div class="section-head">📊 03 · Kết quả phân tích</div><div class="section-note">Kết quả gắn với hồ sơ vừa gửi lên mô hình.</div>', unsafe_allow_html=True)
        if result["mode"] == "model":
            st.success("Kết quả được tạo từ artifact mô hình đã nạp.")
        else:
            st.warning("CHẾ ĐỘ CHƯA TÍCH HỢP MÔ HÌNH — không sử dụng kết quả minh họa để ra quyết định.")
        a, b, c = st.columns(3)
        with a: st.metric("⚠️ Xác suất vỡ nợ (PD)", result.get("pd_display", "Chưa có"))
        with b: st.metric("🏅 Hạng tín dụng", result.get("grade", "Chưa xác định"))
        with c: st.metric("👥 Phân khúc khách hàng", result.get("cluster", "Chưa có mô hình"))
        st.markdown('<div class="section-head">💡 Diễn giải kết quả</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">' + str(result.get("summary", "Chưa có phần diễn giải.")) + '</div>', unsafe_allow_html=True)
        if result.get("factors"):
            st.markdown('<div class="section-head" style="margin-top:1.1rem">🧭 Các yếu tố ảnh hưởng · SHAP</div><div class="section-note">Giá trị SHAP mô tả đóng góp của đặc trưng vào dự báo theo cách biểu diễn của mô hình.</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(result["factors"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Chưa có SHAP explainer/model tương thích. Phần dự báo và phân hạng có thể hiển thị độc lập nếu artifact hợp lệ.")
        st.caption("Hạng A–D sử dụng các ngưỡng Q1/Q2/Q3 đã lưu trong bundle; không suy ra ngưỡng mới từ từng hồ sơ.")

elif page == "Lịch sử hồ sơ":
    st.markdown('<div class="hero"><div class="eyebrow">APPLICATION LOG</div><h1>📚 Lịch sử hồ sơ</h1><p>Theo dõi các hồ sơ đã phân tích trong phiên làm việc hiện tại và xuất dữ liệu để phục vụ báo cáo.</p></div>', unsafe_allow_html=True)
    history = st.session_state.history
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("📂 Tổng hồ sơ trong phiên", len(history))
    with k2: st.metric("🤖 Dự báo từ mô hình", sum(1 for x in history if x.get("mode") == "model"))
    with k3: st.metric("🧪 Chế độ minh họa", sum(1 for x in history if x.get("mode") != "model"))
    st.caption("Lịch sử chỉ lưu trong phiên Streamlit; khi khởi động lại ứng dụng, dữ liệu trong phiên sẽ bị xóa.")
    if not history:
        st.info("Chưa có hồ sơ nào được kiểm tra trong phiên này. Hãy tạo dự báo tại trang Tổng quan & kiểm tra.")
    else:
        rows = []
        for i, item in enumerate(history, start=1):
            app = item["application"]
            rows.append({"STT": i, "Thời gian": item["created_at"], "Tuổi": app["person_age"], "Thu nhập": app["person_income"],
                         "Số tiền vay": app["loan_amnt"], "PD": item.get("pd_display", "Chưa có"), "Hạng": item.get("grade", "Chưa xác định"),
                         "Phân khúc": item.get("cluster", "Chưa có"), "Chế độ": "Mô hình thật" if item["mode"] == "model" else "Chưa tích hợp"})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Tải lịch sử CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="lich_su_ho_so.csv", mime="text/csv")

elif page == "Thông tin hệ thống":
    st.markdown('<div class="hero"><div class="eyebrow">MODEL & SYSTEM</div><h1>⚙️ Thông tin hệ thống</h1><p>Tóm tắt mô hình theo báo cáo Nhóm 14 và kiểm tra trạng thái các thành phần đang có trong project.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">📘 Cơ sở theo báo cáo Nhóm 14</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><b>🎯 Mục tiêu</b><p>Dự báo <code>loan_status</code>: 0 — không vỡ nợ; 1 — vỡ nợ.</p><b>🌳 Mô hình lựa chọn</b><p>Random Forest · ROC-AUC 92,10% · Recall 73,91% theo kết quả báo cáo.</p><b>🏅 Xếp hạng tín dụng</b><p>A–D dựa trên các phân vị PD Q1, Q2, Q3.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><b>🔎 Giải thích mô hình</b><p>SHAP dùng để phân tích các yếu tố đóng góp vào dự báo.</p><b>👥 Phân khúc khách hàng</b><p>K-Means với 4 cụm theo lựa chọn trong báo cáo.</p><b>⚙️ Quy trình</b><p>Tiền xử lý và dự báo được xử lý trong module dịch vụ mô hình.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head" style="margin-top:1.25rem">🧩 Trạng thái artifact</div><div class="section-note">Đường dẫn và trạng thái được đọc từ project hiện tại.</div>', unsafe_allow_html=True)
    state_rows = [
        {"Thành phần": "Random Forest bundle", "Đường dẫn": status["pipeline_path"], "Trạng thái": "Đã tìm thấy" if status["ready"] else "Chưa tìm thấy"},
        {"Thành phần": "K-Means", "Đường dẫn": status["cluster_path"], "Trạng thái": "Đã tìm thấy" if status["cluster_ready"] else "Chưa tìm thấy"},
        {"Thành phần": "SHAP", "Đường dẫn": "Được kiểm tra khi dự báo", "Trạng thái": "Có artifact bundle" if status["shap_available"] else "Chưa khả dụng"},
        {"Thành phần": "Ngưỡng A–D", "Đường dẫn": status["thresholds_path"], "Trạng thái": "Đã tìm thấy bundle" if status["ready"] else "Chưa tìm thấy"},
    ]
    st.dataframe(pd.DataFrame(state_rows), hide_index=True, use_container_width=True)
    st.info("Để tạo/cập nhật artifact theo notebook, đặt `credit_risk_cleaned.csv` ở thư mục gốc rồi chạy `python scripts/train_export.py`.")
    st.warning("Prototype phục vụ học tập. Không nhập dữ liệu định danh hoặc dữ liệu tài chính thật của khách hàng. Chỉ số trong báo cáo không đảm bảo hiệu năng của mọi lần huấn luyện lại.")

st.markdown('<div class="footer-note">CreditRisk AI · Machine Learning Nhóm 14 · Bản demo học tập — không thay thế quy trình thẩm định tín dụng chuyên môn.</div>', unsafe_allow_html=True)
