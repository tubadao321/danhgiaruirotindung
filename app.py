from datetime import datetime
import json
import pandas as pd
import streamlit as st
import psycopg
from google import genai

from src.model_service import evaluate_application, model_status


def get_database_url():
    """Read the shared PostgreSQL URL from Streamlit secrets."""
    try:
        return st.secrets.get("DATABASE_URL", "")
    except Exception:
        return ""


def db_connect():
    url = get_database_url()
    if not url:
        return None
    return psycopg.connect(url, connect_timeout=10, sslmode="require")


def init_history_table():
    """Create the shared history table if it does not exist."""
    conn = db_connect()
    if conn is None:
        return False, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS credit_risk_history (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB NOT NULL
                    )"""
                )
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def save_history_item(item):
    """Save one prediction to the shared PostgreSQL database."""
    conn = db_connect()
    if conn is None:
        return False, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        payload = json.dumps(item, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO credit_risk_history (payload) VALUES (%s::jsonb)",
                    (payload,),
                )
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def get_gemini_client():
    """Create Gemini client from Streamlit secrets or environment variables."""
    import os

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not str(api_key).strip():
        return None
    return genai.Client(api_key=str(api_key).strip())


def _friendly_gemini_error(exc):
    """Return a useful diagnostic without exposing secrets or raw request details."""
    message = str(exc).lower()
    error_type = type(exc).__name__
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status_text = str(status) if status else ""

    if any(term in message for term in ("api_key_invalid", "invalid api key", "api key not valid", "unauthorized", "unauthenticated")) or status_text in {"401", "UNAUTHENTICATED"}:
        return "Gemini từ chối API key. Hãy kiểm tra GEMINI_API_KEY trong Streamlit Secrets và tạo key mới nếu cần."
    if any(term in message for term in ("permission_denied", "permission denied", "forbidden")) or status_text in {"403", "PERMISSION_DENIED"}:
        return "Gemini API không cấp quyền cho yêu cầu này. Kiểm tra API key, project và quyền Gemini API trong Google AI Studio."
    if any(term in message for term in ("resource_exhausted", "quota", "rate limit", "too many requests")) or status_text in {"429", "RESOURCE_EXHAUSTED"}:
        return "Gemini API đã chạm hạn mức hoặc giới hạn tốc độ. Kiểm tra quota/billing trong Google AI Studio rồi thử lại sau."
    if any(term in message for term in ("not_found", "model not found", "not supported for generatecontent", "unknown model")) or status_text == "404":
        return "Model Gemini hiện không khả dụng với API key này. Hãy kiểm tra danh sách model được hỗ trợ hoặc chọn model Flash khả dụng."
    if any(term in message for term in ("deadline_exceeded", "timed out", "timeout", "connection reset", "temporarily unavailable")):
        return "Kết nối tới Gemini bị gián đoạn hoặc quá thời gian chờ. Vui lòng thử lại sau."
    if "billing" in message or "billing_not_enabled" in message:
        return "Project Gemini chưa bật hoặc chưa đủ điều kiện sử dụng API theo chính sách thanh toán hiện tại. Kiểm tra trạng thái project trong Google AI Studio."
    return f"Gemini gặp lỗi {error_type}" + (f" (mã {status_text})" if status_text else ".") + ". Kiểm tra cấu hình API và thử lại."


def ask_gemini(question, current_result, conversation):
    """Answer using current prediction context; fall back only when a model is unavailable."""
    try:
        client = get_gemini_client()
    except Exception as exc:
        return None, _friendly_gemini_error(exc)
    if client is None:
        return None, "Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets."

    # Minimize data sent to the model: only model-relevant application fields/results.
    context = {}
    if isinstance(current_result, dict):
        context = {
            "application": current_result.get("application", {}),
            "mode": current_result.get("mode"),
            "pd_display": current_result.get("pd_display"),
            "grade": current_result.get("grade"),
            "cluster": current_result.get("cluster"),
            "summary": current_result.get("summary"),
            "factors": current_result.get("factors", []),
        }
    system_instruction = """Bạn là CreditRisk AI Assistant, trợ lý giải thích kết quả mô hình rủi ro tín dụng cho mục đích học tập. Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn (thường 100–180 từ, trừ khi người dùng yêu cầu chi tiết). Ưu tiên trả lời trực tiếp câu hỏi. Dựa trên dữ liệu hồ sơ và kết quả được cung cấp; không tự bịa số liệu, không khẳng định quan hệ nhân quả từ SHAP, không coi dự báo là kết luận chắc chắn. Nếu thiếu dữ liệu, hãy nói rõ. Nêu rằng đây chỉ là công cụ tham khảo, không thay thế thẩm định tín dụng chuyên môn và không đưa ra cam kết được duyệt vay. Đề xuất cải thiện chỉ ở mức thông tin chung, không khuyến khích khai sai dữ liệu. Không yêu cầu người dùng cung cấp thông tin định danh như CCCD, số tài khoản, mật khẩu."""
    recent = conversation[-4:] if conversation else []
    prompt = (
        system_instruction
        + "\n\nNGỮ CẢNH KẾT QUẢ HỒ SƠ HIỆN TẠI (JSON):\n"
        + json.dumps(context, ensure_ascii=False, default=str)
        + "\n\nHỘI THOẠI GẦN ĐÂY (JSON):\n"
        + json.dumps(recent, ensure_ascii=False, default=str)
        + "\n\nCÂU HỎI MỚI CỦA NGƯỜI DÙNG:\n"
        + question
    )

    # Use a current Flash model with low thinking effort for a user-facing chatbot.
    # Only fall back when the requested model is unavailable (404/model-not-found).
    models_to_try = ("gemini-3.8-flash", "gemini-3.6-flash")
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "max_output_tokens": 600,
                    "thinking_config": {"thinking_level": "low"},
                },
            )
            answer = getattr(response, "text", None)
            if not answer or not answer.strip():
                return None, "Gemini đã phản hồi nhưng không có nội dung. Vui lòng thử câu hỏi khác."
            return answer.strip(), ""
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            model_unavailable = (
                any(term in message for term in ("not_found", "model not found", "not supported for generatecontent", "unknown model"))
                or str(getattr(exc, "code", "")) == "404"
                or str(getattr(exc, "status_code", "")) == "404"
            )
            if model_unavailable:
                continue
            # Do not expose exception text: some SDK errors may include request details.
            print(f"Gemini request failed: {type(exc).__name__}; model={model_name}; status={getattr(exc, 'code', '') or getattr(exc, 'status_code', '')}")
            return None, _friendly_gemini_error(exc)

    if last_error is not None:
        print(f"Gemini request failed: {type(last_error).__name__}; all configured Flash models unavailable")
        return None, _friendly_gemini_error(last_error)
    return None, "Chưa tìm được model Gemini khả dụng. Vui lòng kiểm tra cấu hình API."


def load_history(limit=500):
    """Return recent shared predictions, newest first."""
    conn = db_connect()
    if conn is None:
        return None, "Chưa cấu hình DATABASE_URL trong Streamlit Secrets."
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM credit_risk_history ORDER BY created_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [row[0] for row in rows], ""
    except Exception as exc:
        return None, str(exc)
    finally:
        conn.close()


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
/* Floating AI assistant launcher and chat panel */
.st-key-ai_floating_button { position: fixed !important; right: 24px !important; bottom: 22px !important; z-index: 99999 !important; width: 64px !important; }
.st-key-ai_floating_button button { width: 58px !important; height: 58px !important; border-radius: 50% !important; border: 0 !important; color: white !important; background: linear-gradient(135deg,#3267e9,#7454e8) !important; box-shadow: 0 8px 28px rgba(50,103,233,.38) !important; font-size: 25px !important; }
.st-key-ai_chat_panel { position: fixed !important; right: 24px !important; bottom: 92px !important; width: min(410px, calc(100vw - 28px)) !important; max-height: min(620px, calc(100vh - 125px)) !important; overflow-y: auto !important; z-index: 99998 !important; background: white !important; border: 1px solid #dce7f5 !important; border-radius: 18px !important; padding: 14px !important; box-shadow: 0 18px 55px rgba(18,45,90,.22) !important; }
.st-key-ai_chat_panel [data-testid="stVerticalBlock"] { gap: .55rem; }
.st-key-ai_chat_panel [data-testid="stChatMessage"] { padding: .65rem .75rem; }
.st-key-ai_chat_panel [data-testid="stForm"] { border: 0; padding: 0; background: transparent; }
@media (max-width: 700px) { .st-key-ai_floating_button { right: 12px !important; bottom: 12px !important; } .st-key-ai_chat_panel { right: 8px !important; bottom: 78px !important; width: calc(100vw - 16px) !important; max-height: calc(100vh - 95px) !important; } }
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

db_ok, db_message = init_history_table()
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
        if db_ok:
            saved, save_error = save_history_item(result)
            if saved:
                st.success("Đã lưu hồ sơ vào lịch sử dùng chung của nhóm.")
            else:
                st.error("Không lưu được hồ sơ vào cơ sở dữ liệu. " + save_error)
        else:
            st.warning("Lịch sử chưa được lưu bền vững vì cơ sở dữ liệu chưa sẵn sàng. Hãy cấu hình DATABASE_URL trong Streamlit Secrets.")
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
    st.markdown('<div class="hero"><div class="eyebrow">APPLICATION LOG</div><h1>📚 Lịch sử hồ sơ</h1><p>Theo dõi lịch sử phân tích dùng chung của nhóm, được lưu trong cơ sở dữ liệu PostgreSQL.</p></div>', unsafe_allow_html=True)
    if db_ok:
        history, history_error = load_history(limit=500)
        if history is None:
            history = []
            st.error("Không đọc được lịch sử từ cơ sở dữ liệu: " + history_error)
    else:
        history = []
        st.warning("Cơ sở dữ liệu chưa kết nối nên không thể hiển thị lịch sử dùng chung. " + db_message)
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("📂 Tổng hồ sơ đã lưu", len(history))
    with k2: st.metric("🤖 Dự báo từ mô hình", sum(1 for x in history if x.get("mode") == "model"))
    with k3: st.metric("🧪 Chế độ minh họa", sum(1 for x in history if x.get("mode") != "model"))
    st.caption("Lịch sử được chia sẻ giữa những người truy cập ứng dụng và vẫn còn sau khi ứng dụng khởi động lại, miễn là cơ sở dữ liệu bên ngoài còn hoạt động.")
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
    if db_ok:
        st.success("Cơ sở dữ liệu lịch sử: đã kết nối PostgreSQL.")
    else:
        st.warning("Cơ sở dữ liệu lịch sử chưa sẵn sàng: " + db_message)
    st.info("Để tạo/cập nhật artifact theo notebook, đặt `credit_risk_cleaned.csv` ở thư mục gốc rồi chạy `python scripts/train_export.py`.")
    st.warning("Prototype phục vụ học tập. Không nhập dữ liệu định danh hoặc dữ liệu tài chính thật của khách hàng. Chỉ số trong báo cáo không đảm bảo hiệu năng của mọi lần huấn luyện lại.")

# Floating Gemini assistant: each visitor gets a private chat in their Streamlit session.
if "ai_chat_open" not in st.session_state:
    st.session_state.ai_chat_open = False
if "ai_chat_messages" not in st.session_state:
    st.session_state.ai_chat_messages = []

with st.container(key="ai_floating_button"):
    if st.button("✦", key="toggle_ai_chat", help="Mở/thu nhỏ trợ lý AI"):
        st.session_state.ai_chat_open = not st.session_state.ai_chat_open
        st.rerun()

if st.session_state.ai_chat_open:
    with st.container(key="ai_chat_panel", border=True):
        head1, head2 = st.columns([5, 1])
        with head1:
            st.markdown("**✦ CreditRisk AI Assistant**")
            st.caption("Trợ lý Gemini · Tư vấn theo kết quả hồ sơ hiện tại")
        with head2:
            if st.button("—", key="minimize_ai_chat", help="Thu nhỏ"):
                st.session_state.ai_chat_open = False
                st.rerun()

        if not st.session_state.ai_chat_messages:
            st.info("Xin chào! Tôi có thể giải thích PD, hạng tín dụng, phân khúc và các yếu tố SHAP của hồ sơ bạn vừa phân tích.")
        for message in st.session_state.ai_chat_messages[-12:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form("ai_chat_form", clear_on_submit=True):
            user_question = st.text_input("Câu hỏi", placeholder="Ví dụ: Vì sao hồ sơ này có PD cao?", label_visibility="collapsed", max_chars=1200)
            send_chat = st.form_submit_button("Gửi câu hỏi", type="primary", use_container_width=True)
        if send_chat and user_question.strip():
            current_result = st.session_state.get("last_result")
            st.session_state.ai_chat_messages.append({"role": "user", "content": user_question.strip()})
            with st.spinner("Trợ lý đang phân tích kết quả hồ sơ..."):
                answer, ai_error = ask_gemini(user_question.strip(), current_result, st.session_state.ai_chat_messages[:-1])
            if answer:
                st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
            else:
                st.session_state.ai_chat_messages.append({"role": "assistant", "content": "⚠️ " + ai_error})
            st.rerun()
        st.caption("Lưu ý: câu trả lời chỉ phục vụ học tập, không phải quyết định tín dụng. Tránh nhập thông tin định danh cá nhân.")

st.markdown('<div class="footer-note">CreditRisk AI · Machine Learning Nhóm 14 · Bản demo học tập — không thay thế quy trình thẩm định tín dụng chuyên môn.</div>', unsafe_allow_html=True)
