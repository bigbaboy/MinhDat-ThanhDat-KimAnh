# streamlit_menu_ui.py
import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from streamlit.runtime.scriptrunner import RerunException

# ------------------------------
# 0. Page config
# ------------------------------
st.set_page_config(
    page_title="Menu Personalization & KMeans Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for centered layout
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# 1. Constants
# ------------------------------
DATA_FILE    = 'final_with_new_items.xlsx'
X_DAYS_NEW   = 30
K_RECS_NEW   = 3
K_RECS_EXIST = 4

if 'acc' not in st.session_state:
    st.session_state.acc = []

# ------------------------------
# Utility: Rerun replacement
# ------------------------------
def rerun():
    raise RerunException("Rerun triggered")

# ------------------------------
# 2. Load Data
# ------------------------------
@st.cache_data
def load_data(path):
    xls = pd.read_excel(path,
        sheet_name=['Menu_Items','Menu_Interaction','Session_Outcome','Order_Details','Customers'])
    menu = xls['Menu_Items']
    inter= xls['Menu_Interaction']
    sess = xls['Session_Outcome']
    od   = xls['Order_Details']
    cust = xls['Customers']

    menu['Launch_Date'] = pd.to_datetime(menu['Launch_Date'], errors='coerce')
    menu['Days_Since_Launch'] = (datetime.now() - menu['Launch_Date']).dt.days

    inter['View_Start_Time'] = pd.to_datetime(inter['View_Start_Time'], errors='coerce')
    inter['View_Duration'] = pd.to_numeric(inter['View_Duration'], errors='coerce')
    inter['Revisit_Count'] = pd.to_numeric(inter['Revisit_Count'], errors='coerce')
    inter['Menu_Zone'] = inter.get('Menu_Zone', '')

    sess['Decision_Time'] = pd.to_numeric(sess['Decision_Time'], errors='coerce')

    return menu, inter, sess, od, cust

menu_items, interactions, session_out, order_details, customers = load_data(DATA_FILE)

# ------------------------------
# 3. KMeans Features & Clustering
# ------------------------------
cust_list = customers['Customer_ID'].dropna().astype(str).unique()
feat_rows = []
for cid in cust_list:
    sids = session_out.loc[session_out['Customer_ID'].astype(str)==cid,'Session_ID']
    hi = interactions[interactions['Session_ID'].isin(sids)].copy()
    hi['Days_Since_Launch'] = hi['Item_ID'].map(menu_items.set_index('Item_ID')['Days_Since_Launch'])
    ho = order_details[order_details['Session_ID'].isin(sids)]

    total_time = hi['View_Duration'].sum()
    new_time = hi.query("Days_Since_Launch<=@X_DAYS_NEW")['View_Duration'].sum()
    pct_new = new_time/total_time if total_time > 0 else 0
    diversity = ho.groupby('Session_ID').tail(5)['Item_ID'].nunique()
    revisit = hi.query("Days_Since_Launch<=@X_DAYS_NEW")['Revisit_Count'].sum()
    pct_top = hi.groupby('Item_ID')['View_Duration'].sum().max()/total_time if total_time > 0 else 0
    zones_cnt = hi['Menu_Zone'].nunique()
    avg_zone = hi.groupby('Menu_Zone')['View_Duration'].mean().mean() if zones_cnt > 0 else 0

    feat_rows.append({
        'Customer_ID': cid,
        'total_time': total_time,
        'pct_new': pct_new,
        'diversity': diversity,
        'revisit': revisit,
        'pct_top': pct_top,
        'zones_cnt': zones_cnt,
        'avg_zone': avg_zone
    })

features_df = pd.DataFrame(feat_rows).set_index('Customer_ID')
scaler = StandardScaler().fit(features_df)
X_scaled = scaler.transform(features_df)
kmeans = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
features_df['cluster'] = kmeans.labels_

cent = pd.DataFrame(kmeans.cluster_centers_, columns=features_df.columns[:-1])
c_ex = cent['pct_new'].idxmax()
c_con = cent['pct_top'].idxmax()
c_imp = cent['zones_cnt'].idxmax()
c_loy = list({0,1,2,3} - {c_ex,c_con,c_imp})[0]
seg_map = {c_ex:'Explorers', c_con:'Conservatives', c_imp:'Impulsive', c_loy:'Loyalists'}
cust_segment = features_df['cluster'].map(seg_map)

# KNN model
nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn_model.fit(X_scaled)

# ------------------------------
# 4. Recommend Functions
# ------------------------------
def recommend_existing(cid):
    seg = cust_segment.get(cid, 'Loyalists')
    sids = session_out.loc[session_out['Customer_ID'].astype(str) == cid, 'Session_ID']
    ho = order_details[order_details['Session_ID'].isin(sids)]

    if seg == 'Explorers':
        tried = ho.merge(menu_items, on='Item_ID')['Category'].unique()
        pool = menu_items[(menu_items['Days_Since_Launch'] <= X_DAYS_NEW) & (~menu_items['Category'].isin(tried))]
        zone_priority = ['Top-Left', 'Center']
        cnt = K_RECS_EXIST

    elif seg == 'Conservatives':
        fav = ho.merge(menu_items, on='Item_ID')['Category'].mode().iat[0]
        pool = menu_items[(menu_items['Category'] == fav) & (menu_items['Days_Since_Launch'] <= X_DAYS_NEW)]
        zone_priority = ['Center', 'Top-Left']
        cnt = K_RECS_EXIST

    elif seg == 'Impulsive':
        top5 = interactions['Item_ID'].value_counts().nlargest(K_RECS_EXIST * 2).index
        pool = menu_items[(menu_items['Item_ID'].isin(top5)) & (menu_items['Days_Since_Launch'] <= X_DAYS_NEW)]
        zone_priority = ['Top-Right', 'Bottom-Left']
        cnt = K_RECS_EXIST

    else:
        pool = menu_items[menu_items['Days_Since_Launch'] <= X_DAYS_NEW].copy()
        pool['Score'] = pool['Days_Since_Launch'].rank(ascending=False)
        pool = pool.sort_values('Score')
        zone_priority = ['Bottom-Right']
        cnt = 2

    recs = pool.sample(min(cnt, len(pool))).reset_index(drop=True)
    return seg, recs[['Item_Name', 'Category', 'Menu_Zone', 'Days_Since_Launch']], zone_priority

def allocate_by_priority_zones(df, zones):
    df = df.sample(frac=1).reset_index(drop=True)
    result = {z: pd.DataFrame(columns=df.columns) for z in ['Top-Left', 'Center', 'Top-Right', 'Bottom-Left', 'Bottom-Right']}
    for i, row in df.iterrows():
        z = zones[i % len(zones)]
        result[z] = pd.concat([result[z], pd.DataFrame([row])], ignore_index=True)
    return result

def recommend_knn(cid):
    if cid not in features_df.index:
        return 'Unknown', pd.DataFrame(), []

    idx = list(features_df.index).index(cid)
    distances, indices = nn_model.kneighbors([X_scaled[idx]])
    neighbor_ids = [features_df.index[i] for i in indices[0] if features_df.index[i] != cid]

    sids = session_out[session_out['Customer_ID'].astype(str) == cid]['Session_ID']
    own_items = order_details[order_details['Session_ID'].isin(sids)]['Item_ID'].unique()

    neighbor_sids = session_out[session_out['Customer_ID'].astype(str).isin(neighbor_ids)]['Session_ID']
    neighbor_orders = order_details[order_details['Session_ID'].isin(neighbor_sids)]

    rec_items = neighbor_orders[~neighbor_orders['Item_ID'].isin(own_items)]['Item_ID'].value_counts().head(K_RECS_EXIST).index
    recs = menu_items[menu_items['Item_ID'].isin(rec_items)].copy()

    return 'KNN-Based', recs[['Item_Name', 'Category', 'Menu_Zone', 'Days_Since_Launch']], ['Top-Left', 'Center']

# ------------------------------
# 5. Sidebar UI
# ------------------------------
with st.sidebar:
    st.header("🔧 Cài đặt")
    mode = st.radio("Chọn trường hợp", ["Khách cũ","Khách mới"])
    # Gợi ý bằng KMeans cho khách cũ, KNN cho khách mới (tự động, không cần checkbox)
    st.markdown("---")

# Trong phần khách cũ, thay đổi logic gọi hàm gợi ý
if mode == "Khách cũ":
    st.header("📋 Cá nhân hoá menu cho khách cũ")
    cnts = cust_segment.value_counts().reindex(['Explorers','Conservatives','Impulsive','Loyalists']).fillna(0)
    percs= (cnts/len(cust_list)*100).round(1).astype(str)+'%'
    dist = pd.concat([cnts.rename('count'), percs.rename('percent')], axis=1).reset_index().rename(columns={'index': 'cluster'})
    fig = px.pie(dist, names='cluster', values='count', title='Phân bố khách theo segment')
    st.plotly_chart(fig, use_container_width=True)

    cid = st.selectbox("Chọn Customer ID", sorted(cust_list))
    if cid:
        seg, df_rec, attention_zones = recommend_existing(cid)

        st.success(f"🔎 {cid} thuộc segment **{seg}**")
        zone_allocs = allocate_by_priority_zones(df_rec, attention_zones)

        st.markdown("### 📌 Món đề xuất theo vùng hiển thị menu")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🔺 Top-Left")
            st.table(zone_allocs['Top-Left'][['Item_Name', 'Category']])
        with c2:
            st.markdown("#### 🎯 Center")
            st.table(zone_allocs['Center'][['Item_Name', 'Category']])
        with c3:
            st.markdown("#### 🔸 Top-Right")
            st.table(zone_allocs['Top-Right'][['Item_Name', 'Category']])
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("#### 🟡 Bottom-Left")
            st.table(zone_allocs['Bottom-Left'][['Item_Name', 'Category']])
        with c5:
            st.markdown("#### ⚪ Bottom-Right")
            st.table(zone_allocs['Bottom-Right'][['Item_Name', 'Category']])
elif mode == "Khách mới":
    st.header("⏱️ Real-time recommendation cho khách mới")
    thr = session_out['Decision_Time'].dropna().mean()
    st.markdown(f"**Ngưỡng đề xuất :** {thr:.2f}s")

    if 'rejected_zones' not in st.session_state:
        st.session_state.rejected_zones = []

    with st.form("rt_form"):
        c1, c2 = st.columns(2)
        with c1:
            dur = st.number_input("⏱ View Duration (s)", 0.0, value=0.0)
            rpt = st.number_input("🔁 Revisit Count", 0, value=0)
            zone = st.selectbox("📍 Menu Zone", menu_items['Menu_Zone'].dropna().unique())
        with c2:
            face = st.selectbox("🙂 Facial Expression", ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry'])
            head = st.selectbox("🧠 Head Pose", ['Tilt_Down', 'Tilt_Up', 'Straight', 'Turn_Left', 'Turn_Right'])
        ok = st.form_submit_button("📥 Thêm tương tác")
        if ok:
            st.session_state.acc.append({
                'View_Duration': dur,
                'Revisit_Count': rpt,
                'Facial_Expression': face,
                'Head_Pose': head,
                'Menu_Zone': zone
            })

    total = sum(x['View_Duration'] for x in st.session_state.acc)
    st.info(f"🕒 Tổng thời gian tương tác: **{total:.2f}s**")

    if st.session_state.acc:
        st.subheader("🧾 Danh sách tương tác đã nhập")
        st.table(pd.DataFrame(st.session_state.acc))

    if total >= thr:
        df_acc = pd.DataFrame(st.session_state.acc)
        zones = df_acc['Menu_Zone'].value_counts().index.tolist()
        valid_zones = [z for z in zones if z not in st.session_state.rejected_zones]
        recs = menu_items[menu_items['Menu_Zone'].isin(valid_zones)].head(K_RECS_NEW)

        st.subheader("🍜 Gợi ý món ăn cho khách mới")
        st.table(recs[['Item_Name','Category','Menu_Zone','Days_Since_Launch']])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Chọn món"):
                st.success("Bạn đã chọn món! Hệ thống sẽ bắt đầu lại.")
                st.session_state.acc = []
                st.session_state.rejected_zones = []
        with c2:
           if st.button("❌ Từ chối gợi ý"):
                if valid_zones:
                    st.session_state.rejected_zones.append(valid_zones[0])
                    # Xoá món của vùng đó ngay lập tức khỏi recs trước khi rerun
                    recs = recs[~recs['Menu_Zone'].isin([valid_zones[0]])]
                rerun() 
