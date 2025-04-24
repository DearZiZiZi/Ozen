import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Fitness Center Analytics",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load data
@st.cache_data
def load_data():
    clients_df = pd.read_csv('clients.csv')
    visits_df = pd.read_csv('visits.csv')
    sales_df = pd.read_csv('sales.csv')
    reviews_df = pd.read_csv('reviews.csv')
    
    # Convert date columns to datetime
    visits_df['VisitDate'] = pd.to_datetime(visits_df['VisitDate'])
    sales_df['SaleDate'] = pd.to_datetime(sales_df['SaleDate'])
    reviews_df['Date'] = pd.to_datetime(reviews_df['Date'])
    
    return clients_df, visits_df, sales_df, reviews_df

# Load data
try:
    clients_df, visits_df, sales_df, reviews_df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    data_loaded = False
    
# Create download link for reports
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main Application
def main():
    if not data_loaded:
        st.warning("Please check your data files and try again.")
        return

    # Sidebar for filtering
    st.sidebar.title("Filters")
    
    # Date range filter
    min_date = min(min(visits_df['VisitDate']), min(sales_df['SaleDate']), min(reviews_df['Date']))
    max_date = max(max(visits_df['VisitDate']), max(sales_df['SaleDate']), max(reviews_df['Date']))
    
    date_options = ['Last 7 days', 'Last 30 days', 'Last 90 days', 'This year', 'All time', 'Custom']
    date_filter = st.sidebar.selectbox('Date Range', date_options)
    
    today = datetime.now().date()
    
    if date_filter == 'Last 7 days':
        start_date = today - timedelta(days=7)
        end_date = today
    elif date_filter == 'Last 30 days':
        start_date = today - timedelta(days=30)
        end_date = today
    elif date_filter == 'Last 90 days':
        start_date = today - timedelta(days=90)
        end_date = today
    elif date_filter == 'This year':
        start_date = datetime(today.year, 1, 1).date()
        end_date = today
    elif date_filter == 'All time':
        start_date = min_date.date()
        end_date = max_date.date()
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input('Start date', min_date.date())
        end_date = col2.date_input('End date', max_date.date())
    
    # Other filters
    membership_types = ['All'] + sorted(clients_df['MembershipType'].unique().tolist())
    selected_membership = st.sidebar.selectbox('Membership Type', membership_types)
    
    trainers = ['All'] + sorted(clients_df['Trainer'].unique().tolist())
    selected_trainer = st.sidebar.selectbox('Trainer', trainers)
    
    gender_options = ['All', 'Male', 'Female']
    selected_gender = st.sidebar.selectbox('Gender', gender_options)
    
    acquisition_sources = ['All'] + sorted(clients_df['AcquisitionSource'].unique().tolist())
    selected_source = st.sidebar.selectbox('Acquisition Source', acquisition_sources)
    
    # Apply filters
    filtered_clients = clients_df.copy()
    if selected_membership != 'All':
        filtered_clients = filtered_clients[filtered_clients['MembershipType'] == selected_membership]
    if selected_trainer != 'All':
        filtered_clients = filtered_clients[filtered_clients['Trainer'] == selected_trainer]
    if selected_gender != 'All':
        filtered_clients = filtered_clients[filtered_clients['Gender'] == selected_gender]
    if selected_source != 'All':
        filtered_clients = filtered_clients[filtered_clients['AcquisitionSource'] == selected_source]
    
    filtered_client_ids = filtered_clients['ClientID'].unique()
    
    # Filter other dataframes
    filtered_visits = visits_df[(visits_df['ClientID'].isin(filtered_client_ids)) & 
                                (visits_df['VisitDate'].dt.date >= start_date) & 
                                (visits_df['VisitDate'].dt.date <= end_date)]
    
    filtered_sales = sales_df[(sales_df['ClientID'].isin(filtered_client_ids)) & 
                              (sales_df['SaleDate'].dt.date >= start_date) & 
                              (sales_df['SaleDate'].dt.date <= end_date)]
    
    filtered_reviews = reviews_df[(reviews_df['ClientID'].isin(filtered_client_ids)) & 
                                  (reviews_df['Date'].dt.date >= start_date) & 
                                  (reviews_df['Date'].dt.date <= end_date)]
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Client Analytics", "Visit Analytics", "Sales Analytics", "Trainer Performance"])
    
    # Tab 1: Overview Dashboard
    with tab1:
        st.header("Fitness Center Overview")
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        total_clients = len(filtered_clients)
        total_visits = len(filtered_visits)
        total_revenue = filtered_sales['Amount'].sum()
        avg_rating = filtered_reviews['Rating'].mean()
        
        col1.metric("Total Clients", f"{total_clients:,}")
        col2.metric("Total Visits", f"{total_visits:,}")
        col3.metric("Total Revenue", f"${total_revenue:,.2f}")
        col4.metric("Average Rating", f"{avg_rating:.2f}/5")
        
        # Second row - charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue over time
            sales_time_series = filtered_sales.groupby(filtered_sales['SaleDate'].dt.date)['Amount'].sum().reset_index()
            sales_time_series.columns = ['Date', 'Revenue']
            
            fig = px.line(sales_time_series, x='Date', y='Revenue', title='Daily Revenue')
            fig.update_layout(xaxis_title='Date', yaxis_title='Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visits over time
            visits_time_series = filtered_visits.groupby(filtered_visits['VisitDate'].dt.date).size().reset_index()
            visits_time_series.columns = ['Date', 'Visits']
            
            fig = px.line(visits_time_series, x='Date', y='Visits', title='Daily Visits')
            fig.update_layout(xaxis_title='Date', yaxis_title='Number of Visits')
            st.plotly_chart(fig, use_container_width=True)
        
        # Third row - charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Membership type distribution
            membership_counts = filtered_clients['MembershipType'].value_counts().reset_index()
            membership_counts.columns = ['MembershipType', 'Count']
            
            fig = px.pie(membership_counts, names='MembershipType', values='Count', 
                         title='Membership Type Distribution', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Acquisition Source distribution
            acquisition_counts = filtered_clients['AcquisitionSource'].value_counts().reset_index()
            acquisition_counts.columns = ['AcquisitionSource', 'Count']
            
            fig = px.pie(acquisition_counts, names='AcquisitionSource', values='Count', 
                         title='Client Acquisition Sources', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Client Analytics
    with tab2:
        st.header("Client Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig = px.histogram(filtered_clients, x='Age', nbins=20,
                              title='Client Age Distribution')
            fig.update_layout(xaxis_title='Age', yaxis_title='Number of Clients')
            st.plotly_chart(fig, use_container_width=True)
            
            # Gender distribution
            gender_counts = filtered_clients['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(gender_counts, names='Gender', values='Count', 
                         title='Gender Distribution', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # New clients acquisition over time
            # For this, we'd need an "Acquisition Date" column which isn't in the data
            # Instead, we can create a synthetic one for demonstration purposes
            if 'AcquisitionDate' not in clients_df.columns:
                np.random.seed(42)
                random_dates = pd.date_range(start='2023-01-01', end='2023-12-31')
                acquisition_dates = np.random.choice(random_dates, size=len(clients_df))
                clients_df['AcquisitionDate'] = acquisition_dates
                filtered_clients = clients_df[clients_df['ClientID'].isin(filtered_clients['ClientID'])]
            
            acquisition_time_series = filtered_clients.groupby(pd.to_datetime(filtered_clients['AcquisitionDate']).dt.month).size().reset_index()
            acquisition_time_series.columns = ['Month', 'New Clients']
            acquisition_time_series['Month'] = acquisition_time_series['Month'].apply(lambda x: calendar.month_name[x])
            
            fig = px.bar(acquisition_time_series, x='Month', y='New Clients', 
                         title='New Client Acquisitions by Month')
            fig.update_layout(xaxis_title='Month', yaxis_title='Number of New Clients')
            st.plotly_chart(fig, use_container_width=True)
            
            # Retention analysis (active clients)
            # For demonstration, we'll define "active" as having visited in the last 30 days
            recent_visits = visits_df[visits_df['VisitDate'] >= (max(visits_df['VisitDate']) - timedelta(days=30))]
            active_clients = recent_visits['ClientID'].nunique()
            inactive_clients = total_clients - active_clients
            
            fig = go.Figure(go.Pie(
                labels=['Active', 'Inactive'],
                values=[active_clients, inactive_clients],
                hole=0.4
            ))
            fig.update_layout(title_text='Client Activity Status (Last 30 Days)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Client table with details
        st.subheader("Client Details")
        st.dataframe(filtered_clients)
        
        # Download link for client report
        st.markdown(get_download_link(filtered_clients, "client_report.csv", "Download Client Report"), unsafe_allow_html=True)
    
    # Tab 3: Visit Analytics
    with tab3:
        st.header("Visit Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visits by day of week
            filtered_visits['DayOfWeek'] = filtered_visits['VisitDate'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            visits_by_day = filtered_visits['DayOfWeek'].value_counts().reindex(day_order).reset_index()
            visits_by_day.columns = ['Day', 'Visits']
            
            fig = px.bar(visits_by_day, x='Day', y='Visits', title='Visits by Day of Week')
            st.plotly_chart(fig, use_container_width=True)
            
            # Visits by hour of day
            filtered_visits['HourOfDay'] = filtered_visits['VisitDate'].dt.hour
            visits_by_hour = filtered_visits.groupby('HourOfDay').size().reset_index()
            visits_by_hour.columns = ['Hour', 'Visits']
            
            fig = px.line(visits_by_hour, x='Hour', y='Visits', title='Visits by Hour of Day')
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visit duration distribution
            fig = px.histogram(filtered_visits, x='VisitDuration', nbins=20,
                              title='Visit Duration Distribution (minutes)')
            fig.update_layout(xaxis_title='Duration (minutes)', yaxis_title='Number of Visits')
            st.plotly_chart(fig, use_container_width=True)
            
            # Average visits per client
            visits_per_client = filtered_visits.groupby('ClientID').size().reset_index()
            visits_per_client.columns = ['ClientID', 'VisitCount']
            
            fig = px.histogram(visits_per_client, x='VisitCount', nbins=20,
                              title='Visits per Client Distribution')
            fig.update_layout(xaxis_title='Number of Visits', yaxis_title='Number of Clients')
            st.plotly_chart(fig, use_container_width=True)
        
        # Visit details
        st.subheader("Visit Details")
        st.dataframe(filtered_visits)
        
        # Download link for visit report
        st.markdown(get_download_link(filtered_visits, "visit_report.csv", "Download Visit Report"), unsafe_allow_html=True)
    
    # Tab 4: Sales Analytics
    with tab4:
        st.header("Sales Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by product
            sales_by_product = filtered_sales.groupby('Product')['Amount'].sum().reset_index()
            sales_by_product = sales_by_product.sort_values('Amount', ascending=False)
            
            fig = px.bar(sales_by_product, x='Product', y='Amount', title='Revenue by Product')
            fig.update_layout(xaxis_title='Product', yaxis_title='Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly revenue trend
            filtered_sales['Month'] = filtered_sales['SaleDate'].dt.strftime('%Y-%m')
            monthly_sales = filtered_sales.groupby('Month')['Amount'].sum().reset_index()
            
            fig = px.line(monthly_sales, x='Month', y='Amount', title='Monthly Revenue Trend')
            fig.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average sale amount by membership type
            sales_with_client_info = pd.merge(filtered_sales, clients_df, on='ClientID')
            avg_sale_by_membership = sales_with_client_info.groupby('MembershipType')['Amount'].mean().reset_index()
            
            fig = px.bar(avg_sale_by_membership, x='MembershipType', y='Amount', 
                         title='Average Sale Amount by Membership Type')
            fig.update_layout(xaxis_title='Membership Type', yaxis_title='Average Amount ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales frequency heatmap (day of week vs hour of day)
            filtered_sales['DayOfWeek'] = filtered_sales['SaleDate'].dt.day_name()
            filtered_sales['HourOfDay'] = filtered_sales['SaleDate'].dt.hour
            
            sales_heatmap = pd.crosstab(filtered_sales['DayOfWeek'], filtered_sales['HourOfDay'])
            # Reorder days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sales_heatmap = sales_heatmap.reindex(day_order)
            
            fig = px.imshow(sales_heatmap, 
                           labels=dict(x="Hour of Day", y="Day of Week", color="Number of Sales"),
                           title="Sales Frequency Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        # Sales details
        st.subheader("Sales Details")
        st.dataframe(filtered_sales)
        
        # Download link for sales report
        st.markdown(get_download_link(filtered_sales, "sales_report.csv", "Download Sales Report"), unsafe_allow_html=True)
    
    # Tab 5: Trainer Performance
    with tab5:
        st.header("Trainer Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trainer ratings
            trainer_ratings = filtered_reviews.groupby('Trainer')['Rating'].mean().reset_index()
            trainer_ratings = trainer_ratings.sort_values('Rating', ascending=False)
            
            fig = px.bar(trainer_ratings, x='Trainer', y='Rating', title='Average Trainer Ratings')
            fig.update_layout(xaxis_title='Trainer', yaxis_title='Average Rating')
            fig.update_yaxes(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
            
            # Number of clients per trainer
            clients_per_trainer = filtered_clients.groupby('Trainer').size().reset_index()
            clients_per_trainer.columns = ['Trainer', 'ClientCount']
            clients_per_trainer = clients_per_trainer.sort_values('ClientCount', ascending=False)
            
            fig = px.bar(clients_per_trainer, x='Trainer', y='ClientCount', title='Number of Clients per Trainer')
            fig.update_layout(xaxis_title='Trainer', yaxis_title='Number of Clients')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig = px.histogram(filtered_reviews, x='Rating', nbins=5,
                              title='Rating Distribution')
            fig.update_layout(xaxis_title='Rating', yaxis_title='Number of Reviews')
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating trend over time
            reviews_by_date = filtered_reviews.groupby(filtered_reviews['Date'].dt.date)['Rating'].mean().reset_index()
            reviews_by_date.columns = ['Date', 'Average Rating']
            
            fig = px.line(reviews_by_date, x='Date', y='Average Rating', title='Rating Trend Over Time')
            fig.update_layout(xaxis_title='Date', yaxis_title='Average Rating')
            fig.update_yaxes(range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        
        # Reviews details
        st.subheader("Review Details")
        st.dataframe(filtered_reviews)
        
        # Download link for reviews report
        st.markdown(get_download_link(filtered_reviews, "reviews_report.csv", "Download Reviews Report"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
