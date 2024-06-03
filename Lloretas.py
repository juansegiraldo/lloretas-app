import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.colors as colors
import plotly.graph_objects as go
import numpy as np
from st_aggrid import AgGrid
import base64
from io import BytesIO

# Load the data from the Excel file
xls = pd.ExcelFile('LloretasSource.xlsx')

# Read the 'Pasos' and 'Carreras' sheets
df_pasos = pd.read_excel(xls, 'Pasos')
df_carreras = pd.read_excel(xls, 'Carreras')

# Read the activities sheets
df_activities_jsg = pd.read_excel(xls, 'ActivitiesJSG')
df_activities_fjv = pd.read_excel(xls, 'ActivitiesFJV')
df_activities_fer = pd.read_excel(xls, 'ActivitiesFER')
df_activities_sum = pd.read_excel(xls, 'ActivitiesSUM')

df_activities_jsg['Participante'] = 'Juan Sebastian Giraldo'
df_activities_fjv['Participante'] = 'FrancoVelez'
df_activities_fer['Participante'] = 'Federico'
df_activities_sum['Participante'] = 'SantiUribem'

# Concatenate all activities dataframes
df_activities = pd.concat([df_activities_jsg, df_activities_fjv, df_activities_fer, df_activities_sum])

# List of activity types to include
included_activity_types = ['Running', 'Entrenamiento en cinta', 'Carrera', 'Treadmill Running', 'Trail Running']

# Filter only activities with specified types
df_activities = df_activities[df_activities['Activity Type'].isin(included_activity_types)]

# Set the date range
start_date = '2024-01-01'
end_date = '2024-05-31'

# Filter df_activities
df_activities = df_activities[(df_activities['Date'] >= start_date) & (df_activities['Date'] <= end_date)]

# Replace ',' with '.' in 'Distance' and convert to float
df_activities['Distance'] = df_activities['Distance'].astype(str).str.replace(',', '.').astype(float)

# Replace '--' with a default value in 'Avg Pace' and convert all values to string
df_activities['Avg Pace'] = df_activities['Avg Pace'].replace('--', '00:00').astype(str)

# Split 'Avg Pace' into a list of [MM, SS, mm]
df_activities['Avg Pace'] = df_activities['Avg Pace'].str.split(':')

# Keep only the first two parts [MM, SS] and join them back into a string
df_activities['Avg Pace'] = df_activities['Avg Pace'].apply(lambda x: ':'.join(x[:2]))

# Now 'Avg Pace' is in the format MM:SS

# Initialize 'Points' column
df_pasos['Points'] = 0
df_carreras['Points'] = 0
df_activities['Points'] = 0

# Function to calculate points
def calculate_points(df, column, first_points, second_points, third_points):
    df['Points'] = 0
    for month in df['Mes'].unique():
        df_month = df[df['Mes'] == month]
        df_month = df_month.sort_values(by=[column], ascending=False)
        df_month.loc[df_month[column].rank(method='min', ascending=False) == 1, 'Points'] += first_points
        df_month.loc[df_month[column].rank(method='min', ascending=False) == 2, 'Points'] += second_points
        df_month.loc[df_month[column].rank(method='min', ascending=False) == 3, 'Points'] += third_points
        df.update(df_month)
    return df

# Calculate points for 'Pasos' and 'Carreras'
df_pasos = calculate_points(df_pasos, 'Pasos', 10, 6, 2)
df_carreras = calculate_points(df_carreras, 'Km', 10, 6, 2)
print(f"Retos de Pasos")
print(df_pasos)
print(f"Retos de Carreras")
print(df_carreras)

total_km_per_participant = df_carreras.groupby('Participante')['Km'].sum().sort_values(ascending=False)
total_steps_per_participant = df_pasos.groupby('Participante')['Pasos'].sum().sort_values(ascending=False)
print(f"Total de Kms por Participante")
print(total_km_per_participant)
print(f"Total de pasos por participante")
print(total_steps_per_participant)


# Calculate points for 'Activities'
# df_activities.loc[df_activities['Distance'] >= 42.2, 'Points'] += 10
# df_activities.loc[(df_activities['Distance'] >= 21.1) & (df_activities['Distance'] < 42.2), 'Points'] += 6

# Define the function for linear interpolation
def calculate_points_long_runs(distance):
    if 21.0975 <= distance <= 42.195:
        return round((4/21.0975) * distance + 2, 1)
    elif distance > 42.195:
        return 10
    else:
        return 0

# Apply the function to the 'Distance' column to calculate 'Points'
df_activities['Points'] = df_activities['Distance'].apply(calculate_points_long_runs)

# Create a Data Frame with only the Long Runs (ie greater than 21.0975 km)
df_long_runs = df_activities[(df_activities['Points'] != 0)]
# Long Runs with only selected columns
df_long_runs_selected_columns = df_long_runs[['Activity Type', 'Date', 'Distance', 'Avg Pace', 'Participante', 'Points']]
print(f"Los actividades de más de 21.0975 kms y sus puntos")
print(df_long_runs_selected_columns)

# Calculate points for the longest running activity per month
df_activities['YearMonth'] = df_activities['Date'].dt.to_period('M')

# Group by YearMonth and Participant, and find the row with max Distance
df_activities['Rank'] = df_activities.groupby(['YearMonth'])['Distance'].rank(method='first', ascending=False)

# Filter rows with Rank 1, these are the rows with max Distance per YearMonth per Participant
df_longest_run = df_activities[df_activities['Rank'] == 1].copy()

# Assign 10 points to these rows
df_longest_run['Points'] = 10

# Drop the Rank and YearMonth columns, we don't need them in the final output
df_longest_run = df_longest_run.drop(columns=['Rank', 'YearMonth'])

# Logest Run with only selected columns
df_longest_run_selected_columns = df_longest_run[['Activity Type', 'Date', 'Distance', 'Avg Pace', 'Participante', 'Points']]
print(f"Las actividades más largas por mes")
print(df_longest_run_selected_columns)

# All activities with only selected columns
df_activities_selected_columns = df_activities[['Activity Type', 'Date', 'Distance', 'Avg Pace', 'Participante', 'Points']]
print(f"Todas las actividades de todos")
print(df_activities_selected_columns)

# Vamos a exponer a las tortugas
df_2 = df_activities_selected_columns.copy()

# Add '00:' prefix to 'Avg Pace' to convert it to 'hh:mm:ss' format
df_2['Avg Pace'] = '00:' + df_2['Avg Pace']

# Now convert 'Avg Pace' to timedelta
df_2['Avg Pace'] = pd.to_timedelta(df_2['Avg Pace'])

# Filter rows where 'Avg Pace' is more than '07:00' min/km
df_activities_tortugas = df_2[df_2['Avg Pace'] > pd.to_timedelta('00:07:00')]
print(f"Las Actividades Tortuga")
print(df_activities_tortugas)

# Sum all points
df_total_points = pd.concat([df_pasos, df_carreras, df_activities, df_longest_run]).groupby('Participante')['Points'].sum().reset_index()
# Sort df_total_points by 'Points' in descending order
df_total_points = df_total_points.sort_values('Points', ascending=False)
# Drop the index column and add a "Position" column
df_total_points.reset_index(drop=True, inplace=True)
#df_total_points['Position'] = df_total_points.index + 1

# Print the modified DataFrame
print("   Así va la tabla de posiciones del Club de Lloretas")
print(df_total_points)

# Vamos a crear el Excel de Fede
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Lloretas2024.xlsx', engine='xlsxwriter')

# Write each DataFrame to a different worksheet.
df_pasos.to_excel(writer, sheet_name='Pasos')
df_carreras.to_excel(writer, sheet_name='Carreras')
df_longest_run_selected_columns.to_excel(writer, sheet_name='Longest Runs')
df_long_runs_selected_columns.to_excel(writer, sheet_name='Long Runs')
df_activities_selected_columns.to_excel(writer, sheet_name='Activities')
df_activities_tortugas.to_excel(writer, sheet_name='Activities Tortugas')
total_steps_per_participant.to_excel(writer, sheet_name='TotalPasos')
total_km_per_participant.to_excel(writer, sheet_name='TotalKms')
df_total_points.to_excel(writer, sheet_name='Total Points')

# Close the Pandas Excel writer and output the Excel file.
writer._save()



# # Vamos a graficar las carreras

## Streamlit app
# Set page configuration
st.set_page_config(layout="wide")

st.title('Club de Lloretas')

## Table with the Total Points
st.markdown("### Así vamos")
df_total_points['Points'] = df_total_points['Points'].map(lambda x: '{:.1f}'.format(x))
st.markdown(df_total_points.style.hide(axis="index").to_html(), unsafe_allow_html=True)

# Define the color mapping for each "Participante"
participante_colors = {
    "FrancoVelez": "pink",
    "Juan Sebastian Giraldo": "lightblue",
    "Federico": "lightgreen",
    "SantiUribem": "lightcoral"
}

# Line Graph of accumulated Kms
# Sort the DataFrame by Date and Participante
df_activities_selected_columns = df_activities_selected_columns.sort_values(['Participante', 'Date'])

# Create a new DataFrame with accumulated distances
accumulated_distances = df_activities_selected_columns.groupby(['Participante', 'Date'])['Distance'].sum().reset_index()
accumulated_distances = accumulated_distances.sort_values(['Participante', 'Date'])
accumulated_distances['Accumulated Distance'] = accumulated_distances.groupby('Participante')['Distance'].cumsum()

# Create the line chart
fig = px.line(accumulated_distances, x='Date', y='Accumulated Distance', color='Participante', title='Accumulated Progress for Runners', color_discrete_map=participante_colors)

# Add text labels in the middle of each line
for participante, group in accumulated_distances.groupby('Participante'):
    middle_index = len(group) // 2
    middle_row = group.iloc[middle_index]
    offset = 20
    fig.add_annotation(
        x=middle_row['Date'],
        y=middle_row['Accumulated Distance'] + offset,
        text=participante,
        showarrow=False,
        font=dict(color=participante_colors[participante])
    )

# Add text labels to the end of each line with offset
for participante, group in accumulated_distances.groupby('Participante'):
    last_row = group.iloc[-1]
    offset = max(accumulated_distances['Accumulated Distance']) * 0.05  # Adjust the offset as needed
    fig.add_annotation(
        x=last_row['Date'],
        y=last_row['Accumulated Distance'] + offset,
        text=f"{last_row['Accumulated Distance']:.2f} km",
        showarrow=False,
        font=dict(color=participante_colors[participante])
    )

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Distance (km)',
    yaxis=dict(showgrid=False),  # Remove y-axis grid lines
    showlegend=False
)

# Display the plot in Streamlit
st.plotly_chart(fig)

# Sort the month order
month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# Create the bar chart for df_pasos
df_pasos['Pasos_k'] = (df_pasos['Pasos'] / 1000).round(1)
df_pasos['Pts_Label'] = df_pasos['Points'].astype(str) + ' pts'
fig_pasos = px.bar(df_pasos, x='Mes', y='Pasos', color='Participante', title='Pasos por Mes', category_orders={'Mes': month_order}, barmode='group', color_discrete_map=participante_colors, text=df_pasos['Pts_Label'] + '<br>' + df_pasos['Pasos_k'].astype(str) + 'k')

# Update bar trace text position
for trace in fig_pasos.data[:len(df_pasos['Participante'].unique())]:
    trace.textposition = 'auto'

# Add trend lines
for participant in df_pasos['Participante'].unique():
    participant_data = df_pasos[df_pasos['Participante'] == participant]
    fig_pasos.add_trace(go.Scatter(x=participant_data['Mes'], y=participant_data['Pasos'], mode='lines', line=dict(color=participante_colors[participant], dash='dot', width=0.5), name=participant, showlegend=False))

fig_pasos.update_layout(
    xaxis_title='Mes',
    yaxis_title='Pasos',
    legend_title='',
    height=400,
    legend=dict(
        orientation="h",  # Place legend horizontally
        yanchor="top",
        y=1.1,
        xanchor="center",
        x=0.5
    )
)
fig_pasos.update_yaxes(showticklabels=False, showgrid=False)
st.plotly_chart(fig_pasos)

# Create the bar chart for df_carreras
df_carreras['Carreras_km'] = (df_carreras['Km']).round(1)
df_carreras['Pts_Label_carreras'] = df_carreras['Points'].astype(str) + ' pts'
fig_carreras = px.bar(df_carreras, x='Mes', y='Km', color='Participante', title='Kms por Mes', category_orders={'Mes': month_order}, barmode='group', color_discrete_map=participante_colors, text=df_carreras['Pts_Label_carreras'] + '<br>' + df_carreras['Carreras_km'].astype(str))

# Update bar trace text position
for trace in fig_carreras.data[:len(df_carreras['Participante'].unique())]:
    trace.textposition = 'auto'

# Add trend lines
for participant in df_carreras['Participante'].unique():
    participant_data = df_carreras[df_carreras['Participante'] == participant]
    fig_carreras.add_trace(go.Scatter(x=participant_data['Mes'], y=participant_data['Km'], mode='lines', line=dict(color=participante_colors[participant], dash='dot', width=0.5), name=participant, showlegend=False))

fig_carreras.update_layout(
    xaxis_title='Mes',
    yaxis_title='Kms',
    legend_title='',
    height=400,
    legend=dict(
        orientation="h",  # Place legend horizontally
        yanchor="top",
        y=1.1,
        xanchor="center",
        x=0.5
    )
)
fig_carreras.update_yaxes(showticklabels=False, showgrid=False)
st.plotly_chart(fig_carreras)

## Table with the Longest runs
st.markdown(" ")
st.markdown("### Carreras más largas por mes")
st.markdown(" ")
df_longest_run_selected_columns = df_longest_run_selected_columns.reset_index(drop=True)
column_order = ['Activity Type', 'Date', 'Distance', 'Avg Pace', 'Participante', 'Points']
df_longest_run_selected_columns = df_longest_run_selected_columns[column_order]
df_longest_run_selected_columns['Date'] = pd.to_datetime(df_longest_run_selected_columns['Date']).dt.strftime('%B')
df_longest_run_selected_columns['Distance'] = df_longest_run_selected_columns['Distance'].map(lambda x: '{:.2f}'.format(x))
df_longest_run_selected_columns['Points'] = df_longest_run_selected_columns['Points'].map(lambda x: '{:.0f}'.format(x))
st.markdown(df_longest_run_selected_columns.style.hide(axis="index").to_html(), unsafe_allow_html=True)


## Table with Long runs
st.markdown(" ")
st.markdown(" ")
st.markdown("### Carreras > 21.095 km")
st.markdown(" ")

df_activities_selected_columns['Date'] = pd.to_datetime(df_activities_selected_columns['Date']).dt.strftime('%d-%b-%Y')
df_activities_selected_columns['Distance'] = df_activities_selected_columns['Distance'].map(lambda x: '{:.2f}'.format(x))
df_activities_selected_columns['Points'] = df_activities_selected_columns['Points'].map(lambda x: '{:.1f}'.format(x))

# Filter the table to show only rows where 'Points' is greater than 0
df_activities_filtered = df_activities_selected_columns[df_activities_selected_columns['Points'].astype(float) > 0]

st.markdown(df_activities_filtered.style.hide(axis="index").to_html(), unsafe_allow_html=True)


def get_all_activities_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in: dataframe
    out: href string
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    output.seek(0)
    xlsx_data = output.getvalue()
    b64 = base64.b64encode(xlsx_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="All_Activities.xlsx">Download All Activities</a>'
    return href

# this will display a download link for the above dataframe
st.markdown(get_all_activities_download_link(df_activities_selected_columns), unsafe_allow_html=True)

# ## Table with the Todas las carreras
# st.markdown("# Todas las Actividades")
# df_activities_selected_columns['Date'] = pd.to_datetime(df_activities_selected_columns['Date']).dt.strftime('%d-%b-%Y')
# df_activities_selected_columns['Distance'] = df_activities_selected_columns['Distance'].map(lambda x: '{:.2f}'.format(x))
# df_activities_selected_columns['Points'] = df_activities_selected_columns['Points'].map(lambda x: '{:.1f}'.format(x))

# # Apply conditional formatting to 'Points' column
# def highlight_positive(val):
#     if float(val) > 0.0:
#         return 'background-color: lime'
#     else:
#         return ''

# df_activities_selected_columns['Points'] = df_activities_selected_columns['Points'].apply(lambda x: f'<span style="{highlight_positive(x)}">{x}</span>')
# html = df_activities_selected_columns.to_html(escape=False, index=False)
# st.markdown(html, unsafe_allow_html=True)
