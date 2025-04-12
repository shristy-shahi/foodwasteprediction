import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import plotly.graph_objs as go

# Load models
rf_model = joblib.load("rfp_model.pkl")
knn_model = joblib.load("knnp_model.pkl")

# Feature list
features = [
    "area", "density_sq_km", "2020_population", "growth_rate", "total_food_waste_tonnes",
    "household_tonnes", "retail_tonnes", "food_service_tonnes", "household_kg_per_capita",
    "retail_kg_per_capita", "food_service_kg_per_capita", "combined_kg_per_capita",
    "2021_last_updated"
]

# Default values for user
default_values = {
    "area": 1000000,
    "density_sq_km": 40,
    "2020_population": 95000000,
    "growth_rate": 2.7,
    "total_food_waste_tonnes": 7000000,
    "household_tonnes": 3000000,
    "retail_tonnes": 2000000,
    "food_service_tonnes": 2000000,
    "household_kg_per_capita": 31,
    "retail_kg_per_capita": 21,
    "food_service_kg_per_capita": 21,
    "combined_kg_per_capita": 73,
    "2021_last_updated": 2021
}

regions = [
    'Southern Asia', 'Southern Europe', 'Northern Africa', 'Sub-Saharan Africa',
    'Latin America and the Caribbean', 'Western Asia', 'Australia and New Zealand',
    'Western Europe', 'Eastern Europe', 'Northern America', 'South-eastern Asia',
    'Eastern Asia', 'Northern Europe', 'Melanesia', 'Polynesia', 'Micronesia', 'Central Asia'
]

app = dash.Dash(__name__)

# Custom style for green outline and layout
input_style = {
    'border': '2px solid #28a745',
    'padding': '5px',
    'borderRadius': '5px',
    'width': '100%',
    'marginBottom': '10px'
}

def create_input_row(feature, label):
    return html.Div([
        html.Label(label, style={'fontWeight': 'bold'}),
        dcc.Input(id=feature, type='number', value=default_values[feature], style=input_style)
    ])

# Layout
app.layout = html.Div([
    html.H1("🌍 Food Waste Per Capita Predictor", style={"textAlign": "center", "color": "#2c3e50"}),

    html.Label("Region", style={'fontWeight': 'bold'}),
    dcc.Dropdown(
        id="region",
        options=[{"label": r, "value": r} for r in regions],
        value="Sub-Saharan Africa",
        style=input_style
    ),

    html.Div([
        html.Div([create_input_row(f, f.replace('_', ' ').capitalize()) for i, f in enumerate(features[:len(features)//2])], style={"width": "48%", "display": "inline-block", "paddingRight": "2%"}),
        html.Div([create_input_row(f, f.replace('_', ' ').capitalize()) for i, f in enumerate(features[len(features)//2:])], style={"width": "48%", "display": "inline-block"}),
    ]),

    html.Button("Predict", id="predict", n_clicks=0, style={
        "backgroundColor": "#28a745", "color": "white", "padding": "10px 20px", "marginTop": "15px",
        "border": "none", "borderRadius": "5px", "fontWeight": "bold"
    }),

    html.Div(id="output", style={"marginTop": "20px", "fontSize": "18px", "color": "#333"}),

    dcc.Graph(id="prediction_graph", style={"marginTop": "30px"})
])

# Callback
@app.callback(
    [Output("output", "children"), Output("prediction_graph", "figure")],
    [Input("predict", "n_clicks")],
    [State("region", "value")] + [State(f, "value") for f in features]
)
def predict(n_clicks, region, *vals):
    if n_clicks == 0:
        return "", go.Figure()

    try:
        input_data = pd.DataFrame([list(vals)], columns=features)
        input_data["region"] = region

        rf_pred = rf_model.predict(input_data)[0]
        knn_pred = knn_model.predict(input_data)[0]

        population = input_data["2020_population"].values[0]
        rf_tonnes = (rf_pred * population) / 1000
        knn_tonnes = (knn_pred * population) / 1000

        output = html.Div([
            html.Div([
                html.Span("✅ Random Forest Prediction: ", style={"fontWeight": "bold", "color": "#2c3e50"}),
                f"{rf_pred:.2f} kg per capita → {rf_tonnes:.2f} tonnes"
            ]),
            html.Div([
                html.Span("✅ KNN Prediction: ", style={"fontWeight": "bold", "color": "#2c3e50"}),
                f"{knn_pred:.2f} kg per capita → {knn_tonnes:.2f} tonnes"
            ])
        ])

        fig = go.Figure(data=[
            go.Bar(name="Random Forest", x=["Prediction"], y=[rf_pred], marker_color="green"),
            go.Bar(name="KNN", x=["Prediction"], y=[knn_pred], marker_color="orange")
        ])
        fig.update_layout(title="📊 Model Predictions (kg per capita)", yaxis_title="kg per capita", barmode='group')

        return output, fig

    except Exception as e:
        return f"Prediction failed: {str(e)}", go.Figure()

if __name__ == "__main__":
    app.run_server(debug=True)
