




import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import sklearn.datasets as datasets
import numpy as np
from kmeans import KMeans  # Import your KMeans implementation

# Initialize the Dash app
app = dash.Dash(__name__)

# Global data and KMeans instance
X = None
kmeans = None
manual_centroids = []  # Store manual centroids selected by the user

# Define the layout of the web application
app.layout = html.Div([
    html.H1("KMeans Clustering Algorithm", style={'textAlign': 'center', 'color': '#2F4F4F'}),
    
    # Number of Clusters (k) input
    html.Div([
        html.Label("Number of Clusters (k):"),
        dcc.Input(id='n-clusters', type='number', value=3, min=1, style={'width': '100px'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Dropdown for Initialization Method
    html.Div([
        html.Label("Initialization Method:"),
        dcc.Dropdown(
            id='init-method',
            options=[
                {'label': 'Random', 'value': 'random'},
                {'label': 'Farthest First', 'value': 'farthest_first'},
                {'label': 'KMeans++', 'value': 'kmeans++'},
                {'label': 'Manual', 'value': 'manual'}
            ],
            value='random',  # Default value
            clearable=False,
            style={'width': '200px'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Instructions for manual initialization
    html.Div(id='manual-instructions', style={'textAlign': 'center', 'color': 'red', 'marginBottom': '20px'}),

    # Buttons for control
    html.Div([
        html.Button('Run To Convergence', id='step-button', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Steps Through KMeans', id='run-button', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Generate New Dataset', id='generate-button', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Reset Algorithm', id='reset-button', n_clicks=0),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Graph to display clustering
    dcc.Graph(id='kmeans-graph'),
    
], style={'fontFamily': 'Arial', 'backgroundColor': '#F0F8FF', 'padding': '20px'})

# Callback to show instructions if Manual Initialization is selected
@app.callback(
    Output('manual-instructions', 'children'),
    Input('init-method', 'value')
)
def display_manual_instructions(init_method):
    if init_method == 'manual':
        return 'Click on the graph to select centroids. You can select up to the specified number of clusters.'
    return ''

# Combined Callback to Generate Dataset, Run KMeans, and Reset Graph
@app.callback(
    Output('kmeans-graph', 'figure'),
    [Input('generate-button', 'n_clicks'),
     Input('step-button', 'n_clicks'),
     Input('run-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('kmeans-graph', 'clickData')],
    [State('n-clusters', 'value'),
     State('init-method', 'value')]
)
def update_graph(gen_clicks, step_clicks, run_clicks, reset_clicks, click_data, n_clusters, init_method):
    global X, kmeans, manual_centroids
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'None'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'generate-button':
        # Generate a dataset of 300 random points
        centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
        X, _ = datasets.make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=0)
        manual_centroids = []  # Reset manual centroids
        return go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers')])

    elif button_id == 'kmeans-graph' and init_method == 'manual' and click_data:
        # Add centroids based on user's click, up to n_clusters
        if len(manual_centroids) < n_clusters:
            x = click_data['points'][0]['x']
            y = click_data['points'][0]['y']
            manual_centroids.append([x, y])
        
        # Plot the centroids
        fig = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers')])
        if manual_centroids:
            centroids_np = np.array(manual_centroids)
            fig.add_trace(go.Scatter(x=centroids_np[:, 0], y=centroids_np[:, 1], mode='markers',
                                     marker=dict(symbol='x', size=12, color='red'),
                                     name='Centroids'))
        return fig

    elif button_id == 'step-button':
        # Step through KMeans
        if kmeans is None:
            kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
            if init_method == 'manual' and manual_centroids:
                kmeans.fit(X, manual_centroids=manual_centroids)
            else:
                kmeans.fit(X)

        kmeans.step(X)  # Perform one step
        centroids = kmeans.centroids
        fig = plot_clusters_with_centroids(X, kmeans.assignment, centroids)
        return fig

    elif button_id == 'run-button':
        # Run to Convergence
        if kmeans is None:
            kmeans = KMeans(n_clusters=n_clusters, init_method=init_method)
            if init_method == 'manual' and manual_centroids:
                kmeans.fit(X, manual_centroids=manual_centroids)
            else:
                kmeans.fit(X)

        kmeans.fit(X)
        centroids = kmeans.centroids
        fig = plot_clusters_with_centroids(X, kmeans.assignment, centroids)
        return fig

    elif button_id == 'reset-button':
        # Reset the graph and algorithm
        kmeans = None
        manual_centroids = []
        return go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers')])

    return go.Figure()

# Helper function to plot data points with centroids
def plot_clusters_with_centroids(X, assignment, centroids):
    fig = go.Figure()

    # Plot data points
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                             marker=dict(color=assignment, showscale=True),
                             name='Data Points'))

    # Plot centroids
    fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                             marker=dict(symbol='x', size=12, color='red'),
                             name='Centroids'))

    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True, port=3000)


