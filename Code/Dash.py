from turtle import width
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.figure_factory as ff
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from plotly.tools import mpl_to_plotly

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP , dbc.icons.BOOTSTRAP])

Landings = pd.read_csv("./Code/Meteorite_Landings.csv")
Impacts = pd.read_csv("./Code/impacts.csv")
Orbit = pd.read_csv("./Code/orbits.csv")

# 3d scatter lat and long

Landings["mass (g)"] = Landings["mass (g)"].fillna(0)
Fallen = Landings[(Landings["fall"] == "Fell") & (Landings["mass (g)"] > 0)]
Fallen = Fallen.sort_values("year")
def plot_3d_landings(withSize=False):
    if(withSize):
        scatter_3d_no_size = px.scatter_3d(Fallen, x='year', y='reclat', z='reclong', size="mass (g)", color='recclass')
    else:
        scatter_3d_no_size = px.scatter_3d(Fallen, x='year', y='reclat', z='reclong', color='recclass')
    return scatter_3d_no_size

def plot_map(size=None,animation_frame="year", projection="orthographic"):
    Fallen["year"] = Fallen["year"].apply(int)
    # if(withSize):
    map_plot = px.scatter_geo(Fallen, lat=Fallen.reclat,
                        lon=Fallen.reclong,
                        color="recclass",
                        hover_name="name",
                        size=size,animation_frame=animation_frame, projection=projection)
    # if(animation_frame=="year"):
    map_plot.update_layout(showlegend=False)
    # else:
        # map_plot.update_layout(showlegend=True)
    return map_plot

rel_plot = px.scatter_matrix(Impacts, dimensions=["Asteroid Velocity", "Asteroid Magnitude", "Asteroid Diameter (km)"], labels={"Asteroid Velocity":"Speed", "Asteroid Magnitude": "Magnitude", "Asteroid Diameter (km)": "Size"}, color="Possible Impacts")

histogram = make_subplots(rows=2, cols=2, subplot_titles=('Possible Impacts','Asteroid Velocity', 'Asteroid Magnitude', "Asteroid Diameter (km)" ))

histogram.add_trace(go.Histogram(x=Impacts['Possible Impacts'], name="Possible Impacts"), row=1, col=1) 
histogram.add_trace(go.Histogram(x=Impacts['Asteroid Velocity'], name="Asteroid Velocity"), row=2, col=1) 
histogram.add_trace(go.Histogram(x=Impacts['Asteroid Magnitude'], name="Asteroid Magnitude"), row=1, col=2) 
histogram.add_trace(go.Histogram(x=Impacts['Asteroid Diameter (km)'], name="Asteroid Diameter (km)"), row=2, col=2) 
histogram.update_layout(showlegend=False, title_text="Distribution of astroid features", height=1000)


pd.options.mode.chained_assignment = None 
YearsPlot = Impacts[Impacts["Period End"] <= 2199 ]
YearsPlot["Start"] = YearsPlot["Period Start"].apply(str)
YearsPlot["Finish"] = YearsPlot["Period End"].apply(str)
timeline = px.timeline(YearsPlot,  x_start="Start", x_end="Finish", y="Object Name")
timeline.update_layout(title_text="Lifetime of Asteroid", height=1000)

x = Impacts.corr().columns
y = Impacts.corr().index
new = Impacts.corr().reset_index(drop=True)
corr_mat = [(Impacts.corr()[d] * 10000)//100  for d in new ]
corr_mat = np.array(corr_mat)
x= np.array(x).tolist()
y= np.array(y).tolist()
heatmap = ff.create_annotated_heatmap(corr_mat, x=x,y=y, annotation_text=corr_mat, colorscale='Viridis')

# helpers

Impacts.drop(['Maximum Torino Scale'], axis=1, inplace=True)
Features = np.array(Impacts.columns[1:]).reshape(3,3)

def drawBoxPlot(Impacts, Features):
    boxPlot = make_subplots(rows=Features.shape[0], cols=Features.shape[1])

    for iy, ix in np.ndindex(Features.shape):
        boxPlot.add_trace(go.Box(y=Impacts[Features[iy][ix]], name=Features[iy][ix]), row=iy+1, col=ix+1) 

    boxPlot.update_layout(height=1000, showlegend=False)
    return boxPlot

def removeOutliers(df, features):
    df_copy = df.copy()

    for feature in features:

        # Calculate q1, q3 and iqr
        q3 = df[feature].quantile(0.75)
        q1 = df[feature].quantile(0.25)
        iqr = q3 - q1

        # Get local minimum and maximum
        local_min = q1 - (1.5 * iqr)
        local_max = q3 + (1.5 * iqr)

        # Remove the outliers
        df_copy = df_copy[(df_copy[feature] >= local_min) & (df_copy[feature] <= local_max)]
        return df_copy

Impacts_Cleaned = removeOutliers(Impacts, Features.flatten())

def sunBurnPlot(start=0,end=9999):
    fig = px.sunburst(
        Impacts_Cleaned[(Impacts_Cleaned['Period Start'] > start) & (Impacts_Cleaned['Period End'] < end)],
        path=['Period Start','Period End', 'Object Name'], values='Asteroid Diameter (km)'
    )
    return fig

def plotElbow(df,features):
    x_cord = []
    y_cord = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k)
        model.fit(df[features])
        y_cord.append(model.inertia_)
        x_cord.append(k)
    fig = px.line(x=x_cord, y=y_cord, title="Clusers list")
    return fig

def plotDendogram(Impacts_Cleaned, Features):
    distance_matrix = linkage(Impacts_Cleaned[Features.flatten()], method='complete')

    fig = ff.create_dendrogram(distance_matrix)
    fig.update_layout(height=1000, width=1500)
    return fig


theme = "plotly"
def drawCluster(X, y, dim=2 , withSize=False):
    pca = PCA(dim)
    X_component = pca.fit_transform(X)
    x1 = X_component[:, 0]
    x2 = X_component[:, 1]

    if (dim == 2):
        if(withSize):
            fig= px.scatter(x=x1, y= x2, color=y, template=theme, size=X["Asteroid Diameter (km)"])
        else:
            fig= px.scatter(x=x1, y= x2, color=y, template=theme)
        return fig

    elif (dim == 3):
        x3 = X_component[:, 2]
        if(withSize):
            fig = px.scatter_3d(X_component, x=x1, y=x2, z=x3, color=y, size=X["Asteroid Diameter (km)"], template=theme)
        else:
            fig = px.scatter_3d(X_component, x=x1, y=x2, z=x3, color=y, template=theme)
        return fig

Impact_model = KMeans(n_clusters=3)
Impact_model_labels = Impact_model.fit_predict(Impacts_Cleaned[Features.flatten()])
centroids = Impact_model.cluster_centers_

Impacts_Cleaned["Labels"] = Impact_model_labels

def drawLabelledCluster(size=None):
    fig = px.scatter(Impacts_Cleaned, x="Maximum Palermo Scale", y="Possible Impacts", color="Labels",
                 size=size)

    return fig
def drawCustomizableLabelledCluster(x=None, y=None, z= None, size=None):
    fig = px.scatter_3d(Impacts_Cleaned, x=x, y=y, z=z, color="Labels",
                 size=size)

    return fig





collapse = html.Div(
[
dbc.Button(
    "View Past Landing Data",
    id="collapse-button",
    className="mb-3",
    color="primary",
    n_clicks=0,
),
dbc.Collapse(
    dash_table.DataTable(Landings.to_dict('records'), [{"name": i, "id": i} for i in Landings.columns], page_size=10),
    id="collapse",
    is_open=False,
),
]
)

collapse_future = html.Div(
[
dbc.Button(
    "Asteroids Currently orbiting Solar System",
    id="collapse-button_future",
    className="mb-3",
    color="primary",
    n_clicks=0,
),
dbc.Collapse(
    dash_table.DataTable(Impacts.to_dict('records'), [{"name": i, "id": i} for i in Impacts.columns], page_size=10),
    id="collapse_future",
    is_open=False,
),
]
)


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_future", "is_open"),
    [Input("collapse-button_future", "n_clicks")],
    [State("collapse_future", "is_open")],
)
def toggle_collapse_future(n, is_open):
    if n:
        return not is_open
    return is_open


tab1_content = dbc.Card(
    dbc.CardBody(
        [
html.H1('Past Landings', className="display-6"),
    html.Hr(className="my-2"),
    collapse,
    # dash_table.DataTable(Landings.to_dict('records'), [{"name": i, "id": i} for i in Landings.columns], page_size=10),
   
#     dcc.Checklist(
#     [' Show Respective to size of the asteroid'],id="size_display"
# ),
html.Div(
    [
         html.H1('3D Plot of year and coordinated landed',  className="display-6"),
    html.Hr(className="my-2"),
        dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
                # {"label": "Flat Map", "value": 2},
                # {"label": "Show Animation Animation", "value": 3},
            ],
            value=[],
            id="size_display",
            switch=True,
        ),
    ]
),
    dcc.Graph(id='3d_scatter'),
#     dcc.Checklist(
#     ['     ', '    ', '    '],id="size_display_map"
# ),
html.Div(
    [
    html.H1('Geo location of the places where Asteroids Hit in past',  className="display-6"),
    html.Hr(className="my-2"),
        dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
                {"label": "Flat Map", "value": 2},
                {"label": "Show Animation Animation", "value": 3},
            ],
            value=[3],
            id="size_display_map",
            switch=True,
        ),
    ]
),
    dcc.Graph(id='3d_map'),
        ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [html.H1('Future Impact Analysis',  className="display-6"),
        collapse_future,
    html.H1('Relation Plot',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=rel_plot),
    html.H1('Histogram',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=histogram),
    html.H1('Total life time of Asteroid',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=timeline),
    html.H1('Heat Map',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=heatmap),
    html.H1('Box Plot before cleaning the data where outliers are found',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=drawBoxPlot(Impacts=Impacts, Features=Features)),
    html.H1('Box Plot after cleaning the data where outliers are removed',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=drawBoxPlot(Impacts=Impacts_Cleaned, Features=Features)),
    html.H1('Various Asteroids Classified based on start and end year',  className="display-6"),
    html.Hr(className="my-2"),
    # dcc.Slider(min=min(Impacts_Cleaned['Period Start']), max=max(Impacts_Cleaned['Period End']), step=1, id='Year_select_sun_burn'),
    dcc.Graph(figure=sunBurnPlot() ,id="SunBurn")]
    ),
    className="mt-3",
)

tab3_content = dbc.Card(
    dbc.CardBody([
html.H1('Model Building:',  className="display-6"),
    html.H1('Decide Number of clusters',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=plotElbow(Impacts_Cleaned, Features.flatten())),
    html.H1('Dendogram Plot',  className="display-6"),
    html.Hr(className="my-2"),
    dcc.Graph(figure=plotDendogram(Impacts_Cleaned, Features.flatten())),
    html.H1('KMean Classification',  className="display-6"),
    html.Hr(className="my-2"),
    dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
            ],
            value=[],
            id="kmean_cluser_with_size",
            switch=True,
        ),
    dcc.Graph(id="kmean_cluser_plotting"),
    html.H1('KMean Classification explorable',  className="display-6"),
    html.Hr(className="my-2"),
    dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
            ],
            value=[],
            id="kmean_cluser_with_size_explorable",
            switch=True,
        ),
        dbc.Select(
            placeholder="Select the Feature for X axis...",
    id="x_explorable",
    options=[{"label":i, "value":i} for i in Features.flatten()],
),

        dbc.Select(
            placeholder="Select the Feature for Y axis...",
    id="y_explorable",
    options=[{"label":i, "value":i} for i in Features.flatten()],
),
        dbc.Select(
            placeholder="Select the Feature for Z axis...",
    id="z_explorable",
    options=[{"label":i, "value":i} for i in Features.flatten()],
),
    dcc.Graph(id="kmean_cluser_plotting_explorable"),
    html.H1('KMean Classification With 2 dimension PCA',  className="display-6"),
    html.Hr(className="my-2"),
    dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
            ],
            value=[],
            id="cluser_with_size",
            switch=True,
        ),
    dcc.Graph(id="cluser_plotting"),
    html.H1('KMean Classification With 3 dimension PCA',  className="display-6"),
    html.Hr(className="my-2"),
    dbc.Checklist(
            options=[
                {"label": "Show Respective to size of the asteroid", "value": 1},
            ],
            value=[],
            id="cluser_with_size_3d",
            switch=True,
        ),
    dcc.Graph(id="cluser_plotting_3d")
    ]
     ),
    className="mt-3",
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Past Astroid hit on Earth"),
        dbc.Tab(tab2_content, label="Future Asteroids impact on Earth"),
        dbc.Tab(tab3_content, label="Model Building"),
    ]
)



app.layout = dbc.Container([
    html.H1('Asteroid Impact on Earth', className="display-3"),
    html.Hr(className="my-3"),
    tabs
    
    

], fluid= True)

@app.callback(
    Output('kmean_cluser_plotting_explorable', 'figure'),
    Input('kmean_cluser_with_size_explorable', 'value'),
    Input('x_explorable', 'value'),
    Input('y_explorable', 'value'),
    Input('z_explorable', 'value')
    )
def kmean_update_fclassification_plot_explore(size,x,y,z):
    print(x,y,z)
    return drawCustomizableLabelledCluster(x=x,y=y,z=z,size='Asteroid Diameter (km)' if (1 in size) else None) # plot_3d_landings(withSize=(1 in size))

@app.callback(
    Output('kmean_cluser_plotting', 'figure'),
    Input('kmean_cluser_with_size', 'value'))
def kmean_update_fclassification_plot(size):
    print(size)
    return drawLabelledCluster(size='Asteroid Diameter (km)' if (1 in size) else None) # plot_3d_landings(withSize=(1 in size))


@app.callback(
    Output('cluser_plotting_3d', 'figure'),
    Input('cluser_with_size_3d', 'value'))
def update_fclassification_plot(size):
    print(size)
    return drawCluster(X=Impacts_Cleaned[Features.flatten()], y=Impacts_Cleaned["Labels"], withSize=(1 in size), dim=3) # plot_3d_landings(withSize=(1 in size))


@app.callback(
    Output('cluser_plotting', 'figure'),
    Input('cluser_with_size', 'value'))
def update_fclassification_plot(size):
    print(size)
    return drawCluster(X=Impacts_Cleaned[Features.flatten()], y=Impacts_Cleaned["Labels"], withSize=(1 in size)) # plot_3d_landings(withSize=(1 in size))

@app.callback(
    Output('3d_scatter', 'figure'),
    Input('size_display', 'value'))
def update_figure(size):
    print(size)
    return plot_3d_landings(withSize=(1 in size))


@app.callback(
    Output('3d_map', 'figure'),
    Input('size_display_map', 'value'))
def update_figure_map(selection):
    send_args = {"animation_frame": None}
    if 1 in selection: send_args["size"]=  "mass (g)"
    if 2 in selection: send_args["projection"]= "natural earth"
    if 3 in selection: send_args["animation_frame"]= "year"
    print(send_args)
    return plot_map(**send_args)

# @app.callback(
#     Output('SunBurn', 'figure')
#     # Input('Year_select_sun_burn', 'value')
#     )
# def sun_burn_plot(selection):
#     print(selection)
#     return sunBurnPlot()

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)