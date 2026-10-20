"""Interactive evidence-linked dependency graph for extracted AUTOSAR relationships."""
import networkx as nx
import plotly.graph_objects as go


def build_dependency_graph(relationships):
    """Create an interactive Plotly graph; nodes/edges come from source-backed rows only."""
    graph = nx.DiGraph()
    for _, item in relationships.iterrows():
        source, target = str(item['Source']), str(item['Target'])
        graph.add_edge(source, target, label=str(item['Relationship']), page=int(item['PDF page']))
    if not graph.nodes:
        return None
    positions = nx.spring_layout(graph, seed=42, k=1.7, iterations=90)
    line_x, line_y = [], []
    for source, target in graph.edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        line_x.extend([x1, x2, None])
        line_y.extend([y1, y2, None])
    lines = go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(width=2, color='#7186a4'), hoverinfo='skip', showlegend=False)
    node_x, node_y, node_labels, hover = [], [], [], []
    for node in graph.nodes:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        connected = [f"{graph[node][t]['label']} → {t} (PDF p. {graph[node][t]['page']})" for t in graph.successors(node)]
        hover.append('<b>' + node + '</b><br>' + '<br>'.join(connected))
    nodes = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_labels,
        textposition='top center', textfont=dict(size=11),
        marker=dict(size=21, color='#4fa4ef', line=dict(color='#ffffff', width=1)),
        customdata=hover, hovertemplate='%{customdata}<extra></extra>', showlegend=False)
    fig = go.Figure(data=[lines, nodes])
    fig.update_layout(height=570, margin=dict(l=50, r=50, t=35, b=35),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Evidence-backed architecture relationships', font=dict(size=16)))
    return fig
