import pandapower as pp
import pandapower.networks as ppnets
import pandapower.plotting as ppplot
import pandapower.topology as pptop

import numpy as np

from pandapower.plotting.generic_geodata import create_generic_coordinates

import pandas as pd

def in_ipynb():
    """
    :return:
    an auxiliary function which checks if located in an jupyter-notebook or not
    """
    import __main__ as main
    return not hasattr(main, '__file__')



import plotly.plotly as pltly
from plotly.graph_objs import Figure, Data, Layout, Scatter, Marker, XAxis, YAxis, Line


def seaborn_to_plotly_palette( scl ):
    ''' converts a seaborn color palette to a plotly colorscale '''
    return [ [ float(i)/float(len(scl)-1), 'rgb'+str((scl[i][0]*255, scl[i][1]*255, scl[i][2]*255)) ] \
            for i in range(len(scl)) ]

def seaborn_to_plotly_color( scl ):
    ''' converts a seaborn color to a plotly color '''
    return 'rgb'+str((scl[0]*255, scl[1]*255, scl[2]*255))

try:
    import pplog as logging
except:
    import logging

try:
    import seaborn
    colors = seaborn_to_plotly_palette(seaborn.color_palette())
    color_yellow = seaborn_to_plotly_color(seaborn.xkcd_palette(["amber"])[0])
except:
    colors = ["b", "g", "r", "c", "y"]

logger = logging.getLogger(__name__)

def create_bus_trace(net, buses=None, size=5, marker="circle", patch_type="circle", colors=None,
                          cmap=None, norm=None, infofunc=None, picker=False, **kwargs):

    bus_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name='buses',
                        marker=Marker(color=colors, size=size, symbol=marker))
    # all the bus coordinates need to be positive in plotly
    # TODO use here copy() in order not to change net.bus_geodata
    bus_geodata = net.bus_geodata
    if (net.bus_geodata.x < 0).any():
        bus_geodata['x'] = bus_geodata.x + abs(bus_geodata.x.min())

    if (net.bus_geodata.y < 0).any():
        bus_geodata['y'] = bus_geodata.y + abs(bus_geodata.y.min())

    if buses is not None:
        buses2plot = net.bus[net.bus.index.isin(buses)]
    else:
        buses2plot = net.bus
    buses_with_geodata = buses2plot.index.isin(bus_geodata.index)
    buses2plot = buses2plot[buses_with_geodata]

    bus_trace['x'], bus_trace['y'] = (bus_geodata.loc[buses2plot.index,'x'].tolist(),
                                      bus_geodata.loc[buses2plot.index, 'y'].tolist())

    bus_trace['text'] = buses2plot.name.tolist()

    return bus_trace

def create_line_trace(net, lines=None, use_line_geodata=True, infofunc=None, cmap=None,
                           norm=None, picker=False, z=None,
                           cbar_title="Line Loading [%]",
                      respect_switches=False, linewidth=1.0, color='grey', **kwargs):

    # initializing line trace
    line_trace = Scatter(x=[], y=[], text=[], line=Line(width=linewidth, color=color),
                         hoverinfo='none', mode='lines', name='lines')
    nogolines = set()
    if respect_switches:
        nogolines = set(net.switch.element[(net.switch.et == "l") &
                                           (net.switch.closed == 0)])
    nogolines_mask = net.line.index.isin(nogolines)

    if lines is not None:
        lines_mask = net.line.index.isin(lines)
        lines2plot = net.line[~nogolines_mask & lines_mask]
    else:
        lines2plot = net.line[~nogolines_mask]

    if use_line_geodata:
        lines_with_geodata = lines2plot.index.isin(net.line_geodata.index)
        lines2plot = lines2plot[lines_with_geodata]

        line_trace['text'] = lines2plot.name.tolist()
        for line_ind, line in lines2plot.iterrows():
            line_coords = net.line_geodata.loc[line_ind, 'coords']
            linex, liney = list(zip(*line_coords))
            line_trace['x'] += linex
            line_trace['x'] += [None]
            line_trace['y'] += liney
            line_trace['y'] += [None]
    else:
        lines_with_geodata = net.line.from_bus.isin(net.bus_geodata.index) & \
                             net.line.to_bus.isin(net.bus_geodata.index)
        lines2plot = lines2plot[lines_with_geodata]
        # getting x and y values from bus_geodata for from and to side of each line
        for xy in ['x', 'y']:
            from_bus = net.bus_geodata.loc[lines2plot.from_bus, xy].tolist()
            to_bus = net.bus_geodata.loc[lines2plot.to_bus, xy].tolist()
            None_list = [None] * len(from_bus)
            line_trace[xy] = np.array([from_bus, to_bus, None_list]).T.flatten()

    return line_trace


def create_trafo_trace(net, trafos=None, trafo_color = 'green', trafo_width = 5, **kwargs):

    trafo_trace = Scatter(x=[], y=[], text=[], line=Line(width=trafo_width, color=trafo_color),
                         hoverinfo='text', mode='lines', name='trafos')
    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) &\
                               net.trafo.lv_bus.isin(net.bus_geodata.index)

    if trafos is not None:
        trafos2create = net.trafo.index.isin(trafos)
        tarfo2plot = net.trafo[trafo_buses_with_geodata & trafos2create]
    else:
        tarfo2plot = net.trafo[trafo_buses_with_geodata]

    trafo_trace['text'] = tarfo2plot.name.tolist()
    for trafo_ind, trafo in tarfo2plot.iterrows():
        trafo_trace['x'].append(net.bus_geodata.loc[trafo.hv_bus, 'x'])
        trafo_trace['x'].append(net.bus_geodata.loc[trafo.lv_bus, 'x'])
        trafo_trace['x'].append(None)

        trafo_trace['y'].append(net.bus_geodata.loc[trafo.hv_bus, 'y'])
        trafo_trace['y'].append(net.bus_geodata.loc[trafo.lv_bus, 'y'])
        trafo_trace['y'].append(None)

    return trafo_trace


def draw_traces(traces):
    # setting Figure object
    fig = Figure(data=Data(traces),   # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=True,
                     width=650,
                     height=650,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text="",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    # check if called from ipynb or not in order to consider appropriate plot function
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig)


def simple_plotly(net=None, respect_switches=False, line_width=1.0, bus_size=10, ext_grid_size=20.0,
                bus_color=colors[0][1], line_color='grey', trafo_color='green', ext_grid_color=color_yellow):

    if net is None:
        import pandapower.networks as nw
        logger.warning("No pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()


    # create geocoord if none are available
    # TODO remove this if not necessary:
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x","y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)

    # ----- Buses ------
    # initializating bus trace
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, colors=bus_color)


    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    use_line_geodata = False if len(net.line_geodata) == 0 else True

    line_trace = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                   color=line_color, linewidths=line_width,
                                   use_line_geodata=use_line_geodata)


    # ----- Trafos ------
    trafo_trace = create_trafo_trace(net, trafo_color=trafo_color, trafo_width=line_width*5)


    # ----- Ext grid ------
    # get external grid from create_bus_trace
    ext_grid_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name='external grid',
                        marker=Marker(color=ext_grid_color, symbol='square', size=ext_grid_size))
    ext_grid_buses_with_geodata = net.ext_grid.bus.isin(net.bus_geodata.index)
    ext_grid_trace['x'] = net.bus_geodata.loc[net.ext_grid.loc[ext_grid_buses_with_geodata, 'bus'], 'x'].tolist()
    ext_grid_trace['y'] = net.bus_geodata.loc[net.ext_grid.loc[ext_grid_buses_with_geodata, 'bus'], 'y'].tolist()
    ext_grid_trace['text'] = net.ext_grid.loc[ext_grid_buses_with_geodata, 'name']


    draw_traces([line_trace, bus_trace, trafo_trace, ext_grid_trace])






def pp_plotly(net=None, respect_switches=False, line_width=1.0, bus_size=10., ext_grid_size=20.0,
                bus_color=colors[0][1], line_color='grey', trafo_color='green', ext_grid_color=color_yellow):

    if net is None:
        import pandapower.networks as nw
        logger.warning("No pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()


    # create geocoord if none are available
    # TODO remove this if not necessary:
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x","y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)

    # ----- Buses ------
    # initializating bus trace
    bus_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name='buses',
                        marker=Marker(color=bus_color, size=bus_size))
    # all the bus coordinates need to be positive in plotly
    # TODO use here copy() in order not to change net.bus_geodata
    bus_geodata = net.bus_geodata
    if (net.bus_geodata.x < 0).any():
        bus_geodata['x'] = bus_geodata.x + abs(bus_geodata.x.min())

    if (net.bus_geodata.y < 0).any():
        bus_geodata['y'] = bus_geodata.y + abs(bus_geodata.y.min())

    bus_trace['x'], bus_trace['y'] = bus_geodata.x.tolist(), bus_geodata.y.tolist()
    bus_trace['text'] = net.bus.name.tolist()


    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    use_line_geodata = False if len(net.line_geodata) == 0 else True
    # initializing line trace
    line_trace = Scatter(x=[], y=[], text=[], line=Line(width=line_width, color=line_color),
                            hoverinfo='none', mode='lines', name='lines')
    nogolines = set()
    if respect_switches:
        nogolines = set(net.switch.element[(net.switch.et == "l") &
                                           (net.switch.closed == 0)])
    nogolines_mask = net.line.index.isin(nogolines)

    if use_line_geodata:
        lines_with_geodata = net.line.index.isin(net.line_geodata.index)
        lines2plot = net.line[~nogolines_mask & lines_with_geodata]

        line_trace['text'] = lines2plot.name.tolist()
        for line_ind, line in lines2plot.iterrows():
            line_coords = net.line_geodata.loc[line_ind, 'coords']
            linex, liney = list(zip(*line_coords))
            line_trace['x'] += linex
            line_trace['x'] += [None]
            line_trace['y'] += liney
            line_trace['y'] += [None]
    else:
        lines_with_geodata = net.line.from_bus.isin(bus_geodata.index) &\
                               net.line.to_bus.isin(bus_geodata.index)
        lines2plot = net.line[~nogolines_mask & lines_with_geodata]
        # getting x and y values from bus_geodata for from and to side of each line
        for xy in ['x', 'y']:
            from_bus = bus_geodata.loc[lines2plot.from_bus, xy].tolist()
            to_bus = bus_geodata.loc[lines2plot.to_bus, xy].tolist()
            None_list = [None] * len(from_bus)
            line_trace[xy] = np.array([from_bus, to_bus, None_list]).T.flatten()


    # ----- Trafos ------
    trafo_trace = Scatter(x=[], y=[], text=[], line=Line(width=line_width*5, color=trafo_color),
                         hoverinfo='text', mode='lines', name='trafos')
    trafo_buses_with_geodata = net.trafo.hv_bus.isin(bus_geodata.index) &\
                               net.trafo.lv_bus.isin(bus_geodata.index)
    tarfo2plot = net.trafo[trafo_buses_with_geodata]
    line_trace['text'] = tarfo2plot.name.tolist()
    for trafo_ind, trafo in tarfo2plot.iterrows():
        trafo_trace['x'].append(bus_geodata.loc[trafo.hv_bus, 'x'])
        trafo_trace['x'].append(bus_geodata.loc[trafo.lv_bus, 'x'])
        trafo_trace['x'].append(None)

        trafo_trace['y'].append(bus_geodata.loc[trafo.hv_bus, 'y'])
        trafo_trace['y'].append(bus_geodata.loc[trafo.lv_bus, 'y'])
        trafo_trace['y'].append(None)


    # ----- Ext grid ------
    ext_grid_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name='external grid',
                        marker=Marker(color=ext_grid_color, symbol='square', size=ext_grid_size))
    ext_grid_buses_with_geodata = net.ext_grid.bus.isin(bus_geodata.index)
    ext_grid_trace['x'] = bus_geodata.loc[net.ext_grid.loc[ext_grid_buses_with_geodata, 'bus'], 'x'].tolist()
    ext_grid_trace['y'] = bus_geodata.loc[net.ext_grid.loc[ext_grid_buses_with_geodata, 'bus'], 'y'].tolist()
    ext_grid_trace['text'] = net.ext_grid.loc[ext_grid_buses_with_geodata, 'name']

    # setting Figure object
    fig = Figure(data=Data([line_trace, bus_trace, trafo_trace, ext_grid_trace]),   # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=True,
                     width=650,
                     height=650,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text="",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    # check if called from ipynb or not in order to consider appropriate plot function
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig)


# net = ppnets.example_simple()
net = ppnets.mv_oberrhein()
del net.bus_geodata #delete the geocoordinates
del net.line_geodata
create_generic_coordinates(net, respect_switches=True)
# ppplot.simple_plot(net)

# G = pptop.create_nxgraph(net)

simple_plotly(net, respect_switches=True)

# ppplot.simple_plot(net)