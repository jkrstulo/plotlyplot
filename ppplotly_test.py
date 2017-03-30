
import pandapower.networks as ppnets
from pandapower.plotting.plotly import simple_plotly, vlevel_plotly,pf_res_plotly
import pandapower.plotting as ppplot

if __name__ == '__main__':

    # net = ppnets.example_simple()
    # net = ppnets.mv_oberrhein()
    # net = ppnets.case24_ieee_rts()
    # net = ppnets.create_cigre_network_lv()

    net = ppnets.panda_four_load_branch()
    # del net.bus_geodata #delete the geocoordinates
    # del net.line_geodata
    # create_generic_coordinates(net, respect_switches=True)
    # ppplot.simple_plot(net)

    # G = pptop.create_nxgraph(net)

    # simple_plotly(net, respect_switches=True)
    # pf_res_plotly(net)

    net.bus_geodata['x'] = ['38.91427','38.91538','38.91458','38.92239','38.93222','38.90842']
    net.bus_geodata['x'] = net.bus_geodata['x'].astype(float)
    net.bus_geodata['y'] = ['-77.02827','-77.02013','-77.03155','-77.04227','-77.02854','-77.02419']
    net.bus_geodata['y'] = net.bus_geodata['y'].astype(float)

    # net = ppnets.create_cigre_network_hv()

    # simple_plotly(net, on_map=False,respect_switches=True)
    pf_res_plotly(net, on_map=True,map_style='dark')
    # vlevel_plotly(net, on_map=True,map_style='dark')

    # ppplot.simple_plot(net)