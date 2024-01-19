from ansys.dpf import post
from ansys.dpf.post import examples

from ansys.dpf import core as dpf
from ansys.dpf.core import examples

# model = dpf.Model(examples.find_static_rst())
# print(model)
#======================================================================
# solution = post.load_solution(examples.find_multishells_rst)
# stress = solution.stress()
# # stress.xx.plot_contour(show_edges=False)
#======================================================================
# from ansys.mapdl import reader as pymapdl_reader
# from ansys.mapdl.reader import examples
#
# # Sample result file
# rstfile = examples.rstfile
#
# # Create result object by loading the result file
# result = pymapdl_reader.read_binary(rstfile)
#
# result.plot_nodal_solution(0)
# result.plot_nodal_solution(1)
# result.plot_nodal_solution(2)
# result.plot_nodal_solution(3)
# result.plot_nodal_solution(4)
# result.plot_nodal_solution(5)
#======================================================================

from ansys.mapdl import reader as pymapdl_reader
# from ansys.mapdl.reader import examples
# import pyansys
#
# # Sample *.cdb
# filename = examples.hexarchivefile
#
# # Read ansys archive file
# archive = pyansys.Archive(filename)
#
# # Print raw data from cdb
# # for key in archive.raw:
# #    print("%s : %s" % (key, archive.raw[key]))
# print(archive.nodes)
#
# # Create a vtk unstructured grid from the raw data and plot it
# print(dir(archive))
# grid = archive._parse_vtk(force_linear=True)
# grid.plot(color='w', show_edges=True)
#
# # write this as a vtk xml file
# grid.save('hex.vtu')
#
# # or as a vtk binary
# grid.save('hex.vtk')
#======================================================================

# from ansys.mapdl import reader as pymapdl_reader
# from ansys.mapdl.reader import examples
#
# # Sample result file
# rstfile = examples.rstfile
#
# # Create result object by loading the result file
# result = pymapdl_reader.read_binary(rstfile)
#
# # Beam natural frequencies
# freqs = result.time_values
#
# # print(freqs)
#
# nnum, disp = result.nodal_solution(0)
# print("nnum=", nnum)
# print("disp=", disp)
# Plot the displacement of Mode 0 in the x direction
# result.plot_nodal_solution(0, 'x', label='Displacement')
# cpos=None
# # result.plot_nodal_solution(0, 'x', label='Displacement', cpos=cpos,
# #                            screenshot='hexbeam_disp.png',
# #                            window_size=[800, 600], interactive=False)
# # Display node averaged stress in x direction for result 6
# # result.plot_nodal_stress(5, 'X')
# result.animate_nodal_solution(0, loop=False, movie_filename='result.gif',
#                              background='grey', displacement_factor=0.01, show_edges=True,
#                              add_text=True,
#                              n_frames=30)
#======================================================================

import pyvista
import numpy as np

# from ansys.mapdl import reader as pymapdl_reader
# from ansys.mapdl.reader import examples
#
# # Download an example shaft modal analysis result file
# shaft = examples.download_shaft_modal()
#
# print('shaft.mesh:\n', shaft.mesh)
# print('-'*79)
# print('shaft.grid:\n', shaft.grid)
#
# ## plot one
# # shaft.grid.plot(color='w', smooth_shading=True)
#
# ## plot two
# x_scalars = shaft.grid.points[:, 0]
# shaft.grid.plot(scalars=x_scalars, smooth_shading=True)
#======================================================================

# import pyvista
# import numpy as np
#
# from ansys.mapdl import reader as pymapdl_reader
# from ansys.mapdl.reader import examples
#
# pontoon = examples.download_pontoon()
# nnum, strain = pontoon.nodal_elastic_strain(0)
#
# scalars = strain[:, 0]
# scalars[:2000] = np.nan  # here, we simulate unknown values
#
# pontoon.grid.plot(scalars=scalars, show_edges=True, lighting=False)
#======================================================================
# sphinx_gallery_thumbnail_number = 6

from ansys.mapdl.reader import examples

# Download an example shaft modal analysis result file
shaft = examples.download_shaft_modal()

print(shaft.mesh)
print(shaft.mesh._grid)

## 绘制节点组件
# cpos = shaft.plot()

cpos = [(-115.35773008378118, 285.36602704380107, -393.9029392590675),
        (126.12852038381345, 0.2179228023931401, 5.236408799851887),
        (0.37246222812978824, 0.8468424028124546, 0.37964435122285495)]

# # 将节点组件绘制为线框
# shaft.plot(element_components=['SHAFT_MESH'], cpos=cpos, style='wireframe',
#            lighting=False)

# # 绘制带有边缘和蓝色的轴
# shaft.plot(show_edges=True, color='cyan')

# 绘制没有照明但有边缘和蓝色的轴
# shaft.plot(lighting=False, show_edges=True, color='cyan')

# 使用“bwr”颜色图绘制没有等高线的振型
# shaft.plot_nodal_solution(9, element_components=['SHAFT_MESH'],
#                           show_displacement=True, cmap='bwr',
#                           displacement_factor=0.3, stitle=None,
#                           overlay_wireframe=True, cpos=cpos)

## 绘制带有轮廓和默认颜色图的模式形状
shaft.plot_nodal_solution(1, element_components=['SHAFT_MESH'],
                          n_colors=10, show_displacement=True,
                          displacement_factor=1, stitle=None,
                          overlay_wireframe=True, cpos=cpos)

# ## 为轴组件的模式设置动画
# shaft.animate_nodal_solution(5, element_components='SHAFT_MESH',
#                              comp='norm', displacement_factor=1,
#                              show_edges=True, cpos=cpos,
#                              loop=False, movie_filename='demo.gif',
#                              n_frames=30)
