
import json
import numpy as np

from pyevtk.hl import polyLinesToVTK, gridToVTK



def toVTK(vtkname,rr, line=False, height=0.16, width=0.16, **kwargs):
    """Write entire coil set into a VTK file
    Args:
        vtkname (str): VTK filename.
        line (bool, optional): Save coils as polylines or surfaces. Defaults to True.
        height (float, optional): Rectangle height when expanded to a finite cross-section. Defaults to 0.1.
        width (float, optional): Rectangle width when expanded to a finite cross-section. Defaults to 0.1.
        kwargs (dict): Optional kwargs passed to "polyLinesToVTK" or "meshio.Mesh.write".
    """
    if line:
        currents = []
        groups = []
        x = []
        y = []
        z = []
        lx = []
        for icoil in list(self):
            currents.append(icoil.I)
            groups.append(icoil.group)
            x.append(icoil.x)
            y.append(icoil.y)
            z.append(icoil.z)
            lx.append(len(icoil.x))
        kwargs.setdefault("cellData", {})
        kwargs["cellData"].setdefault("I", np.array(currents))
        kwargs["cellData"].setdefault("Igroup", np.array(groups))
        polyLinesToVTK(
            vtkname,
            np.concatenate(x),
            np.concatenate(y),
            np.concatenate(z),
            np.array(lx),
            **kwargs
        )
    else:
        import meshio
        points = []
        hedrs = []
        currents = []
        groups = []
        nums = []
        start = 0
        for i, icoil in enumerate(rr):
            # example of meshio.Mesh can be found at https://github.com/nschloe/meshio
            # xx, yy, zz = [icoil].rectangle(width=width, height=height)
            # xx = np.ravel(np.transpose(xx[0:4, :]))
            # yy = np.ravel(np.transpose(yy[0:4, :]))
            # zz = np.ravel(np.transpose(zz[0:4, :]))
            xx = np.ravel(rr[:,:,0])
            yy = np.ravel(rr[:,:,1])
            zz = np.ravel(rr[:,:,2])            
            xyz = np.transpose([xx, yy, zz])
            points += xyz.tolist()
            # number of cells is npoints-1
            ncell = len(xx) // 4 - 1
            ind = np.reshape(np.arange(4 * ncell + 4) + start, (-1, 4))
            hedr = [list(ind[j, :]) + list(ind[j + 1, :]) for j in range(ncell)]
            hedrs += hedr
            currents += (1.62e6 * np.ones(ncell)).tolist()
            groups += (1 * np.ones(ncell, dtype=int)).tolist()
            nums += ((i + 1) * np.ones(ncell, dtype=int)).tolist()
            # update point index number
            start += len(xx)
        kwargs.setdefault("cell_data", {}) 
        kwargs["cell_data"].setdefault("I", [currents])# coil currents  
        kwargs["cell_data"].setdefault("group", [groups])# current groups
        kwargs["cell_data"].setdefault("index", [nums])# coil index, starting from 1
        data = meshio.Mesh(points=points, cells=[("hexahedron", hedrs)], **kwargs)
        data.write(vtkname)
    return


rr = np.load('results/plot/3d_to_vtk/rr.npy')
B = np.load('results/plot/3d_to_vtk/B.npy')
for i in range(5):
    toVTK('results/plot/3d_to_vtk/rr{}.vtk'.format(i+1), rr[i], line=False, height=0.16, width=0.16)



